"""Reproducible evaluation harness for the Advanced RAG challenge.

Runs the challenge brief's 9 official sample questions (3 per document --
see sample_documents/RAG-Hands-On-Use-Cases.docx) plus one additional iMac
no-answer check, against the real pipeline.

Two modes:

  --check-retrieval (default): indexes a small, deterministic chunk subset
    per document into a throwaway temp Chroma collection under the system
    temp directory (never touches production data/chroma/), runs real
    embedding-based retrieval for each question via the existing
    DocumentRetriever, and reports whether the expected page is present
    among the retrieved chunks. Costs only embedding-model API quota. The
    disposable store is built once per document and reused by
    --check-generation, if also requested, for that same document -- it is
    never re-embedded.

  --check-generation: additionally runs each retrieved context through the
    real answer-generation path (llm.prompts.build_rag_prompt +
    a single-attempt ChatGoogleGenerativeAI client) and evaluates the
    generated answer against source-grounded, per-question criteria (see
    EvaluationCriteria / _evaluate_answer below) -- PASS, FAIL, or
    NEEDS_REVIEW, never a bare "did it return text". The underlying client
    is constructed with max_retries=0, which langchain_google_genai threads
    straight into the google-genai SDK's own HttpRetryOptions(attempts=0),
    normalized there to a single, non-retried attempt
    (tenacity.stop_after_attempt(1)) -- confirmed by source inspection, not
    assumed. If any question's generation call raises, the run STOPS
    immediately (no retry, no skipping ahead to the next question or
    document); every result completed so far has already been flushed to
    the results file. This mode spends generate_content quota (free tier:
    20 requests/day) -- opt in deliberately, and prefer --document/
    --question to scope a run to only what you need.

Results (in --check-generation mode) are appended as JSON Lines to a
results file -- by default a fresh, run-ID-stamped file under
scripts/results/ so separate runs are never silently mixed together; pass
--results-file to point multiple runs at one shared file instead (each
record still carries its own run_id). Each record is written and flushed
immediately after that question completes (success OR failure), so a later
failure never discards earlier results. No API key, credential, or other
secret is ever included in a record.

Usage:
    python -m scripts.evaluate_challenge_questions
    python -m scripts.evaluate_challenge_questions --check-generation
    python -m scripts.evaluate_challenge_questions --document imac
    python -m scripts.evaluate_challenge_questions --check-generation --document imac --question 1
    python -m scripts.evaluate_challenge_questions --check-generation --results-file scripts/results/shared.jsonl
"""

import argparse
import hashlib
import json
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.chunking import chunk_documents
from ingestion.loaders import load_pdf_document
from retrieval.retriever import DocumentRetriever, RetrievedChunk
from retrieval.vector_store import ChromaVectorStore

SAMPLE_DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "sample_documents"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

PASS = "PASS"
FAIL = "FAIL"
NEEDS_REVIEW = "NEEDS_REVIEW"
ERROR = "ERROR"


@dataclass
class EvaluationCriteria:
    """Deterministic, source-grounded checks for one question's generated
    answer. Every substring below corresponds to a specific, pre-verified
    fact from the actual source document -- never a generic guess at
    plausible wording:

      required_all / required_any: substrings (case-insensitive) that
        together confirm the SPECIFIC correct fact this question asks for.
        required_all must ALL be present; required_any needs at least one
        (vacuously satisfied if empty).

      wrong_fact_any: a DIFFERENT, real, documented value from the SAME
        source document that a model could plausibly substitute by mistake
        (e.g. a different dollar figure, a different country's stat, a
        different product configuration's total). Checked only once the
        required fact is confirmed ABSENT -- citing the wrong fact
        alongside the right one is not penalized.

      allow_auto_pass: False means this question's criteria are not
        precise enough to safely auto-PASS on their own (typically a
        narrative question without one unique, pre-verified anchor
        phrase) -- required_all/required_any met then caps out at
        NEEDS_REVIEW rather than PASS, while wrong_fact_any can still
        trigger an auto-FAIL.
    """

    required_all: List[str] = field(default_factory=list)
    required_any: List[str] = field(default_factory=list)
    wrong_fact_any: List[str] = field(default_factory=list)
    allow_auto_pass: bool = True


@dataclass
class ChallengeQuestion:
    question: str
    expected_answer_hint: str
    expected_page: int
    criteria: EvaluationCriteria
    is_no_answer_check: bool = False


@dataclass
class DocumentSpec:
    key: str
    file_name: str
    # chunk_index values (from a fresh, deterministic chunk_documents() call)
    # to actually embed and search against -- a realistic subset of the real
    # document, not mock data. None means index every chunk.
    curated_chunk_indices: Optional[set] = None
    questions: List[ChallengeQuestion] = field(default_factory=list)


DOCUMENTS: List[DocumentSpec] = [
    DocumentSpec(
        key="gpo",
        file_name="GPO-Anual-report.pdf",
        curated_chunk_indices=set(range(0, 46)) | set(range(95, 111)),
        questions=[
            ChallengeQuestion(
                question="How many depository libraries did GPO staff visit in FY 2018?",
                expected_answer_hint="166 depository libraries in 25 states",
                expected_page=5,
                # "1,133" is a REAL, different stat from the same document
                # (page 15: "Depository libraries nationwide 1,133") that a
                # model could confuse with the FY2018 visit count asked for.
                criteria=EvaluationCriteria(
                    required_any=["166"], wrong_fact_any=["1,133", "1133"]
                ),
            ),
            ChallengeQuestion(
                question="What new digital system replaced FDsys, and when was it fully deployed?",
                expected_answer_hint="govinfo; FDsys retirement/replacement scheduled December 2018",
                expected_page=4,
                criteria=EvaluationCriteria(required_all=["govinfo", "december", "2018"]),
            ),
            ChallengeQuestion(
                question=(
                    "How much did GPO procure in printing and information products "
                    "in FY 2018 on behalf of federal agencies?"
                ),
                expected_answer_hint="~$375 million / $375.7 million; 84,000+ / 84,111 orders",
                expected_page=5,
                # "874"/"825" are REAL, different FY2018 financial figures
                # from the same document (total operating revenue $874.5M;
                # operating expenses $825.3M) that could be substituted.
                criteria=EvaluationCriteria(
                    required_any=["375"], wrong_fact_any=["874", "825"]
                ),
            ),
        ],
    ),
    DocumentSpec(
        key="gep",
        file_name="GEP-Jun-2026.pdf",
        curated_chunk_indices=(
            set(range(39, 44)) | set(range(318, 325)) | set(range(372, 386)) | set(range(411, 423))
        ),
        questions=[
            ChallengeQuestion(
                question="What is the growth outlook for South Asia in this report?",
                expected_answer_hint="Chapter 2 regional growth narrative for South Asia",
                expected_page=96,
                # No single precise regional-aggregate figure was
                # independently pre-verified for this narrative question,
                # so this never auto-PASSes -- only the brief's own stated
                # failure mode (discussing a DIFFERENT region instead of
                # South Asia) is checked automatically.
                criteria=EvaluationCriteria(
                    required_any=["south asia"],
                    wrong_fact_any=[
                        "sub-saharan africa",
                        "latin america and the caribbean",
                        "east asia and pacific",
                        "middle east, north africa",
                        "europe and central asia",
                    ],
                    allow_auto_pass=False,
                ),
            ),
            ChallengeQuestion(
                question="What does Chapter 3 say about sovereign debt in developing economies?",
                expected_answer_hint=(
                    "Rising EMDE debt raises interest rates/spreads; the relationship is non-linear"
                ),
                expected_page=107,
                # "non-linear" is a distinctive, directly-quoted anchor
                # phrase from the actual chapter text, not a generic guess.
                # The wrong_fact terms belong to Chapter 4's DIFFERENT
                # analytical topic (fiscal policy / commodity swings),
                # which the executive summary lists right next to this one.
                criteria=EvaluationCriteria(
                    required_all=["debt"],
                    required_any=["non-linear", "interest rate", "spread"],
                    wrong_fact_any=["commodity price swing", "sovereign wealth fund", "fiscal rule"],
                ),
            ),
            ChallengeQuestion(
                question=(
                    "According to the executive summary, what are the main risks to the "
                    "global outlook?"
                ),
                expected_answer_hint=(
                    "Conflict escalation/commodity disruption, trade-policy uncertainty, "
                    "geopolitical strains, weather shocks; AI adoption is the upside risk"
                ),
                expected_page=19,
                # "downside" plus a named risk theme are both directly
                # quoted/paraphrased from the actual Executive Summary text
                # ("Risks to the outlook remain skewed to the downside").
                criteria=EvaluationCriteria(
                    required_all=["downside"],
                    required_any=["conflict", "commodity", "trade policy", "geopolitical", "weather"],
                    wrong_fact_any=["skewed to the upside", "primarily upside"],
                ),
            ),
        ],
    ),
    DocumentSpec(
        key="imac",
        file_name="iMac_PER_Oct2024.pdf",
        curated_chunk_indices=None,  # full document -- small (29 chunks), previously confirmed safe
        questions=[
            ChallengeQuestion(
                question=(
                    "How much less energy does iMac use compared to the ENERGY STAR requirement?"
                ),
                expected_answer_hint="58 percent less",
                expected_page=7,
                criteria=EvaluationCriteria(required_any=["58"]),
            ),
            ChallengeQuestion(
                question=(
                    "What is the total carbon footprint of the iMac, and how much of that "
                    "comes from product use versus manufacturing?"
                ),
                expected_answer_hint=(
                    "346 kg CO2e total; Production 50%, Product use 45% (brief's "
                    "'manufacturing' maps to the document's 'Production' category)"
                ),
                expected_page=10,
                # "391" is the REAL total for a DIFFERENT configuration
                # (iMac four ports, 512GB) documented on the same page --
                # a plausible wrong-configuration substitution.
                criteria=EvaluationCriteria(
                    required_all=["346"], required_any=["50", "45"], wrong_fact_any=["391"]
                ),
            ),
            ChallengeQuestion(
                question="What material is the iMac stand made from, and what percentage of it is recycled?",
                expected_answer_hint="Aluminum, 100% recycled",
                expected_page=1,
                # "wood" is the REAL material of a DIFFERENT component
                # (packaging: "100% recycled or responsibly sourced wood
                # fibers") that could be confused with the stand.
                criteria=EvaluationCriteria(
                    required_all=["aluminum", "100"], wrong_fact_any=["wood"]
                ),
            ),
            ChallengeQuestion(
                question=(
                    "What transportation-specific contribution exists within the 99 kg "
                    "emissions reduction shown in the chart?"
                ),
                expected_answer_hint=(
                    "NOT ANSWERABLE -- no per-category value for the page-2 chart exists in "
                    "the source; correct behavior is the configured NO_ANSWER_FALLBACK"
                ),
                expected_page=2,
                is_no_answer_check=True,
                criteria=EvaluationCriteria(),  # unused for no-answer questions; see _evaluate_no_answer
            ),
        ],
    ),
]


class GenerationClient(Protocol):
    """Minimal interface _check_generation needs from a chat client --
    satisfied by langchain's ChatGoogleGenerativeAI, and trivially
    substitutable with a fake/mock in tests so no test ever constructs a
    real Gemini client or touches the network."""

    def invoke(self, messages: List[Any]) -> Any: ...


def _build_throwaway_store(spec: DocumentSpec, temp_root: Path) -> Tuple[ChromaVectorStore, str]:
    """Chunk the real source PDF and index the curated subset into a fresh,
    disposable Chroma collection under temp_root. Never touches production
    data/chroma/. Returns (ChromaVectorStore, document_hash). Called once
    per document per run -- both --check-retrieval and --check-generation
    reuse the same store for that document; neither re-embeds it."""
    file_path = SAMPLE_DOCUMENTS_DIR / spec.file_name
    documents = load_pdf_document(file_path)
    document_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    chunks = chunk_documents(documents, document_hash)

    if spec.curated_chunk_indices is not None:
        chunks = [c for c in chunks if c.metadata["chunk_index"] in spec.curated_chunk_indices]

    store_dir = temp_root / spec.key
    store = ChromaVectorStore(persist_directory=store_dir, collection_name=f"eval_{spec.key}")
    store.add_chunks(chunks)
    return store, document_hash


def _check_retrieval(
    retriever: DocumentRetriever,
    document_hash: str,
    spec: DocumentSpec,
    question_filter: Optional[int],
) -> bool:
    all_passed = True
    for index, q in enumerate(spec.questions, start=1):
        if question_filter is not None and index != question_filter:
            continue

        results: List[RetrievedChunk] = retriever.retrieve(
            q.question, top_k=5, document_hash=document_hash
        )
        pages = [r.page for r in results]
        found = q.expected_page in pages
        rank = pages.index(q.expected_page) + 1 if found else None

        status = "PASS" if found else "FAIL"
        all_passed = all_passed and found
        print(f"[{spec.key.upper()} Q{index}] {status} (retrieval)")
        print(f"  Question: {q.question}")
        print(f"  Expected page {q.expected_page} in top-5 retrieved pages {pages}"
              + (f" -- rank {rank}" if found else " -- NOT FOUND"))
        print(f"  Expected-answer hint: {q.expected_answer_hint}")
        print()

    return all_passed


def _evaluate_no_answer(answer: str) -> Tuple[str, str]:
    """Evaluate the iMac Transportation/99kg no-answer question.

    Exact equality against NO_ANSWER_FALLBACK is checked first and is the
    intended match: llm/prompts.py's SYSTEM_INSTRUCTIONS explicitly tell
    Gemini to "respond with EXACTLY this sentence and nothing else". A
    light normalization (surrounding quotes, trailing period, whitespace,
    case) is tried second, since real model output occasionally adds such
    incidental formatting despite the instruction -- this is documented
    here, not silently assumed, and is reported as a non-exact match when
    it is what matched.

    Explicitly FAILs (does not merely fall through to NEEDS_REVIEW) if the
    answer instead states the unrelated page-10 lifecycle Transportation
    5% figure as if it answered this page-2, 99kg-chart question -- the
    specific, previously-identified confusion risk this question exists to
    catch.
    """
    from llm.prompts import NO_ANSWER_FALLBACK

    stripped = answer.strip()
    if stripped == NO_ANSWER_FALLBACK:
        return PASS, "Exact match to the configured NO_ANSWER_FALLBACK sentence."

    normalized = stripped.strip("\"'").rstrip(".").strip().lower()
    normalized_fallback = NO_ANSWER_FALLBACK.rstrip(".").strip().lower()
    if normalized == normalized_fallback:
        return (
            PASS,
            "Matches NO_ANSWER_FALLBACK after normalizing quotes/trailing period/case "
            "-- NOT an exact match to the configured sentence; the prompt's "
            "\"exactly this sentence\" instruction was not followed verbatim.",
        )

    low = stripped.lower()
    if ("5%" in stripped or "5 percent" in low) and "transport" in low:
        return (
            FAIL,
            "Answer appears to substitute the unrelated page-10 Transportation 5% "
            "lifecycle-share figure for the page-2 chart's unavailable 99kg category "
            "breakdown -- this is the specific confusion this check exists to catch.",
        )

    return (
        NEEDS_REVIEW,
        "Answer does not match NO_ANSWER_FALLBACK and does not match the known "
        "5%-confusion pattern; manual review required to confirm it did not invent "
        "some other unsupported number.",
    )


def _evaluate_generic(answer: str, criteria: EvaluationCriteria) -> Tuple[str, str]:
    low = answer.lower()
    required_all_met = all(term.lower() in low for term in criteria.required_all)
    required_any_met = (not criteria.required_any) or any(
        term.lower() in low for term in criteria.required_any
    )
    wrong_fact_present = any(term.lower() in low for term in criteria.wrong_fact_any)

    if required_all_met and required_any_met:
        if criteria.allow_auto_pass:
            return PASS, "Required source-grounded fact(s) found in the generated answer."
        return (
            NEEDS_REVIEW,
            "Required term(s) found, but this question's criteria are not precise "
            "enough to safely auto-confirm correctness (no single, specific "
            "pre-verified anchor fact) -- manual review required.",
        )

    if wrong_fact_present:
        return (
            FAIL,
            "The expected fact is absent, but the answer references a different, "
            "documented value/region/topic from the same source instead -- likely "
            "confusion with an unrelated fact.",
        )

    return (
        NEEDS_REVIEW,
        "Neither the expected fact nor a known wrong-fact pattern was found; "
        "manual review required.",
    )


def _evaluate_answer(question: ChallengeQuestion, answer: str) -> Tuple[str, str]:
    """Dispatch to the no-answer or generic evaluator. Applies one
    cross-cutting rule first: an answerable question (is_no_answer_check is
    False) that instead returns NO_ANSWER_FALLBACK is a FAIL, not a
    NEEDS_REVIEW -- every one of these 9 questions is stated by the
    challenge brief itself to be answerable from the source."""
    if question.is_no_answer_check:
        return _evaluate_no_answer(answer)

    from llm.prompts import NO_ANSWER_FALLBACK

    if answer.strip() == NO_ANSWER_FALLBACK:
        return FAIL, "Incorrectly abstained on a question the challenge brief states is answerable."

    return _evaluate_generic(answer, question.criteria)


def _write_result(results_file: Path, record: dict) -> None:
    """Append one JSON record and flush immediately, so a later failure
    never discards records already written for this or an earlier run
    sharing the same file."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")
        f.flush()


def _check_generation(
    retriever: DocumentRetriever,
    document_hash: str,
    spec: DocumentSpec,
    question_filter: Optional[int],
    client: GenerationClient,
    results_file: Path,
    run_id: str,
) -> Tuple[bool, int]:
    """Run real generation for spec's questions (or just question_filter).

    Returns (stopped_early, completed_count). On ANY exception from
    client.invoke() for a given question, that failure is recorded (not
    retried), and the function returns immediately -- it does not attempt
    the next question in this document, and the caller must not proceed to
    the next document either.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from llm.prompts import build_rag_prompt

    completed = 0
    for index, q in enumerate(spec.questions, start=1):
        if question_filter is not None and index != question_filter:
            continue

        question_id = f"{spec.key}-{index}"
        results = retriever.retrieve(q.question, top_k=5, document_hash=document_hash)
        system_prompt, user_prompt = build_rag_prompt(q.question, results)
        timestamp = datetime.now(timezone.utc).isoformat()

        print(f"[{spec.key.upper()} Q{index}] GENERATING...")
        print(f"  Question: {q.question}")
        try:
            response = client.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            answer = getattr(response, "text", None)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Gemini returned an empty or unusable response.")
        except Exception as exc:
            print(f"  GENERATION FAILED: {type(exc).__name__}: {exc}")
            print(
                f"  STOPPING: {completed} question(s) completed successfully before this "
                "failure. No retry attempted; no further questions will be run."
            )
            _write_result(
                results_file,
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "document": spec.key,
                    "question_id": question_id,
                    "question_index": index,
                    "question": q.question,
                    "expected_answer_hint": q.expected_answer_hint,
                    "retrieved_pages": [r.page for r in results],
                    "generated_answer": None,
                    "generation_error": f"{type(exc).__name__}: {exc}",
                    "evaluation_outcome": ERROR,
                    "evaluation_reason": "Generation call failed; see generation_error.",
                },
            )
            return True, completed

        outcome, reason = _evaluate_answer(q, answer)
        print(f"  Generated answer: {answer!r}")
        print(f"  Expected-answer hint: {q.expected_answer_hint}")
        print(f"  Evaluation: {outcome} -- {reason}")
        if q.is_no_answer_check:
            print("  (This question should trigger the NO_ANSWER_FALLBACK sentence.)")
        print()

        _write_result(
            results_file,
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "document": spec.key,
                "question_id": question_id,
                "question_index": index,
                "question": q.question,
                "expected_answer_hint": q.expected_answer_hint,
                "retrieved_pages": [r.page for r in results],
                "generated_answer": answer,
                "generation_error": None,
                "evaluation_outcome": outcome,
                "evaluation_reason": reason,
            },
        )
        completed += 1

    return False, completed


def _make_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-generation",
        action="store_true",
        help="Also run real Gemini generation calls (costs generate_content quota).",
    )
    parser.add_argument(
        "--document",
        choices=[spec.key for spec in DOCUMENTS],
        help="Only evaluate this document's questions.",
    )
    parser.add_argument(
        "--question",
        type=int,
        help="Only evaluate this 1-indexed question within the selected document(s).",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help=(
            "JSONL file to append generation results to. Defaults to a fresh, "
            "run-ID-stamped file under scripts/results/ so separate runs are never "
            "silently mixed. Pass this to share one file across runs instead -- each "
            "record still carries its own run_id."
        ),
    )
    args = parser.parse_args()

    specs = [s for s in DOCUMENTS if args.document is None or s.key == args.document]
    run_id = _make_run_id()
    results_file = args.results_file or (DEFAULT_RESULTS_DIR / f"eval_{run_id}.jsonl")
    if args.check_generation:
        print(f"Run ID: {run_id}")
        print(f"Results file: {results_file}")
        print()

    # ignore_cleanup_errors: Chroma/SQLite can still hold its file handle open
    # on Windows when the directory is removed; this is a cleanup nuisance,
    # not a correctness issue, so tolerate it rather than crash the run.
    with tempfile.TemporaryDirectory(prefix="rag_eval_", ignore_cleanup_errors=True) as temp_dir:
        temp_root = Path(temp_dir)
        overall_passed = True
        generation_stopped = False
        for position, spec in enumerate(specs):
            if position > 0:
                # embed_content is rate-limited per rolling 60s window
                # (observed: 100 requests/minute, apparently counted per
                # text rather than per batch call -- GPO's 62-chunk curated
                # subset plus GEP's 38-chunk subset hit exactly this
                # boundary back-to-back). This is separate from, and far
                # more forgiving than, the generate_content DAILY quota. A
                # full 65s pause between documents keeps each one in its
                # own window; a single-document run (--document ...) never
                # pays this cost at all.
                print("(pausing 65s between documents to respect the embedding per-minute rate limit...)")
                time.sleep(65)

            store, document_hash = _build_throwaway_store(spec, temp_root)
            retriever = DocumentRetriever(vector_store=store)

            passed = _check_retrieval(retriever, document_hash, spec, args.question)
            overall_passed = overall_passed and passed

            if args.check_generation:
                # Imported here, not at module scope: constructing this
                # client must never happen in --check-retrieval-only runs.
                from langchain_google_genai import ChatGoogleGenerativeAI

                from config.settings import GEMINI_MODEL, LLM_TEMPERATURE

                # max_retries=0 -> HttpRetryOptions(attempts=0) -> normalized
                # to attempts=1 -> tenacity.stop_after_attempt(1): exactly
                # one HTTP attempt, no retries (verified by source
                # inspection of langchain_google_genai and google-genai).
                client = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL, temperature=LLM_TEMPERATURE, max_retries=0
                )
                stopped, completed = _check_generation(
                    retriever, document_hash, spec, args.question, client, results_file, run_id
                )
                if stopped:
                    generation_stopped = True
                    print(f"Generation run stopped after {completed} completed question(s) in "
                          f"document '{spec.key}'. Remaining documents were not attempted.")
                    break

        if not args.check_generation:
            print("=" * 60)
            print("RETRIEVAL CHECK:", "ALL PASSED" if overall_passed else "SOME FAILED")
            print("Generation-stage validation was NOT run (pass --check-generation to opt in).")
        elif generation_stopped:
            print("=" * 60)
            print("GENERATION CHECK: STOPPED EARLY due to a generation failure.")
            print(f"Partial results preserved in: {results_file}")
        else:
            print("=" * 60)
            print("GENERATION CHECK: completed all requested questions.")
            print(f"Results written to: {results_file}")


if __name__ == "__main__":
    main()
