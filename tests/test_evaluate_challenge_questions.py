"""Tests for the Advanced RAG challenge evaluation harness
(scripts/evaluate_challenge_questions.py).

Fully offline: no real Gemini client is ever constructed or invoked here.
Generation is exercised via a fake client (FakeGenerationClient) that
records every call and can be scripted to raise on a specific call,
mirroring the project's existing FakeChatService pattern (see
tests/test_rag_service.py). No network call happens anywhere in this file.
"""

import json
from types import SimpleNamespace

import pytest

from llm.prompts import NO_ANSWER_FALLBACK
from retrieval.retriever import RetrievedChunk
from scripts.evaluate_challenge_questions import (
    DOCUMENTS,
    ERROR,
    FAIL,
    NEEDS_REVIEW,
    PASS,
    ChallengeQuestion,
    EvaluationCriteria,
    _check_generation,
    _evaluate_answer,
    _evaluate_generic,
    _evaluate_no_answer,
    _write_result,
)


def _make_chunk(page: int, text: str = "Some retrieved context.") -> RetrievedChunk:
    return RetrievedChunk(
        page_content=text,
        distance=0.1,
        source="test.pdf",
        file_name="test.pdf",
        file_type="pdf",
        page=page,
        chunk_index=0,
        chunk_id="chunk-0",
        document_hash="fake-hash",
    )


class FakeRetriever:
    """Stands in for DocumentRetriever -- never touches Chroma or an
    embedding model. Confirms _check_generation only needs the retriever
    interface, never a real (or rebuilt) vector store."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, query, top_k=5, document_hash=None):
        self.calls.append(query)
        return self.chunks


class FakeGenerationClient:
    """Scripted chat client -- returns answers[i] for the i-th invoke()
    call, or raises on the call index given by fail_at. Records every call
    so a test can assert exactly how many attempts were made (no retry,
    no extra calls after a stop)."""

    def __init__(self, answers=None, fail_at=None, exception=None):
        self.answers = answers or []
        self.fail_at = fail_at
        self.exception = exception or RuntimeError("simulated generation failure")
        self.calls = []

    def invoke(self, messages):
        call_index = len(self.calls)
        self.calls.append(messages)
        if self.fail_at is not None and call_index == self.fail_at:
            raise self.exception
        return SimpleNamespace(text=self.answers[call_index])


# --- EvaluationCriteria / _evaluate_generic ---------------------------------


def test_evaluate_generic_pass_on_required_fact():
    criteria = EvaluationCriteria(required_any=["166"], wrong_fact_any=["1,133"])
    outcome, reason = _evaluate_generic("GPO staff visited 166 depository libraries.", criteria)
    assert outcome == PASS


def test_evaluate_generic_fail_on_wrong_fact_without_required():
    criteria = EvaluationCriteria(required_any=["166"], wrong_fact_any=["1,133", "1133"])
    outcome, reason = _evaluate_generic(
        "GPO reports 1,133 depository libraries nationwide.", criteria
    )
    assert outcome == FAIL


def test_evaluate_generic_wrong_fact_alongside_required_is_not_a_fail():
    # Citing the wrong fact ALONGSIDE the correct one (e.g. for contrast)
    # must not be penalized -- only wrong_fact WITHOUT the required fact.
    criteria = EvaluationCriteria(required_any=["166"], wrong_fact_any=["1,133"])
    outcome, _ = _evaluate_generic(
        "GPO staff visited 166 libraries in FY2018; nationwide there are 1,133.", criteria
    )
    assert outcome == PASS


def test_evaluate_generic_needs_review_when_nothing_matches():
    criteria = EvaluationCriteria(required_any=["166"], wrong_fact_any=["1,133"])
    outcome, _ = _evaluate_generic("I'm not sure about the exact number.", criteria)
    assert outcome == NEEDS_REVIEW


def test_evaluate_generic_allow_auto_pass_false_caps_at_needs_review():
    criteria = EvaluationCriteria(required_any=["south asia"], allow_auto_pass=False)
    outcome, reason = _evaluate_generic("South Asia's growth outlook is improving.", criteria)
    assert outcome == NEEDS_REVIEW
    assert "not precise enough" in reason


def test_evaluate_generic_required_all_must_all_be_present():
    criteria = EvaluationCriteria(required_all=["govinfo", "december", "2018"])
    partial = _evaluate_generic("govinfo replaced FDsys.", criteria)[0]
    complete = _evaluate_generic(
        "govinfo replaced FDsys, fully deployed by December 2018.", criteria
    )[0]
    assert partial == NEEDS_REVIEW
    assert complete == PASS


# --- _evaluate_no_answer -----------------------------------------------------


def test_evaluate_no_answer_exact_match_passes():
    outcome, reason = _evaluate_no_answer(NO_ANSWER_FALLBACK)
    assert outcome == PASS
    assert "Exact match" in reason


def test_evaluate_no_answer_normalized_match_passes_with_distinct_reason():
    outcome, reason = _evaluate_no_answer(f'"{NO_ANSWER_FALLBACK.rstrip(".")}."'.upper())
    assert outcome == PASS
    assert "NOT an exact match" in reason


def test_evaluate_no_answer_fails_on_page10_transportation_confusion():
    outcome, reason = _evaluate_no_answer(
        "Transportation contributes 5% to the emissions reduction."
    )
    assert outcome == FAIL
    assert "page-10" in reason


def test_evaluate_no_answer_needs_review_on_unrelated_text():
    outcome, _ = _evaluate_no_answer("The iMac is a popular computer.")
    assert outcome == NEEDS_REVIEW


# --- _evaluate_answer (cross-cutting fallback-on-answerable rule) -----------


def test_evaluate_answer_fails_when_answerable_question_abstains():
    question = ChallengeQuestion(
        question="How much less energy does iMac use?",
        expected_answer_hint="58 percent less",
        expected_page=7,
        criteria=EvaluationCriteria(required_any=["58"]),
    )
    outcome, reason = _evaluate_answer(question, NO_ANSWER_FALLBACK)
    assert outcome == FAIL
    assert "answerable" in reason.lower()


def test_evaluate_answer_routes_no_answer_questions_to_dedicated_evaluator():
    question = ChallengeQuestion(
        question="Transportation contribution to the 99kg reduction?",
        expected_answer_hint="not answerable",
        expected_page=2,
        is_no_answer_check=True,
        criteria=EvaluationCriteria(),
    )
    outcome, _ = _evaluate_answer(question, NO_ANSWER_FALLBACK)
    assert outcome == PASS


# --- All 10 real question definitions, sanity-checked against plausible correct answers ---


def _all_questions():
    for spec in DOCUMENTS:
        for question in spec.questions:
            yield spec.key, question


@pytest.mark.parametrize("doc_key,question", list(_all_questions()), ids=lambda v: str(v)[:40])
def test_every_question_has_criteria_or_is_no_answer_check(doc_key, question):
    assert question.is_no_answer_check or (
        question.criteria.required_all or question.criteria.required_any
    )


def test_gpo_q1_known_wrong_fact_1133_is_caught():
    spec = next(s for s in DOCUMENTS if s.key == "gpo")
    question = spec.questions[0]
    outcome, _ = _evaluate_answer(
        question, "There are 1,133 depository libraries nationwide."
    )
    assert outcome == FAIL


def test_imac_q2_known_wrong_configuration_391_is_caught():
    spec = next(s for s in DOCUMENTS if s.key == "imac")
    question = spec.questions[1]
    outcome, _ = _evaluate_answer(
        question, "The total carbon footprint is 391 kg CO2e."
    )
    assert outcome == FAIL


def test_imac_q3_stand_material_confused_with_packaging_wood_is_caught():
    spec = next(s for s in DOCUMENTS if s.key == "imac")
    question = spec.questions[2]
    outcome, _ = _evaluate_answer(
        question, "The stand is made from 100% recycled wood fibers."
    )
    assert outcome == FAIL


def test_imac_q4_is_the_no_answer_check_on_page_2_not_page_10():
    spec = next(s for s in DOCUMENTS if s.key == "imac")
    no_answer_question = spec.questions[3]
    carbon_question = spec.questions[1]
    assert no_answer_question.is_no_answer_check is True
    assert no_answer_question.expected_page == 2
    assert carbon_question.is_no_answer_check is False
    assert carbon_question.expected_page == 10
    # The 5%-confusion pattern must fail the no-answer question...
    assert _evaluate_answer(no_answer_question, "Transportation is 5% of the total.")[0] == FAIL
    # ...but is a legitimate PASS signal for the *different*, page-10 question.
    assert _evaluate_answer(carbon_question, "346 kg CO2e total; Production 50%.")[0] == PASS


# --- _write_result: incremental persistence ---------------------------------


def test_write_result_appends_and_flushes(tmp_path):
    results_file = tmp_path / "results.jsonl"
    _write_result(results_file, {"question_id": "gpo-1", "evaluation_outcome": PASS})
    # Readable immediately after the first write -- proves it was flushed,
    # not buffered until process exit.
    first_pass_lines = results_file.read_text(encoding="utf-8").splitlines()
    assert len(first_pass_lines) == 1

    _write_result(results_file, {"question_id": "gpo-2", "evaluation_outcome": FAIL})
    lines = results_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["question_id"] == "gpo-1"
    assert records[1]["question_id"] == "gpo-2"


def test_write_result_creates_parent_directory(tmp_path):
    results_file = tmp_path / "nested" / "dir" / "results.jsonl"
    _write_result(results_file, {"question_id": "x"})
    assert results_file.exists()


# --- _check_generation: stop-on-error, no retry, no skip-ahead --------------


def test_check_generation_stops_immediately_on_first_failure(tmp_path):
    spec = next(s for s in DOCUMENTS if s.key == "gpo")
    retriever = FakeRetriever([_make_chunk(page=5)])
    # Only ONE scripted answer exists; if the code attempted a 3rd invoke()
    # call (a retry of the failing question, or proceeding to question 3),
    # this would raise IndexError instead of the intended RuntimeError,
    # failing the test loudly rather than silently passing.
    client = FakeGenerationClient(
        answers=["GPO staff visited 166 depository libraries."], fail_at=1
    )
    results_file = tmp_path / "results.jsonl"

    stopped, completed = _check_generation(
        retriever, "fake-hash", spec, None, client, results_file, "test-run-1"
    )

    assert stopped is True
    assert completed == 1
    assert len(client.calls) == 2  # question 1 (succeeded), question 2 (failed) -- no more

    records = [json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["question_id"] == "gpo-1"
    assert records[0]["evaluation_outcome"] == PASS
    assert records[1]["question_id"] == "gpo-2"
    assert records[1]["evaluation_outcome"] == ERROR
    assert "generation_error" in records[1]
    assert records[1]["generation_error"] is not None
    assert records[1]["generated_answer"] is None


def test_check_generation_runs_all_questions_when_none_fail(tmp_path):
    spec = next(s for s in DOCUMENTS if s.key == "gpo")
    retriever = FakeRetriever([_make_chunk(page=5)])
    client = FakeGenerationClient(
        answers=[
            "166 depository libraries.",
            "govinfo replaced FDsys, fully deployed December 2018.",
            "GPO procured $375 million in printing and information products.",
        ]
    )
    results_file = tmp_path / "results.jsonl"

    stopped, completed = _check_generation(
        retriever, "fake-hash", spec, None, client, results_file, "test-run-2"
    )

    assert stopped is False
    assert completed == 3
    assert len(client.calls) == 3
    records = [json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert all(r["evaluation_outcome"] == PASS for r in records)


def test_check_generation_respects_question_filter(tmp_path):
    spec = next(s for s in DOCUMENTS if s.key == "gpo")
    retriever = FakeRetriever([_make_chunk(page=5)])
    client = FakeGenerationClient(answers=["govinfo, December 2018."])
    results_file = tmp_path / "results.jsonl"

    stopped, completed = _check_generation(
        retriever, "fake-hash", spec, 2, client, results_file, "test-run-3"
    )

    assert stopped is False
    assert completed == 1
    assert len(client.calls) == 1
    records = [json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines()]
    assert records[0]["question_id"] == "gpo-2"


def test_check_generation_never_rebuilds_the_store_itself(tmp_path):
    # FakeRetriever is not backed by any real (or rebuildable) vector store.
    # _check_generation succeeding with it at all proves the function only
    # consumes a pre-built retriever/document_hash -- it has no code path
    # that could re-embed the document itself.
    spec = next(s for s in DOCUMENTS if s.key == "imac")
    retriever = FakeRetriever([_make_chunk(page=7)])
    client = FakeGenerationClient(answers=["iMac uses 58 percent less energy."])
    results_file = tmp_path / "results.jsonl"

    stopped, completed = _check_generation(
        retriever, "fake-hash", spec, 1, client, results_file, "test-run-4"
    )

    assert stopped is False
    assert completed == 1
    assert retriever.calls == ["How much less energy does iMac use compared to the ENERGY STAR requirement?"]


def test_results_record_contains_no_credential_like_keys(tmp_path):
    spec = next(s for s in DOCUMENTS if s.key == "gpo")
    retriever = FakeRetriever([_make_chunk(page=5)])
    client = FakeGenerationClient(answers=["166 depository libraries."])
    results_file = tmp_path / "results.jsonl"

    _check_generation(retriever, "fake-hash", spec, 1, client, results_file, "test-run-5")

    record = json.loads(results_file.read_text(encoding="utf-8").splitlines()[0])
    forbidden_substrings = ("api_key", "apikey", "google_api_key", "secret", "token", "password")
    serialized = json.dumps(record).lower()
    for forbidden in forbidden_substrings:
        assert forbidden not in serialized
