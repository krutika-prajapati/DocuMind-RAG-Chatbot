"""Document chunking: split LangChain Documents into smaller, embeddable chunks.

    LangChain Documents -> CHUNKING -> Chunked Documents

Parsing responsibility stays in ingestion/loaders.py; this module only splits
already-parsed Documents and attaches chunk-level metadata. Embedding and
vector storage are implemented in later phases.
"""

import re
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, validate_chunk_config
from utils.hashing import compute_chunk_id
from utils.logging_config import get_logger

logger = get_logger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Conservative, GPO-style section/note heading detection. A line only counts
# as a heading if, once stripped, it consists of JUST the heading text -- not
# a sentence that merely mentions a note number, and not a table-of-contents
# entry (which has dot leaders before a page number). False positives are
# treated as worse than leaving section_title absent, so this is
# intentionally narrow rather than trying to recognize every heading style.
_NUMBERED_SECTION_PATTERN = re.compile(
    r"^(\d{1,2})\.\s+([A-Z][A-Za-z0-9,’'()/&\-\s]{2,80})$"
)
_LETTERED_SUBSECTION_PATTERN = re.compile(
    r"^([A-Z])\.\s+([A-Z][A-Za-z0-9,’'()/&\-\s]{2,80})$"
)
_TOC_LEADER_PATTERN = re.compile(r"\.{3,}|…")

# GEP-style heading detection, equally conservative. Each pattern requires a
# whole stripped line to consist of just the heading -- confirmed against the
# real extracted text of the June 2026 Global Economic Prospects report,
# not assumed from its conventions in the abstract:
#   - "CHAPTER <N>" (all caps) appears exactly once, standalone, on each
#     chapter's own opening page. The Table of Contents and in-prose
#     cross-references only ever use title case ("Chapter 3") or lowercase,
#     so requiring all caps excludes both.
#   - "BOX <N.N> <Title>" (all caps "BOX") appears on a box's own opening
#     page and is repeated, with a trailing "(continued)"/"( continued )"
#     marker, on every later page the box spans. In-prose cross-references
#     use lowercase "box", so requiring uppercase excludes them.
#   - The six GEP region names appear standalone at the start of their own
#     Chapter 2 section. Two of them ("Latin America and the Caribbean" and
#     "Middle East, North Africa, Afghanistan, and Pakistan") are long enough
#     to wrap across two physical lines there, confirmed to happen at that
#     exact spot and nowhere else in the document, so a two-line join is
#     checked only as a fallback when the single-line match fails.
#   - Decimal subsection numbers ("1.1", "2.3", ...) exist ONLY inside the
#     Table of Contents and figure/table captions in this document, never as
#     a real body heading, so no decimal-subsection pattern is implemented --
#     adding one would do nothing but match those captions and TOC entries.
_CHAPTER_PATTERN = re.compile(r"^CHAPTER\s+(\d{1,2})$")
_CHAPTER_PREFIX_PATTERN = re.compile(r"^(Chapter \d{1,2})(?: - .+)?$")
_BOX_PATTERN = re.compile(r"^BOX\s+(\d\.\d)\s+(.+)$")
_BOX_CONTINUED_SUFFIX_PATTERN = re.compile(r"\s*\(\s*continued\s*\)\s*$", re.IGNORECASE)
_GEP_REGION_NAMES = (
    "East Asia and Pacific",
    "Europe and Central Asia",
    "Latin America and the Caribbean",
    "Middle East, North Africa, Afghanistan, and Pakistan",
    "South Asia",
    "Sub-Saharan Africa",
)
_GEP_REGION_NAME_SET = frozenset(_GEP_REGION_NAMES)


def _chapter_prefix_of(label: Optional[str]) -> Optional[str]:
    """Extract the carried-forward "Chapter N" prefix from a GEP section
    label (e.g. "Chapter 2 - South Asia" -> "Chapter 2"), so a newly
    detected region/box heading can replace just the subsection part while
    keeping the chapter it belongs to. Returns None for GPO-style labels
    (e.g. "Note 9") or when no chapter has been seen yet.
    """
    if label is None:
        return None
    match = _CHAPTER_PREFIX_PATTERN.match(label)
    return match.group(1) if match else None


def _detect_section_heading(
    text: str, current_section: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Scan every line of `text` for a recognized heading:

      - GPO-style: a numbered section/note heading (e.g. "9. Workers'
        Compensation Liability" -> "Note 9") or a lettered sub-heading
        within the most recently seen numbered section (e.g. "I. Workers'
        Compensation Liability" -> "Note 9.I").
      - GEP-style: a "CHAPTER N" banner (-> "Chapter N"), one of the six
        GEP region names standalone at the start of its own section (->
        "Chapter N - <Region>" when a chapter is already active, else just
        "<Region>"), or a "BOX N.N <Title>" heading, including its
        "(continued)" repeats on later pages (-> "Chapter N - Box N.N -
        <Title>", likewise with no chapter prefix if none is active yet).

    Returns (chunk_label, updated_state):
      - `updated_state` is the carried-forward "current section" to pass
        into the NEXT chunk's call -- the last heading seen in `text`,
        same as before.
      - `chunk_label` is what should actually be stored on THIS chunk's own
        metadata. A single chunk can contain the tail of one section
        immediately followed by the next section's heading (e.g. a note's
        closing figures, then the next note's heading, near the end of the
        same chunk); labeling the WHOLE chunk with whichever heading merely
        happens to appear last would misrepresent a chunk that is mostly
        about the EARLIER section. Instead, whichever real section owns the
        most characters of this chunk wins -- content attributed to "no
        section yet" (plain prose/boilerplate before the first real
        heading) never outweighs an actual detected section, so a single
        real heading anywhere in the chunk still wins over surrounding
        boilerplate (preserving the common case), but between two REAL
        competing sections the one that actually owns the majority of this
        chunk is preferred.

    Returns (None, current_section) if no confident heading line is found
    anywhere in `text` -- ordinary prose that merely mentions a note number,
    chapter, region, or box (e.g. "as described in Note 1", "see chapter 3",
    "in the Middle East", "(box 1.1)") never matches, since the whole line
    (or, for the two region names long enough to wrap, the whole of two
    consecutive lines) must consist of just the heading, and a
    table-of-contents line is explicitly excluded.
    """
    raw_lines = text.splitlines(keepends=True)
    stripped_lines: List[str] = []
    line_starts: List[int] = []
    position = 0
    for raw_line in raw_lines:
        line_starts.append(position)
        position += len(raw_line)
        stripped_lines.append(raw_line.strip())
    end_position = position

    segments: List[Tuple[Optional[str], int]] = []
    active_label = current_section
    segment_start = 0

    line_count = len(stripped_lines)
    for index in range(line_count):
        line = stripped_lines[index]
        line_start = line_starts[index]
        if not line or _TOC_LEADER_PATTERN.search(line):
            continue

        new_label = None

        chapter_match = _CHAPTER_PATTERN.match(line)
        box_match = _BOX_PATTERN.match(line)
        if chapter_match:
            new_label = f"Chapter {chapter_match.group(1)}"
        elif box_match:
            title = _BOX_CONTINUED_SUFFIX_PATTERN.sub("", box_match.group(2)).strip()
            if title:
                box_label = f"Box {box_match.group(1)} - {title}"
                chapter_prefix = _chapter_prefix_of(active_label)
                new_label = f"{chapter_prefix} - {box_label}" if chapter_prefix else box_label
        elif line in _GEP_REGION_NAME_SET:
            chapter_prefix = _chapter_prefix_of(active_label)
            new_label = f"{chapter_prefix} - {line}" if chapter_prefix else line
        elif index + 1 < line_count:
            next_line = stripped_lines[index + 1]
            joined = f"{line} {next_line}".strip()
            if next_line and joined in _GEP_REGION_NAME_SET:
                chapter_prefix = _chapter_prefix_of(active_label)
                new_label = f"{chapter_prefix} - {joined}" if chapter_prefix else joined

        if new_label is None:
            numbered_match = _NUMBERED_SECTION_PATTERN.match(line)
            if numbered_match:
                new_label = f"Note {numbered_match.group(1)}"
            elif active_label is not None:
                lettered_match = _LETTERED_SUBSECTION_PATTERN.match(line)
                if lettered_match:
                    note_number = active_label.split(".")[0].removeprefix("Note ").strip()
                    if note_number.isdigit():
                        new_label = f"Note {note_number}.{lettered_match.group(1)}"

        if new_label is not None and new_label != active_label:
            segments.append((active_label, line_start - segment_start))
            active_label = new_label
            segment_start = line_start

    segments.append((active_label, end_position - segment_start))
    updated_state = active_label

    real_segments = [(label, length) for label, length in segments if label is not None]
    if not real_segments:
        return None, updated_state

    totals: Dict[str, int] = {}
    order: List[str] = []
    for label, length in real_segments:
        if label not in totals:
            totals[label] = 0
            order.append(label)
        totals[label] += length

    chunk_label = max(order, key=lambda label: totals[label])
    return chunk_label, updated_state


class NoChunksProducedError(Exception):
    """Raised when a set of documents yields zero usable chunks."""


def _source_key(document: Document) -> str:
    """Build a stable key identifying where a document's content came from."""
    source = document.metadata.get("source", "")
    page = document.metadata.get("page")
    return f"{source}:{page}" if page is not None else str(source)


def chunk_documents(
    documents: List[Document],
    document_hash: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split LangChain Documents into smaller chunks with enriched metadata.

    Each chunk retains the original document's metadata and gains
    chunk_index, chunk_id, chunk_size, and document_hash. chunk_index starts
    at 0 for this call and increases across all input documents (e.g. all
    pages of one uploaded file) — a separate call for a different document
    starts at 0 again. document_hash identifies all chunks that came from
    the same uploaded file, for later bulk deletion. Whitespace-only
    documents and chunks are skipped safely.

    Each chunk also gets an optional section_title (e.g. "Note 9",
    "Note 9.I", "Chapter 2", "Chapter 2 - South Asia", or "Chapter 1 - Box
    1.1 - How much will AI affect global growth?") whenever a confident
    heading has been seen at or before that chunk — see
    _detect_section_heading(). This state is carried across every page of
    one chunk_documents() call (a single section can span several pages),
    but never across separate calls. A chunk that contains both the tail of
    one section and the start of the next is labeled with whichever section
    owns the majority of that chunk's own content, not simply whichever
    heading appears last. The field is left absent from a chunk's metadata
    entirely when no heading has been detected yet, rather than ever being
    set to None.

    Raises ValueError if chunk_size/chunk_overlap is invalid, or
    NoChunksProducedError if no usable text is found across all documents.
    """
    validate_chunk_config(chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
    )

    chunks: List[Document] = []
    chunk_index = 0
    current_section: Optional[str] = None
    for document in documents:
        if not document.page_content or not document.page_content.strip():
            continue

        source_key = _source_key(document)
        for split_text in splitter.split_text(document.page_content):
            if not split_text.strip():
                continue

            chunk_label, current_section = _detect_section_heading(split_text, current_section)

            metadata = dict(document.metadata)
            metadata["chunk_index"] = chunk_index
            metadata["chunk_size"] = len(split_text)
            metadata["document_hash"] = document_hash
            metadata["chunk_id"] = compute_chunk_id(
                document_hash=document_hash,
                source_key=source_key,
                chunk_index=chunk_index,
                chunk_content=split_text,
            )
            if chunk_label is not None:
                metadata["section_title"] = chunk_label
            chunks.append(Document(page_content=split_text, metadata=metadata))
            chunk_index += 1

    if not chunks:
        logger.warning("No usable text found to chunk (document_hash=%s).", document_hash)
        raise NoChunksProducedError("No usable text was found to split into chunks.")

    return chunks
