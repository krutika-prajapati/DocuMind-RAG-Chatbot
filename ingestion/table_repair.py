"""Coordinate-based repair for wide numeric tables whose last column gets
disconnected from its own rows by pypdf's reading-order text extraction.

Root cause (confirmed against the real GEP-Jun-2026.pdf Chapter 2 Table 2.2
and Statistical Appendix tables): a wide table's final column is emitted to
the PDF content stream as a separate run of text-showing operations, placed
at the correct page coordinates but in a different stream order than the
rest of the table. pypdf's default extraction walks the stream in that
order, so the whole table prints first (missing its last column) and the
missing column's values print afterwards as a single unattributed list.

The fix only ever touches a page when a conservative, text-only detector
recognizes this exact signature (a standalone column-header-like token
immediately followed by a long run of bare numbers/placeholders) -- this
never fires on ordinary prose, which has been confirmed directly: applying
pypdf's own coordinate-aware "layout" extraction mode to a two-column prose
page interleaves unrelated sentences from side-by-side columns, so this
module never does anything page-wide or to non-triggering pages. When a
page is flagged, only the specific missing value for each row is located
via real PDF coordinates (pypdf's visitor_text callback) and spliced into
that row's already-correct plain-text value sequence at the ordinal
position implied by the header -- nothing else is touched, and any row
that can't be confidently resolved is left completely unchanged.
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_HEADER_YEAR_PATTERN = re.compile(r"^\d{4}[a-zA-Z]?$")
_HEADER_QUARTER_PATTERN = re.compile(r"^\d{2}Q[1-4][a-zA-Z]?$")
_BARE_NUMBER_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")
_PLACEHOLDER_PATTERN = re.compile(r"^\.{2}$")
_VALUE_PATTERN = re.compile(r"^-?\d+(\.\d+)?$|^\.{2}$")
_DECIMAL_OR_PLACEHOLDER_PATTERN = re.compile(r"^-?\d+\.\d+$|^\.{2}$")
_BARE_INTEGER_PATTERN = re.compile(r"^\d{1,2}$")

MIN_TAIL_RUN = 8
MAX_COLUMN_DISTANCE = 10.0
MIN_HEADER_TOKENS = 3

# A visitor line: (y, [(x, text), ...]) sorted by x within the line, lines
# sorted by y descending (top of page first) -- exactly what
# _collect_visitor_lines() produces from a real page, and what tests build
# directly to exercise the reconstruction logic without a real PDF.
VisitorLine = Tuple[float, List[Tuple[float, str]]]


def _scan_tail_run(lines: List[str], start_index: int) -> Tuple[int, int]:
    """Count the run of bare-number/".." lines starting at `start_index`
    (which must point just after a header-like token line), tolerating
    interspersed blank lines without counting them as entries.

    This is THE definition of "tail entry" used everywhere in this module:
    both the text-only detector (find_disconnected_header_tokens) and the
    tail-removal logic below call this exact function, so they can never
    disagree about what counts as part of a disconnected-column tail or
    where it ends. Returns (entry_count, end_index), where end_index is
    the index of the first line that breaks the run (a real word, a
    footer line, end of page, etc.) -- i.e. the tail's line range is
    exactly lines[start_index:end_index].
    """
    count = 0
    index = start_index
    while index < len(lines) and (
        _BARE_NUMBER_PATTERN.match(lines[index]) or _PLACEHOLDER_PATTERN.match(lines[index]) or not lines[index]
    ):
        if _BARE_NUMBER_PATTERN.match(lines[index]) or _PLACEHOLDER_PATTERN.match(lines[index]):
            count += 1
        index += 1
    return count, index


def find_disconnected_header_tokens(page_text: str) -> List[str]:
    """Return every header-like token (e.g. "2028f") that is immediately
    followed, in plain-text reading order, by a long run of bare-number or
    ".." placeholder lines -- the disconnected-column signature. Empty if
    the page shows no sign of this defect.
    """
    lines = [line.strip() for line in page_text.split("\n")]
    found = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _HEADER_YEAR_PATTERN.match(line):
            run, _end_index = _scan_tail_run(lines, index + 1)
            if run >= MIN_TAIL_RUN:
                found.append(line)
        index += 1
    return found


def _find_tail_line_range(
    plain_lines: List[str], token: str, search_from: int = 0
) -> Optional[Tuple[int, int, int]]:
    """Locate the next occurrence of `token` as a standalone tail-header
    line at or after `search_from`, reusing _scan_tail_run so the exact
    same entries that count for detection also count here. Returns
    (token_line_index, end_index, entry_count) for the first occurrence
    whose run still meets the detector's own MIN_TAIL_RUN threshold, or
    None if no such occurrence exists from `search_from` onward.

    `search_from` lets a page with more than one affected block search
    for each block's own tail independently, in document order, without
    ever re-matching a tail an earlier block already looked at.

    `plain_lines` may be raw (unstripped) page lines -- this strips its
    own working copy before matching, same as find_disconnected_header_
    tokens does, since _scan_tail_run's patterns require an exact match
    and would otherwise be broken by incidental trailing whitespace.
    Stripping never changes line count or order, so the returned indices
    remain valid against the original, unstripped `plain_lines`.
    """
    stripped = [line.strip() for line in plain_lines]
    for index in range(search_from, len(stripped)):
        if stripped[index] == token:
            count, end_index = _scan_tail_run(stripped, index + 1)
            if count >= MIN_TAIL_RUN:
                return index, end_index, count
    return None


def looks_like_disconnected_column_page(page_text: str) -> bool:
    """Conservative, text-only check used to decide whether a page is even
    worth the coordinate-based reconstruction pass below. Never fires on
    ordinary prose -- confirmed directly against this document's narrative
    pages, which never produce 8+ consecutive bare-number lines.
    """
    return len(find_disconnected_header_tokens(page_text)) > 0


def _collect_visitor_lines(page) -> List[VisitorLine]:
    """Group a real pypdf page's text-showing operations by their true
    (x, y) position: every operation sharing a y becomes one line, sorted
    left to right by x, lines sorted top to bottom. This is the one
    function in this module that touches a real pypdf Page object -- every
    other function operates on this plain-data structure, so the
    reconstruction logic itself is testable without a real PDF.
    """
    records: List[Tuple[float, float, str]] = []

    def visitor(text, _cm, tm, _font_dict, _font_size):
        if not text or not text.strip():
            return
        records.append((round(tm[5], 1), tm[4], text))

    page.extract_text(visitor_text=visitor)

    by_y: Dict[float, List[Tuple[float, str]]] = defaultdict(list)
    for y, x, text in records:
        by_y[y].append((x, text))

    return [(y, sorted(by_y[y], key=lambda item: item[0])) for y in sorted(by_y, reverse=True)]


def _header_columns_of(items: List[Tuple[float, str]]) -> List[Tuple[str, float]]:
    columns = []
    for x, text in items:
        for token in text.split():
            if _HEADER_YEAR_PATTERN.match(token) or _HEADER_QUARTER_PATTERN.match(token):
                columns.append((token, x))
    return sorted(columns, key=lambda c: c[1])


def _find_header_blocks(lines: List[VisitorLine]) -> List[Tuple[int, List[Tuple[str, float]]]]:
    """Every line that looks like a table header row, top to bottom. A page
    can hold more than one table (confirmed: GEP's Chapter 2 pages carry
    both a regional-aggregate table and a per-country table, only one of
    which is actually affected), so each is examined independently later.
    """
    blocks = []
    for index, (_y, items) in enumerate(lines):
        columns = _header_columns_of(items)
        if len(columns) >= MIN_HEADER_TOKENS:
            blocks.append((index, columns))
    return blocks


def _split_label_and_values(line_text: str) -> Tuple[List[str], List[str]]:
    """Split a row into (label_words, value_tokens). A bare integer (no
    decimal point) before the first real decimal/placeholder token is
    treated as a footnote marker and kept with the label rather than
    miscounted as a data value -- confirmed these tables' real values are
    always decimal or the ".." placeholder, never a bare integer, so a
    bare integer can only be a footnote superscript.
    """
    words = line_text.split()
    value_start: Optional[int] = None
    for index, word in enumerate(words):
        if _DECIMAL_OR_PLACEHOLDER_PATTERN.match(word):
            value_start = index
            break
        if _VALUE_PATTERN.match(word) and not _BARE_INTEGER_PATTERN.match(word):
            value_start = index
            break
    if value_start is None:
        return words, []
    return words[:value_start], words[value_start:]


def _is_footer_text(text: str) -> bool:
    lowered = text.strip().lower()
    return (
        lowered.startswith("source")
        or lowered.startswith("note")
        or lowered.startswith("table ")
        or lowered.startswith("box ")
    )


def _find_value_near_x(
    items: List[Tuple[float, str]], target_x: float, max_distance: float = MAX_COLUMN_DISTANCE
) -> Optional[str]:
    """Among this row's individually-reported (single-token) items, find
    one whose x is within max_distance of target_x. Returns None (never a
    guess) when zero or more than one candidate is found.
    """
    candidates = []
    for x, text in items:
        words = text.split()
        if len(words) == 1 and _VALUE_PATTERN.match(words[0]) and abs(x - target_x) <= max_distance:
            candidates.append(words[0])
    return candidates[0] if len(candidates) == 1 else None


def _match_plain_line(
    plain_lines: List[str], label_words: List[str], cursor: int
) -> Tuple[Optional[str], int]:
    """Find the next plain-text line starting with label_words, searching
    FORWARD ONLY from `cursor`. The same label text can legitimately appear
    twice within one table's page (confirmed: "Pakistan" as the tail of a
    wrapped region name, and again later as its own country row) -- a
    search from the very start would keep re-matching the first, value-less
    occurrence instead of advancing to the real one.
    """
    if not label_words:
        return None, cursor
    prefix = " ".join(label_words[:2])
    for index in range(cursor, len(plain_lines)):
        if plain_lines[index].strip().startswith(prefix):
            return plain_lines[index].strip(), index + 1
    return None, cursor


def reconstruct_from_visitor_lines(lines: List[VisitorLine], page_plain_text: str) -> str:
    """Repair the disconnected column(s) of every affected table block in
    `lines`, returning the full repaired page text. Rows that can't be
    confidently resolved -- including multi-line-wrapped region names,
    whose label and values can end up on genuinely separate lines in both
    plain text and the coordinate stream -- are left byte-identical to the
    original, never guessed at.

    Once a block's rows are processed, its now-redundant dangling tail
    (the header token line plus its trailing run of bare numbers/".."
    placeholders -- see _scan_tail_run) is removed, but ONLY when the
    number of successful individual value insertions made for that block's
    disconnected token exactly equals the tail's own entry count. A
    "successful insertion" is counted once per (block, token) pair, only
    at the point a row's value is actually used to complete that row --
    never for a merely attempted lookup, an already-complete row, or a
    row left unresolved for any reason (ambiguous deficit, a failed
    coordinate lookup, or a silently-skipped unlabeled values-only line).
    If the counts don't match exactly, the tail is left completely
    unchanged -- this is the only way to be sure no value is ever
    discarded, since a row that was silently skipped (and so never even
    reaches a "resolved" or "unresolved" branch) would otherwise leave a
    tail entry with no home.
    """
    disconnected_tokens = find_disconnected_header_tokens(page_plain_text)
    if not disconnected_tokens:
        return page_plain_text

    blocks = _find_header_blocks(lines)
    if not blocks:
        return page_plain_text

    plain_lines = page_plain_text.split("\n")
    replacements: Dict[str, str] = {}
    tail_removal_ranges: List[Tuple[int, int]] = []
    tail_search_cursor = 0  # page-level, monotonic: each block claims its own tail only

    for block_number, (header_index, header_columns) in enumerate(blocks):
        header_labels = [label for label, _ in header_columns]
        block_disconnected = [token for token in disconnected_tokens if token in header_labels]
        if not block_disconnected:
            continue  # this block's own columns don't include a flagged token

        block_end = blocks[block_number + 1][0] if block_number + 1 < len(blocks) else len(lines)
        header_count = len(header_columns)
        expected_present = header_count - len(block_disconnected)
        missing_x = {label: x for label, x in header_columns if label in block_disconnected}
        missing_positions = {
            label: index for index, label in enumerate(header_labels) if label in block_disconnected
        }
        resolved_counts = {token: 0 for token in block_disconnected}

        pending_label_extra = ""
        plain_cursor = 0  # forward-only within this block; blocks never share row labels
        row_index = header_index + 1
        while row_index < block_end:
            _y, items = lines[row_index]
            row_text_guess = " ".join(text for _, text in items)
            if _is_footer_text(row_text_guess):
                break

            label_words, _ = _split_label_and_values(row_text_guess)
            if not label_words:
                # Values-only line (e.g. a multi-line region name's own
                # aggregate row). Any pending label fragment belongs to
                # THIS row, not whatever real row follows, so it must be
                # dropped here rather than leaking forward.
                pending_label_extra = ""
                row_index += 1
                continue
            if all(_BARE_INTEGER_PATTERN.match(word) for word in label_words):
                # A stray footnote-marker-only fragment. Its short, purely
                # numeric prefix could spuriously match an unrelated plain
                # line and desync the forward-only cursor, so it is never
                # looked up -- and it bounds any pending label to exactly
                # one iteration, same reasoning as above.
                pending_label_extra = ""
                row_index += 1
                continue

            plain_line, plain_cursor = _match_plain_line(plain_lines, label_words, plain_cursor)
            if plain_line is None:
                pending_label_extra = ""
                row_index += 1
                continue

            label, present_values = _split_label_and_values(plain_line)
            if pending_label_extra:
                label = pending_label_extra.split() + label
                pending_label_extra = ""
            if not present_values:
                pending_label_extra = " ".join(label)
                row_index += 1
                continue

            if len(present_values) == header_count:
                row_index += 1
                continue  # already complete for this block -- not affected
            if len(present_values) != expected_present:
                row_index += 1
                continue  # ambiguous deficit -- leave this row untouched

            resolved: Dict[str, str] = {}
            ok = True
            for token, x in missing_x.items():
                value = _find_value_near_x(items, x)
                if value is None:
                    ok = False
                    break
                resolved[token] = value
            if not ok:
                row_index += 1
                continue

            present_iter = iter(present_values)
            full_values = []
            for h_label in header_labels:
                if h_label in missing_positions and h_label in resolved:
                    full_values.append(resolved[h_label])
                else:
                    full_values.append(next(present_iter))

            replacements[plain_line] = " ".join(label) + " " + " ".join(full_values)
            for token in resolved:
                resolved_counts[token] += 1
            row_index += 1

        for token in block_disconnected:
            if resolved_counts[token] == 0:
                continue  # nothing resolved for this token -- no tail to find or remove
            found = _find_tail_line_range(plain_lines, token, tail_search_cursor)
            if found is None:
                continue  # can't locate this block's own tail -- leave everything as-is
            token_index, end_index, tail_entry_count = found
            if tail_entry_count == resolved_counts[token]:
                tail_removal_ranges.append((token_index, end_index))
            tail_search_cursor = end_index  # this occurrence is claimed either way

    if not replacements:
        return page_plain_text

    removed_indexes = set()
    for start, end in tail_removal_ranges:
        removed_indexes.update(range(start, end))

    repaired_lines = [
        replacements.get(line.strip(), line)
        for index, line in enumerate(plain_lines)
        if index not in removed_indexes
    ]
    return "\n".join(repaired_lines)


def repair_wide_table_columns(page, page_plain_text: str) -> str:
    """Public entry point used by ingestion/loaders.py.

    Returns `page_plain_text` unchanged for every page that doesn't match
    the disconnected-column signature -- the overwhelming majority of
    pages, including every page of every other document. Only a page that
    does match is re-examined by real PDF coordinates, and even then only
    the specific rows that resolve unambiguously are changed; anything
    else is returned exactly as it was passed in.
    """
    if not looks_like_disconnected_column_page(page_plain_text):
        return page_plain_text
    lines = _collect_visitor_lines(page)
    return reconstruct_from_visitor_lines(lines, page_plain_text)
