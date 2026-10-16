"""Tests for the coordinate-based wide-table column repair.

Fixtures below replicate the REAL coordinates confirmed by inspecting the
actual GEP-Jun-2026.pdf Chapter 2 Table 2.2 (Vietnam's row genuinely has
2028f=7.4 at x=434.9, matching the "2028f" header at x=430.4 -- see the
module docstring in ingestion/table_repair.py for how this was verified),
so these tests are both deterministic/offline and traceable to real
evidence, without depending on the sample PDF being present in every
environment.
"""

import pytest

from ingestion.chunking import chunk_documents
from ingestion.table_repair import (
    find_disconnected_header_tokens,
    looks_like_disconnected_column_page,
    reconstruct_from_visitor_lines,
    repair_wide_table_columns,
)


class _FakePage:
    """Minimal stand-in for a pypdf Page: replays pre-scripted (text, x, y)
    triples through extract_text(visitor_text=...), exactly the interface
    repair_wide_table_columns() relies on. No real PDF is needed to test
    the full entry point.
    """

    def __init__(self, records):
        self._records = records  # list of (y, x, text)

    def extract_text(self, *args, visitor_text=None, **kwargs):
        for y, x, text in self._records:
            tm = [1, 0, 0, 1, x, y]
            visitor_text(text, None, tm, None, 10.0)
        return ""


# --- Real Table 2.2 (Chapter 2, East Asia and Pacific) fixture, 8 columns:
# 2023 2024 2025e 2026f 2027f 2028f <delta 2026f> <delta 2027f> -----------

_HEADER_Y = 366.8
_HEADER_ITEMS = [
    (179.2, "2023 "),
    (229.6, "2024 "),
    (277.9, "2025e "),
    (329.3, "2026f "),
    (379.7, "2027f  "),
    (430.4, "2028f "),
    (490.4, "2026f "),
    (540.8, "2027f "),
]

_WORLD_PLAIN = "World 2.8 2.9 2.9 2.5 2.8 3.1 3.0 2.9"
_WORLD_ITEMS = [
    (72.1, "World 2.8 2.9 "),
    (283.1, "2.9 "),
    (333.6, "2.5 "),
    (384.0, "2.8 "),
    (494.8, "3.1 "),
    (520.0, "3.0 "),
    (545.2, "2.9 "),
]

_CHINA_PLAIN = "China 5.4 5.0 5.0 4.2 4.3  -0.2 0.1"
_CHINA_ITEMS = [
    (72.1, "China 5.4 5.0 "),
    (283.1, "5.0 "),
    (333.6, "4.2 "),
    (384.0, "4.3 "),
    (434.9, "4.2 "),
    (494.8, "-0.2 "),
    (545.2, "0.1 "),
]

_VIETNAM_PLAIN = "Viet Nam 5.0 7.0 8.0 6.8 7.1  0.5 0.4"
_VIETNAM_ITEMS = [
    (72.1, "Viet Nam 5.0 7.0 "),
    (283.1, "8.0 "),
    (333.6, "6.8 "),
    (384.0, "7.1  "),
    (434.9, "7.4 "),
    (494.8, "0.5 "),
    (545.2, "0.4 "),
]

_KUWAIT_PLAIN = "Kuwait -1.6 -1.5 2.6 -6.4 13.5  .. .."
_KUWAIT_ITEMS = [
    (72.1, "Kuwait -1.6 -1.5 "),
    (283.1, "2.6 "),
    (333.6, "-6.4 "),
    (384.0, "13.5  "),
    (434.9, "2.8 "),
    (494.8, ".. "),
    (545.2, ".. "),
]

_NAURU_PLAIN = "Nauru 3 0.6 1.6 2.1 1.9 1.9  0.0 0.0"
_NAURU_ITEMS = [
    (72.1, "Nauru 3 0.6 1.6 "),
    (283.1, "2.1 "),
    (333.6, "1.9 "),
    (384.0, "1.9  "),
    (434.9, "1.8 "),
    (494.8, "0.0 "),
    (545.2, "0.0 "),
]

_FOOTER_PLAIN = "Source: Test Bank."

_TAIL_LINES = ["2028f", "4.2", "7.4", "2.8", "1.8", "..", "..", "..", ".."]


def _make_row(name, annual5, deltas, missing_value):
    """Build (plain_line, items) for one 8-column row (2023 2024 2025e
    2026f 2027f <missing 2028f> <delta> <delta>), at the same x-offsets as
    _HEADER_ITEMS above (confirmed real pattern: a batched run covering the
    name plus the first two annual values, then one value per subsequent
    run, each landing ~4.5pt right of its header).
    """
    plain = f"{name} {annual5[0]} {annual5[1]} {annual5[2]} {annual5[3]} {annual5[4]}  {deltas[0]} {deltas[1]}"
    items = [
        (72.1, f"{name} {annual5[0]} {annual5[1]} "),
        (283.1, f"{annual5[2]} "),
        (333.6, f"{annual5[3]} "),
        (384.0, f"{annual5[4]} "),
        (434.9, f"{missing_value} "),
        (494.8, f"{deltas[0]} "),
        (545.2, f"{deltas[1]} "),
    ]
    return plain, items


# Eight rows, each needing exactly one 2028f value -- enough entries to
# meet MIN_TAIL_RUN (8) with nothing left over, for the "fully accounted
# for" tail-removal tests below.
_CLEAN_ROW_DEFS = [
    ("Chad", ["4.0", "5.0", "5.6", "5.2", "5.3"], ["-0.1", "0.1"], "5.1"),
    ("Togo", ["6.2", "6.5", "5.9", "5.0", "5.8"], ["-0.2", "0.2"], "5.9"),
    ("Mali", ["4.5", "4.7", "4.1", "5.0", "5.2"], ["0.3", "-0.3"], "5.0"),
    ("Niger", ["2.6", "8.3", "7.0", "6.7", "6.4"], ["0.4", "-0.4"], "6.1"),
    ("Benin", ["6.4", "7.5", "8.1", "7.0", "7.2"], ["0.5", "-0.5"], "7.3"),
    ("Ghana", ["3.1", "5.8", "6.0", "4.8", "4.9"], ["0.6", "-0.6"], "5.0"),
    ("Kenya", ["5.7", "4.7", "4.6", "4.4", "5.0"], ["0.7", "-0.7"], "5.1"),
    ("Egypt", ["3.8", "2.4", "4.4", "4.6", "4.0"], ["0.8", "-0.8"], "4.6"),
]
_CLEAN_ROWS = [_make_row(*row_def) for row_def in _CLEAN_ROW_DEFS]
_CLEAN_TAIL_VALUES = [row_def[3] for row_def in _CLEAN_ROW_DEFS]  # 8 entries, one per row
_CLEAN_TAIL_LINES = ["2028f", *_CLEAN_TAIL_VALUES]

_CLEAN_PLAIN_TEXT = "\n".join(
    [
        " 2023 2024 2025e 2026f 2027f  2026f 2027f ",
        *(plain for plain, _items in _CLEAN_ROWS),
        *_CLEAN_TAIL_LINES,
        _FOOTER_PLAIN,
    ]
)
_CLEAN_LINES = [
    (_HEADER_Y, _HEADER_ITEMS),
    *((300.0 - 20 * i, items) for i, (_plain, items) in enumerate(_CLEAN_ROWS)),
    (10.0, [(72.1, _FOOTER_PLAIN)]),
]

_FULL_PLAIN_TEXT = "\n".join(
    [
        " 2023 2024 2025e 2026f 2027f  2026f 2027f ",
        _WORLD_PLAIN,
        _CHINA_PLAIN,
        _VIETNAM_PLAIN,
        _KUWAIT_PLAIN,
        _NAURU_PLAIN,
        *_TAIL_LINES,
        _FOOTER_PLAIN,
    ]
)

_FULL_LINES = [
    (_HEADER_Y, _HEADER_ITEMS),
    (300.0, _WORLD_ITEMS),
    (200.0, _CHINA_ITEMS),
    (109.6, _VIETNAM_ITEMS),
    (90.0, _KUWAIT_ITEMS),
    (70.0, _NAURU_ITEMS),
    (10.0, [(72.1, _FOOTER_PLAIN)]),
]


# --- Detector ------------------------------------------------------------


def test_detects_the_disconnected_column_signature():
    assert looks_like_disconnected_column_page(_FULL_PLAIN_TEXT) is True


def test_detector_identifies_the_disconnected_header_token():
    assert find_disconnected_header_tokens(_FULL_PLAIN_TEXT) == ["2028f"]


def test_does_not_detect_ordinary_prose():
    prose = (
        "Executive Summary\n"
        "The global economy is facing another major shock. The conflict in "
        "the Middle East has triggered sharp increases in energy prices, "
        "renewed inflationary pressures, and fueled expectations of tighter "
        "monetary policy. Global growth is projected to slow to 2.5 percent "
        "in 2026, from 2.9 percent in 2025."
    )
    assert looks_like_disconnected_column_page(prose) is False


def test_does_not_detect_a_short_run_of_numbers_below_the_threshold():
    """A run of fewer than MIN_TAIL_RUN bare numbers after a year-like token
    is not enough to trigger the detector -- avoids false positives on
    short, incidental numeric lists that aren't a disconnected table tail.
    """
    text = "2028f\n4.2\n7.4\n2.8\nSome prose resumes here."
    assert looks_like_disconnected_column_page(text) is False


def test_gpo_numbered_note_heading_does_not_trigger_detection():
    """Non-regression check for the frozen GPO document: a numbered note
    heading followed by ordinary narrative must never be mistaken for a
    disconnected table column.
    """
    gpo_like_text = (
        "9.  Workers' Compensation Liability\n"
        "The Federal Employees' Compensation Act provides income protection "
        "for approximately 5.2 million and 6 million respectively."
    )
    assert looks_like_disconnected_column_page(gpo_like_text) is False


# --- Reconstruction (direct, using hand-built visitor lines) -------------


def test_vietnam_2028f_value_is_repaired_to_7_4_matching_the_header_column():
    """The specific regression check requested: Vietnam's repaired 2028f
    value must be 7.4, placed at the ordinal position the real "2028f"
    header column occupies (index 5 of 8, confirmed by coordinates: header
    "2028f" at x=430.4, Vietnam's own "7.4" at x=434.9 -- the same ~4.5pt
    offset as every other confirmed column match) -- never found by
    counting through the original disconnected list.
    """
    repaired = reconstruct_from_visitor_lines(_FULL_LINES, _FULL_PLAIN_TEXT)
    vietnam_line = next(l for l in repaired.split("\n") if l.strip().startswith("Viet Nam"))
    vietnam_tokens = vietnam_line.split()
    # Viet(0) Nam(1) 2023(2) 2024(3) 2025e(4) 2026f(5) 2027f(6) 2028f(7) ...
    assert vietnam_tokens[7] == "7.4"


def test_china_row_is_repaired_with_its_own_2028f_value():
    repaired = reconstruct_from_visitor_lines(_FULL_LINES, _FULL_PLAIN_TEXT)
    china_line = next(l for l in repaired.split("\n") if l.strip().startswith("China"))
    assert china_line.split() == ["China", "5.4", "5.0", "5.0", "4.2", "4.3", "4.2", "-0.2", "0.1"]


def test_negative_values_are_preserved_through_repair():
    repaired = reconstruct_from_visitor_lines(_FULL_LINES, _FULL_PLAIN_TEXT)
    kuwait_line = next(l for l in repaired.split("\n") if l.strip().startswith("Kuwait"))
    assert kuwait_line.split() == ["Kuwait", "-1.6", "-1.5", "2.6", "-6.4", "13.5", "2.8", "..", ".."]


def test_missing_value_placeholders_are_preserved_through_repair():
    repaired = reconstruct_from_visitor_lines(_FULL_LINES, _FULL_PLAIN_TEXT)
    kuwait_line = next(l for l in repaired.split("\n") if l.strip().startswith("Kuwait"))
    assert kuwait_line.split().count("..") == 2


def test_footnote_marker_is_preserved_and_not_miscounted_as_a_value():
    repaired = reconstruct_from_visitor_lines(_FULL_LINES, _FULL_PLAIN_TEXT)
    nauru_line = next(l for l in repaired.split("\n") if l.strip().startswith("Nauru"))
    tokens = nauru_line.split()
    assert tokens[:2] == ["Nauru", "3"]
    assert tokens == ["Nauru", "3", "0.6", "1.6", "2.1", "1.9", "1.9", "1.8", "0.0", "0.0"]


def test_already_complete_row_is_left_completely_unchanged():
    """A row whose header this block shares (e.g. an aggregate row that
    already has every column inline) must never be touched -- only rows
    with a genuine deficit are repaired.
    """
    repaired = reconstruct_from_visitor_lines(_FULL_LINES, _FULL_PLAIN_TEXT)
    world_line = next(l for l in repaired.split("\n") if l.strip().startswith("World"))
    assert world_line == _WORLD_PLAIN


def test_footer_and_tail_lines_are_preserved_verbatim():
    repaired = reconstruct_from_visitor_lines(_FULL_LINES, _FULL_PLAIN_TEXT)
    assert _FOOTER_PLAIN in repaired
    for tail_line in _TAIL_LINES:
        assert tail_line in repaired.split("\n")


def test_non_disconnected_page_is_returned_unchanged():
    prose = "Executive Summary\nThe global economy is facing another major shock."
    assert reconstruct_from_visitor_lines(_FULL_LINES, prose) == prose


def test_row_with_ambiguous_deficit_count_is_left_unresolved():
    """A row whose present-value count doesn't match the expected deficit
    (e.g. corrupted or partially-extracted) must never be guessed at."""
    odd_plain = "Oddrow 1.0 2.0  3.0"  # only 3 values, expected 7
    odd_items = [(72.1, "Oddrow 1.0 2.0  3.0 ")]
    lines = [(_HEADER_Y, _HEADER_ITEMS), (50.0, odd_items), (10.0, [(72.1, _FOOTER_PLAIN)])]
    full_text = "\n".join([" 2023 2024 2025e 2026f 2027f  2026f 2027f ", odd_plain, *_TAIL_LINES, _FOOTER_PLAIN])
    repaired = reconstruct_from_visitor_lines(lines, full_text)
    assert odd_plain in repaired.split("\n")


# --- Top-level entry point (via the fake page) ---------------------------


def test_repair_wide_table_columns_end_to_end_via_fake_page():
    records = [(y, x, text) for y, items in _FULL_LINES for x, text in items]
    page = _FakePage(records)
    repaired = repair_wide_table_columns(page, _FULL_PLAIN_TEXT)
    vietnam_line = next(l for l in repaired.split("\n") if l.strip().startswith("Viet Nam"))
    assert vietnam_line.split() == [
        "Viet", "Nam", "5.0", "7.0", "8.0", "6.8", "7.1", "7.4", "0.5", "0.4",
    ]


def test_repair_wide_table_columns_leaves_non_triggering_page_byte_identical():
    prose = (
        "Executive Summary\n"
        "The global economy is facing another major shock, with growth "
        "projected to slow to 2.5 percent in 2026."
    )
    page = _FakePage([(100.0, 72.0, prose)])
    assert repair_wide_table_columns(page, prose) == prose


def test_repair_wide_table_columns_does_not_touch_a_page_with_no_header_row():
    """Detector fires on text alone (e.g. a stray numeric list resembling
    the tail signature), but without a real header row to anchor columns,
    reconstruction must safely decline rather than guess."""
    text = "\n".join(["2028f", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "Some prose."])
    page = _FakePage([(100.0, 72.0, text)])
    assert repair_wide_table_columns(page, text) == text


# --- Dangling tail cleanup: tail_entry_count == successfully_resolved_entry_count ---


def test_clean_block_tail_is_removed_when_every_entry_is_accounted_for():
    """T == R exactly (8 rows, 8 tail entries, all 8 resolved): the
    now-redundant tail header line and every one of its entries must be
    removed, and every row must still be correctly repaired.
    """
    repaired = reconstruct_from_visitor_lines(_CLEAN_LINES, _CLEAN_PLAIN_TEXT)
    repaired_lines = repaired.split("\n")

    for name, annual5, deltas, missing_value in _CLEAN_ROW_DEFS:
        expected = f"{name} {annual5[0]} {annual5[1]} {annual5[2]} {annual5[3]} {annual5[4]} {missing_value} {deltas[0]} {deltas[1]}"
        row_line = next(l for l in repaired_lines if l.strip().startswith(name))
        assert row_line == expected

    assert "2028f" not in repaired_lines
    for tail_value in _CLEAN_TAIL_VALUES:
        assert tail_value not in repaired_lines
    assert _FOOTER_PLAIN in repaired_lines


def test_unresolved_block_retains_full_tail_when_counts_mismatch():
    """T != R (the existing fixture: 8 tail entries, only 4 rows resolved,
    4 unused padding placeholders): the complete original tail -- the
    header token line and every single entry -- must remain, byte for
    byte, exactly as extracted.
    """
    repaired = reconstruct_from_visitor_lines(_FULL_LINES, _FULL_PLAIN_TEXT)
    repaired_lines = repaired.split("\n")
    for tail_line in _TAIL_LINES:
        assert tail_line in repaired_lines


def test_silently_skipped_unlabeled_row_prevents_tail_removal():
    """One row's own label text never reaches the row-matching loop at all
    (e.g. the real MNA aggregate row: its label and values ended up on
    genuinely separate lines in the coordinate stream). It is never logged
    as resolved OR unresolved -- it simply never enters either branch. The
    plain-text tail is otherwise identical to the fully-resolved fixture
    (8 entries), but only 7 rows actually get completed, so the raw count
    (7 resolved vs. 8 tail entries) must still block removal even though
    nothing was ever explicitly flagged as a problem.
    """
    unlabeled_items = [(x, text) for x, text in _CLEAN_ROWS[-1][1] if not text.strip().startswith("Egypt")]
    lines = [
        (_HEADER_Y, _HEADER_ITEMS),
        *((300.0 - 20 * i, items) for i, (_plain, items) in enumerate(_CLEAN_ROWS[:-1])),
        (300.0 - 20 * 7, unlabeled_items),  # Egypt's row, now with no label text of its own
        (10.0, [(72.1, _FOOTER_PLAIN)]),
    ]
    repaired = reconstruct_from_visitor_lines(lines, _CLEAN_PLAIN_TEXT)
    repaired_lines = repaired.split("\n")

    # The 7 labeled rows still get repaired correctly...
    for name, annual5, deltas, missing_value in _CLEAN_ROW_DEFS[:-1]:
        expected = f"{name} {annual5[0]} {annual5[1]} {annual5[2]} {annual5[3]} {annual5[4]} {missing_value} {deltas[0]} {deltas[1]}"
        row_line = next(l for l in repaired_lines if l.strip().startswith(name))
        assert row_line == expected

    # ...but the tail must be fully retained, since only 7 of its 8 entries
    # were ever claimed by a row (7 != 8).
    assert "2028f" in repaired_lines
    for tail_value in _CLEAN_TAIL_VALUES:
        assert tail_value in repaired_lines
    # Egypt's own plain-text row is also left completely untouched.
    egypt_plain = next(plain for plain, _items in _CLEAN_ROWS if plain.startswith("Egypt"))
    assert egypt_plain in repaired_lines


def test_tail_entry_counting_includes_placeholders_and_footnote_bearing_rows():
    """Swap one clean row for a footnoted, placeholder-bearing variant and
    confirm the tail is still removed when T == R -- the counting rule
    must treat a footnote marker as part of the label (not an extra
    value) and ".." as a real, countable tail entry, consistent with
    _scan_tail_run's own definition used for detection.
    """
    footnoted_plain = "Sudan 3 -1.0 2.0 .. .. .. .. .."
    footnoted_items = [
        (72.1, "Sudan 3 -1.0 2.0 "),  # name + footnote marker "3" + 2023 + 2024
        (283.1, ".. "),  # 2025e
        (333.6, ".. "),  # 2026f
        (384.0, ".. "),  # 2027f
        (434.9, "9.9 "),  # 2028f -- the missing value, found via coordinates
        (494.8, ".. "),  # delta 2026f
        (545.2, ".. "),  # delta 2027f
    ]
    plain_text = "\n".join(
        [
            " 2023 2024 2025e 2026f 2027f  2026f 2027f ",
            *(plain for plain, _items in _CLEAN_ROWS[:-1]),
            footnoted_plain,
            "2028f",
            *_CLEAN_TAIL_VALUES[:-1],
            "9.9",
            _FOOTER_PLAIN,
        ]
    )
    lines = [
        (_HEADER_Y, _HEADER_ITEMS),
        *((300.0 - 20 * i, items) for i, (_plain, items) in enumerate(_CLEAN_ROWS[:-1])),
        (300.0 - 20 * 7, footnoted_items),
        (10.0, [(72.1, _FOOTER_PLAIN)]),
    ]
    repaired = reconstruct_from_visitor_lines(lines, plain_text)
    repaired_lines = repaired.split("\n")

    sudan_line = next(l for l in repaired_lines if l.strip().startswith("Sudan"))
    assert sudan_line.split() == ["Sudan", "3", "-1.0", "2.0", "..", "..", "..", "9.9", "..", ".."]
    assert "2028f" not in repaired_lines
    for value in [*_CLEAN_TAIL_VALUES[:-1], "9.9"]:
        assert value not in repaired_lines


_CLEAN_ROW_DEFS_B = [
    (name + "ia", annual5, deltas, missing_value)
    for name, annual5, deltas, missing_value in _CLEAN_ROW_DEFS
]
_CLEAN_ROWS_B = [_make_row(*row_def) for row_def in _CLEAN_ROW_DEFS_B]
_CLEAN_TAIL_LINES_B = ["2029f", *(row_def[3] for row_def in _CLEAN_ROW_DEFS_B)]
_HEADER_ITEMS_B = [(x, text.replace("2028f", "2029f")) for x, text in _HEADER_ITEMS]


def test_multiple_independent_blocks_each_get_their_own_tail_removal_decision():
    """Two separate, fully-resolved blocks on the same page, each with its
    own disconnected column and its own tail: each block's tail must be
    removed independently based on its OWN resolution count, never the
    other block's. (Block B uses a distinct column name, "2029f" --
    reusing the same name twice on one page, as a quick check confirmed,
    hits a separate, pre-existing, never-previously-exercised limitation
    in the existing block_disconnected accounting that is out of scope
    for this cleanup; see the final report.)
    """
    plain_text = "\n".join(
        [
            " 2023 2024 2025e 2026f 2027f  2026f 2027f ",  # block A header
            *(plain for plain, _items in _CLEAN_ROWS),  # block A rows
            *_CLEAN_TAIL_LINES,  # block A's own tail
            " 2023 2024 2025e 2029f 2027f  2026f 2027f ",  # block B header
            *(plain for plain, _items in _CLEAN_ROWS_B),  # block B rows (distinct labels)
            *_CLEAN_TAIL_LINES_B,  # block B's own, separate tail
            _FOOTER_PLAIN,
        ]
    )
    lines = [
        (900.0, _HEADER_ITEMS),
        *((800.0 - 20 * i, items) for i, (_plain, items) in enumerate(_CLEAN_ROWS)),
        (400.0, _HEADER_ITEMS_B),
        *((300.0 - 20 * i, items) for i, (_plain, items) in enumerate(_CLEAN_ROWS_B)),
        (10.0, [(72.1, _FOOTER_PLAIN)]),
    ]

    repaired = reconstruct_from_visitor_lines(lines, plain_text)
    repaired_lines = repaired.split("\n")

    # Both tails removed independently; any leftover tail-header line would
    # mean one block's removal incorrectly depended on the other's count.
    assert repaired_lines.count("2028f") == 0
    assert repaired_lines.count("2029f") == 0
    for value in _CLEAN_TAIL_VALUES:
        assert value not in repaired_lines
    for value in (row_def[3] for row_def in _CLEAN_ROW_DEFS_B):
        assert value not in repaired_lines
    # Both blocks' rows were still correctly repaired.
    chad_line = next(l for l in repaired_lines if l.strip().startswith("Chad") and not l.strip().startswith("Chadia"))
    assert chad_line == "Chad 4.0 5.0 5.6 5.2 5.3 5.1 -0.1 0.1"
    chadia_line = next(l for l in repaired_lines if l.strip().startswith("Chadia"))
    assert chadia_line == "Chadia 4.0 5.0 5.6 5.2 5.3 5.1 -0.1 0.1"


# --- Repaired output through the existing chunker -------------------------


def test_repaired_table_text_chunks_without_unreasonable_boundaries():
    repaired = reconstruct_from_visitor_lines(_FULL_LINES, _FULL_PLAIN_TEXT)
    document = __import__("langchain_core.documents", fromlist=["Document"]).Document(
        page_content=repaired,
        metadata={"source": "gep.pdf", "file_name": "gep.pdf", "file_type": "pdf", "page": 83},
    )
    chunks = chunk_documents([document], document_hash="table-repair-hash", chunk_size=300, chunk_overlap=30)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 300 + 1  # splitter's own small rounding allowance
        assert chunk.page_content.strip() != ""
