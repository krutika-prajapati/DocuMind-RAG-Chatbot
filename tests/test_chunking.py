"""Tests for document chunking and chunk metadata."""

import pytest
from langchain_core.documents import Document

from config.settings import validate_chunk_config
from ingestion.chunking import NoChunksProducedError, chunk_documents


def _make_document(text: str, **metadata) -> Document:
    return Document(page_content=text, metadata=metadata)


def test_normal_document_splits_into_multiple_chunks():
    text = " ".join(f"This is sentence number {i}." for i in range(200))
    document = _make_document(text, source="doc.txt", file_name="doc.txt", file_type="txt")

    chunks = chunk_documents([document], document_hash="hash1", chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1


def test_chunk_overlap_is_applied():
    text = "".join(str(i % 10) for i in range(1000))
    document = _make_document(
        text, source="numbers.txt", file_name="numbers.txt", file_type="txt"
    )

    chunks = chunk_documents([document], document_hash="hash2", chunk_size=50, chunk_overlap=15)

    assert len(chunks) > 1
    for current_chunk, next_chunk in zip(chunks, chunks[1:]):
        overlap_tail = current_chunk.page_content[-15:]
        overlap_head = next_chunk.page_content[:15]
        assert overlap_tail == overlap_head


def test_metadata_is_preserved_and_enriched():
    document = _make_document(
        ("A" * 50 + " ") + ("B" * 50 + " ") + ("C" * 50),
        source="company_policy.pdf",
        file_name="company_policy.pdf",
        file_type="pdf",
        page=4,
    )

    chunks = chunk_documents([document], document_hash="abc123", chunk_size=40, chunk_overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "company_policy.pdf"
        assert chunk.metadata["file_name"] == "company_policy.pdf"
        assert chunk.metadata["file_type"] == "pdf"
        assert chunk.metadata["page"] == 4
        assert "chunk_index" in chunk.metadata
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_size"] == len(chunk.page_content)
        assert chunk.metadata["document_hash"] == "abc123"


def test_chunk_index_starts_at_zero_per_call():
    document = _make_document(
        " ".join(f"word{i}" for i in range(300)),
        source="a.txt",
        file_name="a.txt",
        file_type="txt",
    )

    chunks = chunk_documents([document], document_hash="hash-a", chunk_size=100, chunk_overlap=20)

    assert chunks[0].metadata["chunk_index"] == 0
    indexes = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert indexes == list(range(len(chunks)))


def test_chunk_ids_are_deterministic():
    document = _make_document(
        " ".join(f"item {i}" for i in range(100)),
        source="b.txt",
        file_name="b.txt",
        file_type="txt",
    )

    first_run = chunk_documents([document], document_hash="hash-b", chunk_size=80, chunk_overlap=20)
    second_run = chunk_documents([document], document_hash="hash-b", chunk_size=80, chunk_overlap=20)

    first_ids = [chunk.metadata["chunk_id"] for chunk in first_run]
    second_ids = [chunk.metadata["chunk_id"] for chunk in second_run]
    assert first_ids == second_ids


def test_chunk_ids_are_unique_across_chunks():
    document = _make_document(
        " ".join(f"unique token {i}" for i in range(150)),
        source="c.txt",
        file_name="c.txt",
        file_type="txt",
    )

    chunks = chunk_documents([document], document_hash="hash-c", chunk_size=80, chunk_overlap=20)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_whitespace_only_document_is_ignored_alongside_valid_one():
    whitespace_document = _make_document(
        "   \n\t  ", source="mixed.txt", file_name="mixed.txt", file_type="txt"
    )
    valid_document = _make_document(
        "Real content that should be chunked.",
        source="mixed.txt",
        file_name="mixed.txt",
        file_type="txt",
    )

    chunks = chunk_documents(
        [whitespace_document, valid_document],
        document_hash="hash-d",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) == 1
    assert chunks[0].page_content.strip() == "Real content that should be chunked."


def test_all_whitespace_documents_raise_clear_error():
    whitespace_document = _make_document(
        "   ", source="blank.txt", file_name="blank.txt", file_type="txt"
    )

    with pytest.raises(NoChunksProducedError):
        chunk_documents(
            [whitespace_document], document_hash="hash-e", chunk_size=100, chunk_overlap=20
        )


def test_different_documents_do_not_share_chunk_indexes():
    document_a = _make_document(
        " ".join(f"alpha {i}" for i in range(100)),
        source="a.txt",
        file_name="a.txt",
        file_type="txt",
    )
    document_b = _make_document(
        " ".join(f"beta {i}" for i in range(50)),
        source="b.txt",
        file_name="b.txt",
        file_type="txt",
    )

    chunks_a = chunk_documents([document_a], document_hash="hash-a2", chunk_size=80, chunk_overlap=20)
    chunks_b = chunk_documents([document_b], document_hash="hash-b2", chunk_size=80, chunk_overlap=20)

    assert chunks_a[0].metadata["chunk_index"] == 0
    assert chunks_b[0].metadata["chunk_index"] == 0


def test_invalid_chunk_configuration_is_rejected():
    with pytest.raises(ValueError):
        validate_chunk_config(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        validate_chunk_config(chunk_size=100, chunk_overlap=150)

    validate_chunk_config(chunk_size=100, chunk_overlap=50)  # should not raise


def test_chunk_documents_rejects_invalid_configuration():
    document = _make_document(
        "Some content here.", source="x.txt", file_name="x.txt", file_type="txt"
    )

    with pytest.raises(ValueError):
        chunk_documents([document], document_hash="hash-f", chunk_size=50, chunk_overlap=50)


# --- section_title detection (GPO-style numbered sections/notes) --------


def test_section_title_detects_numbered_note_heading():
    document = _make_document(
        "1.  Summary of Significant Accounting Policies\n"
        "The Agency's mission and authority are derived from various statutes.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=61,
    )

    chunks = chunk_documents([document], document_hash="gpo-hash1", chunk_size=200, chunk_overlap=20)

    assert all(chunk.metadata.get("section_title") == "Note 1" for chunk in chunks)


def test_section_title_detects_lettered_subsection_within_a_note():
    """"Note 1.I"-style combined label: a lettered sub-heading is combined
    with the most recently seen numbered note, even once the chunker has
    moved on to a later chunk.
    """
    document = _make_document(
        "9.  Workers' Compensation Liability\n"
        "The Federal Employees' Compensation Act provides income protection.\n"
        "I. Workers' Compensation Liability\n"
        "This note describes how the liability is calculated for future benefits.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=65,
    )

    chunks = chunk_documents(
        [document], document_hash="gpo-hash2", chunk_size=120, chunk_overlap=20
    )

    assert chunks[0].metadata.get("section_title") == "Note 9"
    assert chunks[-1].metadata.get("section_title") == "Note 9.I"


def test_section_title_tracks_a_second_numbered_note():
    document = _make_document(
        "2.  Fund Balance with Treasury\n"
        "A table of items included in GPO's fund balance with Treasury follows.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=69,
    )

    chunks = chunk_documents([document], document_hash="gpo-hash3", chunk_size=200, chunk_overlap=20)

    assert all(chunk.metadata.get("section_title") == "Note 2" for chunk in chunks)


def test_section_title_detected_even_when_not_the_first_line():
    document = _make_document(
        "Section IV: Notes to the Financial Statements\n"
        "U. S. Government Publishing Office | Financial Year In Review - FY 2018\n"
        "9.  Workers' Compensation Liability\n"
        "Claims incurred and paid by DOL are approximately $5.2 million.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=73,
    )

    chunks = chunk_documents([document], document_hash="gpo-hash4", chunk_size=300, chunk_overlap=20)

    assert chunks[0].metadata.get("section_title") == "Note 9"


def test_section_title_ignores_prose_mentioning_a_note_number():
    document = _make_document(
        "As discussed in Note 1, the Agency's accounting policies require "
        "accrual-basis recognition of revenue for all printing and binding services provided.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=25,
    )

    chunks = chunk_documents([document], document_hash="gpo-hash5", chunk_size=300, chunk_overlap=20)

    assert all("section_title" not in chunk.metadata for chunk in chunks)


def test_section_title_ignores_table_of_contents_entries():
    document = _make_document(
        "9.   Workers' Compensation Liability ................................................. 42\n"
        "10. Commitments  ....................................................................... 42",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=33,
    )

    chunks = chunk_documents([document], document_hash="gpo-hash6", chunk_size=300, chunk_overlap=20)

    assert all("section_title" not in chunk.metadata for chunk in chunks)


def test_section_title_absent_when_no_heading_ever_detected():
    document = _make_document(
        "This is ordinary narrative text with no numbered heading anywhere in it at all.",
        source="plain.txt",
        file_name="plain.txt",
        file_type="txt",
    )

    chunks = chunk_documents([document], document_hash="gpo-hash7", chunk_size=200, chunk_overlap=20)

    assert all("section_title" not in chunk.metadata for chunk in chunks)


def test_section_title_survives_across_multiple_pages_of_the_same_document():
    """A numbered section's heading is only stated once, but its content
    can continue onto a later physical page with no heading of its own --
    the detected section must carry forward across pages.
    """
    page_one = _make_document(
        "1.  Summary of Significant Accounting Policies\n"
        "Introductory accounting policy text follows here.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=61,
    )
    page_two = _make_document(
        "Continuing policy discussion with no new heading on this page at all.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=62,
    )

    chunks = chunk_documents(
        [page_one, page_two], document_hash="gpo-hash8", chunk_size=200, chunk_overlap=20
    )

    assert all(chunk.metadata.get("section_title") == "Note 1" for chunk in chunks)


def test_section_title_does_not_remove_other_metadata():
    document = _make_document(
        "1.  Summary of Significant Accounting Policies\nSome policy text.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=61,
    )

    chunks = chunk_documents([document], document_hash="gpo-hash9", chunk_size=200, chunk_overlap=20)

    for chunk in chunks:
        assert chunk.metadata["source"] == "gpo.pdf"
        assert chunk.metadata["file_name"] == "gpo.pdf"
        assert chunk.metadata["file_type"] == "pdf"
        assert chunk.metadata["page"] == 61
        assert "chunk_id" in chunk.metadata
        assert "document_hash" in chunk.metadata


# --- section_title transition-chunk behavior (majority content wins) ----


def test_transition_chunk_is_labeled_with_the_majority_section_not_the_last_heading():
    """Regression test for the real Note 9 -> Note 10 transition case: a
    chunk whose content is substantially Note 9 (its closing figures), with
    the Note 10 heading only appearing near the very end, must be labeled
    "Note 9" (the section that actually owns most of this chunk) -- not
    "Note 10" merely because that heading happens to appear last.
    """
    document = _make_document(
        "9.  Workers' Compensation Liability\n"
        "Claims incurred and paid by DOL as of September 30, 2018, and September 30, 2017, "
        "but not reimbursed to DOL by GPO, are approximately $5.2 million and $6 million "
        "respectively. This liability is included in accounts payable and accrued expenses. "
        "As of September 30, 2018, GPO estimated discounted current actuarially derived "
        "future workers' compensation liability of approximately $7.5 million and estimated "
        "discounted non-current actuarially derived future workers' liability of "
        "approximately $45.2 million.\n"
        "10. Commitments",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=73,
    )

    # A single large chunk_size keeps all of this in ONE chunk, so the fix
    # is exercised within a single chunk exactly as in the real document.
    chunks = chunk_documents(
        [document], document_hash="gpo-transition", chunk_size=2000, chunk_overlap=100
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.get("section_title") == "Note 9"


def test_transition_chunk_carries_the_new_section_forward_to_the_next_chunk():
    """Even though the transition chunk itself is labeled with the MAJORITY
    section (Note 9), the next chunk -- which is genuinely all Note 10
    content -- must still be labeled "Note 10", confirming the carried-
    forward state is unaffected by the majority-content relabeling.
    """
    page_one = _make_document(
        "9.  Workers' Compensation Liability\n"
        "Claims incurred and paid by DOL as of September 30, 2018, and September 30, 2017, "
        "but not reimbursed to DOL by GPO, are approximately $5.2 million and $6 million "
        "respectively. This liability is included in accounts payable and accrued expenses. "
        "As of September 30, 2018, GPO estimated discounted current actuarially derived "
        "future workers' compensation liability of approximately $7.5 million.\n"
        "10. Commitments",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=73,
    )
    page_two = _make_document(
        "As of September 30, 2018, GPO was committed to various non-cancelable operating "
        "leases, primarily covering warehouse and office space with no new heading on this "
        "page at all.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=74,
    )

    chunks = chunk_documents(
        [page_one, page_two], document_hash="gpo-transition2", chunk_size=2000, chunk_overlap=100
    )

    assert chunks[0].metadata.get("section_title") == "Note 9"
    assert chunks[-1].metadata.get("section_title") == "Note 10"


def test_heading_appearing_early_still_wins_the_normal_case():
    """Symmetry check: when the NEW heading appears early and owns most of
    the chunk (the common case, already covered elsewhere), majority-content
    labeling still picks the new heading -- this isn't a regression that
    only ever prefers whatever came before.
    """
    document = _make_document(
        "9.  Workers' Compensation Liability\n"
        "A short closing note.\n"
        "10.  Commitments\n"
        "As of September 30, 2018, GPO was committed to various non-cancelable operating "
        "leases, primarily covering warehouse and office space across several facilities "
        "nationwide, each with its own remaining lease term and renewal options still "
        "under active negotiation with building owners and regional property managers.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=73,
    )

    chunks = chunk_documents(
        [document], document_hash="gpo-transition3", chunk_size=2000, chunk_overlap=100
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.get("section_title") == "Note 10"


def test_real_heading_beats_a_longer_boilerplate_preamble():
    """A single real heading must still win even when preceded by a MUCH
    longer run of unattributed boilerplate/running-header text -- "no
    section yet" never outweighs an actual detected section, which is what
    keeps the common "heading not on the first line" case working.
    """
    document = _make_document(
        ("Section IV: Notes to the Financial Statements " * 10) + "\n"
        "9.  Workers' Compensation Liability\n"
        "Short note.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=73,
    )

    chunks = chunk_documents(
        [document], document_hash="gpo-transition4", chunk_size=2000, chunk_overlap=100
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.get("section_title") == "Note 9"


# --- section_title detection (GEP-style: chapters, regions, boxes) ------
#
# Every heading form below is the REAL convention confirmed by inspecting
# the actual extracted text of GEP-Jun-2026.pdf, not the hypothetical
# examples assumed up front -- in particular, two of the six region names
# wrap across two physical lines at their own section start, and GEP has no
# real decimal-subsection body heading at all (only Table-of-Contents and
# figure/table caption uses of "N.N").


def test_section_title_detects_chapter_heading():
    document = _make_document(
        "CHAPTER 2\n"
        "Regional outlooks for developing regions follow in this chapter.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=68,
    )

    chunks = chunk_documents([document], document_hash="gep-hash1", chunk_size=300, chunk_overlap=20)

    assert all(chunk.metadata.get("section_title") == "Chapter 2" for chunk in chunks)


def test_section_title_has_no_decimal_subsection_heading_because_none_exists_in_the_real_document():
    """GEP's decimal numbers ("1.1", "2.9", ...) only ever appear inside the
    Table of Contents (with dot leaders) and as the start of a wrapped
    prose line reporting a percentage figure -- confirmed from the real
    extracted text. There is no real "N.N <Title>" body heading anywhere in
    the document, so no decimal-subsection pattern is implemented; this
    locks in that a decimal-prefixed line is never treated as a heading.
    """
    document = _make_document(
        "1.1 Global economic prospects ......................................... 5\n"
        "1.2 Global risks and policy challenges ................................ 6\n"
        "2.9 percent in 2025, with notably weaker prospects in EMDEs.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=10,
    )

    chunks = chunk_documents([document], document_hash="gep-hash2", chunk_size=300, chunk_overlap=20)

    assert all("section_title" not in chunk.metadata for chunk in chunks)


def test_section_title_detects_region_heading_with_active_chapter():
    document = _make_document(
        "CHAPTER 2\nIntroductory chapter text.\n"
        "South Asia\n"
        "Recent developments\n"
        "Growth in SAR is expected to soften to 6.3 percent in 2026.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=96,
    )

    chunks = chunk_documents([document], document_hash="gep-hash3", chunk_size=400, chunk_overlap=20)

    assert all(
        chunk.metadata.get("section_title") == "Chapter 2 - South Asia" for chunk in chunks
    )


def test_section_title_detects_region_heading_without_an_active_chapter():
    document = _make_document(
        "South Asia\nRecent developments\nGrowth in SAR is expected to soften.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=20,
    )

    chunks = chunk_documents([document], document_hash="gep-hash4", chunk_size=300, chunk_overlap=20)

    assert all(chunk.metadata.get("section_title") == "South Asia" for chunk in chunks)


def test_section_title_detects_a_region_name_wrapped_across_two_lines():
    """Two of the six GEP region names are long enough to wrap across two
    physical lines at their own section start, confirmed in the real
    document ("Latin America and" / "the Caribbean", and "Middle East,
    North Africa," / "Afghanistan, and Pakistan") -- this is the real
    convention, not the shorter single-line examples assumed up front.
    """
    document = _make_document(
        "CHAPTER 2\nIntroductory chapter text.\n"
        "Latin America and \n"
        "the Caribbean \n"
        "Recent developments \n"
        "Economic conditions in Latin America and the Caribbean (LAC) this "
        "year have been shaped by the conflict in the Middle East.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=87,
    )

    chunks = chunk_documents([document], document_hash="gep-hash5", chunk_size=500, chunk_overlap=20)

    assert all(
        chunk.metadata.get("section_title") == "Chapter 2 - Latin America and the Caribbean"
        for chunk in chunks
    )


def test_section_title_detects_box_heading():
    document = _make_document(
        "CHAPTER 1\nIntroductory chapter text.\n"
        "BOX 1.1 How much will AI affect global growth?\n"
        "This box addresses three questions about AI's economic impact.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=29,
    )

    chunks = chunk_documents([document], document_hash="gep-hash6", chunk_size=400, chunk_overlap=20)

    expected = "Chapter 1 - Box 1.1 - How much will AI affect global growth?"
    assert all(chunk.metadata.get("section_title") == expected for chunk in chunks)


def test_section_title_box_continued_marker_keeps_the_same_section():
    """A box's "(continued)" repeat on a later page must resolve to the
    SAME label as the box's own opening page, not a different section, and
    must not break the carried-forward state.
    """
    page_one = _make_document(
        "CHAPTER 1\n"
        "BOX 1.1 How much will AI affect global growth?\n"
        "This box addresses three questions about AI's economic impact.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=29,
    )
    page_two = _make_document(
        "BOX 1.1 How much will AI affect global growth? (continued )\n"
        "The box continues its analysis of labor market effects here.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=30,
    )

    chunks = chunk_documents(
        [page_one, page_two], document_hash="gep-hash7", chunk_size=400, chunk_overlap=20
    )

    expected = "Chapter 1 - Box 1.1 - How much will AI affect global growth?"
    assert all(chunk.metadata.get("section_title") == expected for chunk in chunks)


def test_section_title_detected_even_when_heading_is_not_the_first_line():
    document = _make_document(
        "C H A PT ER  2 G L O B AL  EC ON O M I C PR O SPEC T S |  J U N E 2 0 2 6 74 \n"
        "South Asia\n"
        "Recent developments\n"
        "Growth in SAR is expected to soften to 6.3 percent in 2026.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=96,
    )

    chunks = chunk_documents([document], document_hash="gep-hash8", chunk_size=400, chunk_overlap=20)

    assert chunks[0].metadata.get("section_title") == "South Asia"


def test_section_title_carries_forward_through_normal_prose_with_no_new_heading():
    page_one = _make_document(
        "CHAPTER 2\nSouth Asia\nRecent developments\nOpening paragraph of the section.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=95,
    )
    page_two = _make_document(
        "Growth in India is projected to moderate to 6.6 percent, with no new "
        "heading anywhere on this page.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=96,
    )

    chunks = chunk_documents(
        [page_one, page_two], document_hash="gep-hash9", chunk_size=300, chunk_overlap=20
    )

    assert all(
        chunk.metadata.get("section_title") == "Chapter 2 - South Asia" for chunk in chunks
    )


def test_section_title_ignores_prose_mentioning_a_region_name():
    document = _make_document(
        "Economic activity in the EAP region remained resilient in early 2026, "
        "despite signs of spillovers from the conflict in the Middle East "
        "through higher inflation across South Asia and Sub-Saharan Africa too.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=80,
    )

    chunks = chunk_documents([document], document_hash="gep-hash10", chunk_size=400, chunk_overlap=20)

    assert all("section_title" not in chunk.metadata for chunk in chunks)


def test_section_title_ignores_toc_style_box_entry_even_in_uppercase():
    document = _make_document(
        "BOX 1.1 How much will AI affect global growth? ........................... 8\n"
        "BOX 1.2 Unlocking private investment ....................................... 22",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=6,
    )

    chunks = chunk_documents([document], document_hash="gep-hash11", chunk_size=300, chunk_overlap=20)

    assert all("section_title" not in chunk.metadata for chunk in chunks)


def test_section_title_ignores_chart_panel_letter_captions():
    """Real GEP figures use lettered panel captions ("A. Share of economies
    by 2026", "D. Dependence on Middle East energy", ...) that are
    structurally identical in shape to GPO's lettered note sub-headings (a
    single capital letter, period, space, capitalized text). These must
    never become a spurious section label for a GEP document -- confirmed
    safe here because a GEP carried-forward label never reduces to a bare
    digit before its first period, so the GPO lettered-subsection branch
    never actually fires for it.
    """
    document = _make_document(
        "CHAPTER 2\nSouth Asia\nRecent developments\n"
        "D. Dependence on Middle East energy\n"
        "This panel shows the share of oil imports sourced from the region.",
        source="gep.pdf",
        file_name="gep.pdf",
        file_type="pdf",
        page=98,
    )

    chunks = chunk_documents([document], document_hash="gep-hash12", chunk_size=400, chunk_overlap=20)

    assert all(
        chunk.metadata.get("section_title") == "Chapter 2 - South Asia" for chunk in chunks
    )


def test_gpo_numbered_note_detection_is_unaffected_by_the_gep_patterns():
    document = _make_document(
        "9.  Workers' Compensation Liability\n"
        "Claims incurred and paid by DOL are approximately $5.2 million.",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=73,
    )

    chunks = chunk_documents([document], document_hash="gep-regress1", chunk_size=300, chunk_overlap=20)

    assert all(chunk.metadata.get("section_title") == "Note 9" for chunk in chunks)


def test_gpo_transition_majority_labeling_is_unaffected_by_the_gep_patterns():
    document = _make_document(
        "9.  Workers' Compensation Liability\n"
        "Claims incurred and paid by DOL as of September 30, 2018, and September 30, 2017, "
        "but not reimbursed to DOL by GPO, are approximately $5.2 million and $6 million "
        "respectively. This liability is included in accounts payable and accrued expenses. "
        "As of September 30, 2018, GPO estimated discounted current actuarially derived "
        "future workers' compensation liability of approximately $7.5 million and estimated "
        "discounted non-current actuarially derived future workers' liability of "
        "approximately $45.2 million.\n"
        "10. Commitments",
        source="gpo.pdf",
        file_name="gpo.pdf",
        file_type="pdf",
        page=73,
    )

    chunks = chunk_documents(
        [document], document_hash="gep-regress2", chunk_size=2000, chunk_overlap=100
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.get("section_title") == "Note 9"
