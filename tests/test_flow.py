"""Replacement text must use its column without hopping over existing lines."""

import numpy as np
import pymupdf as fitz
import pytest

from src.editor import EditorDraft, OriginalEdit, export_pdf
from src.original import original_regions
from tests.test_original import raster
from tests.test_paragraphs import paragraph_source


def spans(page):
    return [
        span
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
    ]


@pytest.mark.parametrize("scanned", [False, True])
@pytest.mark.parametrize("mode", ["wrap", "shrink", "page"])
def test_short_paragraph_can_fill_the_available_line(scanned, mode):
    data = paragraph_source(scanned, ["Two words", "Short end"])
    item = original_regions(data, 1, "paragraph")[0]
    text = "Two words followed by enough words to fill the available line."
    output = export_pdf(
        data,
        EditorDraft(
            edits=[OriginalEdit(id=item["id"], page=1, text=text, overflow=mode)]
        ),
    )
    with fitz.open(stream=output, filetype="pdf") as doc:
        replacements = [s for s in spans(doc[0]) if s["text"] != "Separate paragraph"]
        assert len(doc) == 1
        assert len(replacements) == 1
        assert replacements[0]["text"] == text
        assert replacements[0]["bbox"][2] <= 462
        assert replacements[0]["size"] == pytest.approx(12, abs=0.1)


@pytest.mark.parametrize("scanned", [False, True])
def test_wrapped_replacement_stops_before_the_next_unchanged_line(scanned):
    data = paragraph_source(
        scanned, ["Two words", "Keep this existing line exactly where it is."]
    )
    item = original_regions(data, 1)[0]
    text = "A replacement that is too long for one line. " * 8
    output = export_pdf(
        data, EditorDraft(edits=[OriginalEdit(id=item["id"], page=1, text=text)])
    )
    with fitz.open(stream=output, filetype="pdf") as edited:
        assert len(edited) > 1
        first = spans(edited[0])
        replacement = [
            s
            for s in first
            if not s["text"].startswith("Keep this")
            and s["text"] != "Separate paragraph"
        ]
        assert len(replacement) == 1
        assert replacement[0]["origin"][1] == pytest.approx(60, abs=1)
        complete = (
            "".join(p.get_text() for p in edited)
            .replace("Keep this existing line exactly where it is.", "")
            .replace("Separate paragraph", "")
        )
        assert "".join(complete.split()) == "".join(text.split())
    assert np.array_equal(raster(data)[130:165], raster(output)[130:165])


def test_wrapping_uses_the_document_margin_and_original_line_spacing():
    data = paragraph_source(False, ["Two words", "Short end"])
    item = original_regions(data, 1, "paragraph")[0]
    text = "A complete replacement paragraph using the width of the page. " * 2
    output = export_pdf(
        data, EditorDraft(edits=[OriginalEdit(id=item["id"], page=1, text=text)])
    )
    with fitz.open(stream=output, filetype="pdf") as doc:
        replacement = [s for s in spans(doc[0]) if s["text"] != "Separate paragraph"]
        assert len(replacement) == 2
        assert [s["origin"][1] for s in replacement] == pytest.approx([60, 78])
        assert all(s["bbox"][2] <= 460.5 for s in replacement)
        # A short final source line must not determine the paragraph's width.
        assert replacement[0]["bbox"][2] > 400


@pytest.mark.parametrize("heading", [False, True])
def test_short_column_does_not_extend_into_the_neighbouring_column(heading):
    doc = fitz.open()
    page = doc.new_page(width=600, height=500)
    if heading:
        page.insert_text(
            (40, 30),
            "A wide heading spanning both columns of the original document",
            fontsize=12,
        )
    for x, y, text in (
        (40, 60, "Two words"),
        (40, 78, "Short end"),
        (340, 60, "Right column first line"),
        (340, 78, "Right column second line"),
        (340, 96, "Right column third line"),
    ):
        page.insert_text((x, y), text, fontsize=12)
    data = doc.tobytes()
    item = next(
        r
        for r in original_regions(data, 1, "paragraph")
        if r["text"].startswith("Two words")
    )
    text = "A replacement using the available left column width. " * 8
    output = export_pdf(
        data, EditorDraft(edits=[OriginalEdit(id=item["id"], page=1, text=text)])
    )
    with fitz.open(stream=output, filetype="pdf") as edited:
        replacement = [
            s
            for s in spans(edited[0])
            if not s["text"].startswith(("Right", "A wide heading"))
        ]
        assert len(replacement) >= 2
        assert replacement[0]["bbox"][2] > 250
        assert all(s["bbox"][2] <= 322 for s in replacement)
        assert "Right column third line" in edited[0].get_text()
    assert np.array_equal(raster(data)[:, 660:], raster(output)[:, 660:])


def test_an_indented_paragraph_below_is_not_mistaken_for_a_column():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text((40, 60), "Two words", fontsize=12)
    page.insert_text((40, 78), "Short end", fontsize=12)
    for y in (100, 118, 136):
        page.insert_text((180, y), "An indented passage.", fontsize=12)
    data = doc.tobytes()
    item = original_regions(data, 1, "paragraph")[0]
    text = "Two words followed by enough words to fill the available line."
    output = export_pdf(
        data, EditorDraft(edits=[OriginalEdit(id=item["id"], page=1, text=text)])
    )
    with fitz.open(stream=output, filetype="pdf") as edited:
        replacement = [
            s for s in spans(edited[0]) if s["text"] != "An indented passage."
        ]
        assert len(replacement) == 1
        assert replacement[0]["text"] == text


def test_repeated_text_edges_define_a_narrower_body_frame():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    body = "An established text column ends here."
    for y in (60, 78, 96):
        page.insert_text((40, y), body, fontsize=12)
    page.insert_text((40, 150), "Two words", fontsize=12)
    data = doc.tobytes()
    item = original_regions(data, 1)[-1]
    text = "A replacement that respects the established text column. " * 2
    output = export_pdf(
        data, EditorDraft(edits=[OriginalEdit(id=item["id"], page=1, text=text)])
    )
    right = 40 + fitz.get_text_length(body, fontsize=12)
    with fitz.open(stream=output, filetype="pdf") as edited:
        replacement = [s for s in spans(edited[0]) if s["text"] != body]
        assert len(replacement) >= 3
        assert all(s["bbox"][2] <= right + 0.6 for s in replacement)
        assert replacement[0]["bbox"][2] > right - 50


def test_an_indented_unchanged_line_is_not_wrapped_around():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text((40, 60), "Two words", fontsize=12)
    page.insert_text((140, 78), "An indented unchanged line.", fontsize=12)
    data = doc.tobytes()
    item = original_regions(data, 1)[0]
    text = (
        "Words that fill the line and continue without weaving through another line. "
        * 6
    )
    output = export_pdf(
        data, EditorDraft(edits=[OriginalEdit(id=item["id"], page=1, text=text)])
    )
    with fitz.open(stream=output, filetype="pdf") as edited:
        replacement = [
            s for s in spans(edited[0]) if not s["text"].startswith("An indented")
        ]
        assert len(replacement) == 1
        assert len(edited) > 1


def test_a_figure_below_the_text_is_not_skipped():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text((40, 60), "Two words", fontsize=12)
    page.draw_rect((40, 80, 250, 150), color=(0, 0, 1))
    data = doc.tobytes()
    item = original_regions(data, 1)[0]
    text = "A replacement that must continue before the figure. " * 7
    output = export_pdf(
        data, EditorDraft(edits=[OriginalEdit(id=item["id"], page=1, text=text)])
    )
    with fitz.open(stream=output, filetype="pdf") as edited:
        replacement = spans(edited[0])
        assert len(replacement) == 2
        assert max(s["bbox"][3] for s in replacement) < 80
        assert len(edited) > 1
        assert "".join("".join(p.get_text() for p in edited).split()) == "".join(
            text.split()
        )
    assert np.array_equal(
        raster(data)[159:302, 79:502], raster(output)[159:302, 79:502]
    )


def test_tightly_spaced_source_lines_keep_their_baseline_grid():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text((40, 60), "Two words", fontsize=12)
    page.insert_text((40, 72), "Short end", fontsize=12)
    data = doc.tobytes()
    item = original_regions(data, 1, "paragraph")[0]
    text = "A complete replacement paragraph using the width of the page. " * 2
    output = export_pdf(
        data, EditorDraft(edits=[OriginalEdit(id=item["id"], page=1, text=text)])
    )
    with fitz.open(stream=output, filetype="pdf") as edited:
        assert len(edited) == 1
        assert [s["origin"][1] for s in spans(edited[0])] == pytest.approx([60, 72])


def test_existing_text_near_the_bottom_is_not_forced_onto_another_page():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text((40, 480), "Two words", fontsize=12)
    data = doc.tobytes()
    item = original_regions(data, 1)[0]
    text = "This text still fits on its original line near the bottom."
    output = export_pdf(
        data, EditorDraft(edits=[OriginalEdit(id=item["id"], page=1, text=text)])
    )
    with fitz.open(stream=output, filetype="pdf") as edited:
        assert len(edited) == 1
        assert edited[0].get_text().strip() == text
        assert spans(edited[0])[0]["origin"][1] == pytest.approx(480)


def test_a_page_number_does_not_widen_the_body_text_frame():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text((40, 60), "Two words", fontsize=12)
    page.insert_text((10, 480), "1", fontsize=12)
    data = doc.tobytes()
    item = original_regions(data, 1)[0]
    text = "Words respecting the main body margin rather than the page number. " * 2
    output = export_pdf(
        data, EditorDraft(edits=[OriginalEdit(id=item["id"], page=1, text=text)])
    )
    with fitz.open(stream=output, filetype="pdf") as edited:
        replacement = [s for s in spans(edited[0]) if s["text"] != "1"]
        assert len(replacement) >= 2
        assert all(s["bbox"][2] <= 460.5 for s in replacement)
