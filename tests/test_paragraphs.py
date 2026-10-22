"""Paragraph grouping must preserve columns, styles and untouched neighbours."""

import io
from types import SimpleNamespace

import numpy as np
import pikepdf
import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.convert import _overlay
from src.editor import FONTS, EditError, EditorDraft, OriginalEdit, export_pdf
from src.original import original_font, original_regions
from src.text import TextLayer
from tests.test_original import draft, raster, source


def paragraph_source(scanned=False, phrases=None):
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    phrases = phrases or [
        "First paragraph line with several words.",
        "Second line in the same paragraph.",
    ]
    if scanned:
        image = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(FONTS / "DejaVuSerif.ttf"), 24)
        for y, text in zip((120, 156), phrases):
            draw.text((80, y), text, font=font, fill="black", anchor="ls")
        draw.rectangle((600, 300, 800, 500), outline="blue", width=4)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        page.insert_image(page.rect, stream=buffer.getvalue())
        with pikepdf.open(io.BytesIO(doc.tobytes())) as pdf:
            layer = TextLayer(pdf)
            result = SimpleNamespace(
                rotation=0,
                lines=[
                    SimpleNamespace(text=text, box=(40, y - 12, 300, y + 3))
                    for text, y in zip(phrases, (60, 78))
                ],
            )
            _overlay(pdf.pages[0], result, 500, 500, [], layer)
            layer.finish()
            buffer = io.BytesIO()
            pdf.save(buffer)
            return buffer.getvalue()
    for y, text in zip((60, 78), phrases):
        page.insert_text((40, y), text, fontsize=12)
    page.insert_text((40, 120), "Separate paragraph", fontsize=12)
    page.draw_rect((300, 150, 400, 250), color=(0, 0, 1))
    return doc.tobytes()


@pytest.mark.parametrize("scanned", [False, True])
def test_group_and_replace_complete_paragraph(scanned):
    data = paragraph_source(scanned)
    group = original_regions(data, 1, "paragraph")[0]
    assert group["id"].endswith("-paragraph")
    assert group["lineCount"] == 2
    assert len(group["memberIds"]) == 2
    assert (
        group["text"]
        == "First paragraph line with several words. Second line in the same paragraph."
    )
    assert "members" not in group
    replacement = (
        "This replaces the entire paragraph, including both of its original lines."
    )
    edit = OriginalEdit(id=group["id"], page=1, text=replacement)
    output = export_pdf(data, EditorDraft(edits=[edit]))
    with fitz.open(stream=output, filetype="pdf") as doc:
        text = " ".join(p.get_text() for p in doc)
        assert "".join(replacement.split()) in "".join(text.split())
        assert "First paragraph" not in text
        assert "Second line" not in text
        if not scanned:
            assert "Separate paragraph" in text
    before, after = raster(data), raster(output)
    assert np.array_equal(before[300:500, 600:800], after[300:500, 600:800])


@pytest.mark.parametrize("scanned", [False, True])
def test_group_formatting_preserves_line_positions_and_styles(scanned):
    data = paragraph_source(scanned)
    group = original_regions(data, 1, "paragraph")[0]
    edit = OriginalEdit(
        id=group["id"], page=1, text=group["text"], color="#bb2244", bold=True
    )
    output = export_pdf(data, EditorDraft(edits=[edit]))
    before, after = raster(data), raster(output)
    mask = np.zeros(before.shape[:2], bool)
    for box in group["boxes"]:
        x, y, w, h = (box[k] * 2 for k in ("x", "y", "width", "height"))
        mask[
            max(0, int(y) - 4) : int(y + h) + 4, max(0, int(x) - 4) : int(x + w) + 4
        ] = True
    assert (before != after).any()
    assert not np.any(before[~mask] != after[~mask])


def test_overlap_is_rejected_and_legacy_line_edits_still_work():
    data = paragraph_source()
    group = original_regions(data, 1, "paragraph")[0]
    line = original_regions(data, 1)[0]
    edits = [
        OriginalEdit(id=item["id"], page=1, text="Changed text")
        for item in (group, line)
    ]
    with pytest.raises(EditError, match="paragraph and its individual lines"):
        export_pdf(data, EditorDraft(edits=edits))
    with fitz.open(
        stream=export_pdf(data, EditorDraft(edits=edits[1:])), filetype="pdf"
    ) as doc:
        assert "Changed text" in doc[0].get_text()
        assert "Second line" in doc[0].get_text()


def test_inline_styles_join_without_inventing_spaces():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 60), "Hello ", fontsize=12)
    x = 40 + fitz.get_text_length("Hello ", fontsize=12)
    page.insert_text((x, 60), "world", fontname="hebo", fontsize=12)
    page.insert_text((40, 77), "A second line continues here.", fontsize=12)
    data = doc.tobytes()
    groups = original_regions(data, 1, "paragraph")
    assert groups[0]["text"] == "Hello world A second line continues here."
    assert len(groups[0]["memberIds"]) == 3
    info = original_font(data, 1, groups[0]["id"])
    assert info["mixed"] is True


def test_columns_headings_lists_and_paragraph_breaks_stay_separate():
    doc = fitz.open()
    page = doc.new_page(width=600, height=600)
    entries = [
        (40, 40, "Heading", 20),
        (40, 70, "Left column first line", 12),
        (40, 86, "Left column second line", 12),
        (340, 70, "Right column first line", 12),
        (340, 86, "Right column second line", 12),
        (40, 140, "Another paragraph", 12),
        (40, 156, "1. First list item", 12),
        (40, 172, "2. Second list item", 12),
    ]
    for x, y, text, size in entries:
        page.insert_text((x, y), text, fontsize=size)
    groups = original_regions(doc.tobytes(), 1, "paragraph")
    assert [item["text"] for item in groups] == [
        "Heading",
        "Left column first line Left column second line",
        "Right column first line Right column second line",
        "Another paragraph",
        "1. First list item",
        "2. Second list item",
    ]


@pytest.mark.parametrize("mode", ["wrap", "shrink", "page"])
@pytest.mark.parametrize("scanned", [False, True])
def test_overflow_modes_preserve_every_character(mode, scanned):
    data = source(scanned)
    text = (
        "Long replacement with explicit breaks.\n"
        + "Repeated words stay selectable. " * 50
    )
    output = export_pdf(data, draft(data, text=text, overflow=mode))
    with fitz.open(stream=output, filetype="pdf") as doc:
        complete = "".join(p.get_text() for p in doc).replace("Other line", "")
        assert "".join(complete.split()) == "".join(text.split())
        spans = [
            s
            for p in doc
            for b in p.get_text("dict")["blocks"]
            for l in b.get("lines", [])
            for s in l["spans"]
            if s["text"] != "Other line"
        ]
        size = 16 if scanned else 18
        assert all(size * 0.85 - 0.1 <= s["size"] <= size + 0.1 for s in spans)
        if mode == "page":
            assert len(doc) > 1
            assert len(doc[0].get_text().replace("Other line", "")) < 80
    assert np.array_equal(
        raster(data)[400:600, 400:600], raster(output)[400:600, 400:600]
    )


def test_slight_shrink_fits_one_line_at_largest_acceptable_size():
    data = source()
    text = "Original words plus quite a few extra words!!!"
    output = export_pdf(data, draft(data, text=text, overflow="shrink"))
    with fitz.open(stream=output, filetype="pdf") as doc:
        spans = [
            s
            for b in doc[0].get_text("dict")["blocks"]
            for l in b.get("lines", [])
            for s in l["spans"]
            if s["text"] == text
        ]
        assert len(doc) == 1
        assert len(spans) == 1
        assert 18 * 0.85 <= spans[0]["size"] < 18


@pytest.mark.parametrize("scanned", [False, True])
@pytest.mark.parametrize(
    "family",
    [
        "DejaVuSans",
        "DejaVuSerif",
        "DejaVuSansMono",
        "LiberationSans",
        "LiberationSerif",
        "LiberationMono",
    ],
)
def test_font_override_uses_selected_family_at_original_size(scanned, family):
    data = source(scanned)
    output = export_pdf(data, draft(data, text="New words", font=family))
    with fitz.open(stream=output, filetype="pdf") as doc:
        spans = [
            s
            for b in doc[0].get_text("dict")["blocks"]
            for l in b.get("lines", [])
            for s in l["spans"]
            if s["text"] == "New words"
        ]
        assert len(spans) == 1
        assert spans[0]["font"].replace(" ", "").startswith(family)
        assert spans[0]["size"] == pytest.approx(16 if scanned else 18, abs=0.1)


@pytest.mark.parametrize("scanned", [False, True])
def test_font_information_matches_actual_rewrite(scanned, tmp_path):
    data = source(scanned)
    path = tmp_path / "source.pdf"
    path.write_bytes(data)
    item = original_regions(data, 1)[0]
    info = original_font(path, 1, item["id"])
    assert info["exact"] is not scanned
    assert info["size"] == pytest.approx(16 if scanned else 18, abs=0.1)
    assert "Serif" in info["name"] if scanned else info["name"] == "Helvetica"


def test_color_only_edit_retains_mixed_native_weights():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 60), "Normal ", fontsize=12)
    x = 40 + fitz.get_text_length("Normal ", fontsize=12)
    page.insert_text((x, 60), "bold", fontsize=12, fontname="hebo")
    data = doc.tobytes()
    group = original_regions(data, 1, "paragraph")[0]
    edit = OriginalEdit(id=group["id"], page=1, text=group["text"], color="#bb2244")
    output = export_pdf(data, EditorDraft(edits=[edit]))
    with fitz.open(stream=output, filetype="pdf") as edited:
        spans = [
            s
            for b in edited[0].get_text("dict")["blocks"]
            for l in b.get("lines", [])
            for s in l["spans"]
        ]
        assert any("bold" in s["text"] and s["flags"] & 16 for s in spans)
        assert all(s["color"] == 0xBB2244 for s in spans)


def test_paragraph_with_short_bold_intro_uses_main_body_font():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text((40, 60), "Note: ", fontsize=12, fontname="hebo")
    x = 40 + fitz.get_text_length("Note: ", fontsize=12, fontname="hebo")
    page.insert_text((x, 60), "the main body is regular text.", fontsize=12)
    page.insert_text((40, 78), "And it continues on the next line.", fontsize=12)
    data = doc.tobytes()
    groups = original_regions(data, 1, "paragraph")
    assert len(groups) == 1
    group = groups[0]
    assert group["lineCount"] == 2
    assert not group["bold"]
    assert original_font(data, 1, group["id"])["name"] == "Helvetica"
    edit = OriginalEdit(
        id=group["id"], page=1, text="A completely rewritten regular paragraph."
    )
    with fitz.open(
        stream=export_pdf(data, EditorDraft(edits=[edit])), filetype="pdf"
    ) as output:
        span = output[0].get_text("dict")["blocks"][0]["lines"][0]["spans"][0]
        assert not span["flags"] & 16
        assert span["origin"] == pytest.approx((40, 60), abs=0.1)
        assert span["size"] == pytest.approx(12, abs=0.1)


def test_scan_font_sample_from_second_line_still_starts_at_original_position():
    data = paragraph_source(
        True, ["Short first line.", "The second line contains many more words."]
    )
    group = original_regions(data, 1, "paragraph")[0]
    info = original_font(data, 1, group["id"])
    assert "DejaVuSerif" in info["name"].replace(" ", "")
    edit = OriginalEdit(id=group["id"], page=1, text="A rewritten paragraph.")
    with fitz.open(
        stream=export_pdf(data, EditorDraft(edits=[edit])), filetype="pdf"
    ) as output:
        span = output[0].get_text("dict")["blocks"][-1]["lines"][0]["spans"][0]
        assert span["origin"] == pytest.approx((40, 60), abs=0.8)
        assert span["size"] == pytest.approx(12, abs=0.5)


@pytest.mark.parametrize("scanned", [False, True])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_paragraph_rewrite_with_crop_rotation_and_pdf_userunit(scanned, rotation):
    with pikepdf.open(io.BytesIO(paragraph_source(scanned))) as pdf:
        pdf.pages[0].CropBox = [20, 20, 480, 480]
        pdf.pages[0].UserUnit = 2
        pdf.pages[0].Rotate = rotation
        buffer = io.BytesIO()
        pdf.save(buffer)
    data = buffer.getvalue()
    group = original_regions(data, 1, "paragraph")[0]
    replacement = "A rewritten paragraph."
    output = export_pdf(
        data,
        EditorDraft(edits=[OriginalEdit(id=group["id"], page=1, text=replacement)]),
    )
    with fitz.open(stream=output, filetype="pdf") as doc:
        assert len(doc) == 1
        assert replacement in doc[0].get_text()
        assert "First paragraph" not in doc[0].get_text()
        spans = [
            s
            for b in doc[0].get_text("dict")["blocks"]
            for l in b.get("lines", [])
            for s in l["spans"]
            if s["text"] == replacement
        ]
        assert spans[0]["size"] == pytest.approx(24, abs=0.7)
        assert spans[0]["origin"] == pytest.approx((40, 80), abs=1.5)
    before, after = raster(data), raster(output)
    # Outside the two original lines and the replacement bounds, all pixels stay.
    allowed = np.zeros(before.shape[:2], bool)
    for box in group["boxes"]:
        x, y, w, h = (box[k] * 2 for k in ("x", "y", "width", "height"))
        allowed[
            max(0, int(y) - 8) : int(y + h) + 8, max(0, int(x) - 8) : int(x + w) + 8
        ] = True
    assert not np.any(before[~allowed] != after[~allowed])
