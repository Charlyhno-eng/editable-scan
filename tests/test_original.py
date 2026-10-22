"""Existing text replacement, scanned glyph fidelity and displayed coordinates."""

import io
from types import SimpleNamespace

import numpy as np
import pikepdf
import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.convert import _overlay
from src.editor import FONTS, EditError, EditorDraft, OriginalEdit, export_pdf
from src.original import original_regions
from src.text import TextLayer


def source(scanned=False, rotation=0):
    doc = fitz.open()
    page = doc.new_page(width=400, height=500)
    if scanned:
        image = Image.new("RGB", (800, 1000), (246, 244, 238))
        draw = ImageDraw.Draw(image)
        draw.text(
            (80, 80),
            "Original words",
            font=ImageFont.truetype(str(FONTS / "DejaVuSerif.ttf"), 32),
            fill=(20, 20, 20),
        )
        draw.rectangle((400, 400, 600, 600), outline=(40, 80, 180), width=5)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        page.insert_image(page.rect, stream=buf.getvalue())
        with pikepdf.open(io.BytesIO(doc.tobytes())) as pdf:
            layer = TextLayer(pdf)
            result = SimpleNamespace(
                rotation=0,
                lines=[SimpleNamespace(text="Original words", box=(40, 40, 161, 59))],
            )
            _overlay(pdf.pages[0], result, 400, 500, [], layer)
            layer.finish()
            buf = io.BytesIO()
            pdf.save(buf)
            doc.close()
            doc = fitz.open(stream=buf.getvalue(), filetype="pdf")
            page = doc[0]
    else:
        page.insert_text((40, 70), "Original words", fontname="helv", fontsize=18)
        page.insert_text((40, 120), "Other line", fontname="tiro", fontsize=16)
        page.draw_rect(fitz.Rect(200, 200, 300, 300), color=(0.2, 0.4, 0.6))
    page.set_rotation(rotation)
    return doc.tobytes()


def raster(data):
    with fitz.open(stream=data, filetype="pdf") as doc:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
        return (
            np.frombuffer(pix.samples, np.uint8)
            .reshape(pix.height, pix.width, 3)
            .copy()
        )


def draft(data, **changes):
    item = original_regions(data, 1)[0]
    return EditorDraft(
        edits=[
            OriginalEdit(
                **{k: item[k] for k in ("id", "page", "text", "bold", "color")}
            ).model_copy(update=changes)
        ]
    )


@pytest.mark.parametrize("scanned", [False, True])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_replace_original_removes_old_searchable_text_preserves_other_content(
    scanned, rotation
):
    data = source(scanned, rotation)
    item = original_regions(data, 1)[0]
    output = export_pdf(data, draft(data, text="Changed words", color="#bb2244"))
    with fitz.open(stream=output, filetype="pdf") as doc:
        text = doc[0].get_text()
        assert "Original words" not in text
        assert text.count("Changed words") == 1
        if not scanned:
            assert "Other line" in text
        assert doc[0].rotation == rotation
    before, after = raster(data), raster(output)
    changed = np.any(before != after, axis=2)
    allowed = np.zeros(changed.shape, bool)
    x, y, w, h = (item[k] * 2 for k in ("x", "y", "width", "height"))
    allowed[
        max(0, int(y) - 4) : int(y + h) + 4, max(0, int(x) - 4) : int(x + w) + 4
    ] = True
    assert changed.any()
    assert not changed[~allowed].any()


def test_recolor_scanned_glyphs_retains_shape_and_paper():
    data = source(True)
    output = export_pdf(data, draft(data, color="#bb2244"))
    before, after = raster(data), raster(output)
    original_ink = before[82:116, 80:325].min(axis=2) < 80
    new_ink = after[82:116, 80:325, 1] < 100
    assert (
        np.logical_and(original_ink, new_ink).sum()
        / np.logical_or(original_ink, new_ink).sum()
        > 0.95
    )
    assert np.array_equal(before[400:600, 400:600], after[400:600, 400:600])


def test_bold_and_color_reuse_digital_font():
    data = source()
    output = export_pdf(data, draft(data, bold=True, color="#bb2244"))
    with fitz.open(stream=output, filetype="pdf") as doc:
        spans = [
            s
            for b in doc[0].get_text("dict")["blocks"]
            for l in b.get("lines", [])
            for s in l["spans"]
        ]
        span = next(s for s in spans if s["text"] == "Original words")
        assert span["font"] in ("Helvetica", "NimbusSans-Regular")
        assert span["color"] == 0xBB2244
    assert (raster(data) != raster(output)).any()


def test_unchanged_draft_byte_identical_and_delete_search_text():
    data = source()
    assert export_pdf(data, EditorDraft()) == data
    output = export_pdf(data, draft(data, text=""))
    with fitz.open(stream=output, filetype="pdf") as doc:
        assert "Original words" not in doc[0].get_text()
        assert "Other line" in doc[0].get_text()


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "p1-b99-l0-s0"},
        {"text": "long " * 200},
        {"text": "two\nlines"},
        {"text": "漢"},
    ],
)
def test_invalid_edits_are_rejected(changes):
    data = source()
    with pytest.raises(EditError):
        export_pdf(data, draft(data, **changes))


@pytest.mark.parametrize("scanned", [False, True])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_cropbox_and_userunit(scanned, rotation):
    with pikepdf.open(io.BytesIO(source(scanned, rotation))) as doc:
        doc.pages[0].CropBox = [20, 20, 380, 480]
        doc.pages[0].UserUnit = 2
        buffer = io.BytesIO()
        doc.save(buffer)
    data = buffer.getvalue()
    item = original_regions(data, 1)[0]
    output = export_pdf(data, draft(data, color="#bb2244"))
    before, after = raster(data), raster(output)
    changed = np.any(before != after, axis=2)
    allowed = np.zeros(changed.shape, bool)
    x, y, w, h = (item[k] * 2 for k in ("x", "y", "width", "height"))
    allowed[
        max(0, int(y) - 10) : int(y + h) + 10, max(0, int(x) - 10) : int(x + w) + 10
    ] = True
    assert changed.any()
    assert not changed[~allowed].any()


def test_embedded_font_glyphs_and_original_metadata_survive_color_change():
    doc = fitz.open()
    page = doc.new_page(width=400, height=500)
    page.insert_font(fontname="Original", fontfile=str(FONTS / "DejaVuSerif.ttf"))
    page.insert_text((40, 70), "Original words", fontname="Original", fontsize=18)
    page.insert_link(
        {
            "kind": fitz.LINK_URI,
            "from": fitz.Rect(200, 200, 300, 250),
            "uri": "https://example.com",
        }
    )
    doc.set_metadata({"title": "Original document"})
    data = doc.tobytes()
    output = export_pdf(data, draft(data, color="#bb2244"))
    with fitz.open(stream=output, filetype="pdf") as edited:
        span = edited[0].get_text("dict")["blocks"][0]["lines"][0]["spans"][0]
        assert span["font"].split("+")[-1] in ("DejaVuSerif", "DejaVu Serif Book")
        assert span["text"] == "Original words"
        assert edited.metadata["title"] == "Original document"
        assert edited[0].get_links()[0]["uri"] == "https://example.com"
    # Compare actual glyph outlines: subset font names and raster antialiasing
    # can differ even when the exact embedded font program is reused.
    from fontTools.pens.recordingPen import RecordingPen
    from fontTools.ttLib import TTFont

    with fitz.open(stream=output, filetype="pdf") as edited:
        embedded = [edited.extract_font(entry[0])[3] for entry in edited[0].get_fonts()]
        glyph_id = next(
            char[1]
            for trace in edited[0].get_texttrace()
            for char in trace["chars"]
            if char[0] == ord("O")
        )
    with TTFont(FONTS / "DejaVuSerif.ttf") as original:
        reference = RecordingPen()
        original.getGlyphSet()[original.getBestCmap()[ord("O")]].draw(reference)
    matched = False
    for data in embedded:
        if not data:
            continue
        with TTFont(io.BytesIO(data)) as font:
            pen = RecordingPen()
            font.getGlyphSet()[font.getGlyphOrder()[glyph_id]].draw(pen)
            matched |= pen.value == reference.value
    assert matched
