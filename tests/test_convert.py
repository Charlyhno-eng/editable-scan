"""Regression checks: original pixels must survive, OCR must stay invisible."""

from contextlib import closing
from functools import partial
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageOps

from src import convert

FIXTURES = Path(__file__).parent / "fixtures"
convert_original = partial(convert.convert_pdf, enhance=False)


def render(source, index=0, scale=1, size=None):
    with pdfium.PdfDocument(source) as document, closing(document[index]) as page:
        if size:
            scale = min(
                (size[0] - 1e-5) / page.get_width(),
                (size[1] - 1e-5) / page.get_height(),
            )
        with closing(page.render(scale=scale)) as bitmap:
            return np.array(bitmap.to_pil().convert("RGB"))


def text(source, index=0):
    with (
        pdfium.PdfDocument(source) as document,
        closing(document[index]) as page,
        closing(page.get_textpage()) as tp,
    ):
        return tp.get_text_bounded()


def result(text="Selectable text: café", rotation=0):
    # Accented document content checks Unicode support regardless of the UI language.
    return SimpleNamespace(
        source="ocr",
        rotation=rotation,
        lines=[SimpleNamespace(text=text, box=(30, 40, 300, 60))],
    )


@pytest.fixture
def fake_ocr(monkeypatch):
    monkeypatch.setattr(convert, "_engine", lambda: lambda _gray: result())


def test_real_ocr_keeps_original_pixels_and_adds_searchable_accents(tmp_path):
    output = tmp_path / "result.pdf"
    progress = []
    convert_original(
        FIXTURES / "scan.png", output, lambda n, t: progress.append((n, t))
    )
    with Image.open(FIXTURES / "scan.png") as image:
        np.testing.assert_array_equal(
            render(output, size=image.size), np.array(image.convert("RGB"))
        )
    extracted = text(output)
    assert "Bonjour" in extracted
    assert "français" in extracted
    assert progress == [(0, 1), (1, 1)]
    with pikepdf.open(output) as pdf:
        forms = pdf.pages[0].Resources.XObject.values()
        assert any(
            b"3 Tr" in obj.read_bytes() for obj in forms if obj.Subtype == "/Form"
        )


def test_real_multipage_scan_keeps_figures_and_pixels(tmp_path):
    source, output = FIXTURES / "scan.pdf", tmp_path / "result.pdf"
    assert convert_original(source, output) == 3
    for i in range(3):
        np.testing.assert_array_equal(render(source, index=i), render(output, index=i))
        assert "Bonjour" in text(output, index=i)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("ocr_rotation", [0, 90, 180, 270])
def test_crop_rotation_and_ocr_turn_preserve_pixels_and_selection(
    tmp_path, monkeypatch, rotation, ocr_rotation
):
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    with pikepdf.open(FIXTURES / "scan.pdf") as pdf:
        del pdf.pages[1:]
        page = pdf.pages[0]
        page.CropBox = pikepdf.Array([20, 30, 330, 450])
        page.Rotate = rotation
        pdf.save(source)

    def ocr(gray):
        h, w = gray.shape
        # OCR boxes are measured after the engine's clockwise upright correction.
        lines = []
        for content, y0, y1 in [("Café selection", 40, 60), ("Second line", 100, 120)]:
            corners = []
            for x, y in [(30, y0), (300, y1)]:
                if ocr_rotation == 90:
                    x, y = h - y, x
                elif ocr_rotation == 180:
                    x, y = w - x, h - y
                elif ocr_rotation == 270:
                    x, y = y, w - x
                corners.append((x, y))
            box = (
                min(p[0] for p in corners),
                min(p[1] for p in corners),
                max(p[0] for p in corners),
                max(p[1] for p in corners),
            )
            lines.append(SimpleNamespace(text=content, box=box))
        return SimpleNamespace(
            source="ocr",
            rotation=ocr_rotation,
            lines=lines,
        )

    monkeypatch.setattr(convert, "_engine", lambda: ocr)
    convert_original(source, output)
    np.testing.assert_array_equal(render(source), render(output))
    assert "Café selection" in text(output)
    assert "Second line" in text(output)
    with pikepdf.open(output) as pdf:
        assert int(pdf.pages[0].Rotate) == rotation
        assert list(pdf.pages[0].CropBox) == [20, 30, 330, 450]
    # Text selection must align with the pre-turn location in the displayed page.
    with (
        pdfium.PdfDocument(output) as doc,
        closing(doc[0]) as page,
        closing(page.get_textpage()) as tp,
        closing(page.render(scale=convert.OCR_DPI / 72)) as bitmap,
    ):
        assert tp.count_rects() == 2
        pos = bitmap.get_posconv(page)
        for index, expected_y in enumerate([50, 110]):
            left, bottom, right, top = tp.get_rect(index)
            a, b = pos.to_bitmap(left, top), pos.to_bitmap(right, bottom)
            center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            assert abs(center[0] - 165) < 5
            assert abs(center[1] - expected_y) < 5


@pytest.mark.parametrize("scale", [1, 4])
def test_digital_pdf_keeps_text_links_and_metadata_without_ocr(
    tmp_path, monkeypatch, scale
):
    source, output = tmp_path / "original.pdf", tmp_path / "output.pdf"
    pdf = FPDF()
    pdf.set_title("Original document")
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(text="Hello, existing text.", link="https://example.com")
    pdf.output(source)

    def never():
        pytest.fail("Digital text must not be recognized again")

    monkeypatch.setattr(convert, "_engine", never)
    convert_original(source, output)
    assert text(source) == text(output)
    np.testing.assert_array_equal(
        render(source, scale=scale), render(output, scale=scale)
    )
    with pikepdf.open(source) as a, pikepdf.open(output) as b:
        assert a.docinfo.Title == b.docinfo.Title
        assert a.pages[0].Annots[0].A.URI == b.pages[0].Annots[0].A.URI


def test_mixed_page_does_not_duplicate_existing_text(tmp_path, monkeypatch):
    source, output = tmp_path / "original.pdf", tmp_path / "output.pdf"
    pdf = FPDF(unit="pt", format=(360, 480))
    pdf.add_page()
    pdf.image(FIXTURES / "scan.png", x=0, y=0, w=360, h=480)
    pdf.set_font("Helvetica", size=15)
    pdf.text(30, 390, "Existing text")
    pdf.output(source)

    def ocr(gray):
        k = gray.shape[1] / 360
        return SimpleNamespace(
            source="ocr",
            rotation=0,
            lines=[
                SimpleNamespace(
                    text="Existing text", box=(30 * k, 377 * k, 125 * k, 394 * k)
                ),
                SimpleNamespace(text="New text", box=(30 * k, 40 * k, 300 * k, 60 * k)),
            ],
        )

    monkeypatch.setattr(convert, "_engine", lambda: ocr)
    convert_original(source, output)
    assert text(output).count("Existing text") == 1
    assert "New text" in text(output)
    np.testing.assert_array_equal(render(source), render(output))


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP", "TIFF", "BMP", "GIF"])
def test_image_formats_preserve_decoded_pixels(tmp_path, fake_ocr, format):
    source, output = tmp_path / f"image.{format}", tmp_path / "output.pdf"
    with Image.open(FIXTURES / "scan.png") as im:
        im.save(source, format=format)
    convert_original(source, output)
    with Image.open(source) as original:
        np.testing.assert_array_equal(
            render(output, size=original.size), np.array(original.convert("RGB"))
        )
        with pikepdf.open(output) as pdf:
            embedded = [
                obj
                for obj in pdf.objects
                if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image"
            ]
            assert len(embedded) == 1
            assert (embedded[0].Width, embedded[0].Height) == original.size
            assert embedded[0].Filter == "/FlateDecode"
            assert embedded[0].Interpolate is False
    assert "café" in text(output)


def test_high_resolution_image_stays_intact_when_ocr_is_downscaled(
    tmp_path, monkeypatch
):
    source, output = tmp_path / "fine-lines.png", tmp_path / "output.pdf"
    image = Image.new("RGB", (2400, 3200), "white")
    drawing = ImageDraw.Draw(image)
    for x in range(20, 2380, 7):
        drawing.line((x, 20, x, 3180), fill="black", width=1)
    image.save(source, dpi=(600, 600))
    seen = []

    def ocr(gray):
        seen.append(gray.shape)
        return result()

    monkeypatch.setattr(convert, "RENDER_PIXELS", 10_000)
    monkeypatch.setattr(convert, "_engine", lambda: ocr)
    convert_original(source, output)
    assert len(seen) == 1
    assert seen[0][0] * seen[0][1] < 10_300
    with pikepdf.open(output) as pdf:
        embedded = next(
            obj
            for obj in pdf.objects
            if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image"
        )
        assert (embedded.Width, embedded.Height) == image.size
        np.testing.assert_array_equal(
            np.array(pikepdf.PdfImage(embedded).as_pil_image()), np.array(image)
        )


def test_nested_pdf_images_and_masks_keep_encoded_samples_without_smoothing(
    tmp_path, fake_ocr
):
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    image = Image.new("RGBA", (200, 100), (20, 60, 120, 128))
    pdf = FPDF(unit="pt", format=(200, 100))
    pdf.add_page()
    pdf.image(image, x=0, y=0, w=200, h=100)
    pdf.output(source)
    with pikepdf.open(source, allow_overwriting_input=True) as doc:
        for obj in doc.objects:
            if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image":
                obj.Interpolate = True
        doc.save(source)

    def samples(path):
        with pikepdf.open(path) as doc:
            return sorted(
                (
                    int(obj.Width),
                    int(obj.Height),
                    str(obj.Filter),
                    sha256(obj.read_raw_bytes()).hexdigest(),
                    bool(obj.Interpolate),
                )
                for obj in doc.objects
                if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image"
            )

    before = samples(source)
    assert len(before) == 2  # The color image and its transparency mask.
    assert all(row[-1] for row in before)
    convert_original(source, output)
    after = samples(output)
    assert [row[:-1] for row in before] == [row[:-1] for row in after]
    assert all(not row[-1] for row in after)
    with pikepdf.open(output) as doc:
        assert any(
            obj.get("/Subtype") == "/Form"
            for obj in doc.pages[0].Resources.XObject.values()
        )


def test_tiff_all_frames_and_dpi(tmp_path, fake_ocr):
    source, output = tmp_path / "input.tiff", tmp_path / "output.pdf"
    images = [Image.new("RGB", (300, 200), color) for color in ["red", "green", "blue"]]
    images[0].save(source, save_all=True, append_images=images[1:], dpi=(72, 72))
    assert convert_original(source, output) == 3
    for i, original in enumerate(images):
        np.testing.assert_array_equal(render(output, index=i), np.array(original))


def test_transparency_is_composited_on_white(tmp_path, fake_ocr):
    source, output = tmp_path / "input.png", tmp_path / "output.pdf"
    image = Image.new("RGBA", (200, 100), (20, 60, 120, 128))
    image.save(source, dpi=(72, 72))
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    np.testing.assert_array_equal(
        render_after_conversion(source, output), np.array(background.convert("RGB"))
    )


def render_after_conversion(source, output):
    convert_original(source, output)
    return render(output)


def test_exif_orientation_preserved(tmp_path, fake_ocr):
    source, output = tmp_path / "input.jpg", tmp_path / "output.pdf"
    image = Image.open(FIXTURES / "scan.png")
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    convert_original(source, output)
    with Image.open(source) as original:
        expected = np.array(ImageOps.exif_transpose(original).convert("RGB"))
    np.testing.assert_array_equal(
        render(output, size=(expected.shape[1], expected.shape[0])), expected
    )


def test_blank_and_illustration_only_pages_are_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(
        convert,
        "_engine",
        lambda: lambda _: SimpleNamespace(source="ocr", rotation=0, lines=[]),
    )
    output = tmp_path / "output.pdf"
    assert convert_original(FIXTURES / "scan.pdf", output) == 3
    assert not text(output)
    np.testing.assert_array_equal(render(FIXTURES / "scan.pdf"), render(output))


def test_ocr_error_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(
        convert, "_engine", lambda: lambda _: SimpleNamespace(source="error")
    )
    with pytest.raises(convert.DocumentError, match="page 1"):
        convert_original(FIXTURES / "scan.png", tmp_path / "output.pdf")


def test_encrypted_pdf_rejected(tmp_path):
    source = tmp_path / "locked.pdf"
    with pikepdf.open(FIXTURES / "scan.pdf") as pdf:
        pdf.save(source, encryption=pikepdf.Encryption(user="secret", owner="owner"))
    with pytest.raises(convert.DocumentError, match="password"):
        convert.inspect_document(source)


def test_page_and_image_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "MAX_PAGES", 2)
    with pytest.raises(convert.DocumentError, match="between 1 and 2"):
        convert.inspect_document(FIXTURES / "scan.pdf")
    monkeypatch.setattr(convert, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(convert.DocumentError, match="40 megapixels"):
        convert.inspect_document(FIXTURES / "scan.png")
