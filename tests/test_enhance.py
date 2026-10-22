"""Restoration quality, layout, Unicode and useful concurrency checks."""

import threading
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pikepdf
import pypdfium2 as pdfium
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from editable_scan import convert
from editable_scan import enhance as restoration

FIXTURES = Path(__file__).parent / "fixtures"


def extracted(path, index=0):
    with (
        pdfium.PdfDocument(path) as doc,
        closing(doc[index]) as page,
        closing(page.get_textpage()) as text,
    ):
        return text.get_text_bounded()


def ocr_page(text="Selectable text", rotation=0):
    return SimpleNamespace(
        source="ocr",
        rotation=rotation,
        lines=[
            SimpleNamespace(text=text, box=(30, 40, 300, 60)),
            SimpleNamespace(text="Second line", box=(30, 90, 300, 110)),
        ],
    )


def raster_objects(pdf):
    return [
        obj
        for obj in pdf.objects
        if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image"
    ]


def reference():
    image = Image.new("L", (900, 360), "white")
    drawing = ImageDraw.Draw(image)
    drawing.text((20, 20), "x2 + y2 = r2", font=ImageFont.load_default(size=70), fill=0)
    drawing.ellipse((20, 130, 160, 270), outline=0, width=6)
    drawing.rectangle((210, 140, 800, 250), outline=0, width=6)
    drawing.line((220, 290, 800, 290), fill=0, width=3)
    return image


def test_sharpening_gets_closer_to_a_known_sharp_reference():
    clean = reference()
    blurred = clean.resize((300, 120), Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(0.45)
    )
    baseline = np.asarray(blurred.resize(clean.size, Image.Resampling.BICUBIC)).astype(
        float
    )
    enhanced = np.asarray(restoration.enhance_image(blurred)).astype(float)
    target = np.asarray(clean).astype(float)
    assert np.mean((enhanced - target) ** 2) < np.mean((baseline - target) ** 2) * 0.9


def test_subpixel_reconstruction_reduces_text_fringe_and_retains_colored_art():
    clean = reference().resize((900, 120), Image.Resampling.LANCZOS)
    pixels = np.asarray(clean).reshape(120, 300, 3).copy()
    pixels[95:110, 240:285] = (235, 20, 85)
    source = Image.fromarray(pixels)
    assert restoration._subpixel_ink(source) is not None
    enhanced = np.asarray(restoration.enhance_image(source))
    baseline = np.asarray(source.resize((900, 360), Image.Resampling.LANCZOS))
    ink = (slice(0, 280), slice(0, 900))
    old_fringe = np.mean(baseline[ink].max(2) - baseline[ink].min(2))
    new_fringe = np.mean(enhanced[ink].max(2) - enhanced[ink].min(2))
    assert new_fringe < old_fringe * 0.6
    np.testing.assert_array_equal(enhanced[300, 780], [235, 20, 85])


def test_restoration_limit_and_transparency(monkeypatch):
    monkeypatch.setattr(restoration, "MAX_ENHANCED_PIXELS", 50_000)
    source = reference().resize((120, 48))
    enhanced = restoration.enhance_image(source)
    assert source.width < enhanced.width
    assert enhanced.width * enhanced.height <= 50_000
    large = reference()
    assert restoration.enhance_image(large).size == large.size
    transparent = Image.new("RGBA", (100, 80), (30, 80, 200, 100))
    assert restoration.enhance_image(transparent) is transparent


def test_image_default_enhances_pixels_and_avoids_pdf_rendering(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "_engine", lambda: lambda _: ocr_page())

    def never(*args, **kwargs):
        raise AssertionError("Image OCR must not encode and render a PDF first")

    monkeypatch.setattr(pdfium.PdfPage, "render", never)
    output = tmp_path / "image.pdf"
    convert.convert_pdf(FIXTURES / "scan.png", output)
    assert "Selectable text" in extracted(output)
    with Image.open(FIXTURES / "scan.png") as source, pikepdf.open(output) as pdf:
        images = raster_objects(pdf)
        assert len(images) == 1
        assert (images[0].Width, images[0].Height) == (
            source.width * 3,
            source.height * 3,
        )
        assert images[0].Filter == "/FlateDecode"
        assert images[0].EditableScanEnhanced is True
        dpi = source.info.get("dpi", (200, 200))
        assert abs(float(pdf.pages[0].MediaBox[2]) - source.width * 72 / dpi[0]) < 0.001


def test_image_restoration_overlaps_ocr(tmp_path, monkeypatch):
    entered = threading.Event()
    recognized = threading.Event()

    def enhance(image):
        entered.set()
        assert recognized.wait(3), "Restoration must overlap OCR"
        return image

    def ocr(gray):
        assert entered.wait(3), "Restoration did not start before OCR completed"
        recognized.set()
        return ocr_page()

    monkeypatch.setattr(convert, "enhance_image", enhance)
    monkeypatch.setattr(convert, "_engine", lambda: ocr)
    convert.convert_pdf(FIXTURES / "scan.png", tmp_path / "image.pdf")


def test_one_shared_font_and_unicode_surrogate_pairs(tmp_path, monkeypatch):
    content = "Café Ελληνικά 中文 𝑥 ∑ 🙂"
    monkeypatch.setattr(convert, "_engine", lambda: lambda _: ocr_page(content))
    output = tmp_path / "unicode.pdf"
    convert.convert_pdf(FIXTURES / "scan.pdf", output)
    for index in range(3):
        assert content in extracted(output, index)
    with pikepdf.open(output) as pdf:
        fonts = [
            obj
            for obj in pdf.objects
            if isinstance(obj, pikepdf.Dictionary) and obj.get("/Subtype") == "/Type0"
        ]
        assert len(fonts) == 1
        assert (
            b"d835dc65" in fonts[0].ToUnicode.read_bytes()
        )  # U+1D465, mathematical italic x.


def test_pdf_scans_are_restored_only_once(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "_engine", lambda: lambda _: ocr_page())
    first, second = tmp_path / "first.pdf", tmp_path / "second.pdf"
    convert.convert_pdf(FIXTURES / "scan.pdf", first)
    convert.convert_pdf(first, second)
    with (
        pikepdf.open(FIXTURES / "scan.pdf") as source,
        pikepdf.open(first) as a,
        pikepdf.open(second) as b,
    ):
        original, images_a, images_b = (
            raster_objects(source),
            raster_objects(a),
            raster_objects(b),
        )
        assert len(original) == len(images_a) == len(images_b)
        for initial, enhanced, repeated in zip(original, images_a, images_b):
            assert int(enhanced.Width) == int(initial.Width) * 3
            assert enhanced.read_raw_bytes() == repeated.read_raw_bytes()
        assert list(source.pages[0].MediaBox) == list(a.pages[0].MediaBox)


def test_indexed_pdf_scan_is_enhanced_and_visible_text_stays_vector(
    tmp_path, monkeypatch
):
    source, output = tmp_path / "indexed.pdf", tmp_path / "result.pdf"
    image = reference().convert("RGB").quantize(colors=128)
    pdf = FPDF(unit="pt", format=(360, 480))
    pdf.add_page()
    pdf.image(image, x=0, y=0, w=360, h=144)
    pdf.set_font("Helvetica", size=16)
    pdf.text(30, 400, "Keep this vector text")
    pdf.set_title("Preserved metadata")
    pdf.output(source)
    monkeypatch.setattr(convert, "_engine", lambda: lambda _: ocr_page())
    convert.convert_pdf(source, output)
    with pikepdf.open(output) as result:
        raster = raster_objects(result)[0]
        # Already sharp indexed scans should no longer be enlarged.
        assert int(raster.Width) == image.width
        assert result.docinfo.Title == "Preserved metadata"
    assert "Keep this vector text" in extracted(output)
    # At 400%, the digital caption must have exactly the same rendered pixels.
    regions = []
    for path in [source, output]:
        with (
            pdfium.PdfDocument(path) as doc,
            closing(doc[0]) as page,
            closing(page.render(scale=4)) as bitmap,
        ):
            regions.append(np.asarray(bitmap.to_pil())[1500:1650, :].copy())
    np.testing.assert_array_equal(*regions)
