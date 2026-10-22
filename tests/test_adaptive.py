"""Adaptive decisions, compact rasters, word selection and honest comparisons."""

from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src import convert
from src.enhance import analyze_image, enhance_image
from src.quality import compare_document, visual_similarity
from src.raster import encode_raster, write_raster
from src.selection import readable_text, word_boxes

FIXTURES = Path(__file__).parent / "fixtures"


def printed_page(size=(900, 400)):
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).text(
        (30, 30),
        "Sharp printed document",
        font=ImageFont.load_default(size=36),
        fill="black",
    )
    return image


@pytest.fixture
def no_ocr(monkeypatch):
    monkeypatch.setattr(
        convert,
        "_engine",
        lambda: lambda _: SimpleNamespace(source="ocr", rotation=0, lines=[]),
    )


def images(pdf):
    return [
        obj
        for obj in pdf.objects
        if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image"
    ]


def test_adaptive_preserves_sharp_pages_and_only_restores_soft_edges():
    clean = printed_page()
    assert not analyze_image(clean)["soft"]
    assert enhance_image(clean) is clean
    soft = clean.filter(ImageFilter.GaussianBlur(0.9))
    assert analyze_image(soft)["soft"]
    restored = enhance_image(soft)
    assert restored.size == (clean.width * 3, clean.height * 3)
    assert restored.info["editable_scan_treatment"] == "soft-edge restoration"
    bilevel = clean.convert("1")
    assert enhance_image(bilevel) is bilevel
    large = printed_page((1800, 2400)).filter(ImageFilter.GaussianBlur(0.9))
    assert enhance_image(large).size == large.size


def test_photographs_are_not_sharpened_or_enlarged():
    pixels = np.random.default_rng(10).integers(0, 256, (200, 300, 3), dtype="uint8")
    photo = Image.fromarray(pixels)
    assert not analyze_image(photo)["document"]
    assert enhance_image(photo) is photo
    white_background = Image.new("RGB", (900, 600), "white")
    white_background.paste(photo, (300, 200))
    assert not analyze_image(white_background)["document"]
    assert enhance_image(white_background) is white_background


def test_jpeg_photograph_reuses_exact_original_samples(tmp_path, no_ocr):
    source, output = tmp_path / "photo.jpg", tmp_path / "photo.pdf"
    photo = Image.fromarray(
        np.random.default_rng(10).integers(0, 256, (200, 300, 3), dtype="uint8")
    )
    photo.save(source, quality=90)
    convert.convert_pdf(source, output)
    with pikepdf.open(output) as pdf:
        obj = images(pdf)[0]
        assert obj.Filter == "/DCTDecode"
        assert obj.read_raw_bytes() == source.read_bytes()
        assert (obj.Width, obj.Height) == photo.size
    assert output.stat().st_size < source.stat().st_size + 3000


def test_compact_scan_keeps_edges_and_bounds_every_channel_error():
    source = printed_page()
    pixels = np.asarray(source).copy()
    pixels[150:230, 100:750] = np.random.default_rng(5).integers(
        0, 256, (80, 650, 3), dtype="uint8"
    )
    source = Image.fromarray(pixels)
    with pikepdf.new() as pdf:
        obj = pdf.make_stream(b"")
        encoded = encode_raster(source)
        write_raster(obj, encoded, source.size)
        result = np.array(pikepdf.PdfImage(obj).as_pil_image().convert("RGB"))
    assert np.abs(result.astype("int16") - pixels.astype("int16")).max() <= 4
    exact = (pixels == 0) | (pixels == 255)
    np.testing.assert_array_equal(result[exact], pixels[exact])
    assert len(encoded[0]) < len(encode_raster(source, compact=False)[0]) * 0.8


def test_grayscale_and_small_palettes_are_lossless():
    for source in [
        printed_page().convert("L"),
        Image.new("RGB", (200, 100), (20, 80, 160)),
    ]:
        with pikepdf.new() as pdf:
            obj = pdf.make_stream(b"")
            encoded = encode_raster(source)
            write_raster(obj, encoded, source.size)
            result = pikepdf.PdfImage(obj).as_pil_image().convert("RGB")
            np.testing.assert_array_equal(
                np.asarray(source.convert("RGB")), np.asarray(result)
            )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("A ﬁne ﬂower: café", "A fine flower: café"),
        (r"Area $\pi r^{2}$", "Area πr²"),
        (r"$\mathcal { N } _ { 6 }(p)=\{q\in O\mid x+y+z=1\}$", "N₆(p)={q∈O|x+y+z=1}"),
        (r"$\frac{1}{\sqrt{x^{2}}}$", "(1)/(√(x²))"),
        (r"$\unknown{x}$", r"$\unknown{x}$"),
        (r"$\frac{x}{y$", r"$\frac{x}{y$"),
        ("x₂ ≠ 𝑥² 中文 🙂", "x₂ ≠ 𝑥² 中文 🙂"),
    ],
)
def test_readable_copy_preserves_unicode_and_keeps_unsupported_math(source, expected):
    assert readable_text(source) == expected


def word_image():
    image = Image.new("L", (600, 100), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    draw.text((20, 20), "Wide", font=font, fill=0)
    draw.text((160, 20), "words", font=font, fill=0)
    pixels = np.asarray(image)
    yy, xx = np.where(pixels < 128)
    box = (float(xx.min()), float(yy.min()), float(xx.max() + 1), float(yy.max() + 1))
    return image, box


def test_selection_uses_real_word_gaps_and_copy_stays_in_reading_order(
    tmp_path, monkeypatch
):
    image, box = word_image()
    runs = word_boxes("Wide words", box, np.asarray(image))
    assert len(runs) == 2
    assert runs[0][1][2] < 120
    assert runs[1][1][0] >= 160
    source, output = tmp_path / "words.png", tmp_path / "words.pdf"
    image.save(source, dpi=(72, 72))
    monkeypatch.setattr(
        convert,
        "_engine",
        lambda: (
            lambda _: SimpleNamespace(
                source="ocr",
                rotation=0,
                lines=[SimpleNamespace(text="Wide words", box=box)],
            )
        ),
    )
    convert.convert_pdf(source, output)
    with (
        pdfium.PdfDocument(output) as doc,
        closing(doc[0]) as page,
        closing(page.get_textpage()) as text,
    ):
        assert text.get_text_bounded().strip() == "Wide words"
        assert text.get_text_bounded(150, 35, 280, 90).strip() == "words"
        # A point in the large gap must not select a letter of either word.
        assert text.get_index(130, 65, 2, 2) is None


def test_similarity_is_identity_bounded_and_does_not_hide_missing_ink():
    source = printed_page()
    assert visual_similarity(source, source.copy()) == 100
    blank = Image.new("RGB", source.size, "white")
    missing = visual_similarity(source, blank)
    assert 0 <= missing < 85
    shifted = Image.new("RGB", source.size, "white")
    shifted.paste(source, (12, 0))
    assert visual_similarity(source, shifted) < 95
    with pytest.raises(ValueError):
        visual_similarity(source, Image.new("RGB", (20, 20)))


def test_report_compares_real_output_and_explains_intentional_restoration(
    tmp_path, no_ocr
):
    source, output = tmp_path / "soft.png", tmp_path / "soft.pdf"
    soft = printed_page().filter(ImageFilter.GaussianBlur(0.9))
    soft.save(source)
    report = {}
    assert convert.convert_pdf(source, output, report=report) == 1
    assert 85 < report["similarity_percent"] < 100
    assert report["output_bytes"] == output.stat().st_size
    assert report["source_bytes"] == source.stat().st_size
    assert report["pages"][0]["source_soft"]
    assert "lower similarity" in report["pages"][0]["summary"]
    assert "not OCR accuracy" in report["method"]
    # Removing an illustration must reduce the score, even if the OCR is unchanged.
    with pikepdf.open(output, allow_overwriting_input=True) as pdf:
        obj = images(pdf)[0]
        write_raster(
            obj,
            encode_raster(Image.new("L", (obj.Width, obj.Height), "white")),
            (obj.Width, obj.Height),
        )
        pdf.save(output)
    changed = compare_document(source, output)
    assert changed["similarity_percent"] < report["similarity_percent"] - 10


def test_report_aggregates_all_pages_and_matches_archive_identity(tmp_path, no_ocr):
    output = tmp_path / "pages.pdf"
    report = {}
    convert.convert_pdf(FIXTURES / "scan.pdf", output, enhance=False, report=report)
    assert len(report["pages"]) == 3
    assert report["similarity_percent"] == 100
    assert all(p["similarity_percent"] == 100 for p in report["pages"])


def test_mixed_quality_report_assesses_each_page_independently(tmp_path, no_ocr):
    source, output = tmp_path / "mixed.tiff", tmp_path / "mixed.pdf"
    clean = printed_page()
    soft = clean.filter(ImageFilter.GaussianBlur(0.9))
    clean.save(source, save_all=True, append_images=[soft], dpi=(72, 72))
    report = {}
    convert.convert_pdf(source, output, report=report)
    first, second = report["pages"]
    assert first["treatments"] == ["preserved"]
    assert second["treatments"] == ["soft-edge restoration"]
    assert first["similarity_percent"] == 100
    assert first["similarity_percent"] > second["similarity_percent"]
    assert report["similarity_percent"] == round(
        (first["similarity_percent"] + second["similarity_percent"]) / 2, 2
    )


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_word_alignment_and_unicode_copy_survive_ocr_quarter_turns(
    tmp_path, monkeypatch, rotation
):
    image, box = word_image()
    image = image.rotate(rotation, expand=True)
    source, output = tmp_path / "rotated.png", tmp_path / "rotated.pdf"
    image.save(source, dpi=(72, 72))
    monkeypatch.setattr(
        convert,
        "_engine",
        lambda: (
            lambda _: SimpleNamespace(
                source="ocr",
                rotation=rotation,
                lines=[SimpleNamespace(text="Wide words", box=box)],
            )
        ),
    )
    convert.convert_pdf(source, output)
    with (
        pdfium.PdfDocument(output) as doc,
        closing(doc[0]) as page,
        closing(page.get_textpage()) as text,
    ):
        assert text.get_text_bounded().strip() == "Wide words"


def test_narrow_single_letter_does_not_insert_spaces_inside_other_words(
    tmp_path, monkeypatch
):
    image = Image.new("L", (180, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 23, 39), fill=0)
    for x in range(40, 103, 7):
        draw.rectangle((x, 20, x + 4, 39), fill=0)
    source, output = tmp_path / "narrow.png", tmp_path / "narrow.pdf"
    image.save(source, dpi=(72, 72))
    box = (20, 20, 101, 40)
    assert len(word_boxes("p voisinage", box, np.asarray(image))) == 2
    monkeypatch.setattr(
        convert,
        "_engine",
        lambda: (
            lambda _: SimpleNamespace(
                source="ocr",
                rotation=0,
                lines=[SimpleNamespace(text="p voisinage", box=box)],
            )
        ),
    )
    convert.convert_pdf(source, output)
    with (
        pdfium.PdfDocument(output) as doc,
        closing(doc[0]) as page,
        closing(page.get_textpage()) as text,
    ):
        assert text.get_text_bounded().strip() == "p voisinage"


def test_columns_keep_model_reading_order_and_math_is_unicode_in_pdf(
    tmp_path, monkeypatch
):
    image = Image.new("L", (700, 280), "white")
    draw, lines = ImageDraw.Draw(image), []
    font = ImageFont.load_default(size=24)
    content = [
        ("Left column one", 20, 20),
        ("Left column two", 20, 70),
        ("Right column one", 380, 20),
        ("Right column two", 380, 70),
        (r"Area $\pi r^{2}$", 20, 180),
    ]
    for text, x, y in content:
        visible = readable_text(text)
        draw.text((x, y), visible, font=font, fill=0)
        lines.append(
            SimpleNamespace(text=text, box=draw.textbbox((x, y), visible, font=font))
        )
    source, output = tmp_path / "columns.png", tmp_path / "columns.pdf"
    image.save(source, dpi=(72, 72))
    monkeypatch.setattr(
        convert,
        "_engine",
        lambda: lambda _: SimpleNamespace(source="ocr", rotation=0, lines=lines),
    )
    convert.convert_pdf(source, output)
    with (
        pdfium.PdfDocument(output) as doc,
        closing(doc[0]) as page,
        closing(page.get_textpage()) as text,
    ):
        extracted = text.get_text_bounded()
        expected = [readable_text(t) for t, x, y in content]
        positions = [extracted.index(t) for t in expected]
        assert positions == sorted(positions)
        assert r"\pi" not in extracted


def test_older_enhanced_pdf_is_compacted_without_enlarging_it_again(tmp_path, no_ocr):
    import zlib

    source, output = tmp_path / "older.pdf", tmp_path / "compact.pdf"
    with Image.open(FIXTURES / "scan.png") as image, pikepdf.new() as pdf:
        convert._embed_frame(pdf, image, 360, 480, False)
        obj = images(pdf)[0]
        obj.write(zlib.compress(image.tobytes(), 1), filter=pikepdf.Name.FlateDecode)
        if "/DecodeParms" in obj:
            del obj.DecodeParms
        obj.EditableScanEnhanced = True
        del obj.EditableScanCompact
        pdf.save(source)
        size = image.size
    report = {}
    convert.convert_pdf(source, output, report=report)
    with pikepdf.open(output) as pdf:
        obj = images(pdf)[0]
        assert (obj.Width, obj.Height) == size
        assert obj.EditableScanCompact is True
        assert obj.EditableScanTreatment == "preserved"
    assert output.stat().st_size < source.stat().st_size * 0.8
    assert report["pages"][0]["treatments"] == ["preserved"]
