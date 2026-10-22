"""Rigid scan rotation, conservative decisions and selectable-text alignment."""

from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageDraw, ImageFont

from editable_scan import convert
from editable_scan.deskew import detect_skew, rotation_geometry


def document():
    image = Image.new("L", (900, 1100), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    for n in range(20):
        draw.text(
            (70, 80 + n * 42),
            "The quick brown fox jumps over the lazy dog.",
            font=font,
            fill=0,
        )
    return image


@pytest.fixture
def no_ocr(monkeypatch):
    monkeypatch.setattr(
        convert,
        "_engine",
        lambda: lambda _: SimpleNamespace(source="ocr", rotation=0, lines=[]),
    )


@pytest.mark.parametrize("angle", [-5, -2, -0.5, 0.5, 2, 5])
def test_detection_corrects_both_directions_and_residual_is_negligible(angle):
    source = document().rotate(angle, expand=True, fillcolor="white")
    correction = detect_skew(source)
    assert correction == pytest.approx(-angle, abs=0.15)
    result = source.rotate(correction, expand=True, fillcolor="white")
    assert detect_skew(result) == 0


def test_uncertain_pages_and_nearly_straight_pages_are_not_rotated():
    source = document()
    for image in [
        source,
        source.rotate(0.15, expand=True, fillcolor="white"),
        source.rotate(12, expand=True, fillcolor="white"),
        source.rotate(90, expand=True),
        Image.new("L", source.size, "white"),
        Image.fromarray(
            np.random.default_rng(3).integers(0, 256, (700, 800), dtype="uint8")
        ),
    ]:
        assert detect_skew(image) == 0
    sparse = Image.new("L", source.size, "white")
    ImageDraw.Draw(sparse).text((50, 50), "Short title", fill=0)
    assert detect_skew(sparse.rotate(3, expand=True, fillcolor="white")) == 0


def test_rigid_matrix_preserves_lengths_angles_and_all_four_corners():
    matrix, (width, height) = rotation_geometry(900, 1100, 3)
    a, b, c, d, e, f = matrix
    assert a * a + b * b == pytest.approx(1)
    assert c * c + d * d == pytest.approx(1)
    assert a * c + b * d == pytest.approx(0)
    for x, y in [(0, 0), (900, 0), (0, 1100), (900, 1100)]:
        assert -1e-6 <= a * x + c * y + e <= width + 1e-6
        assert -1e-6 <= b * x + d * y + f <= height + 1e-6


@pytest.mark.parametrize("kind", ["image", "pdf"])
def test_conversion_deskews_without_resampling_the_embedded_scan(
    tmp_path,
    no_ocr,
    kind,
):
    source = tmp_path / ("scan.png" if kind == "image" else "scan.pdf")
    output = tmp_path / "straight.pdf"
    image = document().rotate(3, expand=True, fillcolor="white").convert("1")
    if kind == "image":
        image.save(source, dpi=(72, 72))
    else:
        # RGB is supported by the adaptive PDF raster path; bilevel ink stays sharp.
        with pikepdf.new() as pdf:
            convert._embed_frame(pdf, image.convert("RGB"), *image.size, False)
            pdf.save(source)
    report = {}
    convert.convert_pdf(source, output, report=report)
    with pikepdf.open(output) as pdf:
        page = pdf.pages[0]
        assert float(page.obj.EditableScanDeskew) == pytest.approx(-3, abs=0.15)
        obj = next(obj for obj in pdf.objects if obj.get("/Subtype") == "/Image")
        assert (obj.Width, obj.Height) == image.size
        np.testing.assert_array_equal(
            np.array(pikepdf.PdfImage(obj).as_pil_image().convert("L")),
            np.array(image.convert("L")),
        )
        assert float(page.mediabox[2]) > image.width
        assert float(page.mediabox[3]) > image.height
    with (
        pdfium.PdfDocument(output) as pdf,
        closing(pdf[0]) as page,
        closing(page.render(scale=1, grayscale=True)) as bitmap,
    ):
        assert detect_skew(bitmap.to_numpy()) == 0
    assert report["pages"][0]["deskew_degrees"] == pytest.approx(-3, abs=0.15)
    assert "Rotation changes pixel positions" in report["pages"][0]["summary"]
    assert report["similarity_percent"] < 100
    # Reprocessing must not rotate or grow the page a second time.
    second = tmp_path / "second.pdf"
    convert.convert_pdf(output, second)
    with pikepdf.open(output) as first, pikepdf.open(second) as again:
        assert list(first.pages[0].mediabox) == list(again.pages[0].mediabox)


def test_ocr_and_selectable_text_follow_the_straightened_page(tmp_path, monkeypatch):
    source, output = tmp_path / "tilted.png", tmp_path / "straight.pdf"
    document().rotate(-3, expand=True, fillcolor="white").save(source, dpi=(72, 72))
    observed = {}

    def recognize(gray):
        assert detect_skew(gray) == 0
        ys, _ = np.where(gray < 100)
        # The first actual ink line after correction supplies a known OCR box.
        top = int(ys.min())
        selected = gray[top : top + 25] < 100
        rows, columns = np.where(selected)
        box = (
            int(columns.min()),
            top + int(rows.min()),
            int(columns.max()) + 1,
            top + int(rows.max()) + 1,
        )
        observed["box"], observed["size"] = box, gray.shape
        return SimpleNamespace(
            source="ocr",
            rotation=0,
            lines=[SimpleNamespace(text="Aligned text", box=box)],
        )

    monkeypatch.setattr(convert, "_engine", lambda: recognize)
    convert.convert_pdf(source, output)
    with (
        pdfium.PdfDocument(output) as pdf,
        closing(pdf[0]) as page,
        closing(page.get_textpage()) as text,
    ):
        assert text.get_text_bounded().strip() == "Aligned text"
        assert text.count_rects() >= 1
        left, _bottom, _right, top = text.get_rect(0)
        box, (h, w) = observed["box"], observed["size"]
        sx, sy = page.get_width() / w, page.get_height() / h
        assert left == pytest.approx(box[0] * sx, abs=1)
        assert top == pytest.approx((h - box[1]) * sy, abs=1)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_existing_pdf_rotation_and_nonzero_crop_origin_remain_valid(
    tmp_path, no_ocr, rotation
):
    source, output = tmp_path / "cropped.pdf", tmp_path / "out.pdf"
    image = document().rotate(2, expand=True, fillcolor="white").convert("1")
    with pikepdf.new() as pdf:
        page = convert._embed_frame(pdf, image, *image.size, False)
        # Move the complete page and crop together, retaining every source sample.
        page.contents_add(b"q 1 0 0 1 40 70 cm\n", prepend=True)
        page.contents_add(b"\nQ", prepend=False)
        page.MediaBox = [40, 70, 40 + image.width, 70 + image.height]
        page.CropBox = list(page.MediaBox)
        page.Rotate = rotation
        pdf.save(source)
    convert.convert_pdf(source, output)
    with pikepdf.open(output) as pdf:
        assert abs(float(pdf.pages[0].obj.EditableScanDeskew)) == pytest.approx(
            2, abs=0.15
        )
    with (
        pdfium.PdfDocument(output) as pdf,
        closing(pdf[0]) as page,
        closing(page.render(scale=1, grayscale=True)) as bitmap,
    ):
        image = bitmap.to_pil()
        if rotation:
            image = image.rotate(rotation, expand=True)
        assert detect_skew(image) == 0
        assert np.mean(np.asarray(image) < 100) > 0.02


def test_pdf_annotations_are_preserved_without_changing_their_coordinates(
    tmp_path, no_ocr
):
    source, output = tmp_path / "linked.pdf", tmp_path / "out.pdf"
    image = document().rotate(3, expand=True, fillcolor="white").convert("1")
    with pikepdf.new() as pdf:
        page = convert._embed_frame(pdf, image, *image.size, False)
        page.Annots = [
            pdf.make_indirect(
                pikepdf.Dictionary(
                    Type=pikepdf.Name.Annot,
                    Subtype=pikepdf.Name.Link,
                    Rect=[30, 40, 300, 90],
                    A=pikepdf.Dictionary(S=pikepdf.Name.URI, URI="https://example.org"),
                )
            )
        ]
        pdf.save(source)
    convert.convert_pdf(source, output)
    with pikepdf.open(output) as pdf:
        assert "/EditableScanDeskew" not in pdf.pages[0].obj
        assert list(pdf.pages[0].Annots[0].Rect) == [30, 40, 300, 90]


def test_archival_mode_does_not_deskew(tmp_path, no_ocr):
    source, output = tmp_path / "tilted.png", tmp_path / "archive.pdf"
    image = document().rotate(3, expand=True, fillcolor="white")
    image.save(source)
    convert.convert_pdf(source, output, enhance=False)
    with pikepdf.open(output) as pdf:
        assert "/EditableScanDeskew" not in pdf.pages[0].obj
