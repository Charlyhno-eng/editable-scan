"""Photographed paper extraction without cropping ordinary scanned content."""

from contextlib import closing
from types import SimpleNamespace

import cv2
import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageDraw, ImageFont

from editable_scan import convert
from editable_scan.crop import _rectified_size, find_sheet, rectify_sheet


def paper(frame=False):
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    if frame:
        draw.rectangle((12, 12, 587, 787), outline="blue", width=3)
        draw.rectangle((22, 22, 577, 777), outline="red", width=3)
    font = ImageFont.load_default(size=24)
    for n in range(14):
        draw.text(
            (45, 50 + n * 40), "Printed paper on a table", font=font, fill="black"
        )
    # Distinctive corner marks must survive rectification, along with the frame.
    for position, color in [((35, 35, 41, 41), "red"), ((560, 755, 566, 761), "blue")]:
        draw.rectangle(position, fill=color)
    return image


def photograph(frame=False, corners=None, background=(115, 85, 60)):
    image = paper(frame)
    corners = np.float32(
        corners
        if corners is not None
        else [[140, 70], [700, 120], [750, 900], [80, 930]]
    )
    source = np.float32([[0, 0], [599, 0], [599, 799], [0, 799]])
    matrix = cv2.getPerspectiveTransform(source, corners)
    canvas = np.full((1000, 900, 3), background, "uint8")
    mask = cv2.warpPerspective(np.full((800, 600), 255, "uint8"), matrix, (900, 1000))
    warped = cv2.warpPerspective(np.asarray(image), matrix, (900, 1000))
    canvas[mask > 0] = warped[mask > 0]
    return Image.fromarray(canvas), corners


@pytest.fixture
def no_ocr(monkeypatch):
    monkeypatch.setattr(
        convert,
        "_engine",
        lambda: lambda _: SimpleNamespace(source="ocr", rotation=0, lines=[]),
    )


@pytest.mark.parametrize("frame", [False, True])
@pytest.mark.parametrize("background", [(115, 85, 60), (35, 35, 35), (160, 175, 190)])
def test_paper_corners_are_detected_instead_of_the_printed_frame(frame, background):
    image, expected = photograph(frame, background=background)
    corners = find_sheet(image)
    assert corners is not None
    assert np.linalg.norm(corners - expected, axis=1).max() < 5
    result, _ = rectify_sheet(image)
    pixels = np.asarray(result)
    # A one-pixel edge is tolerated; the tabletop must not remain inside the page.
    near_background = np.max(np.abs(pixels.astype("int16") - background), axis=2) < 8
    border = np.zeros(near_background.shape, dtype=bool)
    border[8:20, 8:-8] = border[-20:-8, 8:-8] = True
    border[8:-8, 8:20] = border[8:-8, -20:-8] = True
    assert near_background[border].mean() < 0.002
    assert np.count_nonzero((pixels[..., 0] > 180) & (pixels[..., 2] < 80)) > 10
    assert np.count_nonzero((pixels[..., 2] > 180) & (pixels[..., 0] < 80)) > 10
    if frame:
        # Both complete horizontal printed frame lines remain inside the output.
        blue = (pixels[..., 2] > 150) & (pixels[..., 0] < 100)
        assert blue[:50].sum() > result.width
        assert blue[-50:].sum() > result.width


def test_axis_aligned_photo_is_cropped_even_without_any_tilt():
    image, expected = photograph(corners=[[130, 90], [730, 90], [730, 890], [130, 890]])
    assert np.max(np.abs(find_sheet(image) - expected)) < 4
    result, _ = rectify_sheet(image)
    assert result.width < image.width
    assert result.height < image.height


@pytest.mark.parametrize("frame", [False, True])
def test_flat_scans_are_unchanged_including_printed_borders(frame):
    image = paper(frame)
    result, corners = rectify_sheet(image)
    assert result is image
    assert corners is None


def test_uncertain_images_and_transparency_are_unchanged():
    for image in [
        Image.new("RGB", (600, 800), "white"),
        Image.new("RGBA", (600, 800), (255, 255, 255, 0)),
        Image.fromarray(
            np.random.default_rng(8).integers(0, 256, (800, 600, 3), "uint8")
        ),
    ]:
        result, corners = rectify_sheet(image)
        assert result is image
        assert corners is None


def test_multiple_large_sheets_are_not_silently_reduced_to_one():
    image = Image.new("RGB", (1200, 1000), (90, 60, 40))
    image.paste(paper().resize((500, 750)), (40, 100))
    image.paste(paper().resize((500, 750)), (650, 100))
    assert find_sheet(image) is None


@pytest.mark.parametrize("frame", [False, True])
def test_rectified_images_are_not_cropped_repeatedly(frame):
    image, _ = photograph(frame=frame)
    result, corners = rectify_sheet(image)
    assert corners is not None
    assert find_sheet(result) is None


@pytest.mark.parametrize("angles", [(0.15, 0.3, 0.05), (-0.2, 0.4, 0.1)])
def test_camera_geometry_recovers_true_paper_proportions(angles):
    # Known rectangular paper under a centered pinhole camera with square pixels.
    rotation, _ = cv2.Rodrigues(np.array(angles, dtype="float64"))
    paper_points = np.array(
        [[-210, -297, 0], [210, -297, 0], [210, 297, 0], [-210, 297, 0]], "float64"
    )
    xyz = paper_points @ rotation.T + [0, 0, 1000]
    corners = (xyz[:, :2] / xyz[:, 2:] * 1200 + [500, 600]).astype("float32")
    width, height = _rectified_size(corners, (1000, 1200))
    assert width / height == pytest.approx(210 / 297, abs=0.003)


@pytest.mark.parametrize("kind", ["image", "pdf"])
def test_conversion_crops_photo_and_reports_the_intentional_change(
    tmp_path, no_ocr, kind
):
    image, _ = photograph(frame=True)
    source = tmp_path / ("photo.png" if kind == "image" else "photo.pdf")
    output = tmp_path / "cropped.pdf"
    if kind == "image":
        image.save(source, dpi=(144, 144))
    else:
        with pikepdf.new() as pdf:
            convert._embed_frame(pdf, image, image.width / 2, image.height / 2, False)
            pdf.save(source)
    report = {}
    convert.convert_pdf(source, output, report=report)
    with pikepdf.open(output) as pdf:
        assert len(pdf.pages) == 1
        corners = pdf.pages[0].obj.EditableScanSheetCorners
        assert len(corners) == 4
        assert pdf.pages[0].mediabox[2] < image.width / 2
    assert report["pages"][0]["sheet_cropped"]
    assert "perspective was corrected" in report["pages"][0]["summary"]
    assert "lower similarity" in report["pages"][0]["summary"]
    assert report["similarity_percent"] < 100
    # Reprocessing must preserve corrected page geometry.
    repeated = tmp_path / "again.pdf"
    convert.convert_pdf(output, repeated)
    with pikepdf.open(output) as first, pikepdf.open(repeated) as second:
        assert list(first.pages[0].mediabox) == list(second.pages[0].mediabox)


def test_ocr_uses_the_rectified_page_and_selection_follows_it(tmp_path, monkeypatch):
    source, output = tmp_path / "photo.png", tmp_path / "cropped.pdf"
    image, _ = photograph()
    image.save(source, dpi=(72, 72))
    expected, _ = rectify_sheet(image)
    observed = {}

    def recognize(gray):
        assert gray.shape == (expected.height, expected.width)
        y, x = np.where(gray < 80)
        box = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.min()) + 30)
        observed["box"] = box
        return SimpleNamespace(
            source="ocr",
            rotation=0,
            lines=[SimpleNamespace(text="Rectified text", box=box)],
        )

    monkeypatch.setattr(convert, "_engine", lambda: recognize)
    convert.convert_pdf(source, output)
    with (
        pdfium.PdfDocument(output) as pdf,
        closing(pdf[0]) as page,
        closing(page.get_textpage()) as text,
    ):
        assert text.get_text_bounded().strip() == "Rectified text"
        assert text.count_rects() >= 1
        left, _, _, top = text.get_rect(0)
        assert left == pytest.approx(observed["box"][0], abs=1)
        assert top == pytest.approx(expected.height - observed["box"][1], abs=1)


def test_archival_mode_preserves_the_original_photo(tmp_path, no_ocr):
    source, output = tmp_path / "photo.png", tmp_path / "archive.pdf"
    image, _ = photograph()
    image.save(source)
    convert.convert_pdf(source, output, enhance=False)
    with pikepdf.open(output) as pdf:
        assert "/EditableScanSheetCorners" not in pdf.pages[0].obj
