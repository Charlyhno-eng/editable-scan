# SPDX-License-Identifier: MPL-2.0
"""Restore scanned details, preserve page layout and add invisible OCR text."""

from __future__ import annotations

import io
import math
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium
from PIL import Image, ImageOps

from editable_scan.crop import find_sheet, rectify_sheet
from editable_scan.deskew import (
    detect_skew,
    rotate_ocr,
    rotation_geometry,
    straighten_page,
)
from editable_scan.enhance import enhance_image, finish_pdf_images, prepare_pdf_images
from editable_scan.raster import encode_raster, write_raster
from editable_scan.selection import positioned_text, selection_lines
from editable_scan.text import TextLayer

MAX_PAGES = 200
MAX_IMAGE_PIXELS = 40_000_000
RENDER_PIXELS = 8_000_000
OCR_DPI = 200
OCR_LONG_SIDE = 2600
IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "TIFF", "BMP", "GIF"}
_local = threading.local()


class DocumentError(ValueError):
    """An input document cannot be converted."""


def inspect_document(path: Path) -> tuple[str, int]:
    """Check content, page limits and encryption before queuing work."""
    with path.open("rb") as stream:
        is_pdf = b"%PDF-" in stream.read(1024)
    if is_pdf:
        try:
            with pikepdf.open(path) as pdf:
                if pdf.is_encrypted:
                    raise DocumentError(
                        "This PDF is protected. Unlock it and try again."
                    )
                count = len(pdf.pages)
        except pikepdf.PasswordError as exc:
            raise DocumentError("This PDF is password protected.") from exc
        except pikepdf.PdfError as exc:
            raise DocumentError("This PDF is unreadable or damaged.") from exc
        kind = "pdf"
    else:
        try:
            with Image.open(path) as image:
                if image.format not in IMAGE_FORMATS:
                    raise DocumentError(
                        "Unsupported format. Use a PDF or a PNG, JPG, WebP, TIFF, BMP or GIF image."
                    )
                count = getattr(image, "n_frames", 1)
                if count <= MAX_PAGES:
                    for index in range(count):
                        image.seek(index)
                        if image.width * image.height > MAX_IMAGE_PIXELS:
                            raise DocumentError("This image exceeds 40 megapixels.")
        except (OSError, Image.DecompressionBombError) as exc:
            raise DocumentError(
                "Unreadable file. Choose a valid PDF or image."
            ) from exc
        kind = "image"
    if not 1 <= count <= MAX_PAGES:
        raise DocumentError(
            f"The document must contain between 1 and {MAX_PAGES} pages."
        )
    return kind, count


def _engine():
    import indra_ocr

    model_dir = os.environ.get("INDRA_OCR_MODEL_DIR")
    key = (os.getpid(), model_dir)
    if getattr(_local, "key", None) != key:
        _local.engine = indra_ocr.load(path=model_dir, threads=2, math=True).ocr
        _local.key = key
    return _local.engine


def _recognize(gray, index):
    result = _engine()(gray)
    if result.source == "error":
        raise DocumentError(f"Unable to recognize page {index + 1}.")
    result.selection = selection_lines(result, gray)
    return result


def _embed_frame(pdf, frame, width, height, improve, jpeg=None):
    original_frame = frame
    if improve:
        frame = enhance_image(frame)
    if frame.mode not in ("1", "RGB", "L", "CMYK", "RGBA"):
        frame = frame.convert("RGBA")
    alpha = frame.getchannel("A") if frame.mode == "RGBA" else None
    if alpha is not None:
        frame = frame.convert("RGB")

    def raster(image, colorspace):
        obj = pdf.make_stream(b"")
        encoded = encode_raster(image, compact=improve and alpha is None)
        # Indexed/grayscale optimizations provide their own device color space.
        override = colorspace if isinstance(colorspace, pikepdf.Array) else None
        write_raster(obj, encoded, image.size, colorspace=override)
        return obj

    colorspace = {
        "1": pikepdf.Name.DeviceGray,
        "L": pikepdf.Name.DeviceGray,
        "RGB": pikepdf.Name.DeviceRGB,
        "CMYK": pikepdf.Name.DeviceCMYK,
    }[frame.mode]
    profile = frame.info.get("icc_profile", b"")
    signature = {"1": b"GRAY", "L": b"GRAY", "RGB": b"RGB ", "CMYK": b"CMYK"}[
        frame.mode
    ]
    if (
        len(profile) >= 128
        and profile[36:40] == b"acsp"
        and profile[16:20] == signature
    ):
        icc = pdf.make_stream(profile)
        icc.N = len(frame.getbands())
        icc.Alternate = colorspace
        colorspace = pikepdf.Array([pikepdf.Name.ICCBased, icc])
    if jpeg is not None and frame is original_frame:
        # Reuse a photograph's original JPEG without decoding/re-encoding losses.
        obj = pdf.make_stream(b"")
        write_raster(
            obj,
            (jpeg, pikepdf.Name.DCTDecode, None, frame.mode, 8, None),
            frame.size,
            colorspace=colorspace,
        )
    else:
        obj = raster(frame, colorspace)
    obj.EditableScanEnhanced = bool(improve)
    obj.EditableScanCompact = bool(improve)
    obj.EditableScanTreatment = frame.info.get("editable_scan_treatment", "preserved")
    if alpha is not None and alpha.getextrema() != (255, 255):
        obj.SMask = raster(alpha, pikepdf.Name.DeviceGray)
    page = pdf.add_blank_page(page_size=(width, height))
    page.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Scan=obj))
    page.Contents = pdf.make_stream(
        f"q {width:.8f} 0 0 {height:.8f} 0 0 cm /Scan Do Q".encode("ascii")
    )
    return page


def _image_pdf(source, results, improve, progress):
    """Read image pixels directly for OCR instead of encoding and rendering them."""
    with (
        pikepdf.new() as pdf,
        Image.open(source) as image,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="restore") as pool,
    ):
        for index in range(getattr(image, "n_frames", 1)):
            image.seek(index)
            dpi = image.info.get("dpi", (200, 200))
            if not isinstance(dpi, tuple):
                dpi = (dpi, dpi)
            dx, dy = [
                float(v) if math.isfinite(float(v)) and 36 <= float(v) <= 1200 else 200
                for v in dpi[:2]
            ]
            frame = ImageOps.exif_transpose(image.copy())
            if frame.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "CMYK"):
                frame = frame.convert("RGBA")
            frame, corners = rectify_sheet(frame) if improve else (frame, None)
            width, height = frame.width * 72 / dx, frame.height * 72 / dy
            # Restoration and compact encoding overlap the OCR inference.
            jpeg = (
                source.read_bytes()
                if improve
                and corners is None
                and image.format == "JPEG"
                and image.mode in ("RGB", "L")
                and image.getexif().get(274, 1) == 1
                else None
            )
            embedded = pool.submit(
                _embed_frame, pdf, frame, width, height, improve, jpeg
            )
            ocr_frame = frame.convert("RGBA")
            background = Image.new("RGBA", frame.size, "white")
            background.alpha_composite(ocr_frame)
            gray = background.convert("L")
            angle = detect_skew(gray) if improve and abs(dx - dy) < 0.01 else 0
            _, extent = rotation_geometry(*gray.size, angle)
            scale = min(
                1,
                OCR_LONG_SIDE / max(extent),
                math.sqrt(RENDER_PIXELS / (extent[0] * extent[1])),
            )
            if scale < 1:
                gray = gray.resize(
                    (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            if angle:
                # Resample bounded OCR pixels, never the full-resolution visual scan.
                _, size = rotation_geometry(*gray.size, angle)
                gray = rotate_ocr(gray, angle, tuple(math.ceil(v) for v in size))
            results.append(
                (_recognize(np.asarray(gray), index), gray.width, gray.height)
            )
            page = embedded.result()
            straighten_page(page, angle)
            if corners is not None:
                page.obj.EditableScanSheetCorners = pikepdf.Array(
                    [pikepdf.Array(point) for point in corners]
                )
            progress(index + 1, getattr(image, "n_frames", 1))
        stream = io.BytesIO()
        pdf.save(stream)
        return stream.getvalue()


def _rectify_pdf_photo(pdf, target, page, preview):
    """Rectify a single-photo PDF page without flattening text or vector pages."""
    objects = list(page.get_objects())
    if len(objects) != 1 or objects[0].type != pdfium.raw.FPDF_PAGEOBJ_IMAGE:
        return False
    corners = find_sheet(preview)
    if corners is None:
        return False
    w, h = page.get_size()
    px, py = objects[0].get_px_size()
    scale = min(max(px / w, py / h), math.sqrt(MAX_IMAGE_PIXELS / (w * h)))
    with closing(page.render(scale=scale)) as bitmap:
        original = bitmap.to_pil().convert("RGB").copy()
    corners *= np.array(
        [original.width / preview.width, original.height / preview.height]
    )
    frame, corners = rectify_sheet(original, corners)
    width, height = frame.width / scale, frame.height / scale
    replacement = _embed_frame(pdf, frame, width, height, True)
    target.Resources, target.Contents = replacement.Resources, replacement.Contents
    target.MediaBox = [0, 0, width, height]
    target.CropBox = [0, 0, width, height]
    target.Rotate = 0
    for name in ("/TrimBox", "/BleedBox", "/ArtBox"):
        if name in target.obj:
            del target.obj[name]
    target.obj.EditableScanSheetCorners = pikepdf.Array(
        [pikepdf.Array(point) for point in corners]
    )
    del pdf.pages[-1]
    return True


def _straighten_pdf(base):
    """Rectify photos and deskew scans before OCR, keeping digital pages intact."""
    changed = False
    with pikepdf.open(io.BytesIO(base)) as pdf, pdfium.PdfDocument(base) as document:
        for index in range(len(pdf.pages)):
            target = pdf.pages[index]
            # Annotation coordinates and interactive destinations must stay valid.
            if (
                target.obj.get("/Annots")
                or target.obj.get("/EditableScanDeskew")
                or target.obj.get("/EditableScanSheetCorners")
            ):
                continue
            with closing(document[index]) as page:
                if not any(
                    page.get_objects(
                        filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE], max_depth=10
                    )
                ):
                    continue
                scale = min(100 / 72, 1200 / max(page.get_size()))
                with closing(page.render(scale=scale)) as bitmap:
                    preview = bitmap.to_pil().convert("RGB").copy()
                if _rectify_pdf_photo(pdf, target, page, preview):
                    changed = True
                    continue
                angle = detect_skew(preview)
            if angle:
                straighten_page(target, angle)
                changed = True
        if changed:
            stream = io.BytesIO()
            pdf.save(stream)
            return stream.getvalue()
    return base


def _unturn(x, y, rotation, width, height):
    """Upright OCR coordinates back to the original rendered page."""
    if rotation == 90:
        return y, height - x
    if rotation == 180:
        return width - x, height - y
    if rotation == 270:
        return width - y, x
    return x, y


def _covered(box, existing):
    x0, y0, x1, y1 = box
    area = max((x1 - x0) * (y1 - y0), 1)
    return any(
        max(0, min(x1, b[2]) - max(x0, b[0])) * max(0, min(y1, b[3]) - max(y0, b[1]))
        > area * 0.5
        for b in existing
    )


def _overlay(target, result, width, height, existing, text_layer):
    rotation = result.rotation % 360
    upright_width, upright_height = (
        (height, width) if rotation % 180 else (width, height)
    )
    commands = []
    for line in getattr(result, "selection", result.lines):
        text = line.text.strip()
        if not text:
            continue
        x0, y0, x1, y1 = map(float, line.box)
        x0, x1 = max(0, x0), min(upright_width, x1)
        y0, y1 = max(0, y0), min(upright_height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        a, b = (
            _unturn(x0, y0, rotation, width, height),
            _unturn(x1, y1, rotation, width, height),
        )
        box = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        if _covered(box, existing):
            continue
        advance, content = positioned_text(line, text_layer, (x0, y0, x1, y1))
        line_height = y1 - y0
        baseline = upright_height - y1 + 0.2 * line_height
        commands.append(
            f"BT /OCR 1 Tf 3 Tr {advance:.6f} 0 0 {line_height:.6f} "
            f"{x0:.6f} {baseline:.6f} Tm {content} ET"
        )
    if commands:
        text_layer.overlay(target, commands, upright_width, upright_height, rotation)


def convert_pdf(
    source: Path,
    destination: Path,
    progress: Callable[[int, int], None] = lambda _done, _total: None,
    *,
    enhance: bool = True,
    report: dict | None = None,
) -> int:
    """Restore raster detail while keeping layout, vectors and existing text.

    OCR uses a bounded image independently of the higher-resolution visual layer.
    Set enhance=False to keep original raster samples for archival comparisons.
    """
    kind, count = inspect_document(source)
    progress(0, count)
    results = []
    base = (
        _image_pdf(source, results, enhance, progress)
        if kind == "image"
        else source.read_bytes()
    )
    if kind == "pdf" and enhance:
        base = _straighten_pdf(base)
    with (
        pikepdf.open(io.BytesIO(base)) as output,
        pdfium.PdfDocument(base) as document,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="restore") as pool,
    ):
        text_layer = TextLayer(output)
        for index in range(count):
            if kind == "image":
                result, width, height = results[index]
                _overlay(output.pages[index], result, width, height, [], text_layer)
                continue
            rasters = prepare_pdf_images(output.pages[index], pool) if enhance else []
            with (
                closing(document[index]) as page,
                closing(page.get_textpage()) as textpage,
            ):
                # Digital pages already have accurate text, including mathematical glyphs.
                has_text = bool(textpage.get_text_bounded().strip())
                if has_text and not any(
                    page.get_objects(
                        filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE], max_depth=10
                    )
                ):
                    progress(index + 1, count)
                    continue
                w, h = page.get_size()
                scale = min(OCR_DPI / 72, math.sqrt(RENDER_PIXELS / max(w * h, 1)))
                with closing(page.render(scale=scale, grayscale=True)) as bitmap:
                    pixels = bitmap.to_numpy().copy()
                    pos = bitmap.get_posconv(page)
                    existing = []
                    for n in range(textpage.count_rects()):
                        left, bottom, right, top = textpage.get_rect(n)
                        a, b = pos.to_bitmap(left, top), pos.to_bitmap(right, bottom)
                        existing.append(
                            (
                                min(a[0], b[0]),
                                min(a[1], b[1]),
                                max(a[0], b[0]),
                                max(a[1], b[1]),
                            )
                        )
                gray = np.ascontiguousarray(
                    pixels[:, :, 0] if pixels.ndim == 3 else pixels
                )
                result = _recognize(gray, index)
                _overlay(
                    output.pages[index],
                    result,
                    gray.shape[1],
                    gray.shape[0],
                    existing,
                    text_layer,
                )
            finish_pdf_images(rasters)
            progress(index + 1, count)
        text_layer.finish()
        # Keep every encoded sample, including images nested in forms and masks.
        # Ask viewers to avoid interpolation that can soften fine scanned strokes.
        for obj in output.objects:
            if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image":
                obj.Interpolate = False
        output.save(destination, object_stream_mode=pikepdf.ObjectStreamMode.generate)
    if report is not None:
        from editable_scan.quality import compare_document

        report.update(compare_document(source, destination))
    return count
