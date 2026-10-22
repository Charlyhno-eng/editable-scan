# SPDX-License-Identifier: MPL-2.0
"""Compare actual PDF renders with their source at the same page geometry."""

import math
from contextlib import closing

import cv2
import numpy as np
import pikepdf
import pypdfium2 as pdfium
from PIL import Image, ImageOps

from editable_scan.enhance import analyze_image

ANALYSIS_LONG_SIDE = 1600
METHOD = (
    "SSIM on RGB renders at matching dimensions, up to 1600 pixels on the long side "
    "(11-pixel windows, 4-pixel stride). "
    "85% foreground and 15% whole-page weighting prevents blank margins from hiding changes. "
    "The document score is the mean of all page scores. This measures appearance, not OCR accuracy "
    "or recovered detail. Edge contrast is an estimate, not a quality score."
)


def visual_similarity(original, result):
    """A bounded SSIM percentage, with ink and artwork weighted explicitly."""
    if original.size != result.size:
        raise ValueError("Comparison images must have matching dimensions")
    a = np.asarray(original.convert("RGB"))
    b = np.asarray(result.convert("RGB"))
    if np.array_equal(a, b):
        return 100.0

    foreground = (np.minimum(np.minimum(a[..., 0], a[..., 1]), a[..., 2]) < 245) | (
        np.minimum(np.minimum(b[..., 0], b[..., 1]), b[..., 2]) < 245
    )
    foreground = (
        cv2.dilate(foreground.astype("uint8"), np.ones((9, 9), dtype="uint8"))[::4, ::4]
        > 0
    )
    a, b = a.astype("float32"), b.astype("float32")

    def blur(value):
        return cv2.GaussianBlur(value, (11, 11), 1.5)[::4, ::4].copy()

    ma, mb = blur(a), blur(b)
    va = np.maximum(0, blur(a * a) - ma * ma)
    vb = np.maximum(0, blur(b * b) - mb * mb)
    cov = blur(a * b) - ma * mb
    score = ((2 * ma * mb + 6.5025) * (2 * cov + 58.5225)) / (
        (ma * ma + mb * mb + 6.5025) * (va + vb + 58.5225)
    )
    score = np.clip(score.mean(axis=2), 0, 1)
    whole = float(score.mean())
    focus = float(score[foreground].mean()) if foreground.any() else whole
    return round(100 * (0.85 * focus + 0.15 * whole), 2)


def _render(page, *, size=None):
    w, h = page.get_size()
    if size is None:
        scale = min(150 / 72, ANALYSIS_LONG_SIDE / max(w, h))
        size = (math.ceil(w * scale), math.ceil(h * scale))
    scale = min((size[0] - 1e-5) / w, (size[1] - 1e-5) / h)
    with closing(page.render(scale=scale)) as bitmap:
        image = bitmap.to_pil().convert("RGB").copy()
    if size is not None and image.size != size:
        # Fractional page sizes can differ by one raster pixel through rounding.
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _treatments(page):
    resources, seen, treatments = [page.Resources], set(), set()
    while resources:
        for obj in resources.pop().get("/XObject", {}).values():
            if obj.objgen in seen:
                continue
            seen.add(obj.objgen)
            if obj.get("/Subtype") == "/Form":
                resources.append(obj.get("/Resources", {}))
            elif obj.get("/Subtype") == "/Image":
                treatments.add(str(obj.get("/EditableScanTreatment", "preserved")))
    return sorted(treatments) or ["preserved"]


def _page_report(original, rendered, index, treatments):
    similarity = visual_similarity(original, rendered)
    before, after = analyze_image(original), analyze_image(rendered)
    strength = before["edge_strength"]
    change = (
        round(100 * (after["edge_strength"] / strength - 1), 1) if strength else None
    )
    restored = any(value != "preserved" for value in treatments)
    if similarity == 100:
        summary = (
            "The source and PDF are visually identical at the comparison resolution."
        )
    elif before["soft"] and restored and change is not None and change >= 5:
        summary = (
            f"The source has soft text edges. Measured edge contrast increased by {change:g}%. "
            "Restoration intentionally changes the image, which can lower similarity."
        )
    elif restored:
        summary = (
            "Scan edges were restored. Intentional visual changes can lower similarity."
        )
    else:
        summary = "The source was kept without sharpening or enlargement."
    return {
        "page": index + 1,
        "similarity_percent": similarity,
        "source_soft": before["soft"],
        "edge_contrast_change_percent": change,
        "treatments": treatments,
        "summary": summary,
    }


def compare_document(source, destination):
    """Measure every finished page; never infer similarity from compression size."""
    with source.open("rb") as stream:
        is_pdf = b"%PDF-" in stream.read(1024)
    pages = []
    with (
        pikepdf.open(destination) as metadata,
        pdfium.PdfDocument(destination) as output,
    ):
        if is_pdf:
            with pdfium.PdfDocument(source) as original:
                if len(original) != len(output):
                    raise ValueError("Source and output page counts differ")
                for index in range(len(output)):
                    with closing(original[index]) as page:
                        reference = _render(page)
                    with closing(output[index]) as page:
                        rendered = _render(page, size=reference.size)
                    pages.append(
                        _page_report(
                            reference,
                            rendered,
                            index,
                            _treatments(metadata.pages[index]),
                        )
                    )
        else:
            with Image.open(source) as original:
                if getattr(original, "n_frames", 1) != len(output):
                    raise ValueError("Source and output page counts differ")
                for index in range(len(output)):
                    original.seek(index)
                    reference = ImageOps.exif_transpose(original.copy()).convert("RGBA")
                    white = Image.new("RGBA", reference.size, "white")
                    white.alpha_composite(reference)
                    reference = white.convert("RGB")
                    reference.thumbnail(
                        (ANALYSIS_LONG_SIDE, ANALYSIS_LONG_SIDE),
                        Image.Resampling.LANCZOS,
                    )
                    with closing(output[index]) as page:
                        rendered = _render(page, size=reference.size)
                    pages.append(
                        _page_report(
                            reference,
                            rendered,
                            index,
                            _treatments(metadata.pages[index]),
                        )
                    )
    source_bytes, output_bytes = source.stat().st_size, destination.stat().st_size
    return {
        "similarity_percent": round(
            sum(p["similarity_percent"] for p in pages) / len(pages), 2
        ),
        "method": METHOD,
        "pages": pages,
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "size_reduction_percent": round(100 * (1 - output_bytes / source_bytes), 1),
    }
