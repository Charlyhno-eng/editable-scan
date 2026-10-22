# SPDX-License-Identifier: MPL-2.0
"""Conservative text-line deskewing with rigid, non-destructive PDF rotation."""

import math

import cv2
import numpy as np
import pikepdf
from PIL import Image

MAX_ANGLE = 7.0
MIN_ANGLE = 0.25


def detect_skew(image):
    """Return a correction in degrees, or zero when text evidence is uncertain."""
    gray = np.asarray(image.convert("L")) if isinstance(image, Image.Image) else image
    if gray.ndim == 3:
        gray = gray[..., 0]
    scale = min(1, 1200 / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if min(gray.shape) < 100 or np.mean(gray > 220) < 0.5:
        return 0.0
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    h, w = gray.shape
    valid = (
        (stats[:, cv2.CC_STAT_AREA] >= 3)
        & (stats[:, cv2.CC_STAT_AREA] < h * w * 0.002)
        & (stats[:, cv2.CC_STAT_WIDTH] < w * 0.08)
        & (stats[:, cv2.CC_STAT_HEIGHT] < h * 0.08)
    )
    valid[0] = False
    if np.count_nonzero(valid) < 30:
        return 0.0
    mask = valid[labels].astype("uint8")

    def score(angle, pixels):
        ph, pw = pixels.shape
        matrix = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1)
        rotated = cv2.warpAffine(pixels, matrix, (pw, ph), flags=cv2.INTER_NEAREST)
        rows = rotated.sum(axis=1).astype("float64")
        return float(np.square(np.diff(rows)).sum())

    angles = np.arange(-MAX_ANGLE, MAX_ANGLE + 0.01, 0.5)
    scores = np.array([score(a, mask) for a in angles])
    best = float(angles[scores.argmax()])
    fine = np.arange(best - 0.4, best + 0.401, 0.05)
    best = float(max(fine, key=lambda a: score(a, mask)))
    peak, baseline = score(best, mask), score(0, mask)
    if (
        abs(best) < MIN_ANGLE
        or abs(best) >= MAX_ANGLE - 0.25
        or peak < baseline * 1.12
        or peak < np.median(scores) * 1.5
    ):
        return 0.0
    # Independent text bands must agree; a single diagram cannot turn the page.
    votes = 0
    for band in np.array_split(mask, 3):
        if np.count_nonzero(band) < 100:
            continue
        candidates = np.arange(-MAX_ANGLE, MAX_ANGLE + 0.01, 0.25)
        estimate = float(max(candidates, key=lambda a: score(a, band)))
        if abs(estimate - best) <= 0.4:
            votes += 1
    return round(best, 2) if votes >= 2 else 0.0


def rotation_geometry(width, height, angle):
    """Rigid PDF transform and expanded bounds, with no cropped corners."""
    radians = math.radians(angle)
    c, s = math.cos(radians), math.sin(radians)
    new_width = abs(c) * width + abs(s) * height
    new_height = abs(s) * width + abs(c) * height
    return (
        c,
        s,
        -s,
        c,
        new_width / 2 - c * width / 2 + s * height / 2,
        new_height / 2 - s * width / 2 - c * height / 2,
    ), (new_width, new_height)


def straighten_page(page, angle):
    """Rotate original page objects together, without rasterizing or resampling."""
    if not angle:
        return
    x0, y0, x1, y1 = map(float, page.cropbox)
    matrix, size = rotation_geometry(x1 - x0, y1 - y0, angle)
    a, b, c, d, e, f = matrix
    e -= a * x0 + c * y0
    f -= b * x0 + d * y0
    # Clip to the old crop box before rotation, retaining its original visibility.
    page.contents_add(
        (
            f"q {a:.12f} {b:.12f} {c:.12f} {d:.12f} {e:.12f} {f:.12f} cm "
            f"{x0:.12f} {y0:.12f} {x1 - x0:.12f} {y1 - y0:.12f} re W n\n"
        ).encode(),
        prepend=True,
    )
    page.contents_add(b"\nQ", prepend=False)
    page.MediaBox = pikepdf.Array([0, 0, *size])
    page.CropBox = pikepdf.Array([0, 0, *size])
    for name in ("/TrimBox", "/BleedBox", "/ArtBox"):
        if name in page.obj:
            del page.obj[name]
    page.obj.EditableScanDeskew = angle


def rotate_ocr(image, angle, size):
    """Rotate only OCR pixels; the visible PDF keeps its original image samples."""
    w, h = image.size
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    matrix[0, 2] += (size[0] - w) / 2
    matrix[1, 2] += (size[1] - h) / 2
    return Image.fromarray(
        cv2.warpAffine(
            np.asarray(image),
            matrix,
            size,
            flags=cv2.INTER_CUBIC,
            borderValue=255,
        )
    )
