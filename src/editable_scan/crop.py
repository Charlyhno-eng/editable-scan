# SPDX-License-Identifier: MPL-2.0
"""Find photographed paper boundaries and rectify its four corners."""

import cv2
import numpy as np
from PIL import Image


def _ordered(points):
    points = np.asarray(points, dtype="float32").reshape(4, 2)
    center = points.mean(axis=0)
    points = points[
        np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    ]
    return np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)


def _sample(image, points):
    return cv2.remap(
        image,
        points[..., 0].astype("float32"),
        points[..., 1].astype("float32"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _paper_edges(rgb, quad):
    """Look beyond printed frames for the actual paper/background transition."""
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 1), cv2.COLOR_RGB2LAB).astype(
        "float32"
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    reach = max(12, min(h, w) * 0.05)
    offsets = np.arange(-8, int(reach) + 1, dtype="float32")
    lines, evidence = [], 0
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        direction = end - start
        normal = np.array([direction[1], -direction[0]]) / np.linalg.norm(direction)
        centers = start + np.linspace(0.15, 0.85, 25)[:, None] * direction
        points = centers[:, None] + offsets[None, :, None] * normal
        inside = (
            (points[..., 0] >= 1)
            & (points[..., 0] < w - 1)
            & (points[..., 1] >= 1)
            & (points[..., 1] < h - 1)
        )
        colors = _sample(lab, points)
        near = centers[:, None] + np.arange(-4, 5)[None, :, None] * normal
        saturation = float(np.median(_sample(hsv, near)[..., 1].max(axis=1)))
        # A colored printed frame is inside the sheet, not its outer edge.
        outside_sat = float(
            np.median(_sample(hsv, (centers + 8 * normal)[:, None])[..., 1])
        )
        frame = saturation > 45 and outside_sat < max(30, saturation * 0.65)
        first = max(6, min(h, w) * 0.009) if frame else -2
        delta = np.linalg.norm(colors[:, 4:] - colors[:, :-4], axis=2)
        valid = inside[:, 4:] & inside[:, :-4]
        delta[~valid] = 0
        profile = np.median(delta, axis=0)
        distances = offsets[2:-2]
        profile[distances < first] = 0
        best = int(profile.argmax())
        distance = float(distances[best])

        # Printed ink must not count as evidence of a different background.
        # Bright samples on both sides avoid confusing colored frames with paper edges.
        def paper_color(depths, centers=centers, distance=distance, normal=normal):
            samples = _sample(
                lab,
                centers[:, None]
                + (distance + np.array(depths))[None, :, None] * normal,
            )
            return samples[np.arange(len(centers)), samples[..., 0].argmax(axis=1)]

        contrast = np.linalg.norm(
            paper_color([-14, -10, -6]) - paper_color([6, 10, 14]), axis=1
        )
        if profile[best] >= 4 and np.median(contrast) >= 8:
            evidence += 1
            distance = max(1, distance + 1)
            # The physical paper edge need not be parallel to its printed frame.
            nearby = (distances >= max(first, distance - 7)) & (
                distances <= distance + 5
            )
            local = delta[:, nearby]
            peaks = local.argmax(axis=1)
            strengths = local[np.arange(len(centers)), peaks]
            reliable = strengths >= max(3, profile[best] * 0.5)
            if np.count_nonzero(reliable) >= 15:
                positions = centers + (distances[nearby][peaks] + 1)[:, None] * normal
                vx, vy, x, y = cv2.fitLine(
                    positions[reliable].astype("float32"), cv2.DIST_HUBER, 0, 0.01, 0.01
                ).ravel()
                fitted = np.array([vx, vy])
                alignment = abs(
                    float(np.dot(fitted, direction) / np.linalg.norm(direction))
                )
                if alignment > 0.999:
                    lines.append((np.array([x, y]), fitted))
                    continue
        elif frame:
            distance = min(h, w) * 0.015
        else:
            distance = 1
        lines.append((start + normal * distance, direction))
    if evidence < 2:
        return None
    corners = []
    for i in range(4):
        a, da = lines[i - 1]
        b, db = lines[i]
        cross = np.column_stack((da, -db))
        if abs(np.linalg.det(cross)) < 1e-5:
            return None
        t = np.linalg.solve(cross, b - a)[0]
        corners.append(a + t * da)
    return np.asarray(corners, dtype="float32")


def find_sheet(image):
    """Return conservative native-pixel sheet corners, or None for ordinary scans."""
    if (image.mode in ("RGBA", "LA") or "transparency" in image.info) and image.convert(
        "RGBA"
    ).getchannel("A").getextrema() != (255, 255):
        return None
    thumb = image.convert("RGB")
    thumb.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    rgb = np.asarray(thumb)
    h, w = rgb.shape[:2]
    if min(h, w) < 150:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 1)
    edges = cv2.Canny(gray, 25, 75)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), "uint8"))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Two separate large sheets are ambiguous; never silently discard one of them.
    large = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > w * h * 0.22]
    for i, (x, y, bw, bh) in enumerate(large):
        for xx, yy, ww, hh in large[:i]:
            overlap = max(0, min(x + bw, xx + ww) - max(x, xx)) * max(
                0, min(y + bh, yy + hh) - max(y, yy)
            )
            if overlap < min(bw * bh, ww * hh) * 0.1:
                return None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour) / (w * h)
        if not 0.3 < area < 0.985:
            continue
        quad = None
        for tolerance in (0.015, 0.025, 0.04):
            polygon = cv2.approxPolyDP(
                contour, tolerance * cv2.arcLength(contour, True), True
            )
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                quad = _ordered(polygon)
                break
        if quad is None:
            continue
        sides = np.roll(quad, -1, axis=0) - quad
        lengths = np.linalg.norm(sides, axis=1)
        if lengths.min() < min(h, w) * 0.3:
            continue
        cosines = np.sum(sides * np.roll(sides, -1, axis=0), axis=1) / (
            lengths * np.roll(lengths, -1)
        )
        if np.max(np.abs(cosines)) > 0.65:
            continue
        mask = np.zeros((h, w), "uint8")
        cv2.fillConvexPoly(mask, quad.astype("int32"), 1)
        if np.mean(gray[mask > 0] > 140) < 0.55:
            continue
        refined = _paper_edges(rgb, quad)
        if refined is None:
            # An inner printed frame is not a safer alternative to an uncertain boundary.
            return None
        # Small extrapolations accommodate a paper corner just outside the photograph.
        if (
            refined[:, 0].min() < -w * 0.04
            or refined[:, 0].max() > w * 1.04
            or refined[:, 1].min() < -h * 0.04
            or refined[:, 1].max() > h * 1.04
        ):
            return None
        bounds = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        if np.linalg.norm(refined - bounds, axis=1).max() < min(w, h) * 0.03:
            return None
        return refined * np.array([image.width / w, image.height / h], dtype="float32")
    return None


def _rectified_size(corners, source_size):
    top, right, bottom, left = np.linalg.norm(
        np.roll(corners, -1, axis=0) - corners, axis=1
    )
    width, height = max(top, bottom), max(left, right)
    # Recover the rectangle's aspect ratio with a centered, square-pixel camera.
    # For weak/implausible camera estimates, retain the measured edge ratio.
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], "float32")
    homography = cv2.getPerspectiveTransform(unit, corners)
    a, b = homography[:, 0].copy(), homography[:, 1].copy()
    center = np.array(source_size) / 2
    a[:2] -= center * a[2]
    b[:2] -= center * b[2]
    if abs(a[2] * b[2]) > 1e-6:
        focal_squared = -np.dot(a[:2], b[:2]) / (a[2] * b[2])
        diagonal_squared = float(np.square(source_size).sum())
        if 0.09 * diagonal_squared < focal_squared < 64 * diagonal_squared:
            ratio = np.sqrt(
                (np.dot(a[:2], a[:2]) + focal_squared * a[2] ** 2)
                / (np.dot(b[:2], b[:2]) + focal_squared * b[2] ** 2)
            )
            if 0.6 < ratio / (width / height) < 1.67:
                width, height = max(width, height * ratio), max(height, width / ratio)
    scale = min(1, np.sqrt(40_000_000 / (width * height)))
    return max(1, round(width * scale)), max(1, round(height * scale))


def rectify_sheet(image, corners=None):
    """Remove the table and undo perspective once, before OCR and restoration."""
    corners = (
        find_sheet(image) if corners is None else np.asarray(corners, dtype="float32")
    )
    if corners is None:
        return image, None
    width, height = _rectified_size(corners, image.size)
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], "float32"
    )
    matrix = cv2.getPerspectiveTransform(corners, target)
    pixels = cv2.warpPerspective(
        np.asarray(image.convert("RGB")),
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    result = Image.fromarray(pixels)
    result.info.update(image.info)
    return result, corners.tolist()
