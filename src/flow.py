# SPDX-License-Identifier: MPL-2.0
"""Wrap replacement text at its original size without covering existing content."""

import math
from itertools import pairwise

import cv2
import numpy as np
import pymupdf as fitz


def take_line(text, font, size, width):
    """Consume one line, preferring word boundaries and respecting explicit breaks."""
    paragraph, separator, tail = text.partition("\n")
    # Bound the common case to a line-sized prefix, rather than measuring the
    # entire remaining paragraph on every line (quadratic for long documents).
    limit = max(32, int(max(0, width) / max(size, 0.1) * 8) + 1)
    while True:
        advances = np.cumsum(font.char_lengths(paragraph[:limit], fontsize=size))
        if limit >= len(paragraph) or (len(advances) and advances[-1] > width):
            break
        limit *= 2
    count = int(np.searchsorted(advances, max(0, width), side="right"))
    if count >= len(paragraph):
        return paragraph, tail if separator else ""
    if not count:
        return None, text
    boundary = paragraph.rfind(" ", 0, count + 1)
    if boundary > 0:
        count = boundary
    line = paragraph[:count].rstrip()
    remaining = paragraph[count:].lstrip(" ")
    if separator:
        remaining += "\n" + tail
    return line, remaining


class PageFlow:
    def __init__(self, page, items=()):
        self.bounds = page.rect
        self.items = items
        self.cleared = []
        self.rows = []
        for item in items:
            rect = fitz.Rect(item["rect"])
            key = item["id"].rsplit("-s", 1)[0]
            previous = self.rows[-1] if self.rows else None
            if (
                previous
                and previous["key"] == key
                and -rect.height * 0.25
                <= rect.x0 - previous["rect"].x1
                < max(12, rect.height * 1.8)
            ):
                previous["rect"] |= rect
            else:
                self.rows.append(
                    {
                        "key": key,
                        "rect": rect,
                        "baseline": item["origin"][1],
                        "size": item["size"],
                    }
                )
        left = min((r["rect"].x0 for r in self.rows), default=2)
        self.margin = (
            max(2, left)
            if left <= self.bounds.width * 0.2
            else self.bounds.width * 0.06
        )
        self.scale = min(2, 1200 / max(page.rect.width, page.rect.height))
        pix = page.get_pixmap(
            matrix=fitz.Matrix(self.scale, self.scale),
            colorspace=fitz.csGRAY,
            alpha=False,
        )
        gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        ink = ((paper.astype(int) - gray > 18) | (gray < 130)).astype(np.uint8)
        # Protect enclosed figures, but do not fill a border around the whole sheet.
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if 100 < cv2.contourArea(contour) < ink.size * 0.2:
                cv2.drawContours(ink, [contour], -1, 1, thickness=cv2.FILLED)
        self.ink = cv2.dilate(ink, np.ones((3, 3), np.uint8)).astype(bool)
        self.reserved = []

    def frame(self, area, origin, size, members):
        """Infer a local text column, not the width of a short source sentence."""
        reference_size = members[0]["size"] if members else size
        nearby = [
            r
            for r in self.rows
            if abs(r["baseline"] - origin.y) <= max(size * 8, self.bounds.height * 0.18)
            and abs(r["size"] - reference_size) <= reference_size * 0.25
        ]
        aligned = [r for r in nearby if abs(r["rect"].x0 - area.x0) <= size * 1.5]
        left = min((r["rect"].x0 for r in aligned), default=area.x0)
        margin = max(2, left) if left <= self.bounds.width * 0.2 else self.margin
        right = max(area.x1, self.bounds.x1 - margin)
        # Several long lines ending at the same edge provide stronger evidence
        # of a narrower text frame than a ragged final line does. A wide heading
        # must not hide the repeated body edge beneath it.
        edges = [r["rect"].x1 for r in aligned]
        repeated = [
            edge
            for edge in edges
            if edge - area.x0 >= min((right - area.x0) * 0.55, size * 14)
            and sum(abs(other - edge) <= size for other in edges) >= 3
        ]
        if repeated:
            right = max(area.x1, min(right, max(repeated) + 0.5))
        for row in nearby:
            rect = row["rect"]
            if rect.x0 <= max(area.x1 + size * 0.5, area.x0 + size * 4):
                continue
            same_height = rect.y0 < area.y1 and rect.y1 > area.y0
            column = sum(abs(r["rect"].x0 - rect.x0) <= size for r in nearby) >= 2
            paired = any(
                abs(r["baseline"] - row["baseline"]) <= size * 0.5
                and r["rect"].x1 < rect.x0 - size * 0.5
                for r in aligned
            )
            if same_height or (column and paired):
                right = min(right, rect.x0 - max(size * 1.5, margin * 0.5))

        baselines = sorted({m["origin"][1] for m in members})
        gaps = [b - a for a, b in pairwise(baselines) if size <= b - a <= size * 2]
        if not gaps:
            baselines = sorted({r["baseline"] for r in aligned})
            gaps = [
                b - a for a, b in pairwise(baselines) if size <= b - a <= size * 1.9
            ]
        leading = float(np.median(gaps)) if gaps else size * 1.3
        return min(right, self.bounds.x1 - 2), leading

    @staticmethod
    def erased(rect, cleared):
        return any(
            (rect & area).get_area() >= rect.get_area() * 0.98 for area in cleared
        )

    def pixels(self, rect):
        r = rect & self.bounds
        return (
            slice(
                max(0, math.floor(r.y0 * self.scale)),
                max(0, math.ceil(r.y1 * self.scale)),
            ),
            slice(
                max(0, math.floor(r.x0 * self.scale)),
                max(0, math.ceil(r.x1 * self.scale)),
            ),
        )

    def place(
        self,
        text,
        font,
        size,
        origin,
        cleared,
        *,
        right=None,
        bottom=None,
        leading=None,
        reserve=True,
    ):
        ink = self.ink.copy()
        cleared = cleared if isinstance(cleared, list) else [cleared]
        erased = self.cleared + cleared
        for rect in erased:
            ink[self.pixels(rect + (-2, -2, 2, 2))] = False
        for rect in self.reserved:
            ink[self.pixels(rect)] = True
        result = []
        baseline = origin.y
        leading = leading if leading is not None else size * 1.3
        limit_right = min(
            self.bounds.x1 - 2, right if right is not None else self.bounds.x1
        )
        limit_bottom = min(
            self.bounds.y1 - 2,
            max(self.bounds.y1 - self.margin, max(r.y1 for r in cleared)),
            bottom if bottom is not None else self.bounds.y1,
        )
        # An untouched line is a barrier, not a hole to weave through. In
        # particular an indented line must not invite a few words beside it.
        for item in self.items:
            rect = fitz.Rect(item["rect"])
            if (
                item["origin"][1] > origin.y + size * 0.5
                and rect.x1 > origin.x
                and rect.x0 < limit_right
                and not self.erased(rect, erased)
            ):
                limit_bottom = min(
                    limit_bottom, max(origin.y + size * 0.25, rect.y0 - 0.5)
                )
        while text and baseline + size * 0.25 <= limit_bottom:
            band = fitz.Rect(
                origin.x,
                max(0, baseline - size * 0.85),
                limit_right,
                baseline + size * 0.25,
            )
            rows, columns = self.pixels(band)
            occupied = np.flatnonzero(ink[rows, columns].any(axis=0))
            available_right = limit_right
            if len(occupied):
                available_right = min(
                    available_right,
                    columns.start / self.scale + occupied[0] / self.scale - 1,
                )
            line, remaining = take_line(text, font, size, available_right - origin.x)
            if line is not None:
                position = fitz.Point(origin.x, baseline)
                result.append((position, line))
                width = font.text_length(line, fontsize=size)
                area = fitz.Rect(origin.x, band.y0, origin.x + width + 1, band.y1)
                if reserve:
                    self.reserved.append(area)
                text = remaining
                baseline += leading
            else:
                # Keep a contiguous paragraph on its baseline grid. Remaining
                # text belongs on a continuation page, never after a neighbour.
                break
        if reserve:
            self.cleared.extend(cleared)
        return result, text

    def layout(self, text, font, size, origin, cleared, mode, members=()):
        """Keep every character; shrink by at most 15%, never squeeze glyphs."""
        area = fitz.Rect(cleared[0])
        for rect in cleared[1:]:
            area |= rect
        right, leading = self.frame(area, origin, size, members)
        if mode == "shrink":
            for step in range(16):
                candidate = size * (1 - step / 100)
                lines, overflow = self.place(
                    text,
                    font,
                    candidate,
                    origin,
                    cleared,
                    right=right,
                    bottom=area.y1 + 0.5,
                    leading=leading,
                    reserve=False,
                )
                if not overflow:
                    lines, overflow = self.place(
                        text,
                        font,
                        candidate,
                        origin,
                        cleared,
                        right=right,
                        bottom=area.y1 + 0.5,
                        leading=leading,
                    )
                    return lines, overflow, candidate
            size *= 0.85
        lines, overflow = self.place(
            text,
            font,
            size,
            origin,
            cleared,
            right=right,
            bottom=area.y1 + 0.5 if mode == "page" else None,
            leading=leading,
        )
        return lines, overflow, size


def continuation_pages(doc, pending):
    for text, font, size, color, bold, width, height in pending:
        while text:
            page = doc.new_page(width=width, height=height)
            margin = min(36, width * 0.08, height * 0.08)
            baseline = margin + size * font.ascender
            writer = fitz.TextWriter(page.rect)
            previous = text
            while text and baseline + size * 0.25 <= height - margin:
                line, text = take_line(text, font, size, width - 2 * margin)
                if line is None:
                    break
                writer.append((margin, baseline), line, font=font, fontsize=size)
                baseline += size * 1.3
            if text == previous:
                # Very small source pages still need space for a full-size glyph.
                doc.delete_page(-1)
                width, height = max(width, size * 4 + 72), max(height, size * 4 + 72)
                continue
            writer.write_text(
                page, color=color, render_mode=2 if bold and not font.is_bold else 0
            )
