# SPDX-License-Identifier: MPL-2.0
"""Wrap replacement text at its original size without covering existing content."""

import math

import cv2
import numpy as np
import pymupdf as fitz


def take_line(text, font, size, width):
    """Consume one line, preferring word boundaries and respecting explicit breaks."""
    paragraph, separator, tail = text.partition("\n")
    advances = np.cumsum(font.char_lengths(paragraph, fontsize=size))
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
    def __init__(self, page):
        self.bounds = page.rect
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

    def place(self, text, font, size, origin, cleared):
        ink = self.ink.copy()
        ink[self.pixels(cleared + (-2, -2, 2, 2))] = False
        for rect in self.reserved:
            ink[self.pixels(rect)] = True
        result = []
        baseline = origin.y
        leading = size * 1.3
        while text and baseline + size * 0.25 < self.bounds.y1 - 2:
            band = fitz.Rect(
                origin.x,
                max(0, baseline - size * 0.85),
                self.bounds.x1 - 2,
                baseline + size * 0.25,
            )
            rows, columns = self.pixels(band)
            occupied = np.flatnonzero(ink[rows, columns].any(axis=0))
            right = self.bounds.x1 - 2
            if len(occupied):
                right = min(
                    right, columns.start / self.scale + occupied[0] / self.scale - 1
                )
            line, remaining = take_line(text, font, size, right - origin.x)
            if line is not None:
                position = fitz.Point(origin.x, baseline)
                result.append((position, line))
                width = font.text_length(line, fontsize=size)
                area = fitz.Rect(origin.x, band.y0, origin.x + width + 1, band.y1)
                self.reserved.append(area)
                ink[self.pixels(area)] = True
                text = remaining
                baseline += leading
            else:
                # Look below an occupied row rather than painting over a neighbour.
                baseline += max(1, size * 0.2)
        return result, text


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
