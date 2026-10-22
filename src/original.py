# SPDX-License-Identifier: MPL-2.0
"""Edit existing PDF text; retain scanned glyphs for color and weight changes."""

from __future__ import annotations

import io
import math
import threading

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

from src.editor import FONTS, EditError
from src.flow import PageFlow, continuation_pages

PDF_LOCK = threading.RLock()


def display_matrix(page):
    # MuPDF's rotation_matrix translations omit UserUnit on some PDFs.
    width, height = page.rect.width, page.rect.height
    return {
        0: fitz.Matrix(1, 0, 0, 1, 0, 0),
        90: fitz.Matrix(0, 1, -1, 0, width, 0),
        180: fitz.Matrix(-1, 0, 0, -1, width, height),
        270: fitz.Matrix(0, -1, 1, 0, 0, height),
    }[page.rotation]


def regions(page):
    """Stable source IDs and geometry, never supplied by the browser."""
    result = []
    scan_view = None
    scan_scale = min(2, 1600 / max(page.rect.width, page.rect.height))
    for bi, block in enumerate(page.get_text("rawdict", flags=0)["blocks"]):
        for li, line in enumerate(block.get("lines", [])):
            if abs(line["dir"][1]) > 0.02 or line["dir"][0] < 0.98:
                continue  # Do not silently flatten slanted or vertical lettering.
            for si, span in enumerate(line["spans"]):
                span["text"] = "".join(char["c"] for char in span["chars"])
                if not span["text"].strip() or len(span["text"]) > 4000:
                    continue
                rect = fitz.Rect(span["bbox"])
                if rect.is_empty or not page.rect.contains(rect * display_matrix(page)):
                    continue
                scan = span["font"] == "Occulta"
                display = rect * display_matrix(page)
                color = span["color"]
                if scan:
                    if scan_view is None:
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(scan_scale, scan_scale), alpha=False
                        )
                        scan_view = np.frombuffer(pix.samples, np.uint8).reshape(
                            pix.height, pix.width, 3
                        )
                    left, top = (
                        max(0, int(display.x0 * scan_scale)),
                        max(0, int(display.y0 * scan_scale)),
                    )
                    right, bottom = (
                        math.ceil(display.x1 * scan_scale),
                        math.ceil(display.y1 * scan_scale),
                    )
                    sample = scan_view[top:bottom, left:right].reshape(-1, 3)
                    if len(sample):
                        light = sample.mean(axis=1)
                        ink = sample[light <= np.percentile(light, 5)]
                        rgb = np.median(ink, axis=0).astype(int)
                        color = int(rgb[0]) * 65536 + int(rgb[1]) * 256 + int(rgb[2])
                result.append(
                    {
                        "id": f"p{page.number + 1}-b{bi}-l{li}-s{si}",
                        "page": page.number + 1,
                        "text": span["text"],
                        "x": display.x0,
                        "y": display.y0,
                        "width": display.width,
                        "height": display.height,
                        "bold": bool(span["flags"] & 16) if not scan else False,
                        "color": f"#{color:06x}",
                        "font": span["font"],
                        "fontExact": not scan,
                        "scanned": scan,
                        "size": span["size"],
                        "rect": list(rect),
                        "origin": span["origin"],
                        "flags": span["flags"],
                        "positions": [char["origin"] for char in span["chars"]],
                    }
                )
    return result


def original_regions(source, number):
    with (
        PDF_LOCK,
        (
            fitz.open(stream=source, filetype="pdf")
            if isinstance(source, bytes)
            else fitz.open(source)
        ) as doc,
    ):
        if not 1 <= number <= len(doc):
            raise EditError("This page does not exist.")
        return [
            {
                k: v
                for k, v in item.items()
                if k not in ("rect", "origin", "flags", "positions")
            }
            for item in regions(doc[number - 1])
        ]


def patch_image(page, rect):
    # Only a small text region is rasterized; the rest of the page stays intact.
    scale = 4
    for image in page.get_image_info():
        bounds = fitz.Rect(image["bbox"])
        if bounds.contains(rect):
            scale = max(image["width"] / bounds.width, image["height"] / bounds.height)
            break
    scale = min(scale, 12, (2_000_000 / max(1, rect.width * rect.height)) ** 0.5)
    rect = fitz.Rect(
        math.floor(rect.x0 * scale) / scale,
        math.floor(rect.y0 * scale) / scale,
        math.ceil(rect.x1 * scale) / scale,
        math.ceil(rect.y1 * scale) / scale,
    )
    rect &= page.rect * ~display_matrix(page)
    rotation = page.rotation
    page.set_rotation(0)
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    finally:
        page.set_rotation(rotation)
    image = (
        np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    )
    # Closing estimates the paper behind dark lettering, retaining local shading.
    radius = max(3, min(31, round(pix.height * 0.45))) | 1
    background = cv2.morphologyEx(
        image, cv2.MORPH_CLOSE, np.ones((radius, radius), np.uint8)
    )
    contrast = np.max(background.astype(float) - image.astype(float), axis=2)
    denominator = np.maximum(30, np.max(background, axis=2))
    alpha = np.clip(contrast / denominator, 0, 1)
    mask = (alpha > 0.08).astype(np.uint8) * 255
    if mask.any():
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        background = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return rect, background, alpha


def png(image):
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def font_candidates():
    for family in (
        "DejaVuSans",
        "DejaVuSerif",
        "DejaVuSansMono",
        "LiberationSans",
        "LiberationSerif",
        "LiberationMono",
    ):
        italic = (
            "Italic"
            if family == "DejaVuSerif" or family.startswith("Liberation")
            else "Oblique"
        )
        regular = "-Regular" if family.startswith("Liberation") else ""
        for suffix in (regular, "-Bold", f"-{italic}", f"-Bold{italic}"):
            yield FONTS / f"{family}{suffix}.ttf"


def closest_font(item, alpha, rect, bold):
    """Match original ink without distorting its aspect ratio; recover size/baseline."""
    ys, xs = np.where(alpha > 0.25)
    if not len(xs):
        return (
            fitz.Font(fontfile=str(FONTS / "DejaVuSans.ttf")),
            item["size"],
            fitz.Point(item["origin"]),
        )
    left, top, right, bottom = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    target = alpha[top:bottom, left:right]
    height, width = target.shape
    normalized_width = max(1, min(2048, round(width / height * 48)))
    target = cv2.resize(target, (normalized_width, 48))
    best = None
    for path in font_candidates():
        font = ImageFont.truetype(str(path), 128)
        bbox = font.getbbox(item["text"], anchor="ls")
        estimated = round(height / max(1, bbox[3] - bbox[1]) * 128)
        # Render near the actual scan resolution: hinting and character advances
        # at small sizes differ from a large font resized afterwards.
        for pixel_size in range(max(4, estimated - 2), max(5, estimated + 3)):
            font = ImageFont.truetype(str(path), pixel_size)
            bbox = font.getbbox(item["text"], anchor="ls")
            sample = Image.new(
                "L", (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1]))
            )
            ImageDraw.Draw(sample).text(
                (-bbox[0], -bbox[1]), item["text"], font=font, fill=255, anchor="ls"
            )
            ink = np.asarray(sample) / 255.0
            sy, sx = np.where(ink > 0.25)
            if not len(sx):
                continue
            ink = ink[sy.min() : sy.max() + 1, sx.min() : sx.max() + 1]
            ratio = (ink.shape[1] / ink.shape[0]) / (width / height)
            candidate_width = max(1, min(4096, round(normalized_width * ratio)))
            candidate = cv2.resize(ink, (candidate_width, 48))
            canvas = max(normalized_width, candidate_width)
            original = np.zeros((48, canvas))
            rendered = np.zeros_like(original)
            original[:, :normalized_width] = target
            rendered[:, :candidate_width] = candidate
            score = np.mean(
                (
                    cv2.GaussianBlur(original, (3, 3), 0)
                    - cv2.GaussianBlur(rendered, (3, 3), 0)
                )
                ** 2
            )
            score += 0.12 * abs(math.log(ratio))
            if best is None or score < best[0]:
                best = score, path, bbox, (sx.min(), sy.min()), ink.shape[0], pixel_size
    _, path, bbox, offset, sample_height, pixel_size = best
    pixels_per_point = alpha.shape[0] / rect.height
    size = height / sample_height * pixel_size / pixels_per_point
    # PIL's baseline-relative bounds refer to the same font em as the PDF writer.
    origin = fitz.Point(
        rect.x0
        + (left - (bbox[0] + offset[0]) * size / pixel_size * pixels_per_point)
        / (alpha.shape[1] / rect.width),
        rect.y0 + top / pixels_per_point - (bbox[1] + offset[1]) * size / pixel_size,
    )
    if bold and "Bold" not in path.stem:
        stem = path.stem.replace("-Regular", "")
        if "Italic" in stem:
            stem = stem.replace("-Italic", "-BoldItalic")
        elif "Oblique" in stem:
            stem = stem.replace("-Oblique", "-BoldOblique")
        else:
            stem += "-Bold"
        path = path.with_name(stem + ".ttf")
    return fitz.Font(fontfile=str(path)), size, origin


def font_for(doc, page, item, edit):
    # Embedded fonts may be subsets. Reuse only if all requested glyphs exist.
    for entry in page.get_fonts(full=True):
        if entry[3].split("+")[-1] == item["font"].split("+")[-1]:
            try:
                data = doc.extract_font(entry[0])[3]
                font = (
                    fitz.Font(fontbuffer=data) if data else fitz.Font(fontname=entry[3])
                )
                if (edit.bold or not item["bold"]) and all(
                    c.isspace() or font.has_glyph(ord(c)) for c in edit.text
                ):
                    return font
            except (ValueError, RuntimeError):
                pass
    name = item["font"].lower()
    family = (
        "LiberationSerif"
        if "times" in name or "nimbusroman" in name or "liberationserif" in name
        else "LiberationSans"
        if any(
            part in name
            for part in (
                "helvetica",
                "arial",
                "nimbussans",
                "calibri",
                "liberationsans",
            )
        )
        else "LiberationMono"
        if "courier" in name or "liberationmono" in name
        else "DejaVuSerif"
        if ("times" in name or "serif" in name) and "sans" not in name
        else "DejaVuSansMono"
        if "courier" in name or "mono" in name
        else "DejaVuSans"
    )
    italic = bool(item["flags"] & 2)
    suffix = (
        ("-BoldItalic" if edit.bold else "-Italic")
        if italic and (family == "DejaVuSerif" or family.startswith("Liberation"))
        else ("-BoldOblique" if edit.bold else "-Oblique")
        if italic
        else "-Bold"
        if edit.bold
        else "-Regular"
        if family.startswith("Liberation")
        else ""
    )
    return fitz.Font(fontfile=str(FONTS / f"{family}{suffix}.ttf"))


def edit_original(source, edits):
    if not edits:
        return source
    if sum(len(e.text) for e in edits) > 80_000:
        raise EditError("This document contains too much edited text.")
    if len({e.id for e in edits}) != len(edits):
        raise EditError("Each source passage must have a unique ID.")
    with PDF_LOCK, fitz.open(stream=source, filetype="pdf") as doc:
        modified = False
        pending = []
        by_page = {}
        for edit in edits:
            if edit.page > len(doc):
                raise EditError("This page does not exist.")
            by_page.setdefault(edit.page, []).append(edit)
        for number, changes in by_page.items():
            page = doc[number - 1]
            catalog = {r["id"]: r for r in regions(page)}
            rotation = page.rotation
            page.set_rotation(0)
            prepared = []
            flow = None
            for edit in changes:
                item = catalog.get(edit.id)
                if not item:
                    raise EditError("This passage no longer exists. Reopen the editor.")
                if (edit.text, edit.bold, edit.color.lower()) == (
                    item["text"],
                    item["bold"],
                    item["color"],
                ):
                    continue
                rect = fitz.Rect(item["rect"])
                if item["scanned"]:
                    rect = (rect + (-1, -1, 1, 1)) & (page.rect * ~display_matrix(page))
                    rect, background, alpha = patch_image(page, rect)
                else:
                    background = alpha = None
                glyphs = item["scanned"] and edit.text == item["text"]
                font = None
                size, origin = item["size"], fitz.Point(item["origin"])
                if edit.text and not glyphs:
                    if item["scanned"]:
                        font, size, origin = closest_font(item, alpha, rect, edit.bold)
                    else:
                        font = font_for(doc, page, item, edit)
                if font and any(
                    not c.isspace() and not font.has_glyph(ord(c)) for c in edit.text
                ):
                    raise EditError(
                        "The matching font does not support one of these characters."
                    )
                lines = []
                if font and edit.text:
                    if not item["scanned"] and edit.text == item["text"]:
                        lines = (
                            None  # Preserve exact tracking for formatting-only edits.
                        )
                    else:
                        if flow is None:
                            flow = PageFlow(page)
                        lines, overflow = flow.place(
                            edit.text, font, size, origin, rect
                        )
                        if overflow:
                            color = tuple(
                                int(edit.color[i : i + 2], 16) / 255 for i in (1, 3, 5)
                            )
                            pending.append(
                                (
                                    overflow,
                                    font,
                                    size,
                                    color,
                                    edit.bold,
                                    page.rect.width,
                                    page.rect.height,
                                )
                            )
                prepared.append(
                    (
                        edit,
                        item,
                        rect,
                        background,
                        alpha,
                        glyphs,
                        font,
                        size,
                        origin,
                        lines,
                    )
                )
                page.add_redact_annot(rect, fill=False, cross_out=False)
            if not prepared:
                page.set_rotation(rotation)
                continue
            modified = True
            # Remove old searchable text. Keep source images and drawings intact;
            # scanned lettering is replaced by a small local paper patch below.
            page.apply_redactions(images=0, graphics=0, text=0)
            for (
                edit,
                item,
                rect,
                background,
                alpha,
                glyphs,
                font,
                size,
                origin,
                lines,
            ) in prepared:
                color = tuple(int(edit.color[i : i + 2], 16) / 255 for i in (1, 3, 5))
                if item["scanned"]:
                    patch = background
                    if glyphs:
                        if edit.bold:
                            alpha = cv2.dilate(alpha, np.ones((2, 2), np.uint8))
                        patch = np.clip(
                            background * (1 - alpha[:, :, None])
                            + np.array(color) * 255 * alpha[:, :, None],
                            0,
                            255,
                        ).astype(np.uint8)
                    page.insert_image(rect, stream=png(patch), overlay=True)
                if edit.text:
                    if glyphs:
                        # Scanned glyphs stay visible; replace only their hidden text.
                        font = fitz.Font(fontfile=str(FONTS / "DejaVuSans.ttf"))
                        size = rect.height / max(0.1, font.ascender - font.descender)
                        origin = fitz.Point(rect.x0, rect.y0 + size * font.ascender)
                    writer = fitz.TextWriter(page.rect)
                    width = font.text_length(edit.text, fontsize=size)
                    factor = rect.width / max(0.01, width) if glyphs else 1
                    if not item["scanned"] and edit.text == item["text"]:
                        # Preserve the PDF's original tracking and individual origins.
                        factor = 1
                        for char, position in zip(edit.text, item["positions"]):
                            writer.append(position, char, font=font, fontsize=size)
                    elif glyphs:
                        writer.append(origin, edit.text, font=font, fontsize=size)
                    else:
                        for position, line in lines:
                            if line:
                                writer.append(position, line, font=font, fontsize=size)
                    writer.write_text(
                        page,
                        color=color,
                        render_mode=3
                        if glyphs
                        else 2
                        if edit.bold and not font.is_bold
                        else 0,
                        morph=(origin, fitz.Matrix(factor, 1)),
                    )
            page.set_rotation(rotation)
        if not modified:
            return source
        continuation_pages(doc, pending)
        doc.subset_fonts()
        return doc.tobytes(garbage=4, deflate=True)
