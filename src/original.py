# SPDX-License-Identifier: MPL-2.0
"""Edit existing PDF text; retain scanned glyphs for color and weight changes."""

from __future__ import annotations

import hashlib
import io
import math
import threading
from collections import OrderedDict

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

from src.editor import FONTS, EditError
from src.flow import PageFlow, continuation_pages
from src.paragraphs import paragraphs, representative

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


def original_regions(source, number, unit="line"):
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
                if k not in ("rect", "origin", "flags", "positions", "members")
            }
            for item in (
                paragraphs(regions(doc[number - 1]))
                if unit == "paragraph"
                else regions(doc[number - 1])
            )
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


def _closest_font(item, alpha, rect, bold):
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


# Cache only tiny match results, never a user's scanned image. The key includes
# ink, geometry and style, so two different documents cannot share a stale match.
_FONT_MATCHES = OrderedDict()


def closest_font(item, alpha, rect, bold):
    key = (
        hashlib.blake2b(alpha.tobytes(), digest_size=20).digest(),
        alpha.shape,
        hashlib.blake2b(item["text"].encode(), digest_size=20).digest(),
        item["size"],
        tuple(item["origin"]),
        tuple(rect),
        bold,
    )
    with PDF_LOCK:
        if key not in _FONT_MATCHES:
            _FONT_MATCHES[key] = _closest_font(item, alpha, rect, bold)
            if len(_FONT_MATCHES) > 64:
                _FONT_MATCHES.popitem(last=False)
        _FONT_MATCHES.move_to_end(key)
        font, size, origin = _FONT_MATCHES[key]
        return font, size, fitz.Point(origin)


def family_font(family, bold, italic):
    style = (
        "Italic"
        if family == "DejaVuSerif" or family.startswith("Liberation")
        else "Oblique"
    )
    suffix = (
        ("-Bold" if bold else "-") + style
        if italic
        else "-Bold"
        if bold
        else "-Regular"
        if family.startswith("Liberation")
        else ""
    )
    return fitz.Font(fontfile=str(FONTS / f"{family}{suffix}.ttf"))


def original_font(source, number, passage):
    """Resolve one selected passage lazily; matching an entire page is wasteful."""
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
        page = doc[number - 1]
        items = regions(page)
        catalog = {item["id"]: item for item in [*items, *paragraphs(items)]}
        item = catalog.get(passage)
        if not item:
            raise EditError("This passage no longer exists. Reopen the editor.")
        sample = representative(item.get("members", [item]))
        if sample["scanned"]:
            rect = fitz.Rect(sample["rect"])
            rect, _, alpha = patch_image(page, rect + (-1, -1, 1, 1))
            font, size, _ = closest_font(sample, alpha, rect, False)
            name = font.name
        else:
            name, size = sample["font"], sample["size"]
        normalized = name.lower()
        kind = (
            "Mono"
            if "mono" in normalized or "courier" in normalized
            else "Serif"
            if ("serif" in normalized or "times" in normalized or "roman" in normalized)
            and "sans" not in normalized
            else "Sans"
        )
        similar = [f"{family}{kind}" for family in ("DejaVu", "Liberation")]
        if (
            "liberation" in normalized
            or "helvetica" in normalized
            or "times" in normalized
            or "courier" in normalized
        ):
            similar.reverse()
        return {
            "name": name,
            "similar": similar,
            "size": round(size, 2),
            "exact": not sample["scanned"],
            "mixed": len(
                {(m["font"], round(m["size"], 1)) for m in item.get("members", [item])}
            )
            > 1,
        }


def font_for(doc, page, item, edit):
    if edit.font != "auto":
        return family_font(edit.font, edit.bold, bool(item["flags"] & 2))
    # Embedded fonts may be subsets. Reuse only if all requested glyphs exist.
    for entry in page.get_fonts(full=True):
        if entry[3].split("+")[-1] == item["font"].split("+")[-1]:
            try:
                data = doc.extract_font(entry[0])[3]
                font = (
                    fitz.Font(fontbuffer=data) if data else fitz.Font(fontname=entry[3])
                )
                if (edit.bold or not item["bold"]) and all(
                    c.isspace() or font.has_glyph(ord(c)) for c in set(edit.text)
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
    return family_font(family, edit.bold, bool(item["flags"] & 2))


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
        for number, changes in sorted(by_page.items()):
            page = doc[number - 1]
            items = regions(page)
            catalog = {r["id"]: r for r in [*items, *paragraphs(items)]}
            claimed = set()
            resolved = []
            for edit in changes:
                item = catalog.get(edit.id)
                if not item:
                    raise EditError("This passage no longer exists. Reopen the editor.")
                members = item.get("members", [item])
                ids = {m["id"] for m in members}
                if claimed & ids:
                    raise EditError(
                        "A paragraph and its individual lines cannot be edited together."
                    )
                claimed |= ids
                if (edit.text, edit.bold, edit.color.lower(), edit.font) == (
                    item["text"],
                    item["bold"],
                    item["color"],
                    "auto",
                ):
                    continue
                if edit.text == item["text"] and edit.font == "auto":
                    # Formatting alone keeps each source fragment's font, size,
                    # baseline and scanned letter shapes, even in mixed paragraphs.
                    resolved.extend(
                        (
                            edit.model_copy(
                                update={
                                    "id": m["id"],
                                    "text": m["text"],
                                    "bold": edit.bold
                                    if edit.bold != item["bold"]
                                    else m["bold"],
                                    "color": edit.color
                                    if edit.color.lower() != item["color"]
                                    else m["color"],
                                }
                            ),
                            m,
                            [m],
                        )
                        for m in members
                    )
                else:
                    resolved.append((edit, item, members))
            if not resolved:
                continue
            modified = True
            rotation = page.rotation
            page.set_rotation(0)
            prepared = []
            flow = None
            for edit, item, members in sorted(
                resolved, key=lambda row: (row[1]["rect"][1], row[1]["rect"][0])
            ):
                patches = []
                cleared = []
                glyphs = (
                    item["scanned"]
                    and edit.text == item["text"]
                    and edit.font == "auto"
                )
                for member in members:
                    rect = fitz.Rect(member["rect"])
                    if member["scanned"]:
                        rect = (rect + (-1, -1, 1, 1)) & page.rect
                        rect, background, alpha = patch_image(page, rect)
                        patches.append((rect, background, alpha))
                    cleared.append(rect)
                sample = representative(members)
                font = None
                size, origin = sample["size"], fitz.Point(members[0]["origin"])
                if edit.text and not glyphs:
                    if sample["scanned"]:
                        rect, _, alpha = patches[members.index(sample)]
                        font, size, matched_origin = closest_font(
                            sample, alpha, rect, edit.bold
                        )
                        origin = patches[0][0].tl + (matched_origin - rect.tl)
                        if edit.font != "auto":
                            font = family_font(
                                edit.font, edit.bold or font.is_bold, font.is_italic
                            )
                    else:
                        font = font_for(doc, page, sample, edit)
                if font and any(
                    not c.isspace() and not font.has_glyph(ord(c))
                    for c in set(edit.text)
                ):
                    raise EditError(
                        "The matching font does not support one of these characters."
                    )
                color = tuple(int(edit.color[i : i + 2], 16) / 255 for i in (1, 3, 5))
                lines = []
                tracking = (
                    not item["scanned"]
                    and edit.text == item["text"]
                    and edit.font == "auto"
                )
                if font and edit.text and not tracking:
                    if flow is None:
                        flow = PageFlow(page, items)
                    lines, overflow, size = flow.layout(
                        edit.text,
                        font,
                        size,
                        origin,
                        cleared,
                        edit.overflow,
                        members=members,
                    )
                    if overflow:
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
                        cleared,
                        patches,
                        glyphs,
                        tracking,
                        font,
                        size,
                        origin,
                        lines,
                        color,
                    )
                )
            for row in prepared:
                for rect in row[2]:
                    page.add_redact_annot(rect, fill=False, cross_out=False)
            page.apply_redactions(images=0, graphics=0, text=0)
            for (
                edit,
                item,
                cleared,
                patches,
                glyphs,
                tracking,
                font,
                size,
                origin,
                lines,
                color,
            ) in prepared:
                for rect, background, alpha in patches:
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
                if not edit.text:
                    continue
                factor = 1
                if glyphs:
                    rect = cleared[0]
                    font = fitz.Font(fontfile=str(FONTS / "DejaVuSans.ttf"))
                    size = rect.height / max(0.1, font.ascender - font.descender)
                    origin = fitz.Point(rect.x0, rect.y0 + size * font.ascender)
                    factor = rect.width / max(
                        0.01, font.text_length(edit.text, fontsize=size)
                    )
                writer = fitz.TextWriter(page.rect)
                if tracking:
                    for char, position in zip(edit.text, item["positions"]):
                        writer.append(position, char, font=font, fontsize=size)
                elif glyphs:
                    writer.append(origin, edit.text, font=font, fontsize=size)
                else:
                    for position, line in lines:
                        if line:
                            writer.append(position, line, font=font, fontsize=size)
                # TextWriter handles crop offsets but omits PDF UserUnit. Scale
                # its PDF coordinates once, keeping physical sizes and baselines.
                unit = page.rect.width / page.cropbox.width
                matrix = fitz.Matrix(
                    factor / unit, 0, 0, 1 / unit, origin.x * (1 - factor) / unit, 0
                )
                writer.write_text(
                    page,
                    color=color,
                    render_mode=3
                    if glyphs
                    else 2
                    if edit.bold and not font.is_bold
                    else 0,
                    matrix=matrix if unit != 1 or factor != 1 else None,
                )
            page.set_rotation(rotation)
        if not modified:
            return source
        continuation_pages(doc, pending)
        doc.subset_fonts()
        return doc.tobytes(garbage=4, deflate=True)
