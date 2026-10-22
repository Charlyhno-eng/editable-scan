# SPDX-License-Identifier: MPL-2.0
"""Reconstruct text paragraphs from source geometry, independent of PDF order."""

import re
from collections import Counter
from itertools import pairwise
from statistics import median

import pymupdf as fitz


def representative(members):
    """Use the main body style, rather than a short bold introductory fragment."""

    def style(item):
        return item["font"], round(item["size"], 1), item["bold"]

    weights = Counter()
    for item in members:
        weights[style(item)] += len(item["text"].strip())
    main = weights.most_common(1)[0][0]
    return max((m for m in members if style(m) == main), key=lambda m: len(m["text"]))


def source_lines(items):
    """OCR text matrices report a width-dependent size; use their ink height."""
    rows = []
    indexed = sorted(
        enumerate(items), key=lambda entry: (entry[1]["origin"][1], entry[1]["rect"][0])
    )
    for index, item in indexed:
        rect = fitz.Rect(item["rect"])
        baseline = item["origin"][1]
        key = item["id"].rsplit("-s", 1)[0]
        match = None
        for row in reversed(rows):
            height = min(rect.height, row["area"].height)
            if baseline - row["baseline"] > max(rect.height, row["area"].height):
                break
            same_source = key == row["members"][-1]["id"].rsplit("-s", 1)[0]
            gap = max(rect.x0, row["area"].x0) - min(rect.x1, row["area"].x1)
            if (
                abs(baseline - row["baseline"]) <= max(1, height * 0.2)
                and -height * 0.25
                <= gap
                <= (max(12, height * 1.8) if same_source else height * 0.75)
                and sum(len(m["text"]) + 1 for m in row["members"]) + len(item["text"])
                <= 4000
            ):
                match = row
                break
        if match is None:
            rows.append(
                {"members": [item], "area": rect, "baseline": baseline, "order": index}
            )
        else:
            match["members"].append(item)
            match["area"] |= rect
            match["order"] = min(match["order"], index)

    for row in rows:
        members = sorted(row["members"], key=lambda item: item["rect"][0])
        row["members"] = members
        text = members[0]["text"]
        for left, right in pairwise(members):
            gap = right["rect"][0] - left["rect"][2]
            separator = (
                ""
                if text.endswith(" ")
                or right["text"].startswith(" ")
                or gap < row["area"].height * 0.12
                else " "
            )
            text += separator + right["text"]
        row["text"] = text
        row["sample"] = representative(members)
        row["metric"] = (
            row["area"].height if row["sample"]["scanned"] else row["sample"]["size"]
        )
    return sorted(rows, key=lambda row: (row["baseline"], row["area"].x0))


def follows(group, row):
    """Accept body continuations and first-line indents, but retain real breaks."""
    last = group["lines"][-1]
    metric = max(row["metric"], last["metric"])
    small = min(row["metric"], last["metric"])
    delta = row["baseline"] - last["baseline"]
    scanned = row["sample"]["scanned"] and last["sample"]["scanned"]
    # A short OCR line may have a very different horizontal text matrix from a
    # full line. Its font size is not evidence of a heading or paragraph break.
    similar = metric - small <= max(
        3 if scanned else 0.5, metric * (0.35 if scanned else 0.15)
    )
    if not (
        small * 0.65 < delta <= metric * 2.1
        and similar
        and row["sample"]["bold"] == last["sample"]["bold"]
        and not re.match(r"^\s*(?:[•●▪*-]|\d+[.)])\s", row["text"])
        and len(group["text"]) + len(row["text"]) < 4000
    ):
        return False
    gaps = [b["baseline"] - a["baseline"] for a, b in pairwise(group["lines"])]
    if gaps and delta > median(gaps) * 1.35:
        return False
    body_left = median(line["area"].x0 for line in group["lines"][1:] or group["lines"])
    offset = row["area"].x0 - body_left
    aligned = abs(offset) <= max(2, small * 0.55)
    first_indent = len(group["lines"]) == 1 and -metric * 3 <= offset < 0
    hanging = (
        len(group["lines"]) == 1
        and 0 < offset <= metric * 3
        and re.match(r"^\s*(?:[•●▪*-]|\d+[.)])\s", last["text"])
    )
    overlap = min(row["area"].x1, last["area"].x1) - max(
        row["area"].x0, last["area"].x0
    )
    return overlap > 0 and (aligned or first_indent or hanging)


def paragraphs(items):
    groups = []
    seen = []
    for row in source_lines(items):
        # PDF objects and OCR blocks can alternate between columns. Follow the
        # closest compatible paragraph instead of only the preceding object.
        candidates = []
        for group in groups:
            if not follows(group, row):
                continue
            last = group["lines"][-1]
            # A closer heading or differently styled row in this column is a
            # real separator, even if an earlier body paragraph looks similar.
            blocked = any(
                last["baseline"] + last["metric"] * 0.5
                < other["baseline"]
                < row["baseline"] - row["metric"] * 0.5
                and min(last["area"].x1, row["area"].x1, other["area"].x1)
                > max(last["area"].x0, row["area"].x0, other["area"].x0)
                for other in reversed(seen)
            )
            if not blocked:
                candidates.append(group)
        if candidates:
            previous = min(
                candidates,
                key=lambda group: (
                    row["baseline"] - group["lines"][-1]["baseline"],
                    abs(row["area"].x0 - group["lines"][-1]["area"].x0),
                ),
            )
            previous["lines"].append(row)
            previous["text"] += " " + row["text"]
        else:
            groups.append({"lines": [row], "text": row["text"]})
        seen.append(row)
    groups.sort(key=lambda group: min(row["order"] for row in group["lines"]))

    result = []
    for group in groups:
        members = [m for line in group["lines"] for m in line["members"]]
        item = dict(members[0])
        sample = representative(members)
        for key in ("font", "size", "bold", "color", "flags"):
            item[key] = sample[key]
        rect = fitz.Rect(item["rect"])
        display = fitz.Rect(
            item["x"], item["y"], item["x"] + item["width"], item["y"] + item["height"]
        )
        boxes = []
        for member in members:
            rect |= fitz.Rect(member["rect"])
            box = {key: member[key] for key in ("x", "y", "width", "height")}
            boxes.append(box)
            display |= fitz.Rect(
                box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]
            )
        item.update(
            id=item["id"] + "-paragraph" if len(members) > 1 else item["id"],
            text=group["text"],
            rect=list(rect),
            x=display.x0,
            y=display.y0,
            width=display.width,
            height=display.height,
            members=members,
            memberIds=[m["id"] for m in members],
            boxes=boxes,
            lineCount=len(group["lines"]),
        )
        result.append(item)
    return result
