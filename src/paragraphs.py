# SPDX-License-Identifier: MPL-2.0
"""Conservative paragraph detection across native spans and OCR line blocks."""

import re
from collections import Counter
from itertools import pairwise

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


def paragraphs(items):
    lines = []
    for item in items:
        rect = fitz.Rect(item["rect"])
        previous = lines[-1] if lines else None
        same_line = (
            previous
            and item["id"].rsplit("-s", 1)[0] == previous[-1]["id"].rsplit("-s", 1)[0]
            and sum(len(m["text"]) + 1 for m in previous) + len(item["text"]) <= 4000
        )
        if same_line and -rect.height * 0.25 <= rect.x0 - fitz.Rect(
            previous[-1]["rect"]
        ).x1 < max(12, rect.height * 1.8):
            previous.append(item)
        else:
            lines.append([item])

    groups = []
    for line in lines:
        area = fitz.Rect(line[0]["rect"])
        for member in line[1:]:
            area |= fitz.Rect(member["rect"])
        text = line[0]["text"]
        for left, right in pairwise(line):
            gap = right["rect"][0] - left["rect"][2]
            separator = (
                ""
                if text.endswith(" ")
                or right["text"].startswith(" ")
                or gap < area.height * 0.12
                else " "
            )
            text += separator + right["text"]
        previous = groups[-1] if groups else None
        sample = representative(line)
        if previous:
            last = previous["lines"][-1]
            old_sample = representative(last["members"])
            old = last["area"]
            height = min(area.height, old.height)
            overlap = min(area.x1, old.x1) - max(area.x0, old.x0)
            similar = abs(sample["size"] - old_sample["size"]) <= max(
                0.5, sample["size"] * 0.15
            )
            aligned = (
                abs(area.x0 - old.x0) < height * 1.5
                and overlap > min(area.width, old.width) * 0.5
            )
            nearby = old.y0 + height * 0.65 < area.y0 < old.y1 + height * 0.65
            list_start = re.match(r"^\s*(?:[•●▪*-]|\d+[.)])\s", text)
            join = (
                nearby
                and aligned
                and similar
                and not list_start
                and sample["bold"] == old_sample["bold"]
                and len(previous["text"]) + len(text) < 4000
            )
        else:
            join = False
        entry = {"area": area, "members": line}
        if join:
            previous["lines"].append(entry)
            previous["text"] += " " + text
        else:
            groups.append({"lines": [entry], "text": text})

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
