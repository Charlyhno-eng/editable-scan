# SPDX-License-Identifier: MPL-2.0
"""Readable Unicode and image-aligned word selection in OCR reading order."""

import re
import unicodedata
from types import SimpleNamespace

import cv2
import numpy as np

_SYMBOLS = dict(
    zip(
        [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
            "kappa",
            "lambda",
            "mu",
            "nu",
            "xi",
            "omicron",
            "pi",
            "rho",
            "sigma",
            "tau",
            "upsilon",
            "phi",
            "chi",
            "psi",
            "omega",
            "Gamma",
            "Delta",
            "Theta",
            "Lambda",
            "Xi",
            "Pi",
            "Sigma",
            "Phi",
            "Psi",
            "Omega",
        ],
        "αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ",
    )
)
_SYMBOLS.update(
    {
        "in": "∈",
        "notin": "∉",
        "mid": "|",
        "le": "≤",
        "leq": "≤",
        "ge": "≥",
        "geq": "≥",
        "ne": "≠",
        "neq": "≠",
        "times": "×",
        "cdot": "·",
        "cap": "∩",
        "cup": "∪",
        "subset": "⊂",
        "subseteq": "⊆",
        "infty": "∞",
        "sum": "∑",
        "prod": "∏",
        "int": "∫",
        "partial": "∂",
        "nabla": "∇",
        "forall": "∀",
        "exists": "∃",
        "to": "→",
        "rightarrow": "→",
        "leftrightarrow": "↔",
        "ldots": "…",
        "cdots": "⋯",
        "pm": "±",
        "approx": "≈",
        "equiv": "≡",
        "vert": "|",
        "lvert": "|",
        "rvert": "|",
        "max": "max",
        "min": "min",
        "sin": "sin",
        "cos": "cos",
        "log": "log",
        "lim": "lim",
    }
)
_FORMAT = {
    "mathcal",
    "mathbb",
    "mathbf",
    "mathrm",
    "mathit",
    "mathsf",
    "mathtt",
    "text",
    "operatorname",
}
_SUB = str.maketrans(
    "0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"
)
_SUP = str.maketrans("0123456789+-=()in", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱⁿ")


def _math(expression):
    """Translate a supported expression completely, or keep its original LaTeX."""
    value = expression
    position = 0

    def atom():
        nonlocal position
        while position < len(value) and value[position].isspace():
            position += 1
        if position >= len(value):
            raise ValueError
        char = value[position]
        position += 1
        if char == "{":
            return sequence(True)
        if char == "}":
            raise ValueError
        if char != "\\":
            return char
        match = re.match(r"[a-zA-Z]+\*?|.", value[position:])
        if not match:
            raise ValueError
        command = match[0]
        position += len(command)
        command = command.rstrip("*")
        if command in _SYMBOLS:
            return _SYMBOLS[command]
        if command in _FORMAT:
            return atom()
        if command in {
            "left",
            "right",
            "big",
            "Big",
            "displaystyle",
            ",",
            ";",
            "!",
            " ",
            "quad",
            "qquad",
        }:
            return ""
        if command in {"{", "}", "|", "%", "_", "$", "\\"}:
            return command
        if command in {"frac", "dfrac", "tfrac"}:
            return f"({atom()})/({atom()})"
        if command == "sqrt":
            return f"√({atom()})"
        if command in {"hat", "bar", "vec"}:
            return f"{command}({atom()})"
        raise ValueError

    def sequence(group=False):
        nonlocal position
        result = ""
        while position < len(value):
            if value[position].isspace():
                position += 1
                continue
            if value[position] == "}":
                if not group:
                    raise ValueError
                position += 1
                return result
            if value[position] in "_^":
                script = value[position]
                position += 1
                content = atom()
                table = _SUB if script == "_" else _SUP
                result += (
                    content.translate(table)
                    if content and all(ord(c) in table for c in content)
                    else f"{script}({content})"
                )
            else:
                result += atom()
        if group:
            raise ValueError
        return result

    try:
        return sequence()
    except ValueError:
        return "$" + expression + "$"


def readable_text(text):
    text = unicodedata.normalize("NFC", text)
    for ligature, expansion in zip(
        "ﬀﬁﬂﬃﬄﬅﬆ", ["ff", "fi", "fl", "ffi", "ffl", "st", "st"]
    ):
        text = text.replace(ligature, expansion)
    text = "".join(
        c for c in text if unicodedata.category(c) not in {"Cc", "Cs"} or c.isspace()
    )
    text = re.sub(
        r"\$\$(.+?)\$\$|\$(?!\$)([^$\n]+)\$",
        lambda m: _math(m[1] if m[1] is not None else m[2]),
        text,
    )
    return " ".join(text.split())


def character_width(char):
    if unicodedata.combining(char):
        return 0
    if unicodedata.east_asian_width(char) in {"W", "F"}:
        return 1000
    base = unicodedata.normalize("NFD", char)[0]
    if base in " ilI.,:;!'|":
        return 278
    if base in "mwMW@%":
        return 889
    if base in "ftjr()[]{}":
        return 350
    return 600 if base.isupper() else 556


def word_boxes(text, box, gray):
    """Match whitespace to real inter-word gaps; fall back without guessing ink."""
    words = text.split()
    if (
        len(words) < 2
        or gray is None
        or "$" in text
        or any(unicodedata.bidirectional(c) in {"R", "AL"} for c in text)
    ):
        return [(text, box)]
    x0, y0, x1, y1 = box
    h, w = gray.shape
    left, top, right, bottom = (
        max(0, int(x0)),
        max(0, int(y0)),
        min(w, int(np.ceil(x1))),
        min(h, int(np.ceil(y1))),
    )
    crop = gray[top:bottom, left:right]
    if not crop.size or right - left < len(text):
        return [(text, box)]
    _, ink = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    empty = np.count_nonzero(ink, axis=0) == 0
    boundaries = np.flatnonzero(np.diff(np.r_[False, empty, False]))
    gaps = [
        (int(a), int(b))
        for a, b in boundaries.reshape(-1, 2)
        if a > 0 and b < crop.shape[1] and b - a >= max(2, (y1 - y0) * 0.14)
    ]
    if len(gaps) < len(words) - 1:
        return [(text, box)]
    total = sum(character_width(c) for c in text)
    expected, offset = [], 0
    for word in words[:-1]:
        offset += sum(character_width(c) for c in word)
        expected.append((offset + 139) / max(total, 1) * crop.shape[1])
        offset += 278
    # Ordered matching avoids treating an internal letter gap as the next word.
    costs = np.full((len(expected) + 1, len(gaps) + 1), np.inf)
    costs[0, :] = 0
    choices = np.zeros_like(costs, dtype=bool)
    for i, target in enumerate(expected, 1):
        for j, (a, b) in enumerate(gaps, 1):
            use = costs[i - 1, j - 1] + (((a + b) / 2 - target) / max(y1 - y0, 1)) ** 2
            skip = costs[i, j - 1]
            choices[i, j] = use < skip
            costs[i, j] = min(use, skip)
    selected, i, j = [], len(expected), len(gaps)
    while i and j:
        if choices[i, j]:
            selected.append(gaps[j - 1])
            i -= 1
        j -= 1
    selected.reverse()
    if i or any(
        abs((a + b) / 2 - target) > max((y1 - y0) * 2, crop.shape[1] * 0.15)
        for (a, b), target in zip(selected, expected)
    ):
        return [(text, box)]
    starts = [0] + [b for a, b in selected]
    ends = [a for a, b in selected] + [crop.shape[1]]
    advances = [(b - a) / len(word) for word, a, b in zip(words, starts, ends)]
    if min(advances) <= 0 or max(advances) > 2.2 * min(advances):
        return [(text, box)]
    return [
        (word, (left + a, y0, left + b, y1)) for word, a, b in zip(words, starts, ends)
    ]


def selection_lines(result, gray):
    """Keep the OCR layout model's column/paragraph order, aligning words to ink."""
    upright = np.rot90(gray, -(result.rotation % 360) // 90)
    lines = []
    for line in result.lines:
        text = readable_text(line.text)
        runs = word_boxes(text, line.box, upright)
        if text:
            lines.append(SimpleNamespace(text=text, box=line.box, words=runs))
    return lines


def positioned_text(line, text_layer, box):
    """One PDF text object per line preserves order even on rotated pages.

    TJ offsets place word and letter origins on ink without relying on a PDF
    viewer to reconstruct spaces between separate, independently placed objects.
    """
    text = line.text.strip()
    words = getattr(line, "words", [(text, box)])
    if len(words) == 1:
        return (box[2] - box[0]) / len(text), f"<{text_layer.encode(text)}> Tj"
    advance = max(0.01, max((b[2] - b[0]) / len(word) for word, b in words))
    if any(len(word) > 1 and b[2] - b[0] < advance for word, b in words):
        return (box[2] - box[0]) / len(text), f"<{text_layer.encode(text)}> Tj"
    positions = []
    for index, (word, bounds) in enumerate(words):
        step = (bounds[2] - bounds[0] - advance) / max(1, len(word) - 1)
        positions.extend((char, bounds[0] + i * step) for i, char in enumerate(word))
        if index < len(words) - 1:
            positions.append((" ", bounds[2]))
    cursor, parts = box[0], []
    for char, x in positions:
        correction = (cursor - x) * 1000 / advance
        if abs(correction) > 0.01:
            parts.append(f"{correction:.4f}")
        parts.append(f"<{text_layer.encode(char)}>")
        cursor = x + advance
    return advance, "[" + " ".join(parts) + "] TJ"
