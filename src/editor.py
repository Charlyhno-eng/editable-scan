# SPDX-License-Identifier: MPL-2.0
"""Validated changes to existing source passages, exported from the original PDF."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FONTS = Path(__file__).parent / "data" / "fonts"


class EditError(ValueError):
    pass


class OriginalEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(
        pattern=r"^p[0-9]+-b[0-9]+-l[0-9]+-s[0-9]+(?:-paragraph)?$", max_length=80
    )
    page: int = Field(ge=1, le=200, strict=True)
    text: str = Field(max_length=4000)
    bold: bool = False
    color: str = Field(default="#000000", pattern=r"^#[0-9a-fA-F]{6}$")
    overflow: Literal["wrap", "shrink", "page"] = "wrap"
    font: Literal[
        "auto",
        "DejaVuSans",
        "DejaVuSerif",
        "DejaVuSansMono",
        "LiberationSans",
        "LiberationSerif",
        "LiberationMono",
    ] = "auto"

    @field_validator("text")
    @classmethod
    def plain_text(cls, value):
        value = unicodedata.normalize(
            "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
        )
        if any(
            unicodedata.category(char).startswith("C") and char != "\n"
            for char in value
        ):
            raise ValueError("Use plain text without control characters.")
        return value


class EditorDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edits: list[OriginalEdit] = Field(default_factory=list, max_length=600)


def export_pdf(source: bytes, draft: EditorDraft) -> bytes:
    """Start from the untouched OCR result, never stack earlier edits."""
    from src.original import edit_original

    return edit_original(source, draft.edits)
