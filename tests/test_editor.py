"""Reject malformed source edits before they reach PDF operations."""

import pytest
from pydantic import ValidationError

from src.editor import EditorDraft, OriginalEdit


@pytest.mark.parametrize(
    "changes",
    [
        {"page": 0},
        {"page": 1.5},
        {"id": "arbitrary"},
        {"color": "red"},
        {"text": "x\x00y"},
        {"text": "a" * 4001},
        {"x": 30},
        {"overflow": "clip"},
        {"font": "../../some-file"},
        {"font": "UnknownFont"},
    ],
)
def test_invalid_edits(changes):
    with pytest.raises(ValidationError):
        OriginalEdit.model_validate(
            {"id": "p1-b0-l0-s0", "page": 1, "text": "Hello"} | changes
        )


def test_plain_text_normalization():
    item = OriginalEdit(id="p1-b0-l0-s0", page=1, text="cafe\u0301")
    assert item.text == "café"


def test_cannot_add_arbitrary_text_boxes():
    with pytest.raises(ValidationError):
        EditorDraft.model_validate({"additions": []})
