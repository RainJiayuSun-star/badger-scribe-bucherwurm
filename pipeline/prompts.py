"""Versioned transcription instructions used by every backend."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    """A named instruction whose content can be recorded with each run."""

    name: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


VERBATIM_V1 = Prompt(
    name="verbatim-v1",
    text=(
        "You are a careful archival transcription system. Transcribe the page "
        "verbatim in natural reading order. Preserve spelling, capitalization, "
        "punctuation, line breaks, abbreviations, and uncertain or unusual forms. "
        "Return only the transcription; do not explain, summarize, normalize, or "
        "add Markdown."
    ),
)


def prompt_from_file(path: str) -> Prompt:
    """Read an experiment prompt without adding it to command history."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt file is empty: {path}")
    return Prompt(name=file_path.stem, text=text)
