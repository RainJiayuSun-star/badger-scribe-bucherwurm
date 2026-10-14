VERBATIM_PROMPT = """Transcribe all text on this archival page image faithfully.

Rules:
- Verbatim: keep original spelling, casing, punctuation, and abbreviations. Do not modernize, summarize, expand abbreviations, or "correct" the writer.
- Reading order: natural reading order of the page (top to bottom, left to right unless the page is clearly otherwise).
- Output only the transcription. No preamble, no markdown fences, no commentary.
"""


def build_prompt(meta: dict | None, with_metadata: bool) -> str:
    text = VERBATIM_PROMPT.strip()
    if with_metadata and meta:
        bits = []
        for key in ("title", "creator", "date", "language", "summary", "collection"):
            val = (meta.get(key) or "").strip()
            if val:
                bits.append(f"{key}: {val}")
        if bits:
            text += (
                "\n\nCatalog metadata (context only; if the page clearly differs, "
                "transcribe the page):\n" + "\n".join(bits)
            )
    return text
