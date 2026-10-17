"""Detect and cut repetition loops in model transcriptions.

Vision-language models sometimes lose their place on a page and repeat a line or
a phrase until they hit the output limit. Every such page scores the maximum CER
of 1.0, however good the text before the loop was. These helpers find where a
loop starts and keep the text up to and including its first copy.

A loop is the same unit repeated back to back: a block of 1 to `max_period`
lines at least `min_repeats` times, or a run of words (line breaks ignored) at
least three times. Real pages repeat things too (ledger ditto lines, "do" in
survey notes), so the thresholds are set high enough that none of the 619
training answer keys is changed by `cut_loops`.
"""

from __future__ import annotations

import re

MIN_LINE_REPEATS = 4
MAX_PERIOD = 6


def _line_loop_start(lines: list[str], min_repeats: int, max_period: int) -> int | None:
    """Index of the line just after the first copy of the earliest line loop."""
    keys = [line.strip() for line in lines]
    best: int | None = None
    for period in range(1, max_period + 1):
        start = 0
        while start + period * min_repeats <= len(keys):
            unit = keys[start:start + period]
            if not any(unit):
                start += 1
                continue
            repeats = 1
            while keys[start + repeats * period:start + (repeats + 1) * period] == unit:
                repeats += 1
            if repeats >= min_repeats:
                cut = start + period
                best = cut if best is None else min(best, cut)
                break
            start += 1
    return best


_WORD = re.compile(r"\S+")
MIN_WORD_REPEATS = 3
MAX_BLOCK_WORDS = 120
MIN_BLOCK_CHARS = 20


def _word_loop_cut(text: str, min_repeats: int = MIN_WORD_REPEATS) -> int | None:
    """Character offset just after the first copy of the earliest word-level loop.

    Works on words, so a loop whose line breaks fall in different places on each
    repetition is still found.
    """
    matches = list(_WORD.finditer(text))
    words = [m.group() for m in matches]
    n = len(words)
    for start in range(n):
        for size in range(1, min(MAX_BLOCK_WORDS, (n - start) // min_repeats) + 1):
            block = words[start:start + size]
            if sum(len(w) + 1 for w in block) < MIN_BLOCK_CHARS:
                continue
            copies = 1
            while words[start + copies * size:start + (copies + 1) * size] == block:
                copies += 1
            if copies >= min_repeats:
                return matches[start + size - 1].end()
    return None


def cut_loops(text: str, min_repeats: int = MIN_LINE_REPEATS, max_period: int = MAX_PERIOD) -> tuple[str, bool]:
    """Return (text, changed). Keeps everything before the loop plus one copy of it.

    Both detectors run; the earlier cut wins.
    """
    cuts = []
    lines = text.split("\n")
    line_cut = _line_loop_start(lines, min_repeats, max_period)
    if line_cut is not None:
        cuts.append(len("\n".join(lines[:line_cut])))
    word_cut = _word_loop_cut(text)
    if word_cut is not None:
        cuts.append(word_cut)
    if not cuts:
        return text, False
    return text[:min(cuts)].rstrip(), True


def has_loop(text: str) -> bool:
    return cut_loops(text)[1]
