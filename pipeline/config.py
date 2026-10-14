"""Configuration shared by pipeline backends and command-line entry points."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_BASE_URL = "https://llm-gw01.doit.wisc.edu/v1"
DEFAULT_MODEL = "churro-3b"


@dataclass(frozen=True)
class GatewaySettings:
    """Connection and decoding settings for an OpenAI-compatible vision API."""

    base_url: str
    model: str
    max_tokens: int
    timeout: float
    retries: int

    @property
    def chat_completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"
