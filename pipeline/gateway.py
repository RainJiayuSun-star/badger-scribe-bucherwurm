"""Remote OpenAI-compatible Churro backend with streamed OCR responses."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from pipeline.config import GatewaySettings
from pipeline.prompts import Prompt


def encode_image(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_stream(lines: Iterable[bytes]) -> tuple[str, dict[str, Any] | None, str | None]:
    """Collect text, usage, and model metadata from OpenAI SSE response lines."""
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    response_model: str | None = None
    for raw_line in lines:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        event = line[5:].strip()
        if event == "[DONE]":
            break
        try:
            chunk = json.loads(event)
        except json.JSONDecodeError:
            continue
        response_model = chunk.get("model") or response_model
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str):
                pieces.append(content)
    return "".join(pieces), usage, response_model


class GatewayBackend:
    """Remote backend; a local Churro backend can match this API later."""

    def __init__(self, settings: GatewaySettings, api_key: str, prompt: Prompt):
        self.settings = settings
        self.api_key = api_key
        self.prompt = prompt

    def transcribe(self, image: Path) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.settings.model,
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
            "messages": [
                {"role": "system", "content": self.prompt.text},
                {"role": "user", "content": [
                    {"type": "text", "text": "Transcribe this page."},
                    {"type": "image_url", "image_url": {"url": encode_image(image)}},
                ]},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        retryable = {408, 409, 429, 500, 502, 503, 504}
        for attempt in range(self.settings.retries + 1):
            started = time.monotonic()
            request = urllib.request.Request(
                self.settings.chat_completions_url, data=body, method="POST",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout) as response:
                    text, usage, response_model = parse_stream(response)
                if not text:
                    raise RuntimeError("gateway response contained no text content")
                return text, {"elapsed_seconds": round(time.monotonic() - started, 3), "usage": usage, "response_model": response_model, "attempt": attempt + 1}
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                if exc.code not in retryable or attempt == self.settings.retries:
                    raise RuntimeError(f"gateway HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.settings.retries:
                    raise RuntimeError(f"gateway request failed: {exc}") from exc
            time.sleep(2 ** attempt)
        raise AssertionError("unreachable")
