"""Conservative, transcript-preserving scan augmentation."""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def load_profile(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install the fine-tuning extra: uv sync --extra finetune") from exc
    with path.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if not isinstance(profile, dict):
        raise ValueError(f"{path} must contain a mapping")
    return profile


class OCRAugmenter:
    """Keeps page geometry and content intact; never crops or flips text."""

    def __init__(self, profile: dict[str, Any], seed: int):
        self.profile = profile
        self.seed = seed
        self.enabled = bool(profile.get("enabled", True))
        self.probability = float(profile.get("probability", 0.8))

    def apply(self, image: Image.Image, sample_key: str, epoch: int) -> Image.Image:
        if not self.enabled:
            return image.convert("RGB")
        rng = random.Random(f"{self.seed}:{epoch}:{sample_key}")
        if rng.random() > self.probability:
            return image.convert("RGB")
        result = image.convert("RGB")
        if rng.random() < float(self.profile.get("rotation_probability", 0.4)):
            degrees = float(self.profile.get("rotation_degrees", 1.5))
            result = result.rotate(rng.uniform(-degrees, degrees), resample=Image.Resampling.BICUBIC, fillcolor="white")
        if rng.random() < float(self.profile.get("contrast_probability", 0.5)):
            jitter = float(self.profile.get("contrast_jitter", 0.12))
            result = ImageEnhance.Contrast(result).enhance(rng.uniform(1 - jitter, 1 + jitter))
        if rng.random() < float(self.profile.get("brightness_probability", 0.5)):
            jitter = float(self.profile.get("brightness_jitter", 0.10))
            result = ImageEnhance.Brightness(result).enhance(rng.uniform(1 - jitter, 1 + jitter))
        if rng.random() < float(self.profile.get("blur_probability", 0.20)):
            result = result.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, float(self.profile.get("blur_radius", 0.7)))))
        if rng.random() < float(self.profile.get("jpeg_probability", 0.25)):
            quality = rng.randint(int(self.profile.get("jpeg_min_quality", 72)), int(self.profile.get("jpeg_max_quality", 92)))
            buffer = io.BytesIO()
            result.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            result = Image.open(buffer).convert("RGB")
        if rng.random() < float(self.profile.get("noise_probability", 0.18)):
            # Mild monochrome noise emulates scan grain without changing layout.
            import numpy as np

            gray = ImageOps.grayscale(result)
            amount = int(self.profile.get("noise_amount", 5))
            noise_rng = np.random.default_rng(int(rng.random() * (2**32 - 1)))
            values = np.asarray(gray, dtype=np.int16)
            mask = noise_rng.random(values.shape) < 0.035
            noise = noise_rng.integers(-amount, amount + 1, size=values.shape)
            values = np.clip(values + (noise * mask), 0, 255).astype(np.uint8)
            result = Image.fromarray(values, mode="L").convert("RGB")
        return result
