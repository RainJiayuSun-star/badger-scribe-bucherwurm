from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is installed by the project environment")
class AugmentationTests(unittest.TestCase):
    def test_augmentation_preserves_page_extent_and_is_seeded(self):
        from PIL import Image
        from finetune.augmentation import OCRAugmenter

        profile = {"enabled": True, "probability": 1.0, "noise_probability": 0.0, "rotation_probability": 1.0}
        image = Image.new("RGB", (120, 80), "white")
        augmenter = OCRAugmenter(profile, seed=4)
        first = augmenter.apply(image, "page", 0)
        second = augmenter.apply(image, "page", 0)
        self.assertEqual(first.size, image.size)
        self.assertEqual(first.tobytes(), second.tobytes())
