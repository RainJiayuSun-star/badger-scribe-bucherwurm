from __future__ import annotations

import importlib.util
import unittest

from finetune.vlm import max_pixels_for, messages

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ImageBudgetTests(unittest.TestCase):
    CONFIG = {"max_pixels": 1000, "max_pixels_by_category": {"survey_notes": 4000}}

    def test_category_override_and_default(self):
        self.assertEqual(max_pixels_for(self.CONFIG, "survey_notes"), 4000)
        self.assertEqual(max_pixels_for(self.CONFIG, "kade_letters"), 1000)

    def test_cap_is_placed_on_the_image_item(self):
        image = messages("page.jpg", "text", max_pixels=1234)[0]["content"][0]
        self.assertEqual(image, {"type": "image", "image": "page.jpg", "max_pixels": 1234})

    def test_no_cap_leaves_the_item_unchanged(self):
        self.assertNotIn("max_pixels", messages("page.jpg")[0]["content"][0])

    def test_target_goes_in_the_assistant_turn(self):
        self.assertEqual(messages("page.jpg", "hello")[1]["content"][0]["text"], "hello")


class FakeTokenizer:
    vocab = {1: "as", 2: " it", 3: " ####", 4: " put", 5: "#"}

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


@unittest.skipUnless(HAS_TORCH, "torch is installed by the cuda extra")
class MaskIllegibleTests(unittest.TestCase):
    def test_masks_only_target_tokens_containing_hash(self):
        import torch

        from finetune.vlm import mask_illegible

        labels = torch.tensor([[-100, 1, 2, 3, 4, 5]])
        self.assertEqual(mask_illegible(labels, FakeTokenizer()), 2)
        self.assertEqual(labels.tolist(), [[-100, 1, 2, -100, 4, -100]])


@unittest.skipUnless(HAS_TORCH, "torch is installed by the cuda extra")
class LearningRateGroupTests(unittest.TestCase):
    def test_vision_adapters_get_their_own_rate(self):
        import torch

        from finetune.train import trainable_groups

        model = torch.nn.Module()
        model.visual = torch.nn.Module()
        model.visual.block = torch.nn.Linear(2, 2)
        model.visual.merger = torch.nn.Linear(2, 2)
        model.language = torch.nn.Linear(2, 2)
        groups = trainable_groups(model, 1e-4, 2e-5)
        self.assertEqual([g["lr"] for g in groups], [1e-4, 2e-5])
        self.assertEqual(len(groups[1]["params"]), 2)   # visual.block weight and bias
        self.assertEqual(len(groups[0]["params"]), 4)   # merger and language


if __name__ == "__main__":
    unittest.main()
