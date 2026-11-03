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


class HumanRepeatTests(unittest.TestCase):
    def test_human_pages_repeat_and_silver_pages_do_not(self):
        from finetune.train import stage_rows

        train = [
            {"page_id": "a", "category": "survey_notes", "label_source": "human"},
            {"page_id": "b", "category": "survey_notes", "label_source": "silver_claude"},
            {"page_id": "c", "category": "kade_letters", "label_source": "human"},
        ]
        curriculum = {"stages": [["survey_notes"]]}
        stages = stage_rows({"train": train}, curriculum, human_repeats=3)
        self.assertEqual([row["page_id"] for row in stages[0]], ["a", "a", "a", "b"])
        self.assertEqual([row["page_id"] for row in stage_rows({"train": train}, curriculum)[0]], ["a", "b"])

    def test_each_repeat_gets_its_own_augmentation_key(self):
        from finetune.train import stage_rows
        from finetune.vlm import augmentation_key

        train = [{"page_id": "a", "category": "survey_notes", "label_source": "human"},
                 {"page_id": "b", "category": "survey_notes", "label_source": "silver_claude"}]
        rows = stage_rows({"train": train}, {"stages": [["survey_notes"]]}, human_repeats=3)[0]
        self.assertEqual([augmentation_key(row) for row in rows], ["a", "a#1", "a#2", "b"])
        self.assertEqual(train[0], {"page_id": "a", "category": "survey_notes", "label_source": "human"})


class HumanCerTests(unittest.TestCase):
    def test_averages_human_pages_within_then_across_categories(self):
        from finetune.train import human_cer

        pages = [{"category": "a", "label_source": "human", "cer": 0.1},
                 {"category": "a", "label_source": "human", "cer": 0.3},
                 {"category": "b", "label_source": "human", "cer": 0.4},
                 {"category": "b", "label_source": "silver_claude", "cer": 0.9}]
        cer, count = human_cer(pages)
        self.assertAlmostEqual(cer, 0.3)
        self.assertEqual(count, 3)

    def test_no_human_pages_is_none_not_zero(self):
        from finetune.train import human_cer

        self.assertEqual(human_cer([{"category": "a", "label_source": "silver_claude", "cer": 0.9}]), (None, 0))


class AugmentationTests(unittest.TestCase):
    PROFILE = {"probability": 1, "rotation_probability": 0, "contrast_probability": 0, "brightness_probability": 0,
               "blur_probability": 0, "jpeg_probability": 0, "noise_probability": 1, "noise_amount": 5}

    def _page(self):
        from PIL import Image

        return Image.new("RGB", (64, 64), (200, 40, 40))

    def test_noise_keeps_the_page_in_color(self):
        import numpy as np

        from finetune.augmentation import OCRAugmenter

        before = np.asarray(self._page())
        after = np.asarray(OCRAugmenter(self.PROFILE, 1).apply(self._page(), "page", 0))
        self.assertFalse((after == before).all(), "some grain was added")
        self.assertTrue((after[..., 0] > after[..., 1] + 100).all(), "red stays red")

    def test_different_keys_get_different_noise(self):
        from finetune.augmentation import OCRAugmenter

        augmenter = OCRAugmenter(self.PROFILE, 1)
        self.assertEqual(augmenter.apply(self._page(), "page", 0).tobytes(), augmenter.apply(self._page(), "page", 0).tobytes())
        self.assertNotEqual(augmenter.apply(self._page(), "page", 0).tobytes(), augmenter.apply(self._page(), "page#1", 0).tobytes())


if __name__ == "__main__":
    unittest.main()
