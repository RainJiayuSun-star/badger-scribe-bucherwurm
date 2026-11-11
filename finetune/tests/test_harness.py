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


class FakeProcessor:
    """Full chat = 10 tokens, prompt alone = 6, so tokens 6-9 are the target."""

    class tokenizer:
        pad_token_id = 0

    def apply_chat_template(self, item, tokenize=False, add_generation_prompt=False):
        return "prompt" if add_generation_prompt else "full"

    def __call__(self, text, **kwargs):
        import torch

        n = 6 if text[0] == "prompt" else 10
        return {"input_ids": torch.arange(1, n + 1).unsqueeze(0), "attention_mask": torch.ones((1, n), dtype=torch.long)}


class FakeLightOnProcessor(FakeProcessor):
    def apply_chat_template(self, item, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        if enable_thinking:
            raise AssertionError("LightOn must disable thinking")
        return super().apply_chat_template(item, tokenize, add_generation_prompt)


@unittest.skipUnless(HAS_TORCH, "torch is installed by the cuda extra")
class BaseModelOptionTests(unittest.TestCase):
    """image_longest_edge, prompt, prompt_first and target_logits_only (olmOCR); unset, nothing changes."""

    OLMOCR = {"image_longest_edge": 1288, "prompt": "P", "prompt_first": True, "target_logits_only": True,
              "max_pixels": 1000, "max_pixels_by_category": {"survey_notes": 4000}, "mask_illegible": False}

    def collate(self, config, page):
        from unittest import mock

        from finetune import vlm

        seen = []
        with mock.patch.object(vlm, "_vision_inputs", side_effect=lambda batch, patch, family: seen.append(batch) or (None, None)):
            batch = vlm.collate_train([{"page_id": "p", "image_path": str(page), "text": "t", "category": "dominy_accounts"}],
                                      FakeProcessor(), "cpu", 0, None, config)
        return batch, seen[0][0][0]["content"]

    def collate_lighton(self, config, page):
        from finetune import vlm

        return vlm.collate_train([{"page_id": "p", "image_path": str(page), "text": "t", "category": "dominy_accounts"}],
                                 FakeLightOnProcessor(), "cpu", 0, None, config)

    def page(self, size=(100, 200)):
        import tempfile
        from pathlib import Path

        from PIL import Image

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "page.jpg"
        Image.new("RGB", size, "white").save(path)
        return path

    def test_submission_configs_load_a_pinned_public_model(self):
        """Verification reruns these on other hardware: no machine-local path, and the exact weights we used."""
        from pathlib import Path

        from finetune.config import load_config

        for name in ("churro-3b-bf16.yaml", "churro-3b-bf16-english.yaml", "olmocr2-7b-bf16-english.yaml",
                     "lighton3-4b-yale-english.yaml"):
            config = load_config(Path(__file__).parents[1] / "configs" / name)
            self.assertRegex(config["model_id"], r"^[\w.-]+/[\w.-]+$", name)
            self.assertRegex(str(config.get("revision")), r"^[0-9a-f]{40}$", name)

    def test_committed_configs_use_none_of_these_keys(self):
        from pathlib import Path

        from finetune.config import load_config

        configs = [path for path in (Path(__file__).parents[1] / "configs").glob("*.yaml") if path.name != "curricula.yaml"]
        self.assertTrue(configs)
        for path in configs:
            if path.name.startswith(("olmocr", "lighton")):
                continue
            config = load_config(path)
            self.assertFalse({"image_longest_edge", "prompt", "prompt_first", "target_logits_only"} & set(config), path.name)

    def test_unset_keeps_the_churro_prompt_image_and_labels(self):
        from finetune.vlm import PROMPT, page_image

        config = {"max_pixels": 1000, "max_pixels_by_category": {"survey_notes": 4000}, "mask_illegible": False}
        self.assertEqual(page_image(config, "page.jpg", "dominy_accounts"), ("page.jpg", 1000))
        self.assertEqual(messages("page.jpg", None, 1000, config), messages("page.jpg", None, 1000))
        self.assertEqual(messages("page.jpg")[0]["content"][1], {"type": "text", "text": PROMPT})
        page = self.page()
        batch, content = self.collate(config, page)
        self.assertEqual([part["type"] for part in content], ["image", "text"])
        self.assertEqual((content[0]["image"].size, content[0]["max_pixels"], content[1]["text"]), ((100, 200), 1000, PROMPT))
        self.assertNotIn("logits_to_keep", batch)
        self.assertEqual(batch["labels"].tolist(), [[-100] * 6 + [7, 8, 9, 10]])

    def test_olmocr_prompt_first_page_resized_and_logits_on_targets(self):
        page = self.page()
        batch, content = self.collate(self.OLMOCR, page)
        self.assertEqual([part["type"] for part in content], ["text", "image"])
        self.assertEqual(content[0]["text"], "P")
        self.assertEqual(content[1]["image"].size, (644, 1288))     # upscaled to the longest edge
        self.assertNotIn("max_pixels", content[1])
        self.assertEqual(batch["logits_to_keep"], 5)                # last prompt token plus 4 targets
        self.assertEqual(batch["labels"].tolist(), [[-100, 7, 8, 9, 10]])

    def test_longest_edge_shrinks_and_skips_categories_with_their_own_budget(self):
        from PIL import Image

        from finetune.vlm import page_image

        image, cap = page_image(self.OLMOCR, Image.new("RGB", (3000, 2000)), "dominy_accounts")
        self.assertEqual((image.size, cap), ((1288, 859), None))
        survey = Image.new("RGB", (3000, 2000))
        self.assertEqual(page_image(self.OLMOCR, survey, "survey_notes"), (survey, 4000))

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "needs transformers")
    def test_target_logits_give_the_same_loss(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                             num_attention_heads=2, num_key_value_heads=1))
        ids = torch.randint(0, 32, (1, 12))
        labels = ids.clone()
        labels[:, :7] = -100
        keep = 12 - 7 + 1
        full = model(input_ids=ids, labels=labels).loss
        short = model(input_ids=ids, labels=labels[:, -keep:], logits_to_keep=keep).loss
        torch.testing.assert_close(short, full)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "needs transformers")
    def test_lighton_collate_target_logits_give_the_same_loss(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = {"model_family": "lighton", "prompt": "P", "target_logits_only": True,
                  "mask_illegible": False, "max_seq_length": 8192}
        batch = self.collate_lighton(config, self.page())
        self.assertEqual(batch["labels"].tolist(), [[-100, 7, 8, 9, 10]])
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                             num_attention_heads=2, num_key_value_heads=1))
        full_labels = batch["input_ids"].clone()
        full_labels[:, :6] = -100
        full = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=full_labels).loss
        short = model(**batch).loss
        torch.testing.assert_close(short, full)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "needs transformers")
    def test_loss_options_match_the_model_loss_and_drop_the_worst_tokens(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from finetune.train import training_loss

        torch.manual_seed(0)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                             num_attention_heads=2, num_key_value_heads=1))
        ids = torch.randint(0, 32, (1, 12))
        labels = ids.clone()
        labels[:, :3] = -100
        labels[:, 8] = -100  # a masked `#` token inside the target
        batch = {"input_ids": ids, "labels": labels[:, -10:], "logits_to_keep": 10}
        full = model(**batch).loss
        # Truncating a vanishing fraction keeps every token, so the custom path must equal the model's own loss.
        torch.testing.assert_close(training_loss(model, batch, {"loss_truncation": 1e-9}), full)
        logits = model(input_ids=ids, logits_to_keep=10).logits[0, :-1]
        targets = labels[0, -9:]
        per_token = torch.nn.functional.cross_entropy(logits[targets != -100], targets[targets != -100], reduction="none")
        self.assertEqual(per_token.numel(), 8)
        torch.testing.assert_close(training_loss(model, batch, {"loss_truncation": 0.5}), per_token.sort().values[:4].mean())
        smoothed = torch.nn.functional.cross_entropy(logits[targets != -100], targets[targets != -100], label_smoothing=0.1)
        torch.testing.assert_close(training_loss(model, batch, {"label_smoothing": 0.1}), smoothed)


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
