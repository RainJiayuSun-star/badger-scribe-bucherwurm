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


class CheckpointTests(unittest.TestCase):
    def test_prune_keeps_the_newest_by_step_not_by_name(self):
        import tempfile
        from pathlib import Path

        from finetune.train import latest_checkpoint, prune_checkpoints

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (9, 22, 100):
                (root / f"checkpoint-{step}").mkdir()
            (root / "best_adapter").mkdir()
            removed = prune_checkpoints(root, keep=2)
            self.assertEqual([p.name for p in removed], ["checkpoint-9"])
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["best_adapter", "checkpoint-100", "checkpoint-22"])
            self.assertEqual(latest_checkpoint(root).name, "checkpoint-100")
            self.assertEqual(prune_checkpoints(root, keep=0), [])

    def test_checkpoint_root_is_separate_from_the_run_directory(self):
        from pathlib import Path

        from finetune.train import checkpoint_dir

        run_dir = Path("/home/runs/abc")
        self.assertEqual(checkpoint_dir({"run_id": "abc"}, run_dir), run_dir)
        self.assertEqual(checkpoint_dir({"run_id": "abc", "checkpoint_root": "/tmp/ck"}, run_dir), Path("/tmp/ck/abc"))


if __name__ == "__main__":
    unittest.main()
