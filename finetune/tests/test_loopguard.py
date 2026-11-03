from __future__ import annotations

import csv
import importlib.util
import unittest

from finetune.config import default_data_root
from finetune.loopguard import cut_loops, has_loop

HAS_TORCH = importlib.util.find_spec("torch") is not None
TRAIN_CSV = default_data_root() / "train.csv"


class CutLoopsTests(unittest.TestCase):
    def test_repeated_lines_are_cut_after_the_first_copy(self):
        text = "Lieber Sohn!\nWir sind gesund\n" + "Ed. u. H. zu Hause\nIhr\n" * 40
        cut, changed = cut_loops(text)
        self.assertTrue(changed)
        self.assertEqual(cut, "Lieber Sohn!\nWir sind gesund\nEd. u. H. zu Hause\nIhr")

    def test_loop_with_different_line_breaks_each_time_is_found(self):
        sentence = "Ich hab mich doch noch nie so sehr um euch gekümmert, wie jetzt."
        words = sentence.split()
        wrapped = []
        for copy in range(6):  # wrap every copy at a different width
            width = 3 + copy % 3
            wrapped += [" ".join(words[i:i + width]) for i in range(0, len(words), width)]
        text = "Habe die Karte heute erhalten.\n" + "\n".join(wrapped)
        cut, changed = cut_loops(text)
        self.assertTrue(changed)
        self.assertTrue(cut.startswith("Habe die Karte heute erhalten."))
        self.assertEqual(" ".join(cut.split()[5:]), sentence)

    def test_ordinary_repetition_is_left_alone(self):
        ledger = "\n".join(["1834 Jan 5", "To cleaning watch 0 50", "do do 0 50", "do do 0 50", "Feb 2", "do do 0 50"])
        survey = "\n".join(["Set post", "do 7 S 39 W 20", "do 7 S 39 W 20", "Birch 12 N 35 E"])
        for text in (ledger, survey, "short"):
            self.assertEqual(cut_loops(text), (text, False))

    def test_has_loop(self):
        self.assertTrue(has_loop("a b c d e f g h i j k\n" * 10))
        self.assertFalse(has_loop("ein ganz normaler Brief"))


@unittest.skipUnless(TRAIN_CSV.exists(), "needs the competition data")
class AnswerKeyTests(unittest.TestCase):
    def test_no_training_answer_key_is_changed(self):
        with TRAIN_CSV.open(newline="", encoding="utf-8-sig") as handle:
            changed = [row["page_id"] for row in csv.DictReader(handle) if has_loop(row["text"])]
        self.assertEqual(changed, [])


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "loop line here ok\n" * len(ids)


@unittest.skipUnless(HAS_TORCH, "torch is installed by the cuda extra")
class LoopStopperTests(unittest.TestCase):
    def test_stops_only_on_a_check_step_with_a_loop(self):
        import torch

        from finetune.vlm import LoopStopper

        stopper = LoopStopper(FakeTokenizer(), prompt_length=10, every=4)
        self.assertFalse(bool(stopper(torch.zeros((1, 13), dtype=torch.long), None)[0]))  # 3 new tokens: not a check step
        self.assertTrue(bool(stopper(torch.zeros((1, 14), dtype=torch.long), None)[0]))   # 4 new tokens, looping text


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_TORCH, "torch is installed by the cuda extra")
class RetryTests(unittest.TestCase):
    LOOP = "Lieber Sohn!\n" + "Ed. u. H. zu Hause\nIhr\n" * 20
    GOOD = "Lieber Sohn! Wir sind alle gesund und es geht uns gut."

    def run_pages(self, outputs, config):
        import torch
        from unittest import mock

        from finetune import vlm

        class Processor:
            tokenizer = None

            def apply_chat_template(self, *args, **kwargs):
                return ""

            def __call__(self, **kwargs):
                return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}

        calls = []

        def fake_decode(model, processor, inputs, max_new_tokens, guard, temperature):
            calls.append(temperature)
            return outputs.pop(0)

        with mock.patch.object(vlm, "_vision_inputs", return_value=(None, None)), \
                mock.patch.object(vlm, "_decode_once", side_effect=fake_decode):
            texts, infos = vlm.generate_texts(None, Processor(), [{"page_id": "p1", "image_path": "x.jpg"}], "cpu", 10, config)
        return texts[0], infos[0], calls

    def test_off_by_default_keeps_the_cut_greedy_output(self):
        text, info, calls = self.run_pages([self.LOOP], {})
        self.assertEqual(calls, [None])
        self.assertEqual(info, {"loop_cut": 1, "retry_used": 0})
        self.assertTrue(text.endswith("Ihr"))

    def test_first_clean_retry_is_used(self):
        text, info, calls = self.run_pages([self.LOOP, self.LOOP, self.GOOD], {"loop_retry_temperatures": [0.3, 0.6]})
        self.assertEqual(calls, [None, 0.3, 0.6])
        self.assertEqual(text, self.GOOD)
        self.assertEqual(info["retry_used"], 2)

    def test_near_empty_output_is_retried(self):
        text, info, _ = self.run_pages(["Lie", self.GOOD], {"loop_retry_temperatures": [0.3]})
        self.assertEqual((text, info["retry_used"]), (self.GOOD, 1))

    def test_clean_greedy_output_is_not_retried(self):
        _, info, calls = self.run_pages([self.GOOD], {"loop_retry_temperatures": [0.3]})
        self.assertEqual((calls, info["retry_used"]), ([None], 0))
