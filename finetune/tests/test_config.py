from __future__ import annotations

import importlib.util
import unittest

from finetune.config import load_curricula


@unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is installed by the fine-tuning extra")
class ConfigTests(unittest.TestCase):
    def test_matrix_contains_joint_and_reverse_controls(self):
        curricula = load_curricula()
        for name in ("kade-dominy-survey", "kade-then-dominy-then-survey", "survey-then-dominy-then-kade"):
            self.assertIn(name, curricula)
