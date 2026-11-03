from __future__ import annotations

import importlib.util
import os
import unittest
from unittest import mock

from finetune.config import ROOT, default_data_root, load_curricula


class DataRootTests(unittest.TestCase):
    def test_default_is_inside_the_repo(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BADGER_SCRIBE_DATA", None)
            self.assertEqual(default_data_root(), ROOT / "badger-scribe-data")

    def test_environment_variable_overrides_default(self):
        with mock.patch.dict(os.environ, {"BADGER_SCRIBE_DATA": "/tmp/elsewhere"}):
            self.assertEqual(str(default_data_root()), os.path.realpath("/tmp/elsewhere"))


@unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is installed by the fine-tuning extra")
class ConfigTests(unittest.TestCase):
    def test_matrix_contains_joint_and_reverse_controls(self):
        curricula = load_curricula()
        for name in ("kade-dominy-survey", "kade-then-dominy-then-survey", "survey-then-dominy-then-kade"):
            self.assertIn(name, curricula)
