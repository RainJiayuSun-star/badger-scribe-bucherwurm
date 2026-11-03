import tempfile
import unittest
from pathlib import Path
from unittest import mock
import http.client

from pipeline.config import GatewaySettings
from pipeline.gateway import GatewayBackend, IncompleteResponse, parse_stream
from pipeline.prompts import VERBATIM_V1


class GatewayStreamTests(unittest.TestCase):
    def test_collects_streamed_text_and_metadata(self):
        lines = [
            b"data: {\"model\": \"churro-3b\", \"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n",
            b"data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}], \"usage\": {\"total_tokens\": 7}}\n",
            b"data: [DONE]\n",
        ]
        text, usage, model, finish_reason = parse_stream(lines)
        self.assertEqual(text, "Hello world")
        self.assertEqual(usage, {"total_tokens": 7})
        self.assertEqual(model, "churro-3b")
        self.assertIsNone(finish_reason)

    def test_a_stream_that_just_stops_is_incomplete(self):
        with self.assertRaises(IncompleteResponse):
            parse_stream([b"data: {\"choices\": [{\"delta\": {\"content\": \"Partial\"}}]}\n"])

    def test_an_error_chunk_is_incomplete(self):
        with self.assertRaises(IncompleteResponse):
            parse_stream([b"data: {\"choices\": [{\"delta\": {\"content\": \"Partial\"}}]}\n",
                          b"data: {\"error\": \"model crashed\"}\n"])

    def test_a_finish_reason_counts_as_finished_and_is_reported(self):
        lines = [b"data: {\"choices\": [{\"delta\": {\"content\": \"Long\"}, \"finish_reason\": \"length\"}]}\n"]
        self.assertEqual(parse_stream(lines)[0], "Long")
        self.assertEqual(parse_stream(lines)[3], "length")


GOOD = [b"data: {\"choices\": [{\"delta\": {\"content\": \"Page\"}}]}\n", b"data: [DONE]\n"]
CUT_OFF = [b"data: {\"choices\": [{\"delta\": {\"content\": \"Pa\"}}]}\n"]
EMPTY = [b"data: [DONE]\n"]


class RetryTests(unittest.TestCase):
    def _transcribe(self, *responses, retries=2):
        settings = GatewaySettings("http://gateway", "churro", 100, 5.0, retries)
        replies = [mock.MagicMock(**{"__enter__.return_value": r}) if isinstance(r, list) else r for r in responses]
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image, \
                mock.patch("pipeline.gateway.urllib.request.urlopen", side_effect=replies) as urlopen, \
                mock.patch("pipeline.gateway.time.sleep"):
            return GatewayBackend(settings, "key", VERBATIM_V1).transcribe(Path(image.name)), urlopen.call_count

    def test_retries_cut_off_empty_and_dropped_replies(self):
        for failure in (CUT_OFF, EMPTY, http.client.RemoteDisconnected("gone"), ConnectionResetError("reset"),
                        http.client.IncompleteRead(b"")):
            with self.subTest(failure=failure):
                (text, metadata), calls = self._transcribe(failure, GOOD)
                self.assertEqual(text, "Page")
                self.assertEqual((calls, metadata["attempt"]), (2, 2))

    def test_gives_up_with_a_clear_error_after_the_retries(self):
        with self.assertRaisesRegex(RuntimeError, "gateway request failed"):
            self._transcribe(CUT_OFF, CUT_OFF, retries=1)
