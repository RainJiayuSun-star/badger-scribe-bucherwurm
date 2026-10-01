import unittest

from pipeline.gateway import parse_stream


class GatewayStreamTests(unittest.TestCase):
    def test_collects_streamed_text_and_metadata(self):
        lines = [
            b"data: {\"model\": \"churro-3b\", \"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n",
            b"data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}], \"usage\": {\"total_tokens\": 7}}\n",
            b"data: [DONE]\n",
        ]
        text, usage, model = parse_stream(lines)
        self.assertEqual(text, "Hello world")
        self.assertEqual(usage, {"total_tokens": 7})
        self.assertEqual(model, "churro-3b")
