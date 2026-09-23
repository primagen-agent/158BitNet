"""Exact metadata parsing guards; not model answer evaluation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from diagnose_native_prompt import GGUF_TEMPLATE, declared_template


class PromptTests(unittest.TestCase):
    def test_literal_multiline_template_matches_without_truncation(self):
        metadata = "Metadata: tokenizer.chat_template type=string value=" + GGUF_TEMPLATE + "\nTensor: output\n"
        self.assertEqual(declared_template(metadata), GGUF_TEMPLATE)

    def test_unexpected_and_duplicate_templates_fail_closed(self):
        prefix = "Metadata: tokenizer.chat_template type=string value="
        for metadata in ("", prefix + "different\n", (prefix + GGUF_TEMPLATE + "\n") * 2):
            with self.assertRaises(ValueError): declared_template(metadata)


if __name__ == "__main__": unittest.main()
