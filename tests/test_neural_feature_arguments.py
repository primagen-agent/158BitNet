"""Reader CLI must not silently discard supplied C feature-cache paths."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from train_neural_dialogue_reader import reader_argument_parser


class ReaderFeatureArgumentTests(unittest.TestCase):
    def parse(self, extra):
        return reader_argument_parser().parse_args([
            "--gguf", "models/bitcpm4-1b-tq2_0.gguf",
            "--probe", "build/neural_slot_probe", "--output", "build/reader",
            "--features", "build/base/features.pt", *extra,
        ])

    def test_multiple_and_repeated_cache_options_keep_every_path_in_order(self):
        args = self.parse(["--extra-features", "build/replay/features.pt", "build/current/features.pt",
                           "--extra-features", "build/binding/features.pt"])
        self.assertEqual(args.features, "build/base/features.pt")
        self.assertEqual(args.extra_features, ["build/replay/features.pt", "build/current/features.pt",
                                              "build/binding/features.pt"])

    def test_omitted_option_is_empty_without_leaking_previous_parse(self):
        self.parse(["--extra-features", "build/replay/features.pt"])
        self.assertEqual(self.parse([]).extra_features, [])

    def test_empty_cache_option_fails_before_feature_extraction(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as result:
            self.parse(["--extra-features", "--steps", "120"])
        self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
