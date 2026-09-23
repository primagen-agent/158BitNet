"""Native backend guards; actual model certification is DG-004, not these fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from native_continuous_generation import NativeContinuousBackend, check_trace, tokenizer_stop_ids
from native_memory_encoder import sha_file


class BackendTests(unittest.TestCase):
    def test_trace_requires_exact_fresh_prefill_and_dispatch(self):
        trace = {"initial_position": 0, "final_position": 7, "eval_calls": 1, "prefix_tokens": 7}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log"
            valid = "[bitnet] cpu tier: arm_neon\nBNC_TRACE " + json.dumps(trace) + "\n"
            path.write_text(valid)
            self.assertEqual(check_trace(path, 7), trace)
            for text in (valid.replace('"initial_position": 0', '"initial_position": 1'),
                         valid.replace('"eval_calls": 1', '"eval_calls": 2'),
                         valid.replace("arm_neon", "scalar"), valid + valid, ""):
                path.write_text(text)
                with self.assertRaises(ValueError): check_trace(path, 7)

    def test_tokenizer_stop_is_single_exact_token_not_text_guess(self):
        class Tokenizer:
            def encode(self, *_): return [5]
            def eos(self): return 4
            def decode_pieces(self, *_): return [b"<|im_end|>"]
        self.assertEqual(tokenizer_stop_ids(Tokenizer()), (4, 5))
        tokenizer = Tokenizer(); tokenizer.encode = lambda *_: [5, 6]
        with self.assertRaises(ValueError): tokenizer_stop_ids(tokenizer)
        tokenizer = Tokenizer(); tokenizer.decode_pieces = lambda *_: [b"wrong"]
        with self.assertRaises(ValueError): tokenizer_stop_ids(tokenizer)

    def test_sentencepiece_dummy_space_is_not_a_stop_token(self):
        class Tokenizer:
            def encode(self, *_): return [3, 5]
            def eos(self): return 5
            def decode_pieces(self, *_): return [b" ", b"<|im_end|>"]
        self.assertEqual(tokenizer_stop_ids(Tokenizer()), (5,))
        tokenizer = Tokenizer(); tokenizer.decode_pieces = lambda *_: [b"word", b"<|im_end|>"]
        with self.assertRaises(ValueError): tokenizer_stop_ids(tokenizer)

    def test_label_or_batch_prefix_fails_before_model_access(self):
        backend = object.__new__(NativeContinuousBackend)
        for prefix in ((1, "gold"), [1, 2], (1,), (1, -1), (1, 73448), (True, 2)):
            with self.assertRaises(ValueError): backend(prefix, ())

    def test_teacher_forced_continuation_is_rejected(self):
        backend = object.__new__(NativeContinuousBackend)
        backend.previous_prefix, backend.previous_prediction = (1, 2), 3
        with self.assertRaisesRegex(ValueError, "actual prediction"): backend((1, 2, 4), ())

    def test_source_swap_and_multiple_events_are_rejected(self):
        backend = object.__new__(NativeContinuousBackend)
        backend.bound_sources = ("source-a",)
        backend._bind_sources(("source-a",))
        for sources in (("source-b",), ("a", "b"), ["a"], ("",)):
            with self.assertRaises(ValueError): backend._bind_sources(sources)

    def test_trained_controller_accepts_structurally_empty_memory(self):
        backend=object.__new__(NativeContinuousBackend)
        backend.bound_sources=None; backend.prepared=None; backend.condition='availability_trained'
        backend._bind_sources(())
        self.assertEqual(backend.bound_sources,())
        self.assertIsNone(backend.prepared)

    def test_trained_zero_residual_is_recorded_without_relaxing_base_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); binary=root/'probe'; binary.write_bytes(b'fixture')
            backend=object.__new__(NativeContinuousBackend)
            backend.probe=backend.reference_probe=str(binary); backend.gguf='not-read'
            backend.root=root; backend.identity={'probe_sha256':sha_file(binary),'reference_probe_sha256':sha_file(binary)}
            backend.condition='availability_trained'; backend.verify_reference=True
            backend.bound_sources=(); backend.prepared=None
            backend.previous_prefix=backend.previous_prediction=None; backend.steps=[]
            backend.fusion=SimpleNamespace(inspect=lambda *_:SimpleNamespace(residual=torch.zeros(1,1024),state_probabilities=torch.tensor([[1.,0.,0.]])))
            base={'hidden':np.zeros(1024,dtype=np.float32),'base':np.zeros(73448,dtype=np.float32),'logits':np.zeros(73448,dtype=np.float32)}
            with patch('native_continuous_generation.native_forward',return_value=base), patch('native_continuous_generation.forward',return_value=base), patch('native_continuous_generation.check_trace',return_value={}):
                backend((1,2),())
            self.assertFalse(backend.steps[0]['neural_residual_nonzero'])
            self.assertTrue(backend.steps[0]['base_reference_checked'])
            self.assertTrue(backend.steps[0]['output_equals_base'])
            wrong={**base,'hidden':np.ones(1024,dtype=np.float32)}
            with patch('native_continuous_generation.native_forward',return_value=wrong), patch('native_continuous_generation.forward',return_value=base), patch('native_continuous_generation.check_trace',return_value={}):
                with self.assertRaisesRegex(ValueError,'ordinary native C'): backend((1,2,0),())


if __name__ == "__main__": unittest.main()
