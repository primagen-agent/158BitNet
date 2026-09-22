"""Research backend: fresh native C prefix and native continuous projection.

The fusion module is still Python. Each C call is a new process/context. Encoding
one supplied source episode is not a trained event selector or automatic writer.
"""
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_continuous_memory import forward
from diagnose_native_generation_gradient import exact, native_forward
from native_memory_encoder import NativeMemoryEncoder, BACKBONE_SHA256, sha_file, text_key


def check_trace(log_path, prefix_length):
    lines = Path(log_path).read_text().splitlines()
    traces = [json.loads(line.removeprefix("BNC_TRACE ")) for line in lines if line.startswith("BNC_TRACE ")]
    expected = {"initial_position": 0, "final_position": prefix_length, "eval_calls": 1, "prefix_tokens": prefix_length}
    if traces != [expected] or "[bitnet] cpu tier: arm_neon" not in lines:
        raise ValueError("native fresh-context trace or CPU dispatch mismatch")
    return expected


def tokenizer_stop_ids(tokenizer):
    end = tokenizer.encode("<|im_end|>", False)
    pieces = tokenizer.decode_pieces(end)
    # The model's SentencePiece dummy-prefix policy may prepend one space even
    # when add_bos=False. That space is not a stop token. Do not strip arbitrary
    # tokens or assume a hardcoded vocabulary ID.
    if pieces == [b" ", b"<|im_end|>"] and len(end) == 2:
        end = end[1:]
    elif pieces != [b"<|im_end|>"] or len(end) != 1:
        raise ValueError("ChatML end marker must be one exact C token after optional dummy space")
    eos = tokenizer.eos()
    if type(eos) is not int or not 0 <= eos < 73448:
        raise ValueError("invalid native EOS")
    return tuple(sorted({eos, end[0]}))


class NativeContinuousBackend:
    def __init__(self, *, gguf, probe, reference_probe, encoder_probe, fusion, output,
                 condition, verify_reference=True):
        if condition not in ("reference", "disabled", "empty", "zero_gain", "active_untrained", "availability_empty", "availability_active", "availability_trained"):
            raise ValueError("unknown diagnostic condition")
        if condition in ("availability_empty", "availability_active", "availability_trained"):
            from memory_availability import AvailabilityMemoryFusion
            if not isinstance(fusion, AvailabilityMemoryFusion):
                raise ValueError("availability condition requires the neural availability module")
        if (fusion.hidden, fusion.layers) != (1024, 24) or any(p.device.type != "cpu" or p.dtype != torch.float32
                                                            for p in fusion.parameters()):
            raise ValueError("native diagnostic requires FP32 CPU 0.5B fusion")
        self.gguf, self.probe, self.reference_probe = (str(Path(p).resolve()) for p in (gguf, probe, reference_probe))
        self.encoder = NativeMemoryEncoder(self.gguf, encoder_probe)
        self.root = Path(output); self.root.mkdir(parents=True, exist_ok=False)
        self.fusion = fusion.eval()
        self.condition, self.verify_reference = condition, verify_reference
        self.bound_sources, self.prepared = None, None
        self.steps, self.previous_prefix, self.previous_prediction = [], None, None
        self.identity = {"backbone_sha256": BACKBONE_SHA256, "probe_sha256": sha_file(self.probe),
                         "reference_probe_sha256": sha_file(self.reference_probe), "encoder": self.encoder.identity}

    def _bind_sources(self, sources):
        if type(sources) is not tuple or len(sources) > 1 or any(type(s) is not str or not s for s in sources):
            raise ValueError("diagnostic supports zero or one immutable supplied episode")
        if self.bound_sources is not None:
            if sources != self.bound_sources: raise ValueError("memory changed during generation")
            return
        self.bound_sources = sources
        if self.condition in ("empty", "availability_empty") and sources:
            raise ValueError("empty control must receive an empty state")
        if self.condition in ("zero_gain", "active_untrained", "availability_active", "availability_trained"):
            if not sources and self.condition != "availability_trained": raise ValueError("active control requires a supplied episode")
            if not sources: return
            encoded = self.encoder.encode(list(sources), self.root / "sources")
            with torch.no_grad():
                self.prepared = self.fusion.prepare(torch.from_numpy(encoded[0].features.copy()))

    def __call__(self, prefix, sources):
        if (type(prefix) is not tuple or not 2 <= len(prefix) <= 128 or
                any(type(t) is not int or not 0 <= t < 73448 for t in prefix)):
            raise ValueError("complete immutable native prefix required")
        if self.previous_prefix is not None and prefix != self.previous_prefix + (self.previous_prediction,):
            raise ValueError("generation prefix is not the preceding actual prediction")
        self._bind_sources(sources)
        # Detect changed binaries before another process can silently change domains.
        if sha_file(self.probe) != self.identity["probe_sha256"] or sha_file(self.reference_probe) != self.identity["reference_probe_sha256"]:
            raise ValueError("native executable changed during generation")
        name = f"step-{len(self.steps):03d}"
        zero = np.zeros(1024, dtype="<f4")
        reference = None
        if self.verify_reference or self.condition == "reference":
            reference = native_forward(self.reference_probe, self.gguf, prefix, zero, self.root, name + "-reference", False)
        traces = []
        availability_probabilities = None
        if self.condition == "reference":
            logits = reference["logits"]
            delta_nonzero, reference_exact = False, True
        else:
            base = forward(self.probe, self.gguf, prefix, zero, self.root, name + "-base", False)
            traces.append(check_trace(self.root / (name + "-base.log"), len(prefix)))
            if reference is not None and not all(exact(base[k], reference[k]) for k in ("hidden", "logits")):
                raise ValueError("continuous baseline differs from ordinary native C")
            delta = zero
            if self.prepared is not None or self.condition in ("availability_empty", "availability_trained"):
                with torch.no_grad():
                    hidden = torch.from_numpy(base["hidden"].copy())[None, :]
                    if self.condition in ("availability_empty", "availability_active", "availability_trained"):
                        control = self.fusion.inspect(23, hidden, self.prepared)
                        delta = control.residual[0].numpy().copy()
                        availability_probabilities = control.state_probabilities[0].tolist()
                    else:
                        delta = self.fusion.residual(23, hidden, self.prepared)[0].numpy().copy()
                if self.condition == "zero_gain" and np.any(delta != 0):
                    raise ValueError("zero-gain control is not zero")
            # Disabled gets a deliberately nonzero vector to exercise the C switch.
            actual_delta = np.ones(1024, dtype="<f4") if self.condition == "disabled" else delta
            value = forward(self.probe, self.gguf, prefix, actual_delta, self.root, name + "-output",
                            self.condition != "disabled")
            traces.append(check_trace(self.root / (name + "-output.log"), len(prefix)))
            if not exact(value["base"], base["base"]) or not exact(value["hidden"], base["hidden"]):
                raise ValueError("fresh prefix recomputation changed base features")
            logits = value["logits"]
            delta_nonzero = bool(np.any(delta != 0))
            reference_exact = exact(logits, base["logits"])
            active_conditions = ("active_untrained", "availability_empty", "availability_active")
            if self.condition not in active_conditions + ("availability_trained",) and not reference_exact:
                raise ValueError("disabled/empty/zero control changed native output")
            if self.condition in active_conditions and (not delta_nonzero or reference_exact):
                raise ValueError("active diagnostic did not apply a nonzero memory residual")
            # A trained controller may legitimately emit zero; record the actual
            # effect without turning a no-memory decision into a harness failure.
        prediction = int(logits.argmax())
        self.steps.append({"prefix_ids": list(prefix), "predicted_token_id": prediction, "condition": self.condition,
                           "source_text_sha256": [text_key(s) for s in sources], "native_traces": traces,
                           "base_reference_checked": reference is not None, "output_equals_base": reference_exact,
                           "neural_residual_nonzero": delta_nonzero,
                           "availability_probabilities": availability_probabilities})
        self.previous_prefix, self.previous_prediction = prefix, prediction
        return logits

    def finish(self):
        # Recheck identity after the final call; a mid-run model change invalidates the run.
        if sha_file(self.gguf) != self.identity["backbone_sha256"]:
            raise ValueError("backbone changed during generation")
        if sha_file(self.probe) != self.identity["probe_sha256"] or sha_file(self.reference_probe) != self.identity["reference_probe_sha256"]:
            raise ValueError("native executable changed during generation")
        report = {"identity": self.identity, "condition": self.condition, "steps": self.steps,
                  "cross_step_kv_reuse": False, "internal_prefill_kv_buffers": True,
                  "neural_fusion_implementation": "python", "source_episode_selection": "supplied_not_learned",
                  "raw_sha256": {str(p.relative_to(self.root)): sha_file(p) for p in self.root.rglob("*") if p.is_file()}}
        with (self.root / "backend.json").open("x") as stream: json.dump(report, stream, indent=2); stream.write("\n")
        return report
