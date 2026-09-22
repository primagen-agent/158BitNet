"""Fresh C decoder/projection, Python neural matching/copy. No bank fallback."""
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_continuous_memory import forward
from diagnose_native_generation_gradient import exact, native_forward
from episode_memory_inputs import encoder_texts
from native_continuous_generation import check_trace
from native_memory_encoder import NativeMemoryEncoder, BACKBONE_SHA256, sha_file
from native_prefix_bank import validate_prefix
from native_token_payload import native_token_input, input_binding
from neural_memory_generation import encode_generation_input
from token_memory_composition import ComposedTokenMemory, mix_content_copy


def check_extension(initial, previous, prediction, prefix, expected_sources, sources):
    validate_prefix(prefix)
    if sources != expected_sources or type(sources) is not tuple:
        raise ValueError('source changed or not bound to natural input')
    if prefix != (initial if previous is None else previous + (prediction,)):
        raise ValueError('prefix must start at natural prompt and extend actual prediction')


class NativeTokenBackend:
    def __init__(self, *, runtime, tokenizer, tok_probe, gguf, probe, reference_probe,
                 encoder_probe, model, head, scale, output, enabled=True):
        if type(model) is not ComposedTokenMemory or type(enabled) is not bool:
            raise ValueError('explicit composed research model and enable flag required')
        if model.reader.hidden != 1024 or model.reader.vocab != 73448 or model.roles.layers != 24:
            raise ValueError('bound 0.5B geometry required')
        if any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('FP32 CPU model required')
        if head.shape != (73448, 1024) or head.requires_grad or head.dtype != torch.float32 or head.device.type != 'cpu':
            raise ValueError('frozen matching head required for numerical reference')
        self.model, self.head, self.scale, self.enabled = model, head, scale, enabled
        self.gguf, self.probe, self.reference_probe = [str(Path(p).resolve()) for p in (gguf, probe, reference_probe)]
        self.root = Path(output); self.root.mkdir(parents=True, exist_ok=False)
        self.encoder = NativeMemoryEncoder(self.gguf, encoder_probe)
        if input_binding(self.encoder.identity, sha_file(tok_probe)) != model.reader.binding:
            raise ValueError('encoder/tokenizer/model binding mismatch')
        self.request = encode_generation_input(runtime, tokenizer)
        query, sources = encoder_texts(runtime)
        rows = self.encoder.encode([query, *sources], self.root / 'inputs')
        if len(sources) > 1: raise ValueError('single supplied event only')
        if tokenizer.encode(query, True) != rows[0].token_ids.tolist(): raise ValueError('query token identity mismatch')
        raw = b''.join(tokenizer.decode_pieces(rows[0].token_ids[1:].tolist()))
        if raw not in (query.encode(), b' ' + query.encode()): raise ValueError('query byte mismatch')
        inputs = native_token_input(rows[0], rows[1] if sources else None,
            runtime['episodes'][0] if sources else None, sources[0] if sources else None, tokenizer, model.reader.binding)
        with torch.no_grad(): self.state = model.prefill(inputs)
        self.identity = {'backbone_sha256': BACKBONE_SHA256, 'probe_sha256': sha_file(self.probe),
            'reference_probe_sha256': sha_file(self.reference_probe), 'encoder': self.encoder.identity}
        self.steps = []; self.previous = self.prediction = None

    def _project(self, prefix, delta, name, base, traces):
        value = forward(self.probe, self.gguf, prefix, delta.numpy()[0].copy(), self.root, name, True)
        traces.append(check_trace(self.root / (name + '.log'), len(prefix)))
        if not all(exact(value[k], base[k]) for k in ('base', 'hidden')):
            raise ValueError('native base changed during projection')
        return torch.from_numpy(value['logits'].copy())[None]

    @staticmethod
    def compare(actual, expected):
        if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4) or not torch.equal(actual.argmax(-1), expected.argmax(-1)):
            raise ValueError('native/Torch composed output mismatch')
        return float((actual - expected).abs().max())

    def __call__(self, prefix, sources):
        check_extension(self.request.prompt_token_ids, self.previous, self.prediction, prefix, self.request.source_texts, sources)
        self.state.validate(self.model)
        for name in ('probe', 'reference_probe'):
            if sha_file(getattr(self, name)) != self.identity[name + '_sha256']: raise ValueError('native binary changed')
        label = f'step-{len(self.steps):03d}'; zero = np.zeros(1024, dtype='<f4')
        reference = native_forward(self.reference_probe, self.gguf, prefix, zero, self.root, label + '-reference', False)
        base = forward(self.probe, self.gguf, prefix, zero, self.root, label + '-base', False)
        traces = [check_trace(self.root / (label + '-base.log'), len(prefix))]
        if not all(exact(base[k], reference[k]) for k in ('hidden', 'logits')): raise ValueError('ordinary C reference mismatch')
        with torch.no_grad():
            h = torch.from_numpy(base['hidden'].copy())[None]; b = torch.from_numpy(base['base'].copy())[None]
            route = int(self.state.state_logits.argmax()); qualification = None; mass = 0.
            # Explicit isolated qualification, never used as a serving route.
            if not self.steps and self.enabled:
                branches = self.model.branches(h, b, self.head, self.scale, self.state)
                content, uncertainty = self.model.roles.branches(h, self.state.prepared)
                cc = self._project(prefix, content, label + '-qual-content', base, traces)
                cu = self._project(prefix, uncertainty, label + '-qual-uncertainty', base, traces)
                cm = mix_content_copy(cc, branches.positions, branches.copy_mass, self.state.pointer.inputs.token_ids)
                qualification = {'supported_max_error': self.compare(cm, branches.supported),
                    'uncertainty_max_error': self.compare(cu, branches.uncertainty), 'top1_equal': True,
                    'scope': 'branch-only, not selected or oracle-routed generation'}
            if not self.enabled or route == 0:
                output = b
            elif route == 2:
                # No content/pointer calls on this actual decoding route.
                delta = self.model.roles.uncertainty(h)
                output = self._project(prefix, delta, label + '-output', base, traces)
            else:
                delta = self.model.roles.content.residual(self.model.roles.layers - 1, h, self.state.prepared)
                content = self._project(prefix, delta, label + '-output', base, traces)
                pointer = self.model.reader.proposal(h, b, self.state.pointer)
                output = mix_content_copy(content, pointer.position_probabilities, pointer.copy_mass, self.state.pointer.inputs.token_ids)
                mass = float(pointer.copy_mass[0, 0])
            expected = self.model.read(h, b, self.head, self.scale, self.state, enabled=self.enabled)
            error = self.compare(output, expected)
        logits = output.numpy()[0].copy(); prediction = int(logits.argmax())
        same = exact(logits, base['logits'])
        if (not self.enabled or route == 0) and not same: raise ValueError('disabled/normal not bit-exact')
        self.steps.append({'prefix_ids': list(prefix), 'predicted_token_id': prediction, 'selected_route': route,
            'effective_route': route if self.enabled else 'disabled', 'copy_mass': mass,
            'state_probabilities': self.state.state_logits.exp().tolist(), 'base_reference_checked': True,
            'output_equals_base': same, 'native_traces': traces, 'torch_composition_max_error': error,
            'branch_only_qualification': qualification})
        self.previous, self.prediction = prefix, prediction
        return logits

    def finish(self):
        self.state.validate(self.model)
        if sha_file(self.gguf) != BACKBONE_SHA256: raise ValueError('GGUF changed')
        for name in ('probe', 'reference_probe'):
            if sha_file(getattr(self, name)) != self.identity[name + '_sha256']: raise ValueError('native binary changed')
        report = {'identity': self.identity, 'enabled': self.enabled, 'steps': self.steps,
            'cross_step_kv_reuse': False, 'internal_prefill_kv_buffers': True,
            'prefix_bank_used': False, 'gold_answer_prefix_used': False, 'neural_fusion_implementation': 'python',
            'source_episode_selection': 'supplied_not_learned',
            'raw_sha256': {str(p.relative_to(self.root)): sha_file(p) for p in self.root.rglob('*') if p.is_file()}}
        with (self.root / 'backend.json').open('x') as f: json.dump(report, f, indent=2); f.write('\n')
        return report
