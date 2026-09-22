"""DG-014: read-only attribution on certified actual C trajectories, not accuracy."""
import argparse
import json
from pathlib import Path
import struct

import torch

from audit_joint_generation import continuous_file
from c_tokenizer import CTokenizer
from episode_memory_inputs import encoder_texts
from eval_joint_baseline import fixed_panel
from joint_checkpoint_transport import restore_certified_initial
from joint_memory_model import create_joint_model
from joint_optimizer import FEATURE_DIGEST, load_checkpoint, tensor_digest
from native_memory_encoder import BACKBONE_SHA256, read_encoded, sha_file
from native_token_payload import input_binding, native_token_input
from neural_memory_generation import encode_generation_input
from prepare_joint_binding_curriculum import sentence
from prepare_neural_memory_protocol import digest
from token_memory_composition import mix_content_copy


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'training/memory/neural-system'


def annotate_source(message, framed, pieces, allowed, meta):
    """POST-FORWARD only. Ambiguous token boundaries are never forced to a fact."""
    facts = meta['source_facts']
    if message is None:
        if facts or pieces or allowed: raise ValueError('empty source mismatch')
        return [], []
    if meta['scenario'] not in ('correct', 'role_swap', 'value_swap', 'ordinary_source'):
        raise ValueError('outside fixed diagnostic panel')
    clauses = [sentence(f['subject'], f['relation'], f['value'], meta['language']) for f in facts]
    text = (' ' if meta['language'] == 'en' else '').join(clauses)
    if text != message['text']: raise ValueError('metadata does not reconstruct source')
    raw = framed.encode(); decoded = b''.join(pieces)
    pad = 0 if decoded == raw else 1 if decoded == b' ' + raw else None
    if pad is None or len(pieces) != len(allowed): raise ValueError('source byte alignment failed')
    literal = json.dumps(text, ensure_ascii=False).encode()
    if literal[1:-1] != text.encode() or not raw.endswith(literal + b'}'):
        raise ValueError('escaped or noncanonical source')
    start = len(raw) - len(literal) + pad
    spans = []; offset = start
    for i, (clause, fact) in enumerate(zip(clauses, facts)):
        value = fact['value'].encode(); clause = clause.encode()
        if clause.count(value) != 1: raise ValueError('ambiguous value annotation')
        value_start = offset + clause.index(value)
        spans.append({'fact_index': i, 'fact': fact, 'start': offset, 'end': offset + len(clause),
                      'value_start': value_start, 'value_end': value_start + len(value),
                      'target': fact['subject'] == meta['query_subject'] and fact['relation'] == meta['query_relation']})
        offset += len(clause) + (meta['language'] == 'en')
    rows = []; offset = 0
    for i, (piece, can_copy) in enumerate(zip(pieces, allowed)):
        end = offset + len(piece)
        # Ignore only boundary whitespace, never punctuation or partial letters.
        left = offset + len(piece) - len(piece.lstrip())
        right = end - (len(piece) - len(piece.rstrip()))
        containing = [s for s in spans if left < right and left >= s['start'] and right <= s['end']]
        values = [s for s in spans if left < right and left >= s['value_start'] and right <= s['value_end']]
        overlaps_value = any(left < s['value_end'] and right > s['value_start'] for s in spans)
        row = {'position': i, 'byte_start': offset, 'byte_end': end, 'piece_hex': piece.hex(),
               'piece': piece.decode('utf-8', errors='replace'), 'copy_allowed': bool(can_copy),
               'fact_index': containing[0]['fact_index'] if len(containing) == 1 else None}
        row['category'] = ('structural' if not can_copy else
                           ('target_value' if values[0]['target'] else 'other_value') if len(values) == 1 else
                           'ambiguous_value_boundary' if overlaps_value else
                           'fact_nonvalue' if containing else 'other_payload')
        rows.append(row); offset = end
    return spans, rows


def mass_by_category(positions, annotations):
    if len(positions) != len(annotations) + 1: raise ValueError('position alignment mismatch')
    out = {'null': float(positions[0])}
    for p, row in zip(positions[1:], annotations):
        key = row['category']; out[key] = out.get(key, 0.) + float(p)
    if abs(sum(out.values()) - 1.) > 2e-5: raise ValueError('probability partition failed')
    return out


def qualify_raw(directory, backend, runtime):
    if backend['cross_step_kv_reuse'] or backend['prefix_bank_used'] or backend['gold_answer_prefix_used']:
        raise ValueError('invalid generation policy')
    for name, expected in backend['raw_sha256'].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or sha_file(path) != expected:
            raise ValueError('raw artifact changed')
    q, sources = encoder_texts(runtime); texts = [q, *sources]
    wire = struct.pack('<I', len(texts)) + b''.join(struct.pack('<I', len(t.encode())) + t.encode() for t in texts)
    if (directory / 'inputs/batch-0.input').read_bytes() != wire: raise ValueError('source changed')
    return q, sources, read_encoded(directory / 'inputs/batch-0.bin', len(texts))


def run(a):
    registration = RESEARCH / 'experiments/DG-014.json'
    reg = json.loads(registration.read_text())
    for name, expected in reg['inputs'].items():
        if sha_file(ROOT / name) != expected: raise ValueError('registered input changed: ' + name)
    features = ROOT / 'build/neural-memory-cg003-full-features'
    manifest = json.loads((features / 'manifest.json').read_text())
    gguf = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'; probe = ROOT / 'build/tok_probe'
    if digest(manifest) != FEATURE_DIGEST or sha_file(gguf) != BACKBONE_SHA256 or sha_file(probe) != manifest['tokenizer_sha256']:
        raise ValueError('feature/backbone/tokenizer identity changed')
    identity = input_binding(manifest['encoder_identity'], manifest['tokenizer_sha256'])
    records, panel = fixed_panel(RESEARCH / 'data/JB-001', json.loads((RESEARCH / 'checks/CG-003-data.json').read_text()))
    metadata = {r['id']: r for r in map(json.loads, (RESEARCH / 'data/JB-001/dev.index.jsonl').read_text().splitlines())}
    torch.set_num_threads(4); tokenizer = CTokenizer(str(probe), str(gguf)); cases = []
    try:
        with torch.no_grad():
            for arm in ('joint_aux', 'joint_product'):
                short = arm.removeprefix('joint_'); status_path = ROOT / 'build/neural-memory-cg003-training-results/training' / arm / 'status.json'
                status = json.loads(status_path.read_text()); binding = status['binding']; certs = {c['step']: c for c in status['checkpoints']}
                if identity != binding['encoder_binding']: raise ValueError('encoder binding changed')
                for name, expected in binding['source_sha256'].items():
                    if sha_file(ROOT / 'python' / name) != expected: raise ValueError('trained source changed')
                model = create_joint_model(identity, manifest['blocked_ids'], arm).eval()
                restore_certified_initial(status_path.parent / certs[0]['file'], expected_sha256=certs[0]['sha256'], model=model,
                                          binding=binding, initial_digest=certs[0]['state_digest'])
                load_checkpoint(status_path.parent / certs[50]['file'], expected_sha256=certs[50]['sha256'], model=model,
                                expected_binding=binding, expected_initial_digest=certs[0]['state_digest'])
                before = tensor_digest(model.state_dict())
                for experiment in ('CG-003', 'DG-013'):
                    retained = RESEARCH / 'reviews' / experiment
                    report = json.loads((retained / f'{short}-generation.json').read_text())
                    audit = json.loads((retained / f'{short}-native-audit.json').read_text())
                    if (not audit['passed'] or audit['generation_sha256'] != sha_file(retained / f'{short}-generation.json') or
                            report['status_sha256'] != sha_file(status_path) or report['checkpoint_sha256'] != certs[50]['sha256']):
                        raise ValueError('uncertified generation')
                    root = ROOT / 'build' / (f'neural-memory-cg003-{arm}-generation' if experiment == 'CG-003' else f'neural-memory-dg013-{arm}')
                    if sha_file(root / 'generation.json') != sha_file(retained / f'{short}-generation.json'): raise ValueError('report mismatch')
                    for j, pred in enumerate(report['predictions']):
                        i = j if experiment == 'CG-003' else pred['original_case_index']
                        runtime = records[i]; directory = root / f'case-{i:02d}'
                        backend = json.loads((directory / 'backend.json').read_text())
                        if json.loads((directory / 'generation.json').read_text()) != pred or pred['input_sha256'] != digest(runtime): raise ValueError('case changed')
                        q, sources, encoded = qualify_raw(directory, backend, runtime)
                        if tokenizer.encode(q, True) != encoded[0].token_ids.tolist(): raise ValueError('query changed')
                        memory = native_token_input(encoded[0], encoded[1] if sources else None, runtime['episodes'][0] if sources else None,
                                                    sources[0] if sources else None, tokenizer, identity)
                        state = model.prefill(memory); route = int(state.core.state_logits.argmax())
                        if route != pred['selected_route'] or state.core.state_logits.exp().tolist() != pred['initial_state_probabilities']:
                            raise ValueError('route replay changed')
                        # Complete neural work BEFORE accessing case annotations.
                        observations = []; prefix = encode_generation_input(runtime, tokenizer).prompt_token_ids
                        if route == 1:
                            for k, token in enumerate(pred['generated_token_ids']):
                                step = backend['steps'][k]
                                if step['prefix_ids'] != list(prefix) or step['predicted_token_id'] != token: raise ValueError('not actual prefix')
                                h, base, correction, logits = continuous_file(directory, f'step-{k:03d}-base', prefix, False)
                                if torch.count_nonzero(correction) or not torch.equal(base, logits): raise ValueError('invalid base')
                                p = model.reader.proposal(h, base, state.core.pointer)
                                if float(p.copy_mass[0, 0]) != step['copy_mass']: raise ValueError('copy mass changed')
                                branch = base if experiment == 'DG-013' else continuous_file(directory, f'step-{k:03d}-output', prefix, True,
                                            model.roles.content.residual(model.roles.layers-1, h, state.core.prepared))[3]
                                mixed = mix_content_copy(branch, p.position_probabilities, p.copy_mass, memory.token_ids)
                                if int(mixed.argmax()) != token: raise ValueError('selected token changed')
                                pos = p.position_probabilities[0]; w = p.copy_mass.double()[0, 0]
                                cp = pos[1:].double()[memory.token_ids == token].sum() / pos[1:].double().sum()
                                observations.append({'position': k, 'token_id': token, 'base_top1': int(base.argmax()),
                                    'branch_top1': int(branch.argmax()), 'copy_mass': float(w), 'positions': pos.tolist(),
                                    'selected_copy_contribution': float(w * cp),
                                    'selected_branch_contribution': float((1-w) * branch.double().softmax(-1)[0, token])})
                                prefix = prefix + (token,)
                        meta = metadata[runtime['id']]
                        pieces = tokenizer.decode_pieces(memory.token_ids.tolist()) if sources else []
                        spans, annotations = annotate_source(runtime['episodes'][0] if sources else None, sources[0] if sources else None,
                                                             pieces, memory.copy_allowed.tolist(), meta)
                        for row in observations:
                            row['mass_by_category'] = mass_by_category(row['positions'], annotations)
                            top = max(range(len(row['positions'])), key=row['positions'].__getitem__)
                            row['top_source_position'] = top-1 if top else None
                            row['selected_source_positions'] = [n for n, t in enumerate(memory.token_ids.tolist()) if t == row['token_id'] and annotations[n]['copy_allowed']]
                            row['piece'] = tokenizer.decode_pieces([row['token_id']])[0].decode('utf-8', errors='replace')
                        slot_masses = []
                        for attention in state.core.pointer.match_attention.tolist():
                            slot_masses.append({'categories': mass_by_category(attention, annotations),
                                'facts': [sum(attention[n+1] for n, r in enumerate(annotations) if r['fact_index'] == s['fact_index']) for s in spans]})
                        target_values = [s['fact']['value'] for s in spans if s['target']]
                        row = {'experiment': experiment, 'arm': arm, 'case_index': i, 'id': runtime['id'], 'scenario': meta['scenario'],
                               'language': meta['language'], 'query_relation': meta['query_relation'], 'route': route,
                               'state_probabilities': pred['initial_state_probabilities'], 'presence_probabilities': state.core.factor_logits.sigmoid().tolist(),
                               'slot_attention': slot_masses, 'source_spans': spans, 'source_tokens': annotations,
                               'source_token_ids': memory.token_ids.tolist(), 'target_values': target_values,
                               'target_value_byte_inclusion': [v in pred['text'] for v in target_values],
                               'reply': pred['text'], 'truncated': pred['truncated'], 'steps': observations}
                        cases.append(row)
                    print(json.dumps({'arm': arm, 'experiment': experiment, 'cases_complete': len(cases)}), flush=True)
                if tensor_digest(model.state_dict()) != before: raise ValueError('weights changed')
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()
    if len(cases) != 64 or sum(bool(c['steps']) for c in cases) != 32: raise ValueError('incomplete registered inventory')
    output = {'format': 'dg014-value-path-diagnostic-v1', 'registration_sha256': sha_file(registration),
              'source_sha256': sha_file(__file__), 'panel': panel, 'optimizer_steps': 0, 'parameters_unchanged': True,
              'new_generation': False, 'semantic_accuracy_measured': False, 'cases': cases,
              'supported_trajectories': 32, 'replayed_supported_positions': sum(len(c['steps']) for c in cases)}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.output).open('x') as f: json.dump(output, f, ensure_ascii=False, indent=2); f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    run(parser.parse_args())
