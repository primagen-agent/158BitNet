#!/usr/bin/env python3
"""Export the frozen identity research model, not a production promotion."""
import argparse
import hashlib
import struct
import zlib
from pathlib import Path
import numpy as np
import torch
from train_resident_identity import IdentityResidentMemorySet

KEYS = ('raw_scale', 'identity_bias', 'baseline.read_pool',
        'baseline.project.weight', 'baseline.project.bias',
        'baseline.pair.0.weight', 'baseline.pair.0.bias',
        'baseline.score.weight', 'baseline.score.bias',
        'baseline.count.0.weight', 'baseline.count.0.bias',
        'baseline.count.2.weight', 'baseline.count.2.bias',
        'baseline.link_pair.0.weight', 'baseline.link_pair.0.bias',
        'baseline.link_score.weight', 'baseline.link_score.bias',
        'baseline.scope.0.weight', 'baseline.scope.0.bias',
        'baseline.scope.2.weight', 'baseline.scope.2.bias',
        'entity_role.0.weight', 'entity_role.0.bias',
        'entity_role.2.weight', 'entity_role.2.bias')

def export(checkpoint, output):
    item = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if item['format'] != 'RESIDENT_IDENTITY_RESEARCH_V1' or item['configuration']['architecture'] != 'identity':
        raise ValueError('only identity research checkpoints are supported')
    state = item['state_dict']; hidden = state['baseline.project.weight'].shape[1]
    model = IdentityResidentMemorySet(hidden)
    model.load_state_dict(state, strict=True)
    if set(state) != set(KEYS) or hidden != 1024:
        raise ValueError('unsupported geometry')
    buffers = [np.asarray(state[k].float().numpy(), dtype='<f4').tobytes() for k in KEYS]
    if not all(torch.isfinite(v).all() for v in state.values()): raise ValueError('nonfinite weights')
    with Path(output).open('xb') as f:
        f.write(b'BNRESID1' + struct.pack('<4I', 1, hidden, 128, len(KEYS)))
        f.write(bytes.fromhex(item['backbone_sha256']))
        f.write(hashlib.sha256(Path(checkpoint).read_bytes()).digest())
        for b in buffers: f.write(struct.pack('<2I', len(b), zlib.crc32(b)))
        for b in buffers: f.write(b)
    return hashlib.sha256(Path(output).read_bytes()).hexdigest()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint'); p.add_argument('output'); a = p.parse_args()
    print(export(a.checkpoint, a.output))
