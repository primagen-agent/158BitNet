"""EC-001 deployment: append the trained episode-relevance head to the
deployed .bnmodel as version 3 (base payload preserved byte-exactly).

Usage: python3 python/export_ep_rel_v3.py [ckpt] [base] [out]
Gate check: refuses to export when the registered gate (holdout top-1 >= 0.80)
failed — a failed head must not reach the C server.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_bnmodel import append_tensors

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / 'build/neural-memory-ec001/ckpt-2000.pt'
BASE = ROOT / 'build/neural-c-weights/model.bnmodel'
OUT = ROOT / 'build/neural-c-weights/model_v3.bnmodel'


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else CKPT
    base = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT

    ck = torch.load(ckpt, weights_only=False)
    if not ck.get('gate_pass', False):
        print(f'GATE FAILED (holdout top1={ck.get("dev_top1", float("nan")):.3f} < 0.80); '
              'refusing to export a failed head', file=sys.stderr)
        return 1
    sd = ck['model']
    tensors = {
        'ep_rel.0.weight': sd['0.weight'].detach().numpy(),
        'ep_rel.0.bias': sd['0.bias'].detach().numpy(),
        'ep_rel.2.weight': sd['2.weight'].detach().numpy(),
        'ep_rel.2.bias': sd['2.bias'].detach().numpy(),
    }
    index = append_tensors(base, tensors, out)
    print(f'exported {out} ({out.stat().st_size} bytes, '
          f'{len(index)} tensors)')
    for name, (shape, offset) in index.items():
        if name.startswith('ep_rel'):
            print(f'  {name}: shape={shape} offset={offset}')
    print(f'holdout top1 = {ck["dev_top1"]:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
