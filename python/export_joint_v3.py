"""EC-005 deployment: append the trained joint-pair relevance head to the
deployed .bnmodel as version 3 (base payload preserved byte-exactly).

Gate policy (registered in reviews/EC-005/PROCESS.md):
- holdout >= 0.80 and fit >= 0.90: export.
- fit >= 0.95 and holdout in [0.75, 0.80): the registered fallback band —
  export ONLY with --fallback (ranking + tau-refusal deployment).
- otherwise: refuse.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_bnmodel import append_tensors

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / 'build/neural-memory-ec005/ckpt-best.pt'
BASE = ROOT / 'build/neural-c-weights/model.bnmodel'
OUT = ROOT / 'build/neural-c-weights/model_v3.bnmodel'


def main():
    fallback = '--fallback' in sys.argv
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else CKPT
    base = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT

    ck = torch.load(ckpt, weights_only=False)
    fit, hold = ck.get('fit_top1', 0.0), ck.get('holdout_top1', 0.0)
    if ck.get('gate_pass'):
        mode = 'gate pass'
    elif fit >= 0.95 and 0.75 <= hold < 0.80 and fallback:
        mode = 'REGISTERED FALLBACK (ranking + tau-refusal)'
    else:
        print(f'refusing export: fit={fit:.3f} holdout={hold:.3f} '
              f'(gate 0.80; fallback band needs fit>=0.95, holdout>=0.75, --fallback)',
              file=sys.stderr)
        return 1

    sd = ck['model']
    tensors = {
        'ep_joint.pair.weight': sd['pair.weight'].detach().numpy(),
        'ep_joint.pair.bias': sd['pair.bias'].detach().numpy(),
        'ep_joint.mix.weight': sd['mix.weight'].detach().numpy(),
        'ep_joint.mix.bias': sd['mix.bias'].detach().numpy(),
        'ep_joint.out.weight': sd['out.weight'].detach().numpy(),
        'ep_joint.out.bias': sd['out.bias'].detach().numpy(),
    }
    index = append_tensors(base, tensors, out)
    print(f'[{mode}] exported {out} ({out.stat().st_size} bytes, '
          f'{len(index)} tensors)')
    print(f'fit={fit:.3f} holdout={hold:.3f} (step {ck.get("step", -1)})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
