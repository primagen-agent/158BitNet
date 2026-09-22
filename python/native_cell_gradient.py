"""DG-002 local scale derivative conditioned on native, unchanged int8 codes.

Not an STE, not a derivative across code changes, and not an approved trainer.
"""
from pathlib import Path
import struct

import numpy as np
import torch
from torch.nn import functional as F

from native_generation_bridge import native_value_with_surrogate_gradient as anchor

FLOAT_STAGES = {11: ("ffn_norm", 1024), 12: ("gate", 4096), 13: ("up", 4096),
                14: ("activation", 4096), 15: ("down", 1024), 16: ("residual", 1024)}
QUANT_STAGES = {18: ("ffn_input", 1024), 20: ("ffn_down_input", 4096), 21: ("output_input", 1024)}


def read_cell(path):
    result = {"quantizers": {}}
    with Path(path).open("rb") as stream:
        if stream.read(8) != b"BNGC0001": raise ValueError("invalid cell trace format")
        seen = set()
        while header := stream.read(12):
            if len(header) != 12: raise ValueError("truncated cell header")
            stage, kind, size = struct.unpack("<III", header)
            if stage in seen: raise ValueError("duplicate cell record")
            seen.add(stage)
            if stage in FLOAT_STAGES and kind == 1:
                name, expected = FLOAT_STAGES[stage]
                if size != expected: raise ValueError("float cell geometry mismatch")
                data = stream.read(4 * size)
                if len(data) != size * 4: raise ValueError("truncated float cell")
                value = np.frombuffer(data, dtype="<f4").copy()
                if not np.isfinite(value).all(): raise ValueError("nonfinite cell")
                result[name] = value
            elif stage in QUANT_STAGES and kind == 2:
                name, expected = QUANT_STAGES[stage]
                if size != expected: raise ValueError("quantized cell geometry mismatch")
                data = stream.read(4 + size)
                if len(data) != 4 + size: raise ValueError("truncated quantized cell")
                scale = struct.unpack_from("<f", data)[0]
                if not np.isfinite(scale) or scale <= 0: raise ValueError("invalid native scale")
                codes = np.frombuffer(data, dtype=np.int8, offset=4).copy()
                if (codes == -128).any(): raise ValueError("invalid symmetric int8 code")
                result["quantizers"][name] = {"scale": scale, "codes": codes}
            else: raise ValueError("unknown cell stage")
    if seen != FLOAT_STAGES.keys() | QUANT_STAGES.keys(): raise ValueError("incomplete cell trace")
    return result


def cell_changes(base, changed, base_output, changed_output):
    inputs = {"ffn_input": "ffn_norm", "ffn_down_input": "activation"}
    report = {}
    for name in QUANT_STAGES.values():
        key = name[0]
        a = base[inputs[key]] if key in inputs else base_output
        b = changed[inputs[key]] if key in inputs else changed_output
        ia, ib = int(np.abs(a).argmax()), int(np.abs(b).argmax())
        ties = int((np.abs(a) == abs(a[ia])).sum()) != 1 or int((np.abs(b) == abs(b[ib])).sum()) != 1
        report[key] = {"changed_codes": int((base["quantizers"][key]["codes"] != changed["quantizers"][key]["codes"]).sum()),
                       "max_index_changed": ia != ib, "max_sign_changed": bool(np.sign(a[ia]) != np.sign(b[ib])), "ambiguous_max": ties}
    stable = all(v["changed_codes"] == 0 and not v["max_index_changed"] and not v["max_sign_changed"] and not v["ambiguous_max"] for v in report.values())
    return {"same_cell": stable, "quantizers": report}


def cell_local_suffix(backbone, after_attention, native, cell):
    """Real-arithmetic local derivative evaluated at native intermediates.

    With int8 codes fixed, each linear output varies only with its input scale.
    Native tensor anchoring prevents intermediate numerical drift at this point.
    """
    cfg, layer = backbone.cfg, backbone.layers[-1]
    if any(w.requires_grad for w in tuple(layer.values()) + (backbone.out_norm, backbone.out_proj)):
        raise ValueError("backbone must be frozen")
    tensor = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float64).copy()).reshape(1, -1)
    rms64 = lambda value, weight: value * torch.rsqrt(value.square().mean(-1, keepdim=True) + cfg.rms_eps) * weight.double()
    x = anchor(tensor(native["after"]), after_attention.double())
    norm = anchor(tensor(cell["ffn_norm"]), rms64(x, layer["ffn_norm"]))
    def scale(value, key):
        # Use FP64 here to compare the local mathematical derivative; native
        # scale values themselves remain the recorded FP32 values.
        actual = torch.tensor(cell["quantizers"][key]["scale"], dtype=torch.float64)
        return anchor(actual, value.abs().amax() / 127.)
    s = scale(norm, "ffn_input") / cell["quantizers"]["ffn_input"]["scale"]
    gate, up = tensor(cell["gate"]) * s, tensor(cell["up"]) * s
    activation = anchor(tensor(cell["activation"]), F.silu(gate) * up)
    s = scale(activation, "ffn_down_input") / cell["quantizers"]["ffn_down_input"]["scale"]
    residual = anchor(tensor(cell["residual"]), x + tensor(cell["down"]) * s * cfg.residual_scale)
    hidden = anchor(tensor(native["hidden"]), rms64(residual, backbone.out_norm))
    s = scale(hidden, "output_input") / cell["quantizers"]["output_input"]["scale"]
    proxy = (tensor(native["logits"]) * s).float()[0]
    return anchor(torch.from_numpy(native["logits"].copy()), proxy)
