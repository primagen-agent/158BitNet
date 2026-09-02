"""ggw.py -- ctypes bridge to ggwshim.so (Phase-A trainer).

Loads the standalone C shim (python/ggwshim.c, compiled on the server) and
exposes GGUF weights as torch tensors. Layouts are the repo's own dequant
semantics reimplemented 1:1 in the shim (TQ2_0 66-byte blocks, Q6_K
210-byte blocks); this module only marshals buffers.

Weight layout note: GGUF tensor dims are [in, out] but storage is row-major
with dims[0] contiguous, so the flat fp32 buffer reshapes directly to
nn.Linear.weight layout [out, in] -- no transpose.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import numpy as np
import torch

_F32P = ctypes.POINTER(ctypes.c_float)
_I32P = ctypes.POINTER(ctypes.c_int)
_U64P = ctypes.POINTER(ctypes.c_uint64)
_F64P = ctypes.POINTER(ctypes.c_double)


@dataclass
class LayerWeights:
    q: torch.Tensor      # [q_dim, hidden] fp32 cpu
    k: torch.Tensor      # [kv_dim, hidden]
    v: torch.Tensor      # [kv_dim, hidden]
    o: torch.Tensor      # [hidden, q_dim]
    gate: torch.Tensor   # [ffn, hidden]
    up: torch.Tensor     # [ffn, hidden]
    down: torch.Tensor   # [hidden, ffn]


_GEOMETRY_KEYS = ("n_layers", "hidden", "kv_dim", "q_dim", "ffn", "vocab",
                  "rope_dim")


class GGUFWeights:
    def __init__(self, gguf_path: str, lib_path: str | None = None):
        self.gguf_path = gguf_path
        self.lib = self._load_lib(lib_path)
        self.model = self.lib.shim_open(gguf_path.encode())
        if not self.model:
            raise RuntimeError(f"shim_open failed for {gguf_path}")
        geo = (ctypes.c_int * 7)()
        if self.lib.shim_geometry(self.model, geo) != 0:
            raise RuntimeError("shim_geometry failed")
        self.geometry = dict(zip(_GEOMETRY_KEYS, geo))
        self.hidden = self.geometry["hidden"]
        self.n_layers = self.geometry["n_layers"]
        self.q_dim = self.geometry["q_dim"]
        self.kv_dim = self.geometry["kv_dim"]
        self.ffn = self.geometry["ffn"]
        self.vocab = self.geometry["vocab"]
        self.rope_dim = self.geometry["rope_dim"]
        # rms eps from metadata (default 1e-6 like the C runtime)
        self.rms_eps = 1e-6
        t, v64, f64 = ctypes.c_uint32(), ctypes.c_uint64(), ctypes.c_double()
        if self.lib.shim_meta(self.model, b"llama.attention.layer_norm_rms_epsilon",
                              ctypes.byref(t), ctypes.byref(v64),
                              ctypes.byref(f64)) == 0 and f64.value != 0.0:
            self.rms_eps = f64.value
        self.rope_freq_base = 10000.0
        if self.lib.shim_meta(self.model, b"llama.rope.freq_base",
                              ctypes.byref(t), ctypes.byref(v64),
                              ctypes.byref(f64)) == 0 and f64.value != 0.0:
            self.rope_freq_base = f64.value
        self.n_heads = 0
        if self.lib.shim_meta(self.model, b"llama.attention.head_count",
                              ctypes.byref(t), ctypes.byref(v64),
                              ctypes.byref(f64)) == 0:
            self.n_heads = int(v64.value)
        self.n_kv_heads = 0
        if self.lib.shim_meta(self.model, b"llama.attention.head_count_kv",
                              ctypes.byref(t), ctypes.byref(v64),
                              ctypes.byref(f64)) == 0:
            self.n_kv_heads = int(v64.value)
        if self.n_heads == 0 or self.n_kv_heads == 0:
            raise RuntimeError("missing head counts in GGUF metadata")
        self.head_dim = self.q_dim // self.n_heads
        self._layer_cache: dict[int, LayerWeights] = {}
        self._f32_cache: dict[str, np.ndarray] = {}

    # ---- public ----

    def get_layer(self, n: int) -> LayerWeights:
        lw = self._layer_cache.get(n)
        if lw is not None:
            return lw
        g = self.geometry
        shapes = {
            "q": (g["q_dim"], g["hidden"]),
            "k": (g["kv_dim"], g["hidden"]),
            "v": (g["kv_dim"], g["hidden"]),
            "o": (g["hidden"], g["q_dim"]),
            "gate": (g["ffn"], g["hidden"]),
            "up": (g["ffn"], g["hidden"]),
            "down": (g["hidden"], g["ffn"]),
        }
        tensors = {}
        for name in shapes:
            if name in ("q", "k", "v"):
                tname = f"blk.{n}.attn_{name}.weight"
            elif name == "o":
                tname = f"blk.{n}.attn_output.weight"
            else:
                tname = f"blk.{n}.ffn_{name}.weight"
            tensors[name] = torch.from_numpy(self._dequant(tname, shapes[name]))
        lw = LayerWeights(**tensors)
        self._layer_cache[n] = lw
        return lw

    def get_f32(self, name: str, shape) -> np.ndarray:
        key = f"{name}:{shape}"
        hit = self._f32_cache.get(key)
        if hit is not None:
            return hit
        arr = self._dequant(name, shape)
        self._f32_cache[key] = arr
        return arr

    def get_rope_factors(self, which: str = "short") -> np.ndarray | None:
        """rope_factors_{short,long}.weight fp32 [head_dim/2] or None."""
        try:
            return self.get_f32(f"rope_factors_{which}.weight",
                                (self.rope_dim // 2,))
        except RuntimeError:
            return None

    def close(self):
        m = getattr(self, "model", None)
        if m is not None:
            self.lib.shim_close(m)
            self.model = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---- internals ----

    def _dequant(self, name: str, shape) -> np.ndarray:
        n = 1
        for s in shape:
            n *= s
        buf = np.empty(n, dtype=np.float32)
        rc = self.lib.shim_dequant_tensor(
            self.model, name.encode(), buf.ctypes.data_as(_F32P), buf.size)
        if rc != 0:
            raise RuntimeError(f"shim_dequant_tensor({name!r}) failed rc={rc}")
        return buf.reshape(shape)

    @staticmethod
    def _load_lib(lib_path: str | None) -> ctypes.CDLL:
        candidates = []
        if lib_path:
            candidates.append(lib_path)
        env = os.environ.get("GGWSHIM_LIB")
        if env:
            candidates.append(env)
        here = os.path.dirname(os.path.abspath(__file__))
        candidates += [
            os.path.join(here, "..", "build", "libggwshim.so"),
            os.path.join(here, "libggwshim.so"),
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                lib = ctypes.CDLL(os.path.abspath(cand))
                return GGUFWeights._bind(lib)
        raise RuntimeError("libggwshim.so not found (tried: "
                           + ", ".join(candidates) + ")")

    @staticmethod
    def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
        lib.shim_open.argtypes = [ctypes.c_char_p]
        lib.shim_open.restype = ctypes.c_void_p
        lib.shim_close.argtypes = [ctypes.c_void_p]
        lib.shim_close.restype = None
        lib.shim_geometry.argtypes = [ctypes.c_void_p, _I32P]
        lib.shim_geometry.restype = ctypes.c_int
        lib.shim_meta.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                  ctypes.POINTER(ctypes.c_uint32), _U64P, _F64P]
        lib.shim_meta.restype = ctypes.c_int
        lib.shim_dequant_tensor.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                            _F32P, ctypes.c_size_t]
        lib.shim_dequant_tensor.restype = ctypes.c_int
        return lib
