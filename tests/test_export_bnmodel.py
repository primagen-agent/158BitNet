"""export_bnmodel layout contract, shared byte-for-byte with src/neural_memory.c.

Covers the EC-001 deployment step: appending ep_rel.* tensors to a v2 base
file to produce v3, with the existing payload preserved bit-exactly.
"""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from export_bnmodel import append_tensors, read_bnmodel_index, replace_tensors, write_bnmodel  # noqa: E402


def read_header(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        version, count, index_size = struct.unpack("<III", f.read(12))
        index = f.read(index_size)
        return magic, version, count, index_size, index


class ExportBnmodelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_layout(self):
        a = np.arange(4, dtype="<f4").reshape(2, 2)
        b = np.array([1.5, -2.5], dtype="<f4")
        out = self.dir / "m.bnmodel"
        write_bnmodel(out, {"a.weight": a, "a.bias": b}, version=3)

        magic, version, count, index_size, index = read_header(out)
        self.assertEqual(magic, b"BNMW")
        self.assertEqual(version, 3)
        self.assertEqual(count, 2)
        lines = index.decode().splitlines()
        self.assertEqual(lines, [
            '{"n":"a.weight","s":[2,2],"o":0}',
            '{"n":"a.bias","s":[2],"o":16}',
        ])
        self.assertEqual(sum(len(line) + 1 for line in lines), index_size)

        with open(out, "rb") as f:
            f.seek(16 + index_size + 1)
            payload = f.read()
        self.assertEqual(len(payload), 24)
        self.assertEqual(payload, a.tobytes() + b.tobytes())

    def test_read_index_roundtrip(self):
        a = np.zeros((3, 4), dtype="<f4")
        out = self.dir / "m.bnmodel"
        write_bnmodel(out, {"x": a}, version=2)
        index = read_bnmodel_index(out)
        self.assertEqual(index["x"], ((3, 4), 0))

    def test_append_preserves_base_payload(self):
        base_a = np.random.default_rng(1).standard_normal(8, dtype="<f4")
        base_b = np.random.default_rng(2).standard_normal((2, 3), dtype="<f4")
        base = self.dir / "base.bnmodel"
        write_bnmodel(base, {"query.weight": base_a, "route.weight": base_b}, version=2)
        with open(base, "rb") as f:
            base_bytes = f.read()

        ep_rel = {
            "ep_rel.0.weight": np.zeros((256, 4096), dtype="<f4"),
            "ep_rel.0.bias": np.zeros(256, dtype="<f4"),
            "ep_rel.2.weight": np.zeros((1, 256), dtype="<f4"),
            "ep_rel.2.bias": np.zeros(1, dtype="<f4"),
        }
        out = self.dir / "v3.bnmodel"
        append_tensors(base, ep_rel, out)

        magic, version, count, index_size, index = read_header(out)
        self.assertEqual(magic, b"BNMW")
        self.assertEqual(version, 3)
        self.assertEqual(count, 6)
        new_index = read_bnmodel_index(out)
        self.assertEqual(new_index["query.weight"], ((8,), 0))
        self.assertEqual(new_index["route.weight"], ((2, 3), 32))
        self.assertEqual(
            new_index["ep_rel.0.weight"], ((256, 4096), 32 + 24))

        # base payload bytes preserved verbatim at identical offsets
        with open(out, "rb") as f:
            f.seek(16 + index_size + 1)
            payload = f.read(32 + 24)
        self.assertEqual(payload, base_a.tobytes() + base_b.tobytes())
        self.assertNotEqual(out.read_bytes(), base_bytes)

    def test_replace_tensors(self):
        base_a = np.arange(8, dtype="<f4")
        base = self.dir / "base.bnmodel"
        write_bnmodel(base, {"query.weight": base_a, "wide_fact.0.weight": np.zeros((2, 4), dtype="<f4")}, version=2)
        repl = {"wide_fact.0.weight": np.ones((2, 4), dtype="<f4"),
                "ep_joint.mix.bias": np.full(3, 2.5, dtype="<f4")}
        out = self.dir / "repl.bnmodel"
        index = replace_tensors(base, repl, out, new_version=3)

        magic, version, count, index_size, idx = read_header(out)
        self.assertEqual(magic, b"BNMW")
        self.assertEqual(version, 3)
        self.assertEqual(count, 3)
        # untouched tensor payload preserved verbatim
        with open(out, "rb") as f:
            f.seek(16 + index_size + 1 + index["query.weight"][1])
            self.assertEqual(f.read(32), base_a.tobytes())
        # replaced tensor payload
        with open(out, "rb") as f:
            f.seek(16 + index_size + 1 + index["wide_fact.0.weight"][1])
            self.assertEqual(f.read(32), np.ones((2, 4), dtype="<f4").tobytes())
        # added tensor
        with open(out, "rb") as f:
            f.seek(16 + index_size + 1 + index["ep_joint.mix.bias"][1])
            self.assertEqual(f.read(12), np.full(3, 2.5, dtype="<f4").tobytes())

    def test_deterministic(self):
        tensors = {"t": np.arange(5, dtype="<f4")}
        one, two = self.dir / "one.bnmodel", self.dir / "two.bnmodel"
        write_bnmodel(one, tensors, version=2)
        write_bnmodel(two, tensors, version=2)
        self.assertEqual(one.read_bytes(), two.read_bytes())


if __name__ == "__main__":
    unittest.main()
