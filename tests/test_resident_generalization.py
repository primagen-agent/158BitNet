import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from diagnose_resident_generalization import build_controls, transform_world, load_diagnostic_features
from prepare_memory_set_curriculum import make_world, prepare
from train_resident_memory_set import load_worlds


class ResidentGeneralizationTest(unittest.TestCase):
    def test_diagnostic_feature_cache_rejects_identity_and_geometry_mismatch(self):
        worlds = build_controls(1)["replay"]
        features = [{"events": [torch.randn(3, 8) for _ in worlds[0]["events"]],
                     "queries": [torch.randn(3, 8) for _ in worlds[0]["queries"]]}]
        binding = {"format": "RESIDENT_FACTORIAL_DIAGNOSTIC_FEATURES_V1", "backbone_sha256": "bound-backbone",
                   "diagnostic_corpus_sha256": "bound-corpus", "evaluation_only": True}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.pt"
            torch.save({**binding, "features": features}, path)
            actual = load_diagnostic_features(path, binding, worlds)
            self.assertTrue(torch.equal(actual[0]["events"][0], features[0]["events"][0]))
            for key in binding:
                wrong = dict(binding); wrong[key] = "wrong"
                with self.assertRaises(ValueError): load_diagnostic_features(path, wrong, worlds)
            torch.save({**binding, "features": []}, path)
            with self.assertRaises(ValueError): load_diagnostic_features(path, binding, worlds)

    def test_annotation_extension_preserves_frozen_corpus_and_old_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = prepare(Path(folder) / "data", diverse_train=True)
            self.assertEqual(manifest["splits"]["train"]["sha256"], "5ead2b1e64d46ada1cd0135cfbf2b27b72844dbf4620750bf98c4775c6552ee8")
            self.assertEqual(manifest["splits"]["valid"]["sha256"], "8b352c38ec96185892c55b311e2377f6e1accdea9a592e020f60f54b1c2ed29c")
            self.assertEqual(manifest["splits"]["test"]["sha256"], "24fedf583effcfaee91e563602fa6002b16c062dffdfa4d795ed667d94b08c76")
            self.assertEqual(manifest["training_supervision_sha256"], "d712879c7b3f904865c8aa65ac17a6c1af253e3d0bd07cb47b2c52874992246d")

    def test_crossed_controls_change_only_declared_surfaces(self):
        controls = build_controls(3)
        self.assertEqual(len(controls), 13)
        for i, base in enumerate(controls["fresh_e0_s0_q0"]):
            for key, worlds in controls.items():
                w = worlds[i]
                for event in w["events"]:
                    self.assertEqual(event["text"].encode()[event["value_start"]:event["value_end"]].decode(), event["value"])
                self.assertTrue(w["evaluation_only"])
                if not key.startswith("fresh_e"): continue
                self.assertEqual([q["targets"] for q in w["queries"]], [q["targets"] for q in base["queries"]])
                self.assertEqual([e["value"] for e in w["events"]], [e["value"] for e in base["events"]])
            source = controls["fresh_e0_s1_q0"][i]
            query = controls["fresh_e0_s0_q1"][i]
            self.assertEqual(base["queries"], source["queries"])
            self.assertEqual(base["events"], query["events"])
            self.assertNotEqual(base["events"], source["events"])
            self.assertNotEqual(base["queries"], query["queries"])
            self.assertEqual([q["text"] for q in base["queries"]], [q["text"] for q in controls["fresh_prefix_shift"][i]["queries"]])

    def test_replay_and_binding_controls_preserve_questions_and_expected_cardinality(self):
        controls = build_controls(4)
        for i, replay in enumerate(controls["replay"]):
            original = make_world(i, "train", 2810916, True)
            self.assertEqual(replay["events"], original["events"])
            self.assertEqual(replay["queries"], original["queries"])
            changed_values = controls["replay_new_values"][i]
            self.assertEqual([q["text"] for q in replay["queries"]], [q["text"] for q in changed_values["queries"]])
            self.assertEqual([q["targets"] for q in replay["queries"]], [q["targets"] for q in changed_values["queries"]])
            self.assertNotEqual([e["value"] for e in replay["events"]], [e["value"] for e in changed_values["events"]])
            base, swapped = controls["fresh_e0_s0_q0"][i], controls["fresh_binding_swap"][i]
            self.assertEqual([q["text"] for q in base["queries"]], [q["text"] for q in swapped["queries"]])
            self.assertEqual([len(q["targets"]) for q in base["queries"]], [len(q["targets"]) for q in swapped["queries"]])
            self.assertEqual(sum(a["targets"] != b["targets"] for a, b in zip(base["queries"], swapped["queries"])), 6)
            self.assertEqual([e["value"] for e in base["events"]], [e["value"] for e in swapped["events"]])

    def test_diagnostic_worlds_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "diagnostic.jsonl"
            p.write_text(json.dumps(build_controls(1)["fresh_e0_s0_q0"][0]) + "\n")
            with self.assertRaises(ValueError): load_worlds(p, "diagnostic")
        with self.assertRaises(ValueError): build_controls(129)
        with self.assertRaises(ValueError): build_controls(1, 2810916)


if __name__ == "__main__": unittest.main()
