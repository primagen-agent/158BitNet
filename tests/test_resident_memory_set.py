import copy
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from prepare_memory_set_curriculum import make_world, prepare
from train_resident_memory_set import (ResidentMemorySet, collate, compile_states, read_batch,
                                       create_model, load_worlds, select_sets, support_loss, pairwise_binding_loss, evaluate,
                                       factorized_labels, training_loss, version_log_gate,
                                       counterfactual_memory_batch, training_version_groups, activation_count_scores,
                                       distractor_donors, distractor_pair_batch, paired_activation_consistency,
                                       initialize_training_model, DEFAULT_TRAINING_SEED)
from diagnose_resident_memory_set import row_error_metrics, memory_interventions


class ResidentMemorySetTest(unittest.TestCase):
    def test_training_seed_reproduces_initialization_and_sampling_not_corpus(self):
        torch.manual_seed(DEFAULT_TRAINING_SEED)
        legacy = create_model(8, initial_order_scale=.025, architecture="factorized")
        legacy_draws = torch.randint(1000, (32,))
        current = initialize_training_model(8, "factorized", .025)
        draws = torch.randint(1000, (32,))
        for name, value in legacy.state_dict().items(): self.assertTrue(torch.equal(value, current.state_dict()[name]))
        self.assertTrue(torch.equal(legacy_draws, draws))
        first = initialize_training_model(8, "factorized", .025, 2810917)
        first_draws = torch.randint(1000, (32,))
        second = initialize_training_model(8, "factorized", .025, 2810917)
        second_draws = torch.randint(1000, (32,))
        self.assertTrue(torch.equal(first_draws, second_draws))
        for name, value in first.state_dict().items(): self.assertTrue(torch.equal(value, second.state_dict()[name]))
        self.assertFalse(torch.equal(first.project.weight, current.project.weight))
        self.assertFalse(torch.equal(first_draws, draws))
        a = make_world(0, "train", DEFAULT_TRAINING_SEED, True)
        initialize_training_model(8, "factorized", .025, 2810918)
        self.assertEqual(a, make_world(0, "train", DEFAULT_TRAINING_SEED, True))
        with self.assertRaises(ValueError): initialize_training_model(8, "factorized", .025, -1)
        with self.assertRaises(ValueError): initialize_training_model(8, "factorized", .025, 2 ** 63)

    def test_distractor_supervision_is_separate_and_identity_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "data"
            manifest = prepare(root, 8, 2, diverse_train=True)
            worlds = load_worlds(root / "train.jsonl", "train")
            path = root / "train_supervision.json"
            annotations = json.loads(path.read_text())
            self.assertFalse((root / "valid_supervision.json").exists())
            self.assertFalse((root / "test_supervision.json").exists())
            for i, world in enumerate(worlds):
                self.assertEqual(world, make_world(i, "train", 2810916, True))
                self.assertNotIn("event_keys", world)
            donors = distractor_donors(worlds, path, manifest["splits"]["train"]["sha256"])
            for wi, pool in enumerate(donors):
                recipient = annotations["worlds"][worlds[wi]["world_id"]]
                for di, keep in pool:
                    donor = annotations["worlds"][worlds[di]["world_id"]]
                    self.assertFalse(set(recipient["entities"]) & {donor["event_keys"][i][0] for i in keep})
                    self.assertTrue({x[1] for x in recipient["event_keys"]} & {donor["event_keys"][i][1] for i in keep})
            with self.assertRaises(ValueError): distractor_donors(worlds, path, "wrong_hash")
            changed = copy.deepcopy(worlds); changed[0]["split"] = "valid"
            with self.assertRaises(ValueError): distractor_donors(changed, path, manifest["splits"]["train"]["sha256"])
            with self.assertRaises(ValueError): make_world(0, "valid", 1, supervision={})

    def test_distractor_pairs_preserve_supports_features_and_version_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "data"
            manifest = prepare(root, 8, 2, diverse_train=True)
            worlds = load_worlds(root / "train.jsonl", "train")
            before = copy.deepcopy(worlds)
            features = [{"events": [torch.randn(5, 8).half() for _ in w["events"]],
                         "queries": [torch.randn(5, 8).half() for _ in w["queries"]]} for w in worlds]
            donors = distractor_donors(worlds, root / "train_supervision.json", manifest["splits"]["train"]["sha256"])
            indices = [(0, i) for i in range(len(worlds[0]["queries"]))]
            data, batch_worlds, batch_indices = distractor_pair_batch(worlds, features, indices, donors, "cpu")
            self.assertEqual(worlds, before)
            self.assertEqual(len(batch_indices), 24)
            self.assertTrue(torch.equal(data[2][:12], data[2][12:]))
            self.assertTrue(torch.equal(data[-1][:12], data[-1][12:]))
            self.assertTrue((data[-3][12:].sum(-1) > data[-3][:12].sum(-1)).all())
            self.assertTrue((data[-1][12:] <= data[-3][:12]).all())
            for wi, qi in batch_indices[12:]:
                self.assertEqual(batch_worlds[wi]["queries"][qi], worlds[0]["queries"][qi])
                self.assertEqual(batch_worlds[wi]["events"][:len(worlds[0]["events"])], worlds[0]["events"])
                self.assertEqual(len(set(e["source_id"] for e in batch_worlds[wi]["events"])), len(batch_worlds[wi]["events"]))
                training_version_groups(batch_worlds[wi])
            model = create_model(8, 16, .025, "factorized")
            loss = training_loss(model, data, batch_worlds, batch_indices, invariance_pairs=12)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(float(model.project.weight.grad.abs().sum()), 0)

    def test_invariance_loss_ignores_added_slots_but_penalizes_changed_original_predictions(self):
        scores = torch.tensor([[2., -2., -1e4], [2., -2., 9.]], requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        counts = torch.tensor([[1., 2., 3., 0., -1.]] * 2, requires_grad=True)
        self.assertAlmostEqual(float(paired_activation_consistency(scores, counts, mask, 1).detach()), 0., places=6)
        changed = scores.detach().clone(); changed[1, 0] = -2.; changed.requires_grad_()
        count_changed = counts.detach().clone(); count_changed[1] = count_changed[1].flip(0); count_changed.requires_grad_()
        loss = paired_activation_consistency(changed, count_changed, mask, 1)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertEqual(float(changed.grad[1, 2]), 0)
        self.assertGreater(float(count_changed.grad.abs().sum()), 0)
        invalid = mask.clone(); invalid[1, 0] = False
        with self.assertRaises(ValueError): paired_activation_consistency(scores, counts, invalid, 1)

    def test_activation_count_is_joint_set_energy_not_query_classification(self):
        import itertools
        scores = torch.tensor([[2., -3., 1., 999.]], requires_grad=True)
        mask = torch.tensor([[True, True, True, False]])
        prior = torch.tensor([.1, -.2, .3, -.4, .5], requires_grad=True)
        energy = activation_count_scores(scores, mask, prior)
        for k in range(4):
            brute = max(sum(float(scores.detach()[0, j]) for j in subset) + float(prior.detach()[k])
                        for subset in itertools.combinations(range(3), k))
            self.assertAlmostEqual(float(energy.detach()[0, k]), brute, places=5)
        self.assertEqual(select_sets(scores, energy, mask).tolist(), [[True, False, True, False]])
        torch.nn.functional.cross_entropy(energy, torch.tensor([1])).backward()
        self.assertGreater(float(scores.grad[0, 2]), 0)  # Discourage the unnecessary second activation.
        self.assertEqual(float(scores.grad[0, 3]), 0)
        self.assertGreater(float(prior.grad.abs().sum()), 0)
        empty = torch.zeros_like(mask)
        self.assertEqual(int(activation_count_scores(scores, empty, prior).argmax()), 0)
        self.assertFalse(select_sets(scores, activation_count_scores(scores, empty, prior), empty).any())

    def test_activation_count_model_trains_and_reads_only_persisted_state(self):
        torch.manual_seed(17); baseline = create_model(8, 16, .025, "factorized")
        torch.manual_seed(17); coupled = create_model(8, 16, .025, "activation_count")
        for name, value in coupled.state_dict().items():
            expected = baseline.count[-1].bias if name == "cardinality_prior" else baseline.state_dict()[name]
            self.assertTrue(torch.equal(value, expected), name)
        torch.manual_seed(9)
        w = make_world(0, "train", 1)
        f = {"events": [torch.randn(5, 8).half() for _ in w["events"]],
             "queries": [torch.randn(5, 8).half() for _ in w["queries"]]}
        indices = [(0, i) for i in range(len(w["queries"]))]
        data = collate([w], [f], indices, "cpu")
        model = create_model(8, 16, .025, "activation_count").eval()
        self.assertFalse(hasattr(model, "count"))
        loss = training_loss(model, data, [w], indices)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        state = compile_states(model, [f], "cpu")
        buffer = io.BytesIO(); torch.save(state, buffer); buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        inputs = read_batch([{"queries": f["queries"]}], restored, indices, "cpu")
        scores, counts = model.read(*inputs)
        original = model(*data[:-1])
        self.assertTrue(torch.allclose(scores, original[0], atol=1e-5))
        self.assertTrue(torch.allclose(counts, original[1], atol=1e-5))
        self.assertTrue(torch.equal(counts, activation_count_scores(scores, inputs[-2], model.cardinality_prior)))
        perm = torch.arange(inputs[2].shape[1] - 1, -1, -1)
        ps, pc = model.read(inputs[0], inputs[1], inputs[2][:, perm], inputs[3][:, perm], inputs[4][:, perm])
        self.assertTrue(torch.allclose(scores[:, perm], ps, atol=1e-5))
        self.assertTrue(torch.allclose(counts, pc, atol=1e-5))

    def test_erasure_control_is_not_misrepresented_as_memory_success(self):
        torch.manual_seed(3)
        w = make_world(0, "valid", 1)
        f = {"events": [torch.randn(5, 8).half() for _ in w["events"]],
             "queries": [torch.randn(5, 8).half() for _ in w["queries"]]}
        model = create_model(8, 16, .025, "factorized").eval()
        result = memory_interventions(model, [w], [f], "cpu")
        self.assertTrue(result["oracle_constructed_alternative_stores"])
        self.assertFalse(result["gold_at_read"])
        # Default safe count head abstains everywhere: erasure alone proves nothing.
        self.assertEqual(result["controls"]["erase_relevant"]["correct"], 10)
        self.assertEqual(result["controls"]["erase_relevant"]["total"], 10)
        self.assertEqual(result["controls"]["keep_relevant"]["correct"], 0)
        self.assertEqual(result["controls"]["keep_relevant"]["total"], 10)

    def test_counterfactual_pairs_change_memory_not_questions_or_scope(self):
        w = make_world(0, "train", 1, True)
        f = {"events": [torch.randn(5, 8).half() for _ in w["events"]],
             "queries": [torch.randn(5, 8).half() for _ in w["queries"]]}
        indices = [(0, i) for i in range(len(w["queries"]))] * 2
        data = collate([w], [f], indices, "cpu")
        torch.manual_seed(2)
        changed = counterfactual_memory_batch(data, [w], indices, 12)
        self.assertTrue(torch.equal(changed[-3][:12], data[-3][:12]))
        self.assertTrue(torch.equal(changed[-1][:12], data[-1][:12]))
        for i in (0, 1, 2, 3, 6): self.assertTrue(torch.equal(changed[i], data[i]))
        self.assertTrue(torch.equal(changed[2][:12], changed[2][12:]))
        self.assertTrue(torch.equal(changed[-1], data[-1] * changed[-3]))
        self.assertTrue((changed[-3][12:] != data[-3][12:]).any())
        for group in training_version_groups(w):
            present = changed[-3][:, sorted(group)]
            self.assertTrue(torch.equal(present, present[:, :1].expand_as(present)))
        model = create_model(8, 16, .025, "factorized")
        loss = training_loss(model, changed, [w], indices)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.project.weight.grad.abs().sum()), 0)
        rejected = copy.deepcopy(w); rejected["split"] = "valid"
        with self.assertRaises(ValueError): counterfactual_memory_batch(data, [rejected], indices, 12)
        with self.assertRaises(ValueError): counterfactual_memory_batch(data, [w], indices, -1)

    def test_removed_memory_cannot_influence_reader_or_hide_old_versions(self):
        torch.manual_seed(7)
        w = make_world(0, "train", 1)
        f = {"events": [torch.randn(5, 8).half() for _ in w["events"]],
             "queries": [torch.randn(5, 8).half() for _ in w["queries"]]}
        model = create_model(8, 16, .025, "factorized").eval()
        resident = compile_states(model, [f], "cpu")
        query, qm, state, mask, positions = read_batch([f], resident, [(0, 0)], "cpu")
        removed = next(q["targets"] for q in w["queries"] if q["kind"] == "history")
        mask[:, removed] = False
        a, ac = model.read(query, qm, state, mask, positions)
        corrupted = state.clone(); corrupted[:, removed, :, :-1] = torch.randn_like(corrupted[:, removed, :, :-1]) * 100
        b, bc = model.read(query, qm, corrupted, mask, positions)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        self.assertTrue(torch.allclose(ac, bc, atol=1e-5))
        # Physically removing the unavailable slots has identical active outputs.
        keep = mask[0].nonzero().flatten()
        c, cc = model.read(query, qm, state[:, keep], mask[:, keep], positions[:, keep])
        self.assertTrue(torch.allclose(a[:, keep], c, atol=1e-5))
        self.assertTrue(torch.allclose(ac, cc, atol=1e-5))
        malformed = copy.deepcopy(w)
        malformed["queries"].append({"kind": "history", "targets": [removed[0], len(w["events"]) - 1]})
        with self.assertRaises(ValueError): training_version_groups(malformed)

    def test_diverse_training_changes_only_language_and_keeps_holdouts_sealed(self):
        for i in range(12):
            original = make_world(i, "train", 2810916)
            diverse = make_world(i, "train", 2810916, True)
            for a, b in zip(original["events"], diverse["events"]):
                self.assertEqual(a["value"], b["value"])
                self.assertEqual(a["source_id"], b["source_id"])
                self.assertEqual(b["text"].encode()[b["value_start"]:b["value_end"]].decode(), b["value"])
            for a, b in zip(original["queries"], diverse["queries"]):
                for key in ("kind", "targets", "answers"): self.assertEqual(a[key], b[key])
            self.assertNotEqual(original["events"], diverse["events"])
            self.assertNotEqual(original["queries"], diverse["queries"])
        with tempfile.TemporaryDirectory() as folder:
            baseline = prepare(Path(folder) / "base", 4, 2)
            diverse = prepare(Path(folder) / "diverse", 4, 2, diverse_train=True)
            for split in ("valid", "test"):
                self.assertEqual(baseline["splits"][split], diverse["splits"][split])
            self.assertNotEqual(baseline["splits"]["train"]["sha256"], diverse["splits"]["train"]["sha256"])
            load_worlds(Path(folder) / "diverse/train.jsonl", "train")

    def test_version_gate_only_suppresses_learned_older_same_fact(self):
        # An unrelated distractor is newest; it must not suppress either fact.
        links = torch.full((1, 4, 4), -30.)
        links[0, 0, 2] = links[0, 2, 0] = 30.
        mask = torch.tensor([[True, True, True, True]])
        positions = torch.tensor([[0., .3, .7, 1.]])
        current = version_log_gate(links, torch.tensor([-30.]), mask, positions).exp()
        self.assertLess(float(current[0, 0]), 1e-6)
        self.assertTrue(torch.allclose(current[0, 1:], torch.ones(3), atol=1e-6))
        history = version_log_gate(links, torch.tensor([30.]), mask, positions).exp()
        self.assertTrue(torch.allclose(history, torch.ones_like(history), atol=1e-6))
        # Padding and physical storage order are not chronology.
        perm = torch.tensor([3, 2, 0, 1])
        swapped = version_log_gate(links[:, perm][:, :, perm], torch.tensor([-30.]), mask[:, perm], positions[:, perm])
        self.assertTrue(torch.allclose(current[:, perm], swapped.exp(), atol=1e-6))
        mask[0, 2] = False
        masked = version_log_gate(links, torch.tensor([-30.]), mask, positions).exp()
        self.assertGreater(float(masked[0, 0]), .999)

    def test_factorized_labels_are_training_only_and_version_complete(self):
        w = make_world(0, "train", 1)
        indices = [(0, i) for i in range(len(w["queries"]))]
        address, links, history = factorized_labels([w], indices, len(w["events"]), "cpu")
        pairs = [set(q["targets"]) for q in w["queries"] if q["kind"] == "history"]
        self.assertEqual(int((links[0] - torch.eye(len(w["events"]))).sum()), 4)
        self.assertTrue(torch.equal(links, links.transpose(1, 2)))
        for i, q in enumerate(w["queries"]):
            expected = set(q["targets"])
            for group in pairs:
                if expected & group: expected |= group
            self.assertEqual(address[i].nonzero().flatten().tolist(), sorted(expected))
            self.assertEqual(bool(history[i]), q["kind"] == "history")

    def test_factorized_reader_gradients_persistence_and_no_label_inputs(self):
        torch.manual_seed(6)
        w = make_world(0, "train", 1)
        f = {"events": [torch.randn(4 + i % 2, 8).half() for i in range(len(w["events"]))],
             "queries": [torch.randn(5, 8).half() for _ in w["queries"]]}
        indices = [(0, i) for i in range(len(w["queries"]))]
        model = create_model(8, 16, .025, "factorized").eval()
        data = collate([w], [f], indices, "cpu")
        loss = training_loss(model, data, [w], indices)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            # Safe count initialization zeros the last weight, so its hidden
            # layer starts receiving gradients only after the first update.
            if not name.startswith("count.0."):
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        scores, counts = model(*data[:-1])
        changed = copy.deepcopy(w)
        for q in changed["queries"]: q["targets"] = []; q["answers"] = []; q["kind"] = "null"
        for e in changed["events"]: e["value"] = "not_an_input"
        second = model(*collate([changed], [f], indices, "cpu")[:-1])
        self.assertTrue(torch.equal(scores, second[0]))
        self.assertTrue(torch.equal(counts, second[1]))
        resident = compile_states(model, [f], "cpu")
        buffer = io.BytesIO(); torch.save(resident, buffer); buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        inputs = read_batch([{"queries": f["queries"]}], restored, indices, "cpu")
        read_scores, read_counts, components = model.read_components(*inputs)
        self.assertTrue(torch.allclose(scores, read_scores, atol=1e-5))
        self.assertTrue(torch.allclose(counts, read_counts, atol=1e-5))
        perm = torch.arange(inputs[2].shape[1] - 1, -1, -1)
        ps, pc = model.read(inputs[0], inputs[1], inputs[2][:, perm], inputs[3][:, perm], inputs[4][:, perm])
        self.assertTrue(torch.allclose(read_scores[:, perm], ps, atol=1e-5))
        self.assertTrue(torch.allclose(read_counts, pc, atol=1e-5))
        # Fact matching and scope cannot use chronology as a shortcut.
        _, _, reversed_time = model.read_components(*inputs[:-1], 1 - inputs[-1])
        for key in ("address", "history", "links"):
            self.assertTrue(torch.equal(components[key], reversed_time[key]))
        padded = torch.nn.functional.pad(inputs[2], (0, 0, 0, 3))
        pads, padc = model.read(inputs[0], inputs[1], padded, inputs[3], inputs[4])
        self.assertTrue(torch.allclose(read_scores, pads, atol=1e-5))
        self.assertTrue(torch.allclose(read_counts, padc, atol=1e-5))
        empty = read_batch([{"queries": f["queries"]}], [restored[0][:0]], indices, "cpu")
        es, ec = model.read(*empty)
        self.assertTrue(torch.isfinite(ec).all())
        self.assertTrue(torch.isfinite(es).all())
        self.assertFalse(select_sets(es, ec, empty[-2]).any())

    def test_token_addresses_are_persistent_masked_and_differentiable(self):
        torch.manual_seed(4)
        w = make_world(0, "train", 1)
        features = {"events": [torch.randn(4 + i % 3, 8).half() for i in range(len(w["events"]))],
                    "queries": [torch.randn(5, 8).half() for _ in w["queries"]]}
        indices = [(0, i) for i in range(len(w["queries"]))]
        model = create_model(8, 16, .025, "token").eval()
        data = collate([w], [features], indices, "cpu")
        scores, counts = model(*data[:-1])
        support_loss(scores, counts, data[-1], data[-3]).backward()
        self.assertGreater(float(model.project.weight.grad.abs().sum()), 0)
        resident = compile_states(model, [features], "cpu")
        self.assertEqual(resident[0].ndim, 3)
        query_only = [{"queries": features["queries"]}]
        inputs = read_batch(query_only, resident, indices, "cpu")
        actual, _ = model.read(*inputs)
        self.assertTrue(torch.allclose(scores, actual, atol=1e-5))
        perm = torch.arange(inputs[2].shape[1] - 1, -1, -1)
        swapped, _ = model.read(inputs[0], inputs[1], inputs[2][:, perm], inputs[3][:, perm], inputs[4][:, perm])
        self.assertTrue(torch.allclose(actual[:, perm], swapped, atol=1e-5))
        padded = torch.nn.functional.pad(inputs[2], (0, 0, 0, 4))
        padded_scores, _ = model.read(inputs[0], inputs[1], padded, inputs[3], inputs[4])
        self.assertTrue(torch.allclose(actual, padded_scores, atol=1e-5))
        empty_inputs = read_batch(query_only, [resident[0][:0]], indices, "cpu")
        es, ec = model.read(*empty_inputs)
        self.assertTrue(torch.isfinite(ec).all())
        self.assertFalse(select_sets(es, ec, empty_inputs[-2]).any())

    def test_binding_loss_rewards_correct_ranking_and_ignores_padding(self):
        target = torch.tensor([[1., 0., 0.]])
        mask = torch.tensor([[True, True, False]])
        good = torch.tensor([[3., -1., 10000.]], requires_grad=True)
        bad = torch.tensor([[-1., 3., -10000.]])
        loss = pairwise_binding_loss(good, target, mask)
        self.assertLess(float(loss.detach()), float(pairwise_binding_loss(bad, target, mask)))
        loss.backward()
        self.assertLess(float(good.grad[0, 0]), 0)
        self.assertGreater(float(good.grad[0, 1]), 0)
        self.assertEqual(float(good.grad[0, 2]), 0)
        self.assertEqual(float(pairwise_binding_loss(good, target * 0, mask).detach()), 0)

    def test_error_decomposition_does_not_count_oracle_answers_as_success(self):
        scores = torch.tensor([1., 4., 3., 0.])
        row = row_error_metrics(scores, 1, [1, 2], [])
        self.assertFalse(row["set_correct"])
        self.assertFalse(row["count_correct"])
        self.assertTrue(row["oracle_count_set_correct"])
        self.assertFalse(row["threshold_set_correct"])
        threshold = row_error_metrics(torch.tensor([-2., 3., 2., -1.]), 1, [1, 2], [])
        self.assertTrue(threshold["threshold_set_correct"])
        self.assertTrue(threshold["threshold_count_correct"])
        self.assertFalse(threshold["set_correct"])
        version = row_error_metrics(scores, 1, [2], [{1, 2}])
        self.assertTrue(version["previous_version_error_with_oracle_count"])
        self.assertFalse(version["other_binding_error_with_oracle_count"])
        binding = row_error_metrics(scores, 1, [3], [{2, 3}])
        self.assertTrue(binding["other_binding_error_with_oracle_count"])

    def test_order_scale_ablation_changes_only_order_initialization(self):
        torch.manual_seed(3); base = ResidentMemorySet(8, 16)
        torch.manual_seed(3); scaled = ResidentMemorySet(8, 16, initial_order_scale=.025)
        for name, tensor in base.state_dict().items():
            expected = tensor * .025 if name == "order.weight" else tensor
            self.assertTrue(torch.equal(scaled.state_dict()[name], expected), name)
        with self.assertRaises(ValueError): ResidentMemorySet(8, 16, initial_order_scale=-1)

    def test_curriculum_coverage_and_sealed_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "data"
            manifest = prepare(root, 4, 2)
            train = load_worlds(root / "train.jsonl", "train")
            valid = load_worlds(root / "valid.jsonl", "valid")
            self.assertEqual(manifest["splits"]["train"]["questions"], 48)
            self.assertFalse({e["value"] for w in train for e in w["events"]} &
                             {e["value"] for w in valid for e in w["events"]})
            for w in train + valid:
                self.assertEqual({q["kind"] for q in w["queries"]}, {"current", "multi", "history", "null"})
                self.assertEqual({len(q["targets"]) for q in w["queries"]}, {0, 1, 2, 4})
                self.assertNotIn("active", w["events"][0])
            with self.assertRaises(ValueError): load_worlds(root / "test.jsonl", "test")
            with self.assertRaises(FileExistsError): prepare(root)

    def test_evaluation_and_locomo_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "rows.jsonl"
            w = make_world(0, "train", 1)
            w["locomo_used"] = True; p.write_text(json.dumps(w))
            with self.assertRaises(ValueError): load_worlds(p, "train")

    def test_selection_is_learned_cardinality_not_fixed_top_one(self):
        scores = torch.tensor([[4., 3., 2., 1., 99.]] * 3)
        counts = torch.full((3, 5), -10.); counts[0, 0] = counts[1, 2] = counts[2, 4] = 10.
        mask = torch.tensor([[True, True, True, True, False]] * 3)
        selected = select_sets(scores, counts, mask)
        self.assertEqual(selected.sum(1).tolist(), [0, 2, 4])
        self.assertFalse(selected[:, -1].any())

    def test_read_uses_resident_tensors_and_is_slot_permutation_equivariant(self):
        torch.manual_seed(1)
        model = ResidentMemorySet(8, 16).eval()
        query = torch.randn(2, 5, 8); qm = torch.ones(2, 5, dtype=torch.bool)
        state = torch.randn(2, 4, 16); mask = torch.tensor([[True, True, True, False]] * 2)
        positions = torch.tensor([[0., .5, 1., 0.]] * 2)
        a, ac = model.read(query, qm, state, mask, positions)
        perm = torch.tensor([2, 0, 3, 1])
        b, bc = model.read(query, qm, state[:, perm], mask[:, perm], positions[:, perm])
        self.assertTrue(torch.allclose(a[:, perm], b, atol=1e-5))
        self.assertTrue(torch.allclose(ac, bc, atol=1e-5))
        self.assertEqual(list(inspect.signature(model.read).parameters),
                         ["query", "query_mask", "state", "slot_mask", "positions"])
        empty = torch.zeros_like(mask)
        logits, counts = model.read(query, qm, state * 0, empty, positions * 0)
        self.assertTrue(torch.isfinite(counts).all())
        self.assertFalse(select_sets(logits, counts, empty).any())

    def test_gradients_reach_writer_and_reader_without_gold_input_leakage(self):
        torch.manual_seed(2)
        w = make_world(0, "train", 1)
        f = {"events": [torch.randn(5, 8).half() for _ in w["events"]],
             "queries": [torch.randn(5, 8).half() for _ in w["queries"]]}
        indices = [(0, i) for i in range(len(w["queries"]))]
        model = ResidentMemorySet(8, 16).eval()
        data = collate([w], [f], indices, "cpu")
        scores, counts = model(*data[:-1])
        support_loss(scores, counts, data[-1], data[-3]).backward()
        for name in ("project.weight", "write_pool", "read_pool", "pair.0.weight"):
            grad = dict(model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0., name)
        modified = copy.deepcopy(w)
        for q in modified["queries"]: q["targets"] = []; q["answers"] = []
        for e in modified["events"]: e["value"] = "not_an_input"
        changed = collate([modified], [f], indices, "cpu")
        s2, c2 = model(*changed[:-1])
        self.assertTrue(torch.equal(scores, s2)); self.assertTrue(torch.equal(counts, c2))
        resident = compile_states(model, [f], "cpu")
        snapshot = io.BytesIO()
        torch.save(resident, snapshot); snapshot.seek(0)
        restored = torch.load(snapshot, weights_only=True)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(resident, restored)))
        # Query-time batches need no event features; only resident addresses.
        query_only = [{"queries": f["queries"]}]
        recalled = model.read(*read_batch(query_only, restored, indices, "cpu"))
        self.assertTrue(torch.allclose(scores, recalled[0], atol=1e-5))
        self.assertTrue(torch.allclose(counts, recalled[1], atol=1e-5))
        metrics, _ = evaluate(model, [w], [f], "cpu")
        self.assertEqual(metrics["total"], 12)
        self.assertEqual(metrics["worst_group_accuracy"], 0.)  # Ignore-all is not success.


if __name__ == "__main__": unittest.main()
