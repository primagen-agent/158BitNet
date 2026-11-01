"""Current model format, learned source isolation and numerical invariants."""

from pathlib import Path
import sys
import struct
import tempfile
import unittest
import zlib
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from neural_slot_model import (
    NeuralSlots,
    INPUT,
    HIDDEN,
    WIDTH,
    SOURCE_CONTEXT_WIDTH,
    SOURCE_CONTEXT_INPUT,
    RELATION_HEADS,
    SEMANTIC,
    QUERY,
    QUERY_KEYS,
    LEXICAL,
    IDENTITY,
    KEYS,
    export,
    require_parameters,
    pack_memories,
    identity_word_features,
    decode_plan,
)


class Contract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8201)

    def test_zero_attention_temperature_preserves_unscaled_read_and_scope(self):
        model = NeuralSlots()
        self.assertTrue(torch.equal(model.attention_scales(), torch.ones(4)))
        self.assertTrue(torch.equal(model.attention_scales(scope=True), torch.ones(4)))
        queries, memories = torch.randn(2, QUERY), torch.randn(3, 5, WIDTH)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
        q = queries[:, :SEMANTIC].reshape(2, RELATION_HEADS, WIDTH)
        attention = (torch.einsum("qhd,ntd->qnht", q, memories) * 0.125)
        attention = attention.masked_fill(~mask[None, :, None], -torch.inf).softmax(-1)
        attended = torch.einsum("qnht,ntd->qnhd", attention, memories)
        cue = q[:, None].expand_as(attended)
        pairs = torch.cat((cue, attended, cue * attended, (cue-attended).abs()), -1).flatten(-2)
        for scope in (False, True):
            expected = (model.scope if scope else model.read)(pairs).squeeze(-1)
            torch.testing.assert_close(model.semantic_bank_logits(queries, memories, mask, scope=scope, memory_context_latents=(queries).new_zeros((len(mask), 16))), expected, atol=0, rtol=0)

    def test_nonzero_attention_temperature_has_finite_gradients_and_masked_padding(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.memory_attention_logits.copy_(torch.tensor([0.3, -1.0, 2.0, 4.0]))
            model.scope_memory_attention_logits.copy_(torch.tensor([-0.5, .2, .7, 1.2]))
        queries, memories = torch.randn(2, QUERY), torch.randn(3, 5, WIDTH)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
        changed = memories.clone()
        changed[~mask] = 1e8
        for scope in (False, True):
            score = model.semantic_bank_logits(queries, memories, mask, scope=scope, memory_context_latents=(queries).new_zeros((len(mask), 16)))
            torch.testing.assert_close(model.semantic_bank_logits(queries, changed, mask, scope=scope, memory_context_latents=(queries).new_zeros((len(mask), 16))), score, atol=0, rtol=0)
            score.sum().backward()
        self.assertTrue(torch.isfinite(model.memory_attention_logits.grad).all())
        self.assertTrue((model.memory_attention_logits.grad != 0).all())
        self.assertTrue(torch.isfinite(model.scope_memory_attention_logits.grad).all())
        self.assertTrue((model.scope_memory_attention_logits.grad != 0).all())
        with torch.no_grad():
            model.memory_attention_logits.copy_(torch.tensor([1e6, -1e6, 10, -10]))
        scale = model.attention_scales()
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue(((scale >= torch.exp(torch.tensor(-4.0))) & (scale <= torch.exp(torch.tensor(4.0)))).all())

    def test_current_export_has_exact_tensor_geometry_and_crcs(self):
        model = NeuralSlots()
        self.assertEqual(set(model.state_dict()), set(KEYS))
        self.assertEqual(len(KEYS), 91)
        self.assertEqual(KEYS[19:21], ("identity_match.0.weight", "identity_match.0.bias"))
        self.assertEqual(KEYS[27], "identity_match.2.weight")
        self.assertEqual(KEYS[69], "identity_match.2.bias")
        self.assertEqual(KEYS[71:75], ("plan_boundary.0.weight", "plan_boundary.0.bias",
                                    "plan_boundary.2.weight", "plan_boundary.2.bias"))
        self.assertEqual(KEYS[75], "plan_boundary_raw.weight")
        self.assertEqual(tuple(model.plan_boundary_raw.weight.shape), (2, 3 * INPUT))
        self.assertEqual(KEYS[76:80], ("scope_query_encode.weight", "scope_query_encode.bias",
                                    "scope_query_pool.weight", "scope_memory_attention_logits"))
        self.assertEqual(KEYS[80], "plan_membership_raw.weight")
        self.assertEqual(tuple(model.plan_membership_raw.weight.shape), (1, WIDTH + 3 * INPUT))
        self.assertEqual(float(model.plan_membership_raw.weight.detach().abs().sum()), 0)
        self.assertEqual(KEYS[81:85], ("plan_source_context.0.weight", "plan_source_context.0.bias",
                                    "plan_source_context.2.weight", "plan_source_context.2.bias"))
        self.assertEqual(tuple(model.plan_source_context[0].weight.shape), (SOURCE_CONTEXT_WIDTH, SOURCE_CONTEXT_INPUT))
        self.assertEqual(tuple(model.plan_source_context[0].bias.shape), (SOURCE_CONTEXT_WIDTH,))
        self.assertEqual(tuple(model.plan_source_context[2].weight.shape), (1, SOURCE_CONTEXT_WIDTH))
        self.assertEqual(tuple(model.plan_source_context[2].bias.shape), (1,))
        self.assertGreater(float(model.plan_source_context[0].weight.detach().abs().sum()), 0)
        self.assertEqual(float(model.plan_source_context[2].weight.detach().abs().sum()), 0)
        self.assertEqual(float(model.plan_source_context[2].bias.detach().abs().sum()), 0)
        self.assertEqual(tuple(model.scope_query_encode.weight.shape), (WIDTH, INPUT + 2))
        self.assertEqual(tuple(model.scope_query_encode.bias.shape), (WIDTH,))
        self.assertEqual(tuple(model.scope_query_pool.weight.shape), (RELATION_HEADS, WIDTH))
        self.assertEqual(tuple(model.scope_memory_attention_logits.shape), (RELATION_HEADS,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model"
            export(model, "00" * 32, path)
            raw = path.read_bytes()
            self.assertEqual(raw[:8], b"BNSMODEL")
            self.assertEqual(struct.unpack("<4I", raw[8:24]), (1, INPUT, WIDTH, len(KEYS)))
            offset = 56 + len(KEYS) * 8
            for i, key in enumerate(KEYS):
                size, crc = struct.unpack_from("<II", raw, 56 + i * 8)
                self.assertEqual(size, model.state_dict()[key].numel() * 4)
                self.assertEqual(zlib.crc32(raw[offset : offset + size]), crc)
                offset += size
            self.assertEqual(offset, len(raw))

    def test_zero_writer_edges_preserve_tag_logits_and_static_lattice(self):
        model = NeuralSlots()
        body, context = torch.randn(8, INPUT), torch.randn(5, INPUT)
        encoded, logits, ends = model.plan(body, context, body)
        self.assertTrue(torch.equal(model.plan_boundary_scores(encoded, body), torch.zeros(8, 2)))
        torch.testing.assert_close(logits, model.plan_tags(encoded), atol=0, rtol=0)
        _, static = model.plan_potentials()
        torch.testing.assert_close(model.plan_edge_potentials(ends), static[None].expand(7, -1, -1), atol=0, rtol=0)
        torch.testing.assert_close(model.plan_terminal_potentials(ends), model.plan_end, atol=0, rtol=0)

    def test_zero_context_residual_preserves_parent_and_learns_without_dead_hidden_branch(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model = NeuralSlots().to(dtype=dtype)
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(name.startswith("plan_source_context."))
                with torch.no_grad():
                    model.plan_membership_raw.weight.normal_(0, .03)
                    model.plan_boundary[2].weight.normal_(0, .02)
                body = torch.randn(7, INPUT, dtype=dtype)
                context = torch.randn(4, INPUT, dtype=dtype)
                full, body_only, logits, ends = model.plan_with_encodings(body, context, body)
                parent_encoded = model.plan_encoding(body, context, body)
                boundaries = model.plan_boundary_scores(parent_encoded, body)
                expected = model.plan_tags(parent_encoded)
                expected += F.pad(boundaries[:, :1].expand(-1, 3), (1, 1))
                expected += F.pad(model.plan_membership_scores(parent_encoded, body)[:, None].expand(-1, 4), (1, 0))
                torch.testing.assert_close(full, parent_encoded, atol=0, rtol=0)
                torch.testing.assert_close(logits, expected, atol=0, rtol=0)
                torch.testing.assert_close(ends, boundaries[:, 1], atol=0, rtol=0)
                residual = model.plan_context_residual(full, body_only, body)
                self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
                residual.sum().backward()
                for parameter in model.plan_source_context.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                for parameter in model.plan_source_context[2].parameters():
                    self.assertGreater(float(parameter.grad.abs().sum()), 0)
                for parameter in model.plan_source_context[0].parameters():
                    self.assertEqual(float(parameter.grad.abs().sum()), 0)
                optimizer = torch.optim.SGD(model.plan_source_context.parameters(), lr=.01)
                optimizer.step()
                optimizer.zero_grad()
                model.plan_context_residual(full, body_only, body).sum().backward()
                for parameter in model.plan_source_context[0].parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0)
                self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters()
                                    if not name.startswith("plan_source_context.")))

    def test_body_stream_excludes_context_and_retains_body_summary_and_encoder_bias(self):
        model = NeuralSlots()
        body, context = torch.randn(6, INPUT), torch.randn(4, INPUT)
        full, body_only, _, _ = model.plan_with_encodings(body, context, body)
        other_full, other_body, _, _ = model.plan_with_encodings(body, context * -3, body)
        self.assertFalse(torch.equal(full, other_full))
        torch.testing.assert_close(body_only, other_body, atol=0, rtol=0)
        torch.testing.assert_close(body_only, model.plan_body_encoding(body, body), atol=0, rtol=0)
        changed_body = body.clone()
        changed_body[-1] += 1
        self.assertFalse(torch.equal(body_only, model.plan_body_encoding(changed_body, body)))
        with torch.no_grad():
            model.plan_context.bias.add_(.4)
        self.assertFalse(torch.equal(body_only, model.plan_body_encoding(body, body)))

    def test_active_context_residual_changes_common_source_membership_and_preserves_references(self):
        model = NeuralSlots()
        body, context = torch.randn(7, INPUT), torch.randn(5, INPUT)
        source, kinds = torch.randn(4, INPUT), torch.eye(4)
        with torch.no_grad():
            model.plan_membership_raw.weight.normal_(0, .03)
            model.plan_boundary[2].weight.normal_(0, .02)
        full, body_only, before, before_ends = model.plan_with_encodings(body, context, body)
        references = (model.plan_reference(full.mean(0), source, kinds),
                      model.query_reference(full.mean(0), source, kinds, raw_context=model.query_binding_context(body, context)))
        with torch.no_grad():
            model.plan_source_context[2].weight.normal_(0, .07)
            model.plan_source_context[2].bias.fill_(.5)
        after_full, after_body, after, after_ends = model.plan_with_encodings(body, context, body)
        residual = model.plan_context_residual(full, body_only, body)
        self.assertGreater(float(residual.detach().abs().sum()), 0)
        expected = before + F.pad(residual[:, None].expand(-1, 4), (1, 0))
        torch.testing.assert_close(after, expected, atol=3e-7, rtol=3e-7)
        torch.testing.assert_close(after[:, 0], before[:, 0], atol=0, rtol=0)
        torch.testing.assert_close(after_full, full, atol=0, rtol=0)
        torch.testing.assert_close(after_body, body_only, atol=0, rtol=0)
        torch.testing.assert_close(after_ends, before_ends, atol=0, rtol=0)
        for reference, expected_reference in zip((model.plan_reference, model.query_reference), references):
            for actual, expected_value in zip(reference(after_full.mean(0), source, kinds, **(dict(raw_context=model.query_binding_context(body, context)) if reference == model.query_reference else {})), expected_reference):
                torch.testing.assert_close(actual, expected_value, atol=0, rtol=0)

    def test_nonlinear_context_residual_has_no_frozen_margin_limit(self):
        model = NeuralSlots().double()
        full = torch.zeros(3, WIDTH, dtype=torch.float64)
        body = full.clone()
        body[:, 0] = torch.tensor([1., 2., -1.], dtype=torch.float64)
        words = torch.randn(3, INPUT, dtype=torch.float64)
        with torch.no_grad():
            model.plan_tags.weight.zero_()
            model.plan_membership_raw.weight.zero_()
            model.plan_source_context[0].weight.zero_()
            model.plan_source_context[0].bias.zero_()
            model.plan_source_context[0].weight[0, 0] = 1
            model.plan_source_context[0].weight[1, WIDTH] = 1
            model.plan_source_context[2].weight.zero_()
            model.plan_source_context[2].weight[0, :2] = torch.tensor([5., -2.])
            model.plan_source_context[2].bias.fill_(2.)
        expected = 7 * body[:, 0].tanh() + 2
        torch.testing.assert_close(model.plan_context_residual(full, body, words), expected, atol=1e-14, rtol=1e-14)
        self.assertTrue((expected.abs() > 1).all())
        with torch.no_grad():
            model.plan_tags.weight.normal_()
            model.plan_membership_raw.weight.fill_(100)
        torch.testing.assert_close(model.plan_context_residual(full, body, words), expected, atol=1e-14, rtol=1e-14)
        for malformed_full, malformed_body in ((full[:, :-1], body), (full, body[:-1]), (full[0], body[0])):
            with self.assertRaises(ValueError):
                model.plan_context_residual(malformed_full, malformed_body, words)

    def test_source_context_raw_hidden_channel_is_complete_normalized_and_lexical_independent(self):
        model = NeuralSlots().double()
        full = body = torch.zeros(3, WIDTH, dtype=torch.float64)
        words = torch.zeros(3, INPUT, dtype=torch.float64)
        words[:, HIDDEN - 1] = torch.tensor([3., -3., 0.])
        words[:, 0] = 4
        with torch.no_grad():
            model.plan_source_context[0].weight.zero_()
            model.plan_source_context[0].bias.zero_()
            model.plan_source_context[0].weight[0, 2 * WIDTH + HIDDEN - 1] = 1
            model.plan_source_context[2].weight.zero_()
            model.plan_source_context[2].weight[0, 0] = 5
            model.plan_source_context[2].bias.fill_(2)
        expected = 5 * torch.tensor([.6, -.6, 0.], dtype=torch.float64).tanh() + 2
        torch.testing.assert_close(model.plan_context_residual(full, body, words), expected, atol=1e-14, rtol=1e-14)
        scaled = words.clone()
        scaled[:, :HIDDEN] *= 7
        scaled[:, HIDDEN:] = torch.randn(3, LEXICAL, dtype=torch.float64) * 100
        torch.testing.assert_close(model.plan_context_residual(full, body, scaled), expected, atol=1e-14, rtol=1e-14)
        with self.assertRaises(ValueError):
            model.plan_context_residual(full, body, words[:, :-1])

    def test_writer_edges_can_change_begin_and_end_without_changing_binding_input(self):
        model = NeuralSlots()
        body, context = torch.randn(8, INPUT), torch.randn(5, INPUT)
        before, _, _ = model.plan(body, context, body)
        with torch.no_grad():
            model.plan_boundary[2].weight.normal_(0, 0.1)
            model.plan_boundary[2].bias.copy_(torch.tensor([1.0, -0.5]))
        after, logits, ends = model.plan(body, context, body)
        self.assertTrue(torch.equal(before, after))
        starts = model.plan_boundary_scores(after, body)[:, 0]
        expected = model.plan_tags(after).clone()
        expected[:, 1:4] += starts[:, None]
        torch.testing.assert_close(logits, expected, atol=0, rtol=0)
        target = torch.tensor([0, 1, 4, 4, 0, 2, 4, 0])
        model.sequence_loss(logits, target, ends).backward()
        self.assertTrue(torch.isfinite(model.plan_boundary[2].weight.grad).all())
        self.assertGreater(float(model.plan_boundary[2].weight.grad.abs().sum()), 0)

    def test_writer_edge_probabilities_stay_finite_for_extreme_scores(self):
        ends = torch.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
        continuing, stopping = NeuralSlots.plan_boundary_potentials(ends)
        self.assertTrue(torch.isfinite(continuing).all())
        self.assertTrue(torch.isfinite(stopping).all())
        self.assertEqual(float(continuing[1].detach()), 0)
        self.assertEqual(float(stopping[1].detach()), 0)
        (continuing.sum() + stopping.sum()).backward()
        self.assertTrue(torch.isfinite(ends.grad).all())

    def test_raw_boundary_zero_warm_and_independent_channels(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.plan_boundary[2].weight.normal_(0, 0.1)
        encoded, words = torch.randn(7, WIDTH), torch.randn(7, INPUT)
        original = words.clone()
        padded = F.pad(encoded, (0, 0, 1, 1))
        previous = model.plan_boundary(torch.cat((padded[:-2], encoded, padded[2:]), -1))
        torch.testing.assert_close(model.plan_boundary_scores(encoded, words), previous, atol=0, rtol=0)
        with torch.no_grad():
            model.plan_boundary_raw.weight.normal_(0, 0.1)
        actual = model.plan_boundary_scores(encoded, words)
        changed = words.clone()
        changed[:, HIDDEN:] *= -1
        self.assertFalse(torch.equal(model.plan_boundary_scores(encoded, changed), actual))
        scaled = words.clone()
        scaled[:, :HIDDEN] *= 7
        scaled[:, HIDDEN:] *= 0.25
        torch.testing.assert_close(model.plan_boundary_scores(encoded, scaled), actual, atol=2e-6, rtol=2e-6)
        self.assertTrue(torch.equal(words, original))
        self.assertTrue(torch.isfinite(model.plan_boundary_scores(encoded, torch.zeros_like(words))).all())
        with self.assertRaises(ValueError):
            model.plan_boundary_scores(encoded, words[:-1])

    def test_raw_boundary_has_direct_gradient_without_changing_binding(self):
        model = NeuralSlots()
        for key, parameter in model.named_parameters():
            parameter.requires_grad_(key == "plan_boundary_raw.weight")
        body, context = torch.randn(8, INPUT), torch.randn(5, INPUT)
        original = body.clone()
        before, _, _ = model.plan(body, context, body)
        types = torch.eye(4)[torch.arange(8) % 4]
        bindings = (model.plan_reference(before.mean(0), body, types),
                    model.query_reference(before.mean(0), body, types, raw_context=model.query_binding_context(body, context)))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        target = torch.tensor([0, 1, 4, 4, 0, 2, 4, 0])
        for step in range(2):
            _, logits, ends = model.plan(body, context, body)
            optimizer.zero_grad()
            model.sequence_loss(logits, target, ends).backward()
            gradient = model.plan_boundary_raw.weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
            optimizer.step()
        after, _, _ = model.plan(body, context, body)
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        self.assertTrue(torch.equal(body, original))
        for head, expected in zip((model.plan_reference, model.query_reference), bindings):
            actual = head(after.mean(0), body, types, **(dict(raw_context=model.query_binding_context(body, context)) if head == model.query_reference else {}))
            for old, new in zip(expected, actual):
                torch.testing.assert_close(new, old, atol=0, rtol=0)

    def test_raw_boundary_can_exceed_previous_output_range(self):
        model = NeuralSlots()
        encoded, words = torch.zeros(1, WIDTH), torch.zeros(1, INPUT)
        words[0, 0] = 1
        with torch.no_grad():
            model.plan_boundary_raw.weight[0, INPUT] = 3
            model.plan_boundary_raw.weight[1, INPUT] = -4
        actual = model.plan_boundary_scores(encoded, words)
        torch.testing.assert_close(actual, torch.tensor([[3., -4.]]), atol=0, rtol=0)

    def test_raw_lookahead_changes_endpoint_without_changing_centre_representation(self):
        model = NeuralSlots()
        words, encoded = torch.zeros(3, INPUT), torch.zeros(3, WIDTH)
        words[0, 0], words[1, 1], words[2, 2] = 1, 1, 1
        with torch.no_grad():
            model.plan_boundary_raw.weight[1, 2 * INPUT + 1] = 4
            model.plan_boundary_raw.weight[0, 0] = .7
        actual = model.plan_boundary_scores(encoded, words)
        torch.testing.assert_close(actual[:, 1], torch.tensor([4., 0., 0.]), atol=0, rtol=0)
        self.assertEqual(float(actual[0, 0].detach()), 0)  # Missing previous word.
        changed = words.clone()
        changed[1, 1], changed[1, 2] = 0, 1
        self.assertTrue(torch.equal(changed[0], words[0]))
        self.assertEqual(float(model.plan_boundary_scores(encoded, changed)[0, 1].detach()), 0)
        self.assertTrue(torch.equal(encoded, torch.zeros_like(encoded)))

    def test_current_word_initialization_preserves_the_existing_residual(self):
        model = NeuralSlots()
        words = torch.randn(5, INPUT)
        old_weight = torch.randn(2, INPUT) / 20
        with torch.no_grad():
            model.plan_boundary_raw.weight[:, INPUT:2 * INPUT].copy_(old_weight)
        normalized = torch.cat((F.normalize(words[:, :HIDDEN], dim=-1),
                                F.normalize(words[:, HIDDEN:], dim=-1)), -1)
        expected = F.linear(normalized, old_weight)
        torch.testing.assert_close(model.plan_boundary_scores(torch.zeros(5, WIDTH), words),
                                   expected, atol=2e-6, rtol=2e-6)

    def test_parallel_crf_partition_and_gradients_match_sequential_oracle(self):
        model = NeuralSlots().double()
        for count in (1, 2, 3, 7, 16, 17, 96):
            logits = torch.randn(count, 5, dtype=torch.float64, requires_grad=True)
            ends = torch.randn(count, dtype=torch.float64, requires_grad=True)
            start, _ = model.plan_potentials()
            alpha = start + logits[0]
            for row, edge in zip(logits[1:], model.plan_edge_potentials(ends)):
                alpha = torch.logsumexp(alpha[:, None] + edge, dim=0) + row
            expected = torch.logsumexp(alpha + model.plan_terminal_potentials(ends), 0)
            actual = model.plan_log_partition(logits, ends)
            variables = (logits, ends, model.plan_transition, model.plan_start, model.plan_end)
            old = torch.autograd.grad(expected, variables, retain_graph=True, allow_unused=True)
            new = torch.autograd.grad(actual, variables, allow_unused=True)
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
            for a, b in zip(old, new):
                if a is None:
                    self.assertIsNone(b)
                    continue
                self.assertTrue(torch.isfinite(b).all())
                torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)

    def test_checkpoint_rejects_missing_extra_or_wrong_shape_tensors(self):
        model = NeuralSlots()
        state = dict(model.state_dict())
        require_parameters(model, state)
        missing = dict(state)
        del missing["query_ref_pair.2.bias"]
        extra = dict(state, unused=torch.ones(1))
        shape = dict(state)
        shape["plan_ref_encode.weight"] = torch.zeros(64, 1031)
        old81 = {key: state[key] for key in KEYS[:81]}
        partial83 = {key: state[key] for key in KEYS[:83]}
        partial84 = dict(state)
        del partial84["plan_source_context.2.bias"]
        context_shapes = [dict(state, **{key: torch.zeros(*shape)}) for key, shape in (
            ("plan_source_context.0.weight", (SOURCE_CONTEXT_WIDTH, WIDTH)),
            ("plan_source_context.0.bias", (1,)),
            ("plan_source_context.2.weight", (1, WIDTH)),
            ("plan_source_context.2.bias", (2,)),
        )]
        for bad in (missing, extra, shape, old81, partial83, partial84, *context_shapes):
            with self.assertRaises(ValueError):
                require_parameters(model, bad)

    def test_query_source_learning_does_not_change_write_decoder(self):
        model = NeuralSlots()
        candidate, words, kinds = torch.randn(64), torch.randn(4, INPUT), torch.eye(4)
        before = model.plan_reference(candidate, words, kinds)
        need, tags = model.query_reference(candidate, words, kinds, raw_context=candidate.new_zeros(4 * INPUT))
        torch.nn.functional.cross_entropy(tags, torch.tensor([0, 1, 2, 0])).backward()
        self.assertGreater(float(model.query_ref_encode.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.query_ref_pair[0].weight.grad.abs().sum()), 0)
        for name, p in model.named_parameters():
            if name.startswith(("plan_ref_", "plan_need.", "plan_origin.")):
                self.assertIsNone(p.grad, name)
        with torch.no_grad():
            model.query_ref_pair[2].bias[1].add_(4)
        after = model.plan_reference(candidate, words, kinds)
        for actual, expected in zip(after, before):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_reference_selection_requires_original_source_locations(self):
        model = NeuralSlots()
        with self.assertRaises(ValueError):
            model.query_reference(torch.randn(64), torch.randn(4, INPUT), None, raw_context=torch.zeros(4 * INPUT))

    def test_plan_rejects_orphan_inside_tag(self):
        units = [(0, 4), (5, 9), (10, 15), (16, 20)]
        self.assertEqual(decode_plan([4, 1, 4, 0], units), [(5, 15, 1, 1, 3)])

    def test_padded_bank_matches_individual_slot_scores(self):
        model = NeuralSlots()
        tokens, words = torch.randn(6, INPUT) * 0.05, torch.randn(3, INPUT) * 0.05
        a = model.prepare((tokens, words))
        b = model.prepare((tokens[:2], words[:1]))
        bank, ids, mask = pack_memories([a, b])
        meta = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        for scope in (False, True):
            q, qi = model.query((tokens, words), scope=scope)
            actual = model.score_bank(
                q[None], qi[None], bank, ids, meta, mask, scope=scope
            , memory_context_latents=(q[None]).new_zeros((len(mask), 16)))[0]
            expected = torch.stack(
                [
                    model.score(q, qi, *a, meta[0], scope=scope, memory_context_latents=(a[0]).new_zeros(16)),
                    model.score(q, qi, *b, meta[1], scope=scope, memory_context_latents=(b[0]).new_zeros(16)),
                ]
            )
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_identity_word_units_preserve_subtoken_byte_mass(self):
        x = torch.randn(5, INPUT)
        bounds = [(0, 1), (1, 5), (5, 6), (6, 8), (8, 10)]
        words, units = identity_word_features("Soren live", dict(x=x, bounds=bounds))
        self.assertEqual(units, [(0, 5), (6, 10)])
        torch.testing.assert_close(words[0], (x[0] + 4 * x[1]) / 5)

    def test_semantic_bank_logits_ignore_padding_and_intent(self):
        model = NeuralSlots()
        queries, bank = torch.randn(2, QUERY), torch.randn(3, 4, WIDTH)
        mask = torch.tensor([[True, True, False, False], [True, True, True, True], [True, False, False, False]])
        changed_queries, changed_bank = queries.clone(), bank.clone()
        changed_queries[:, SEMANTIC:] = 1000
        changed_bank[~mask] = 1000
        for scope in (False, True):
            expected = model.semantic_bank_logits(queries, bank, mask, scope=scope, memory_context_latents=(queries).new_zeros((len(mask), 16)))
            actual = model.semantic_bank_logits(changed_queries, changed_bank, mask, scope=scope, memory_context_latents=(changed_queries).new_zeros((len(mask), 16)))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_query_encoder_learning_is_independent_of_memory_encoder(self):
        model = NeuralSlots()
        features = (torch.randn(5, INPUT), torch.randn(3, INPUT))
        q, _ = model.query(features)
        self.assertEqual(q.shape, (QUERY,))
        q[:SEMANTIC].square().sum().backward()
        self.assertGreater(float(model.query_encode.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.query_pool.weight.grad.abs().sum()), 0)
        self.assertIsNone(model.encode.weight.grad)
        self.assertIsNone(model.scope_query_encode.weight.grad)
        self.assertIsNone(model.scope_query_pool.weight.grad)
        self.assertEqual(float(model.intent.weight.grad.abs().sum()), 0)
        self.assertNotIn("query_encode.weight", QUERY_KEYS)
        self.assertNotIn("query_pool.weight", QUERY_KEYS)
        self.assertEqual(len(QUERY_KEYS), 13)
        model.zero_grad(set_to_none=True)
        encoded, _ = model.prepare(features)
        encoded.square().sum().backward()
        self.assertGreater(float(model.encode.weight.grad.abs().sum()), 0)
        self.assertIsNone(model.query_encode.weight.grad)

    def test_scope_query_copy_preserves_warm_math_and_is_independent_afterward(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.memory_attention_logits.copy_(torch.tensor([.2, -.4, .7, 1.1]))
            model.scope_query_encode.load_state_dict(model.query_encode.state_dict())
            model.scope_query_pool.load_state_dict(model.query_pool.state_dict())
            model.scope_memory_attention_logits.copy_(model.memory_attention_logits)
        features = (torch.randn(6, INPUT), torch.randn(3, INPUT))
        read_q, read_id = model.query(features)
        scope_q, scope_id = model.query(features, scope=True)
        torch.testing.assert_close(scope_q, read_q, atol=0, rtol=0)
        torch.testing.assert_close(scope_id, read_id, atol=0, rtol=0)
        torch.testing.assert_close(model.attention_scales(scope=True), model.attention_scales(), atol=0, rtol=0)
        self.assertNotEqual(model.scope_query_encode.weight.data_ptr(), model.query_encode.weight.data_ptr())
        self.assertNotEqual(model.scope_query_pool.weight.data_ptr(), model.query_pool.weight.data_ptr())
        memories = torch.randn(3, 4, WIDTH)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        cue = read_q[:SEMANTIC].reshape(1, RELATION_HEADS, WIDTH)
        attention = (torch.einsum("qhd,ntd->qnht", cue, memories) * .125
                     * model.attention_scales()[None, None, :, None])
        attention = attention.masked_fill(~mask[None, :, None], -torch.inf).softmax(-1)
        attended = torch.einsum("qnht,ntd->qnhd", attention, memories)
        repeated = cue[:, None].expand_as(attended)
        pairs = torch.cat((repeated, attended, repeated * attended, (repeated-attended).abs()), -1).flatten(-2)
        expected = model.scope(pairs).squeeze(-1)
        actual = model.semantic_bank_logits(scope_q[None], memories, mask, scope=True, memory_context_latents=(scope_q[None]).new_zeros((len(mask), 16)))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with torch.no_grad():
            model.query_encode.weight.add_(.01)
            model.query_pool.weight.add_(.4)
            model.memory_attention_logits.add_(1)
            model.read_update.weight.normal_()
            model.read_update.bias.fill_(4)
        after, identity = model.query(features, scope=True)
        torch.testing.assert_close(after, scope_q, atol=0, rtol=0)
        torch.testing.assert_close(identity, scope_id, atol=0, rtol=0)
        torch.testing.assert_close(model.semantic_bank_logits(after[None], memories, mask, scope=True, memory_context_latents=(after[None]).new_zeros((len(mask), 16))),
                                   actual, atol=0, rtol=0)
        self.assertFalse(torch.equal(model.query(features)[0], read_q))

    def test_scope_query_gradients_use_only_scope_query_and_attention_channels(self):
        model = NeuralSlots()
        features = (torch.randn(6, INPUT), torch.randn(3, INPUT))
        scope_q, _ = model.query(features, scope=True)
        logits = model.semantic_bank_logits(scope_q[None], torch.randn(2, 5, WIDTH),
                                            torch.ones(2, 5, dtype=torch.bool), scope=True, memory_context_latents=(scope_q[None]).new_zeros((len(torch.ones(2, 5, dtype=torch.bool)), 16)))
        logits.sum().backward()
        for parameter in (model.scope_query_encode.weight, model.scope_query_pool.weight,
                          model.scope_memory_attention_logits):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        for parameter in (model.query_encode.weight, model.query_pool.weight,
                          model.memory_attention_logits, model.encode.weight):
            self.assertIsNone(parameter.grad)

    def test_query_heads_can_pool_different_tokens_and_mask_padding(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.query_pool.weight.zero_()
            model.query_pool.weight[:, :RELATION_HEADS] = 30 * torch.eye(RELATION_HEADS)
        encoded = torch.zeros(1, RELATION_HEADS + 1, WIDTH)
        encoded[0, :RELATION_HEADS, :RELATION_HEADS] = torch.eye(RELATION_HEADS)
        encoded[0, -1] = 1000
        mask = torch.tensor([[True] * RELATION_HEADS + [False]])
        pooled = model.pool_query(encoded, mask).reshape(RELATION_HEADS, WIDTH)
        self.assertEqual(pooled.argmax(-1).tolist(), list(range(RELATION_HEADS)))
        torch.testing.assert_close(pooled[:, :RELATION_HEADS], torch.eye(RELATION_HEADS), atol=1e-6, rtol=0)

    def test_relation_heads_attend_different_slot_tokens(self):
        model = NeuralSlots()
        query = torch.zeros(1, QUERY)
        query[0, :SEMANTIC].reshape(RELATION_HEADS, WIDTH)[:, :RELATION_HEADS] = 100 * torch.eye(RELATION_HEADS)
        memory = torch.zeros(1, RELATION_HEADS, WIDTH)
        memory[0, :, :RELATION_HEADS] = 8 * torch.eye(RELATION_HEADS)
        captured = []
        hook = model.read[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        model.semantic_bank_logits(query, memory, torch.ones(1, RELATION_HEADS, dtype=torch.bool), memory_context_latents=(query).new_zeros((len(torch.ones(1, RELATION_HEADS, dtype=torch.bool)), 16)))
        hook.remove()
        pairs = captured[0].reshape(1, 1, RELATION_HEADS, 4 * WIDTH)
        attended = pairs[0, 0, :, WIDTH:2 * WIDTH]
        self.assertEqual(attended.argmax(-1).tolist(), list(range(RELATION_HEADS)))
        torch.testing.assert_close(attended[:, :RELATION_HEADS], 8 * torch.eye(RELATION_HEADS), atol=1e-6, rtol=0)

    def test_identity_update_is_learned_and_normalizes_both_blocks(self):
        model = NeuralSlots()
        original_q, current_q = torch.zeros(QUERY), torch.ones(QUERY)
        original, current = torch.zeros(IDENTITY), torch.zeros(IDENTITY)
        original[[0, LEXICAL]] = 1
        current[[1, LEXICAL + 1]] = 1
        gate = torch.sigmoid(torch.tensor(-4.0))
        mixed = (1 - gate) * original + gate * current
        expected = torch.cat((torch.nn.functional.normalize(mixed[:LEXICAL], dim=-1),
                              torch.nn.functional.normalize(mixed[LEXICAL:], dim=-1)))
        actual = model.advance_identity(original_q, current_q, original, current)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for block in (actual[:LEXICAL], actual[LEXICAL:]):
            torch.testing.assert_close(block.norm(), torch.tensor(1.0))
        actual[[1, LEXICAL + 1]].sum().backward()
        self.assertGreater(float(model.identity_update.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.identity_update.bias.grad.abs().sum()), 0)
        with torch.no_grad():
            model.identity_update.bias.fill_(20)
        changed = model.advance_identity(original_q, current_q, original, current)
        torch.testing.assert_close(changed, current, atol=1e-7, rtol=0)
        empty = model.advance_identity(original_q, current_q, torch.zeros_like(original), torch.zeros_like(current))
        self.assertTrue(torch.isfinite(empty).all())
        self.assertEqual(float(empty.detach().abs().sum()), 0)

    def test_identity_matcher_uses_both_cosines_and_preserves_batch_semantics(self):
        model = NeuralSlots()
        queries = torch.randn(3, IDENTITY)
        memories = torch.randn(4, IDENTITY)
        for values in (queries, memories):
            values[:, :LEXICAL] = torch.nn.functional.normalize(values[:, :LEXICAL], dim=-1)
            values[:, LEXICAL:] = torch.nn.functional.normalize(values[:, LEXICAL:], dim=-1)
        batch = model.identity_logits(queries, memories)
        scalar = torch.stack([torch.stack([model.identity_logits(q, m) for m in memories]) for q in queries])
        torch.testing.assert_close(batch, scalar, atol=1e-7, rtol=1e-6)
        batch.square().sum().backward()
        self.assertGreater(float(model.identity_match[0].weight.grad[:, 0].abs().sum()), 0)
        self.assertGreater(float(model.identity_match[0].weight.grad[:, 1].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
