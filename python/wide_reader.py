"""V3-021: Direct 2048-dim fact/value heads (bypass 64-dim bottleneck).

Architecture change: the fact/value scoring heads now take the FULL 2048-dim
query and source hidden states directly, instead of the lossy 64-dim pooled
projections. This preserves all backbone information needed for subject
identity — the model CAN learn to distinguish "Brenna" from "Caius".

The route/mode heads stay on the 64-dim path (already calibrated from V3-009/
V3-013). Only the fact/value pathway is widened to 2048 dims.
"""
import math
import types
from pathlib import Path

import torch
from torch import nn

from fine_span_reader import FineSpanReader, FineOutput
from joint_span_reader import candidate_spans
from memory_token_read import versions
from value_transport import require


class WideFactValueReader(nn.Module):
    """Reader with 2048-dim fact/value heads for subject-aware selection.

    The route/mode pathway uses the original 64-dim projections (frozen,
    calibrated). The fact/value pathway bypasses the bottleneck by using
    the full 2048-dim query and source hidden states.
    """
    def __init__(self, base_reader: FineSpanReader, hidden_dim=2048):
        super().__init__()
        self.base = base_reader
        self.hidden_dim = hidden_dim
        W = base_reader.width  # 64

        # New wide fact/value heads: take 2048*2 = 4096 dims → score
        # fact head: 4096 → 256 → 1 (fact score per span)
        self.wide_fact = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        # value head: 4096 → 256 → 1 (value score per span, conditioned on fact)
        self.wide_value = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, inputs, layout, prefix, actual_ids):
        """Forward pass: compute spans + wide fact/value scores."""
        device = next(self.parameters()).device
        W = self.base.width
        H = self.hidden_dim

        # Run the base reader to get spans and route/mode (64-dim path)
        base_out = self.base(inputs, layout, prefix, actual_ids)

        # If no spans, return base output (route/mode are still valid)
        if not base_out.spans:
            return base_out

        # Get the full 2048-dim representations for wide heads
        # Query: pool over query positions (simple mean for now)
        with torch.no_grad():
            qrows = self.base.query(inputs.query).tanh()  # [n_q, 64]
            # For the wide path, we need the ORIGINAL 2048-dim features
            # (not the 64-dim projections)
            q_full = inputs.query  # [n_query, 2048]
            q_wide = q_full.mean(dim=0)  # [2048] — mean pooling

            # Source: for each span, get the source features in that span range
            # The SpanFeatures has .source which is [n_source, 2048]
            src_full = inputs.source  # [n_source, 2048]

        # Compute wide fact scores for each span
        n_spans = len(base_out.spans)
        fact_scores = []
        value_scores = []

        for span_start, span_end in base_out.spans:
            # Get source representation for this span (mean of source rows in span)
            # Note: span indices refer to source token positions
            if span_end <= src_full.shape[0]:
                span_src = src_full[span_start:span_end].mean(dim=0)  # [2048]
            else:
                span_src = torch.zeros(H, device=device)

            # Concatenate query + span representations
            wide_input = torch.cat([q_wide, span_src], dim=0)  # [4096]

            # Score this span
            fs = self.wide_fact(wide_input)  # [1]
            vs = self.wide_value(wide_input)  # [1]
            fact_scores.append(fs)
            value_scores.append(vs)

        # Stack and softmax
        fact_logits = torch.stack(fact_scores).squeeze(-1)  # [n_spans]
        value_logits = torch.stack(value_scores).squeeze(-1)  # [n_spans]

        fact_logp = fact_logits.log_softmax(0)  # [n_spans]
        # Value is conditioned on each fact span (simplified: same scores)
        value_logp = value_logits.log_softmax(0).unsqueeze(0).expand(n_spans, -1)  # [n_spans, n_spans]

        # Build a new output with wide fact/value scores
        # Keep route/mode from base output
        temp = FineOutput(
            self.base, inputs, prefix, layout,
            tuple(base_out.spans),
            base_out.route_logits,
            fact_logp,
            value_logp,
            base_out.mode_logits,
            base_out.boundary_scores,  # boundary scores from base
            (),
        )
        return FineOutput(
            self.base, inputs, prefix, layout,
            tuple(base_out.spans),
            base_out.route_logits,
            fact_logp,
            value_logp,
            base_out.mode_logits,
            base_out.boundary_scores,
            versions(self.base, temp.tensors()),
        )

    def predict(self, output):
        """Predict using wide fact/value scores."""
        output.validate(self.base)
        route = int(output.route_logits.argmax())
        if route != 1 or not output.spans or int(output.mode_logits.argmax()) != 1:
            return {'route': route, 'mode': 'generate', 'fact': None, 'value': None,
                    'fact_bytes': None, 'value_bytes': None, 'confidence': 0.}
        joint = output.fact_logp[:, None] + output.value_logp
        f, v = divmod(int(joint.argmax()), len(output.spans))
        fact_span, value_span = output.spans[f], output.spans[v]
        fs, fe = fact_span
        vs, ve = value_span
        confidence = float(
            output.route_logits.softmax(0)[1] *
            output.mode_logits.softmax(0)[1] *
            math.exp(float(fact_logp[f] + value_logp[f, v]))
        )
        return {'route': route, 'mode': 'start',
                'fact': fact_span, 'value': value_span,
                'fact_bytes': (fs, fe), 'value_bytes': (vs, ve),
                'confidence': confidence}

    def parameters(self, recurse=True):
        """Yield wide head parameters + base parameters."""
        for p in self.wide_fact.parameters():
            yield p
        for p in self.wide_value.parameters():
            yield p
        for p in self.base.parameters():
            yield p

    def named_parameters(self, prefix='', recurse=True):
        for name, p in self.wide_fact.named_parameters(prefix='wide_fact'):
            yield name, p
        for name, p in self.wide_value.named_parameters(prefix='wide_value'):
            yield name, p
        for name, p in self.base.named_parameters(prefix='base'):
            yield name, p

    def state_dict(self):
        return {
            'wide_fact': self.wide_fact.state_dict(),
            'wide_value': self.wide_value.state_dict(),
            'base': self.base.state_dict(),
        }

    def load_state_dict(self, sd):
        self.wide_fact.load_state_dict(sd['wide_fact'])
        self.wide_value.load_state_dict(sd['wide_value'])
        self.base.load_state_dict(sd['base'])

    def eval(self):
        self.base.eval()
        return super().eval()

    def train(self, mode=True):
        self.base.train(mode)
        return super().train(mode)

    @property
    def binding(self):
        return self.base.binding

    @property
    def width(self):
        return self.base.width
