"""V3-003 model factory: structural-scalars route input + detached mode input.

The pinned FineSpanReader class and file stay untouched. Instances keep the
exact FineSpanReader type (the pinned controller checks `type(model) is
FineSpanReader`) by re-binding forward via MethodType; the route Linear is
replaced with a wider one over concat(q, evidence, structural[3]).
Structural standardization constants are frozen from the 892-item training set
(DG-029 evidence); they are recorded in every artifact that uses this factory.
"""
import math
from pathlib import Path
import types

import torch
from torch import nn

from fine_span_reader import FineSpanReader, FineOutput
from joint_span_reader import candidate_spans
from memory_token_read import versions
from value_transport import require

STRUCT_MU = (0.838565, 136.708527, 0.455858)
STRUCT_SD = (0.368138, 102.237572, 0.264822)


def make_struct_route_reader(key, *, width=64, seed=1018):
    torch.manual_seed(seed)
    model = FineSpanReader(2048, 1024, key, width=width).eval()
    torch.manual_seed(seed + 1)
    model.route = nn.Linear(2 * width + 3, 3)

    def forward(self, inputs, layout, prefix, actual_ids):
        device = next(self.parameters()).device
        for x in (inputs.query, inputs.source):
            require(x.ndim == 2 and x.shape[1] == self.input_size and x.dtype == torch.float32
                    and x.device == device and not x.requires_grad and len(x) <= 512
                    and bool(torch.isfinite(x).all()), 'invalid frozen C features')
        require(len(inputs.query) > 0 and inputs.allowed.shape == (len(inputs.source),)
                and inputs.allowed.device == device, 'invalid structural mask')
        prefix.validate(actual_ids, self.binding, self.hidden_size)
        require(prefix.hidden.device == device, 'mixed prefix device')
        candidates = candidate_spans(inputs.allowed); starts, ends = layout.endpoints()
        spans = tuple((s, e) for s, e in candidates if starts[s] and ends[e - 1] and starts[s][0] < ends[e - 1][-1])
        qrows = self.query(inputs.query).tanh(); q = (self.query_pool(qrows).softmax(0) * qrows).sum(0)
        source = self.source(inputs.source).tanh(); n = len(source)
        offset = self.offset(torch.arange(65, device=device))
        joined = torch.cat((source[:, None].expand(n, 65, self.width),
                            q[None, None].expand(n, 65, self.width),
                            offset[None].expand(n, 65, self.width)), -1)
        scores = self.boundaries(joined).permute(2, 0, 1)
        if spans:
            s = torch.tensor([a for a, b in spans], device=device)
            e = torch.tensor([b for a, b in spans], device=device)
            sums = torch.cat((source.new_zeros(1, self.width), source.cumsum(0)))
            h = self.span(torch.cat((source[s], source[e - 1], (sums[e] - sums[s]) / (e - s)[:, None]), -1)).tanh()
            qq = q.expand_as(h)
            fp = self.fact(torch.cat((qq, h, qq * h, (qq - h).abs()), -1)).flatten().log_softmax(0)
            scores_coarse = (self.fact_value(h) + self.value_query(q)[None]) @ self.value(h).T / math.sqrt(self.width)
            inside = (s[None] >= s[:, None]) & (e[None] <= e[:, None])
            vp = scores_coarse.masked_fill(~inside, -torch.inf).log_softmax(-1)
            evidence = fp.exp() @ h
        else:
            fp = q.new_empty(0); vp = q.new_empty(0, 0); evidence = torch.zeros_like(q)
        structural = torch.tensor(STRUCT_MU, device=device)
        structural_sd = torch.tensor(STRUCT_SD, device=device)
        raw = torch.tensor([float(n > 0), float(len(spans)),
                            float(int(inputs.allowed.sum())) / 32.0], device=device)
        route = self.route(torch.cat((q, evidence, (raw - structural) / structural_sd)).detach())
        if not spans:
            route = route.masked_fill(torch.tensor([False, True, False], device=device), -torch.inf)
        # V3-003: the mode head trains on a detached input (readout-only learning).
        mode = self.mode(torch.cat((prefix.hidden, q, evidence)).detach())
        temporary = FineOutput(self, inputs, prefix, layout, spans, route, fp, vp, mode, scores, ())
        return FineOutput(self, inputs, prefix, layout, spans, route, fp, vp, mode, scores,
                          versions(self, temporary.tensors()))

    model.forward = types.MethodType(forward, model)
    return model
