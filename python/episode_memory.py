"""Neural episode binding, with source boundaries preserved through readout.

UNVALIDATED DRAFT: follow training/memory/neural-system/PLAN.md before training
or integrating this prototype. It is not an approved general memory schema.

Inference receives only contextual/lexical token vectors. Subject strings,
span labels, relation labels and target event indices are training diagnostics.
"""
import torch
from torch import nn
from torch.nn import functional as F

from memory_fusion import GatedMemoryFusion

READER_FORMAT = "BITNET_EPISODE_BINDING_RESEARCH_V1"


class EpisodeBindingReader(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.owner_role = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))
        self.query_relation = nn.Linear(hidden, hidden, bias=False)
        self.memory_relation = nn.Linear(hidden, hidden, bias=False)
        self.entity_scale = nn.Parameter(torch.tensor(12.))
        self.entity_bias = nn.Parameter(torch.tensor(-6.))
        self.relation_scale = nn.Parameter(torch.tensor(8.))
        self.relation_bias = nn.Parameter(torch.tensor(-2.))

    def address(self, tokens, query=False):
        if tokens.ndim != 2 or tokens.shape[1] != 2 * self.hidden or not len(tokens):
            raise ValueError("invalid episode geometry")
        contextual, lexical = tokens.float().split(self.hidden, dim=-1)
        roles = self.owner_role(contextual).squeeze(-1)
        identity = F.normalize((roles.softmax(0)[:, None] * lexical).sum(0), dim=0)
        projection = self.query_relation if query else self.memory_relation
        relation = F.normalize(projection(contextual.mean(0)), dim=0)
        return identity, relation, roles

    def forward(self, query_tokens, episodes):
        owner, relation, query_roles = self.address(query_tokens, True)
        owners, relations, memory_roles = [], [], []
        for episode in episodes:
            entity, predicate, roles = self.address(episode)
            owners.append(entity); relations.append(predicate); memory_roles.append(roles)
        if not episodes:
            empty = query_tokens.new_empty(0)
            return empty, {"entity": empty, "relation": empty, "query_roles": query_roles, "memory_roles": []}
        entity_logits = F.softplus(self.entity_scale) * (torch.stack(owners) @ owner) + self.entity_bias
        relation_logits = F.softplus(self.relation_scale) * (torch.stack(relations) @ relation) + self.relation_bias
        log_probability = (F.logsigmoid(entity_logits) + F.logsigmoid(relation_logits)).clamp_max(-1e-6)
        score = log_probability - torch.log(-torch.expm1(log_probability))
        return score, {"entity": entity_logits, "relation": relation_logits,
                       "query_roles": query_roles, "memory_roles": memory_roles}


class EpisodeGatedMemoryFusion(GatedMemoryFusion):
    """Use a separately learned episode gate and protect prompt representations.

The reader is frozen only for this component experiment to isolate generation.
Rejected nonempty memory provides a learned NULL signal, never a text answer.
Empty memory is still an exact bypass. No K is supplied or predicted.
"""
    def __init__(self, hidden, layers, heads=8):
        super().__init__(hidden, layers, heads, evidence_gate=False)
        self.reader = EpisodeBindingReader(hidden).requires_grad_(False)
        self.null_episode = nn.Parameter(torch.randn(1, 2 * hidden) * .02)

    def train(self, mode=True):
        super().train(mode); self.reader.eval()
        return self

    def bind_episodes(self, episodes, query_tokens, response_start):
        if response_start < 0:
            raise ValueError("invalid response boundary")
        if not episodes:
            return lambda layer, hidden: hidden
        with torch.no_grad():
            scores, _ = self.reader(query_tokens, episodes)
            active = (scores > 0).nonzero().flatten().tolist()
        prepared = [self.prepare(episodes[i]) for i in active]
        if not prepared:
            prepared = [self.prepare(self.null_episode)]
        def fuse(layer, hidden):
            if len(hidden) <= response_start:
                return hidden
            response = hidden[response_start:]
            # Keep episode token attention separate; token counts cannot give
            # one episode more weight merely because it contains more tokens.
            deltas = [self(layer, response, state) - response for state in prepared]
            changed = response + torch.stack(deltas).mean(0)
            return torch.cat((hidden[:response_start], changed), dim=0)
        return fuse
