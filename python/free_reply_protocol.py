"""DG-023 free-reply protocol pieces. Zero updates; never a serving path.

The generation loop itself lives in the check harness (it drives real C
forwards); this module owns the reply record, the blind packet projection,
the adjudication schema and the gold-side aggregation. Reviewers only ever
see packet() output - no routes, labels or gold answers.
"""
from dataclasses import dataclass
import hashlib
import json

from value_transport import require

STOP_REASONS = ('eos', 'budget', 'failed')


@dataclass(frozen=True)
class FreeReply:
    record_id: str
    route_target: int
    language: str
    question_text: str
    question_subject: str
    gold_value: str
    prompt_ids: tuple
    token_ids: tuple
    text: str
    stop_reason: str
    positions_c_verified: int
    zero_residual_bitwise: bool
    decision_matches_argmax: bool

    def __post_init__(self):
        require(self.route_target in (0, 1, 2) and self.stop_reason in STOP_REASONS and
                type(self.token_ids) is tuple and self.positions_c_verified == len(self.token_ids) and
                self.zero_residual_bitwise and self.decision_matches_argmax and self.stop_reason != 'failed',
                'incomplete or unverifiable free reply')

    def packet(self):
        """Blind projection for reviewers: no route, label, gold or prompt ids.

        The trailing end-marker a serving stack strips is not part of the
        reply text a user sees; everything else (including reserved-token
        garbage or truncation) is kept verbatim."""
        text = self.text.removesuffix('<|im_end|>')
        payload = {'language': self.language, 'question': self.question_text,
                   'reply': text, 'truncated': self.stop_reason == 'budget'}
        pid = hashlib.sha256(json.dumps({'record_id': self.record_id, 'language': self.language,
                                         'question': self.question_text, 'reply': text,
                                         'truncated': payload['truncated']},
                                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        return {'id': pid, **payload}


def packet_sha256(packet):
    return hashlib.sha256(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_adjudication(adj):
    require(type(adj) is dict and type(adj.get('packet_id')) is str and len(adj['packet_id']) == 64 and
            type(adj.get('reviewer_id')) is str and bool(adj['reviewer_id']),
            'adjudication identity required')
    require(type(adj.get('claims')) is list and all(
        type(c) is dict and all(type(c.get(k)) is str and bool(c[k]) for k in ('subject', 'relation'))
        and type(c.get('value')) is str
        for c in adj['claims']), 'claims must be subject/relation strings plus a value string '
                                '(an empty value annotates a value-less assertion)')
    require(all(type(adj.get(k)) is bool for k in
                ('acknowledges_missing_evidence', 'natural_reply', 'unsolicited_memory_narration')),
            'boolean adjudication fields required')


def item_verdict(reply, adj):
    """Gold-side scoring, computed only after both blind reviews are in."""
    require(type(reply) is FreeReply and type(adj) is dict, 'typed reply and adjudication required')
    claims = adj['claims']
    asked = [c for c in claims if c['subject'].lower() == reply.question_subject.lower()]
    if reply.route_target == 2:
        correct = adj['acknowledges_missing_evidence'] and not asked
    elif reply.route_target == 0:
        # Ordinary no-memory items on this panel are arithmetic; the answer is
        # the gold value carried by any claim, not a subject attribution.
        correct = any(c['value'].strip() == reply.gold_value for c in claims) and adj['natural_reply']
    else:
        correct = any(c['value'].strip() == reply.gold_value and c['subject'].lower() == reply.question_subject.lower()
                      for c in claims) and adj['natural_reply']
    return {'route_target': reply.route_target, 'semantic_pass': bool(correct),
            'natural_reply': adj['natural_reply'],
            'unsolicited_memory_narration': adj['unsolicited_memory_narration'],
            'acknowledges_missing_evidence': adj['acknowledges_missing_evidence']}


def aggregate(replies, review_a, review_b):
    """Full-denominator aggregation; disagreements recorded, never dropped."""
    require(type(replies) is list and bool(replies) and type(review_a) is list and type(review_b) is list,
            'reviews and reply list required')
    by_pid_a = {a['packet_id']: a for a in review_a}
    by_pid_b = {a['packet_id']: a for a in review_b}
    rows = []
    for r in replies:
        p = r.packet()
        require(p['id'] in by_pid_a and p['id'] in by_pid_b, 'missing adjudication in a reviewer')
        va, vb = item_verdict(r, by_pid_a[p['id']]), item_verdict(r, by_pid_b[p['id']])
        rows.append({'packet_id': p['id'], 'record_id': r.record_id, 'route_target': r.route_target,
                     'reviewer_a': va, 'reviewer_b': vb,
                     'agree': va['semantic_pass'] == vb['semantic_pass']})
    counts = {t: {'denominator': sum(1 for x in rows if x['route_target'] == t),
                  'a_pass': sum(1 for x in rows if x['route_target'] == t and x['reviewer_a']['semantic_pass']),
                  'b_pass': sum(1 for x in rows if x['route_target'] == t and x['reviewer_b']['semantic_pass']),
                  'agreement': sum(1 for x in rows if x['route_target'] == t and x['agree'])}
              for t in (0, 1, 2)}
    return {'rows': rows, 'counts': counts,
            'truncated': sum(1 for r in replies if r.stop_reason == 'budget'),
            'total': len(replies)}
