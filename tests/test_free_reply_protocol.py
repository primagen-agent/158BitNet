from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from free_reply_protocol import FreeReply,aggregate,item_verdict,packet_sha256,validate_adjudication


def reply(route=2, subject='Brenna', gold='Riga', text="I do not know where Brenna lives.",
          rid='rec-1', question='Where does Brenna live?'):
    return FreeReply(rid,route,'en',question,subject,gold,
                     (1,2,3),(4,5),text,'eos',2,True,True)


def adj(packet_id,pass_uncertain=True,claims=(),narration=False):
    return {'packet_id':packet_id,'reviewer_id':'test',
            'claims':list(claims),
            'acknowledges_missing_evidence':pass_uncertain,
            'natural_reply':not narration,'unsolicited_memory_narration':narration}


class ReplyTests(unittest.TestCase):
    def test_unverified_reply_rejected(self):
        from dataclasses import replace
        r=reply()
        with self.assertRaises(ValueError):replace(r,zero_residual_bitwise=False)
        with self.assertRaises(ValueError):replace(r,decision_matches_argmax=False)
        with self.assertRaises(ValueError):replace(r,positions_c_verified=1)
        with self.assertRaises(ValueError):replace(r,stop_reason='failed')

    def test_packet_is_blind(self):
        p=reply().packet()
        self.assertEqual(set(p),{'id','language','question','reply','truncated'})
        self.assertNotIn('Riga',json_dump(p) if False else str(p))

    def test_packet_id_stable_and_bound(self):
        p=reply().packet();self.assertEqual(packet_sha256(p),packet_sha256(p))


class AdjudicationTests(unittest.TestCase):
    def test_schema_enforced(self):
        with self.assertRaises(ValueError):validate_adjudication({'claims':[]})
        with self.assertRaises(ValueError):validate_adjudication(adj('x'*64,claims=[{'subject':'a'}]))
        validate_adjudication(adj('b'*64))

    def test_uncertainty_verdict(self):
        r=reply(route=2)
        self.assertTrue(item_verdict(r,adj('a'*64))['semantic_pass'])
        bad=adj('a'*64,claims=[{'subject':'Brenna','relation':'home_city','value':'Tokyo'}])
        self.assertFalse(item_verdict(r,bad)['semantic_pass'])
        self.assertFalse(item_verdict(r,adj('a'*64,pass_uncertain=False))['semantic_pass'])

    def test_normal_and_supported_verdicts(self):
        n=reply(route=0,gold='10')
        ok=adj('a'*64,pass_uncertain=False,claims=[{'subject':'sum','relation':'equals','value':'10'}])
        self.assertTrue(item_verdict(n,ok)['semantic_pass'])
        wrong=adj('a'*64,pass_uncertain=False,claims=[{'subject':'sum','relation':'equals','value':'11'}])
        self.assertFalse(item_verdict(n,wrong)['semantic_pass'])
        s=reply(route=1,gold='Riga')
        ok=adj('a'*64,pass_uncertain=False,claims=[{'subject':'Brenna','relation':'home_city','value':'Riga'}])
        self.assertTrue(item_verdict(s,ok)['semantic_pass'])
        self.assertFalse(item_verdict(s,adj('a'*64))['semantic_pass'])  # refusal fails supported


class AggregateTests(unittest.TestCase):
    def test_full_denominator_and_agreement(self):
        rs=[reply(route=0,gold='10'),reply(route=2,rid='rec-2',question='Where does Kai live?')]
        pa=[adj(rs[0].packet()['id'],pass_uncertain=False,claims=[{'subject':'sum','relation':'equals','value':'10'}]),
            adj(rs[1].packet()['id'])]
        pb=[adj(rs[0].packet()['id'],pass_uncertain=False),adj(rs[1].packet()['id'])]  # normal: no claim
        out=aggregate(rs,pa,pb)
        self.assertEqual(out['total'],2)
        self.assertEqual(out['counts'][0]['denominator'],1)
        self.assertEqual(out['counts'][2]['a_pass'],1)
        self.assertEqual(out['counts'][0]['a_pass'],1)
        self.assertEqual(out['counts'][0]['b_pass'],0)

    def test_missing_review_rejected(self):
        r=reply()
        with self.assertRaises(ValueError):aggregate([r],[adj('deadbeef')],[adj(r.packet()['id'])])


def json_dump(x):
    import json;return json.dumps(x)

if __name__=='__main__':unittest.main()
