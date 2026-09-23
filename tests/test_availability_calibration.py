"""Calibration plumbing fixtures only; independent reviewer outputs are separate."""
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from calibrate_availability_review import make_pack, score
from neural_memory_protocol import Adjudication, Claim, response_hash
from prepare_availability_curriculum import materialize


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        corpus=Path(self.tmp.name)/"corpus"
        materialize(ROOT/"training/memory/neural-system/data/CG-001-pilot.json",corpus)
        self.fixtures=make_pack(corpus)

    def vote(self,fixture,reviewer,claims=()):
        p=fixture["packet"]
        adjudication=Adjudication(response_hash(p["response"]),reviewer,claims,False,True,False)
        return {"id":p["id"],"packet_sha256":p["packet_sha256"],"adjudication":asdict(adjudication)}

    def test_blind_packet_excludes_gold_and_scenario(self):
        self.assertEqual(len(self.fixtures),16)
        for fixture in self.fixtures:
            self.assertEqual(set(fixture["packet"]),{"id","packet_sha256","context","episodes","response"})
        result=score(self.fixtures,[],["a","b"])
        self.assertEqual((result["total"],result["reviews_complete"],result["matching_authored_case_count"]),(16,0,0))
        self.assertFalse(result["memory_accuracy_measured"])

    def test_disagreement_kept_even_if_both_extract_incorrectly(self):
        f=self.fixtures[0]
        reviews=[self.vote(f,"a"),self.vote(f,"b",(Claim("Other","home_city","OtherCity","current","actual"),))]
        row=score(self.fixtures,reviews,["a","b"])["cases"][0]
        self.assertTrue(row["reviews_complete"])
        self.assertFalse(row["reviewers_agree"])
        self.assertFalse(row["matches_authored_case"])

    def test_unknown_duplicate_and_wrong_hash_votes_rejected(self):
        vote=self.vote(self.fixtures[0],"a")
        for reviews in ([vote,vote],[self.vote(self.fixtures[0],"c")],[{**vote,"packet_sha256":"0"*64}]):
            with self.assertRaises(ValueError): score(self.fixtures,reviews,["a","b"])


if __name__=="__main__":unittest.main()
