"""Fixed denominators and fail-closed launch checks for CG-003."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from review_joint_generation import apply_gates, repeated_block
from joint_launch import require_launch

ROOT=Path(__file__).resolve().parents[1]/'training/memory/neural-system'


class JointReviewLaunchTests(unittest.TestCase):
    def setUp(self):
        self.base=json.loads((ROOT/'reviews/CG-003/baseline.json').read_text())
        self.score=json.loads((ROOT/'reviews/CG-003/baseline-score.json').read_text())

    def test_baseline_has_full_denominators_and_no_promotion(self):
        score=apply_gates(self.score,self.base,self.base,self.base['panel'])
        self.assertEqual(score['total'],24);self.assertEqual(score['pass'],8)
        self.assertEqual(score['denominators']['grounded']['pass'],0)
        self.assertEqual(score['denominators']['normal']['pass'],8)
        self.assertEqual(score['value_pairs']['total'],4)
        self.assertFalse(score['pilot_gates_passed']);self.assertFalse(score['promotion_eligible'])

    def test_repetition_requires_three_adjacent_blocks_at_least_four_tokens(self):
        self.assertTrue(repeated_block([1,2,3,4]*3))
        self.assertFalse(repeated_block([1,2,3,4]*2))
        self.assertFalse(repeated_block([1,2,3]*3))

    def test_normal_route_changed_fails_even_with_identical_text(self):
        candidate=copy.deepcopy(self.base);candidate['memory_enabled']=True
        for r in candidate['predictions']:r['selected_route']=0
        normal=next(p['id'] for p in candidate['panel'] if p['scenario']=='ordinary_source')
        next(r for r in candidate['predictions'] if r['id']==normal)['selected_route']=1
        score=apply_gates(self.score,candidate,self.base,candidate['panel'])
        self.assertEqual(score['denominators']['normal']['pass'],7)
        self.assertIn(normal,score['mechanically_invalid_ids'])

    def test_unresolved_review_survives_mechanical_failure(self):
        row=self.base['predictions'][0];row['truncated']=True
        case=next(c for c in self.score['cases'] if c['id']==row['id']);case['status']='needs_review'
        score=apply_gates(self.score,self.base,self.base,self.base['panel'])
        self.assertEqual(score['needs_review'],1);self.assertFalse(score['pilot_gates_passed'])

    def test_missing_pair_cannot_shrink_denominator(self):
        self.score['cases'].pop(0)
        with self.assertRaises(ValueError): apply_gates(self.score,self.base,self.base,self.base['panel'])

    def test_user_budget_does_not_bypass_evidence(self):
        config=json.loads((ROOT/'experiments/CG-003.json').read_text())
        self.assertTrue(config['training_approved'])
        with self.assertRaisesRegex(ValueError,'Training blocked'):require_launch(config=config)
        with self.assertRaisesRegex(ValueError,'Training blocked'):require_launch(training_approved=True,prerequisites_complete=True)


if __name__=='__main__':unittest.main()
