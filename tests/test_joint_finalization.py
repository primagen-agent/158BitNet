import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from finalize_joint_evaluation import qualify,select
from review_joint_generation import BASELINE_SHA


class JointFinalizationTests(unittest.TestCase):
    def fixture(self):
        ids=[str(i) for i in range(24)]
        report={'format':'cg003-fixed-native-checkpoint-v1','arm':'joint_aux','step':50,'baseline_sha256':BASELINE_SHA,
            'memory_enabled':True,'route_policy':'predicted_initial_prefill_argmax_fixed_for_reply',
            'max_new_tokens':64,'context_capacity':128,'checkpoint_sha256':'checkpoint','status_sha256':'status',
            'predictions':[{'id':i,'generated_token_ids':[1,2],'backend_cache_policy_verified':True,'every_base_reference_checked':True} for i in ids]}
        audit={'format':'cg003-native-generation-audit-v1','arm':'joint_aux','complete':True,'passed':True,'expected_cases':24,
            'cases':[{'id':i,'passed':True} for i in ids],'positions':48,'checkpoint_sha256':'checkpoint','status_sha256':'status'}
        return report,audit,ids

    def test_complete_matching_audit_required(self):
        report,audit,ids=self.fixture();qualify(report,audit,'joint_aux',ids)
        for key,value in (('complete',False),('passed',False),('positions',47),('checkpoint_sha256','other'),('status_sha256','other')):
            bad=copy.deepcopy(audit);bad[key]=value
            with self.assertRaises(ValueError):qualify(report,bad,'joint_aux',ids)

    def test_panel_missing_or_reordered_rejected(self):
        report,audit,ids=self.fixture()
        audit['cases'].reverse()
        with self.assertRaises(ValueError):qualify(report,audit,'joint_aux',ids)
        report,audit,ids=self.fixture();report['predictions'].pop()
        with self.assertRaises(ValueError):qualify(report,audit,'joint_aux',ids)

    def test_fixed_selection_and_no_promotion_of_failed_candidate(self):
        def score(pairs,grounded,passed=True):return {'pilot_gates_passed':passed,'value_pairs':{'pass':pairs},'denominators':{'grounded':{'pass':grounded}}}
        self.assertIsNone(select({'joint_aux':score(4,8,False),'joint_product':score(4,8,False)}))
        self.assertEqual(select({'joint_aux':score(3,6),'joint_product':score(3,6)}),'joint_aux')
        self.assertEqual(select({'joint_aux':score(3,8),'joint_product':score(4,6)}),'joint_product')
        self.assertEqual(select({'joint_aux':score(4,8,False),'joint_product':score(3,6)}),'joint_product')


if __name__=='__main__':unittest.main()
