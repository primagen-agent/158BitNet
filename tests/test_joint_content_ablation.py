from pathlib import Path
import sys
import copy
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from ablate_joint_content import read_content_off,selected_indices
from joint_memory_model import create_joint_model
from joint_optimizer import tensor_digest
from memory_token_read import TokenReadInput
from qualify_joint_native import fixture_parameters
from token_memory_composition import mix_content_copy
from review_content_ablation import qualify_fresh,finish_score
from ablate_joint_content import ORIGINAL_SHA


class JointContentAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.x=TokenReadInput(torch.randn(4,16),torch.randn(5,16),torch.tensor([2,3,4,5,6]),torch.ones(5,dtype=torch.bool),'fixture')
        self.h=torch.randn(2,8);self.b=torch.randn(2,13);self.head=torch.randn(13,8)

    def model(self,arm,route):
        m=create_joint_model('fixture',(0,),arm,hidden=8,vocab=13,layers=2,heads=2)
        fixture_parameters(m,route);return m

    def test_supported_only_uses_unchanged_pointer_and_base(self):
        for arm in ('joint_aux','joint_product'):
            m=self.model(arm,1);s=m.prefill(self.x);before=tensor_digest(m.state_dict())
            p=m.reader.proposal(self.h,self.b,s.core.pointer)
            expected=mix_content_copy(self.b,p.position_probabilities,p.copy_mass,self.x.token_ids)
            ordinary=m.read(self.h,self.b,self.head,4.,s)
            with patch.object(m.roles.content,'residual',side_effect=AssertionError('content used')):
                actual=read_content_off(m,self.h,self.b,self.head,4.,s)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            self.assertFalse(torch.equal(actual,ordinary));self.assertEqual(before,tensor_digest(m.state_dict()))

    def test_other_routes_and_disabled_are_unchanged(self):
        for arm in ('joint_aux','joint_product'):
            for route in (0,2):
                m=self.model(arm,route);s=m.prefill(self.x);expected=m.read(self.h,self.b,self.head,4.,s)
                with patch.object(m.reader,'proposal',side_effect=AssertionError('pointer used')):
                    actual=read_content_off(m,self.h,self.b,self.head,4.,s)
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                self.assertIs(read_content_off(m,self.h,self.b,self.head,4.,s,enabled=False),self.b)

    def test_selection_depends_on_predicted_route_not_outcome(self):
        r={'arm':'joint_aux','step':50,'route_policy':'predicted_initial_prefill_argmax_fixed_for_reply',
           'predictions':[{'selected_route':1 if i<4 else 2,'text':'wrong'} for i in range(24)]}
        self.assertEqual(selected_indices(r,'joint_aux'),[0,1,2,3])
        for row in r['predictions']:row['text']='correct'
        self.assertEqual(selected_indices(r,'joint_aux'),[0,1,2,3])
        r['predictions'][4]['selected_route']=1
        with self.assertRaises(ValueError):selected_indices(r,'joint_aux')

    def test_fresh_subset_and_audit_cannot_shrink_or_change_checkpoint(self):
        old={'arm':'joint_aux','step':50,'checkpoint_sha256':'checkpoint',
            'route_policy':'predicted_initial_prefill_argmax_fixed_for_reply',
            'predictions':[{'id':str(i),'selected_route':1 if i<4 else 2} for i in range(24)]}
        new={'format':'dg013-content-off-generation-v1','arm':'joint_aux','fresh_indices':[0,1,2,3],
            'optimizer_steps':0,'parameters_unchanged':True,'checkpoint_sha256':'checkpoint',
            'original_sha256':ORIGINAL_SHA['joint_aux'],'all_predicted_routes_unchanged':True,
            'max_new_tokens':64,'context_capacity':128,
            'predictions':[{'id':str(i),'generated_token_ids':[1,2]} for i in range(4)]}
        audit={'format':'dg013-content-off-audit-v1','arm':'joint_aux','passed':True,'fresh_cases':4,
            'unchanged_cases_not_rerun':20,'original_sha256':ORIGINAL_SHA['joint_aux'],
            'cases':[{'id':str(i),'passed':True} for i in range(4)],'positions':8}
        self.assertEqual(qualify_fresh(new,audit,old,'joint_aux'),[0,1,2,3])
        for key,value in [('passed',False),('positions',7),('unchanged_cases_not_rerun',0)]:
            bad=copy.deepcopy(audit);bad[key]=value
            with self.assertRaises(ValueError):qualify_fresh(new,bad,old,'joint_aux')
        bad=copy.deepcopy(new);bad['predictions'].pop()
        with self.assertRaises(ValueError):qualify_fresh(bad,audit,old,'joint_aux')

    def test_repetitive_pass_is_rejected_without_hiding_review_disagreement(self):
        score={'cases':[{'id':'a','status':'pass','reasons':[],'scenario':'correct','language':'en','world_id':'x'},
                        {'id':'b','status':'needs_review','reasons':['reviewer_disagreement'],'scenario':'role_swap','language':'en','world_id':'x'}],
               'groups':{'correct/en':{},'role_swap/en':{}},'worlds':{'x':{}}}
        report={'predictions':[{'id':n,'truncated':True,'utf8_complete':True,'generated_token_ids':[1,2,3,4]*3} for n in ('a','b')]}
        s=finish_score(score,report)
        self.assertEqual(s['fail'],1);self.assertEqual(s['needs_review'],1);self.assertFalse(s['adjudication_complete'])
        self.assertEqual(s['grounded']['total'],1);self.assertEqual(s['role_swap_negative']['total'],1)


if __name__=='__main__':unittest.main()
