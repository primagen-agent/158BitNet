"""DG-008 measurement guard fixtures, not capability evidence."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from diagnose_role_content_path import token_span,validate_control,first_divergence,trace_content,summarize_trace,difference
from memory_role_separated import RoleSeparatedMemory


class ContentPathTests(unittest.TestCase):
    def runtime(self,text):
        return {'context':[{'text':'Where does Ada live?'}],'episodes':[{'role':'user','speaker':'user','text':text}]}

    def test_subject_control_rejects_simultaneous_value_change(self):
        a=self.runtime('Ada lives in Turin.');b=self.runtime('Boris lives in Graz.')
        fa={'subject':'Ada','value':'Turin'};fb={'subject':'Boris','value':'Graz'}
        with self.assertRaisesRegex(ValueError,'confounded'):validate_control(a,b,fa,fb,'subject')
        validate_control(self.runtime('Ada lives in Graz.'),b,{'subject':'Ada','value':'Graz'},fb,'subject')
        validate_control(a,self.runtime('Ada lives in Graz.'),fa,{'subject':'Ada','value':'Graz'},'value')

    def test_other_text_or_query_changes_rejected(self):
        a=self.runtime('Ada lives in Turin.');b=self.runtime('Ada used to live in Graz.')
        with self.assertRaises(ValueError):validate_control(a,b,{'subject':'Ada','value':'Turin'},{'subject':'Ada','value':'Graz'},'value')
        b=self.runtime('Ada lives in Graz.')
        with self.assertRaises(ValueError):validate_control(a,b,{'subject':'Ada','value':'Turin','time':'current'},
                                                           {'subject':'Ada','value':'Graz','time':'2020'},'value')
        b=self.runtime('Ada lives in Graz.');b['context']=[{'text':'Different question'}]
        with self.assertRaises(ValueError):validate_control(a,b,{'subject':'Ada','value':'Turin'},{'subject':'Ada','value':'Graz'},'value')

    def test_spans_cover_split_tokens_and_only_exact_optional_dummy_space(self):
        self.assertEqual(token_span('A in Turin',[b'A',b' in',b' Tu',b'rin'],'Turin'),[2,3])
        self.assertEqual(token_span('小安',[b' ',b'\xe5',b'\xb0\x8f',b'\xe5\xae\x89'],'小安'),[1,2,3])
        for text,pieces,value in [('AA',[b'AA'],'A'),('A',[b'  A'],'A'),('A',[b'B'],'A')]:
            with self.assertRaises(ValueError):token_span(text,pieces,value)

    def test_first_difference_cannot_use_an_identical_or_prefix_only_target(self):
        self.assertEqual(first_divergence([1,2,3],[1,2,4]),2)
        for b in ([1,2,3],[1,2]):
            with self.assertRaises(ValueError):first_divergence([1,2,3],b)

    def test_nonzero_trace_reconstructs_and_attention_annotations_are_post_forward(self):
        torch.manual_seed(4);m=RoleSeparatedMemory(8,2,2)
        with torch.no_grad():m.content.output.weight.normal_(std=.1)
        h=torch.randn(1,8);source=torch.randn(6,16)
        with torch.no_grad():
            with patch.object(m,'decide',side_effect=AssertionError('No state decision on a value-probe prefix')):
                t=trace_content(m,h,source)
            a=summarize_trace(t,[0],[1]);b=summarize_trace(t,[3],[4,5])
            self.assertEqual(a['content_delta_l2'],b['content_delta_l2'])
            self.assertNotIn('state_probabilities',a)
            self.assertGreater(float(t['delta'].norm()),0)
            self.assertEqual(difference(t['delta'],t['delta'])['relative_l2'],0)
            self.assertAlmostEqual(float(t['attention'].sum(-1).mean()),1.,places=6)
            self.assertAlmostEqual(a['value_attention']['uniform_mass'],1/7)


if __name__=='__main__':unittest.main()
