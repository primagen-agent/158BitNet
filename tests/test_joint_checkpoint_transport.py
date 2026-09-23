import tempfile
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from joint_memory_model import create_joint_model
from joint_optimizer import save_checkpoint, tensor_digest, load_checkpoint
from joint_checkpoint_transport import restore_certified_initial


class JointCheckpointTransportTests(unittest.TestCase):
    def fresh(self):return create_joint_model('fixture',(0,),'joint_aux',hidden=8,vocab=13,layers=2,heads=2)

    def test_certified_initial_replaces_cross_domain_roundoff_without_relaxing_final_checks(self):
        model=self.fresh();binding={'fixture':True}
        with torch.no_grad():model.reader.slot_queries.add_(1e-7)
        initial=tensor_digest(model.state_dict());opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'initial.pt';cert=save_checkpoint(path,model,opt,0,binding,initial)
            local=self.fresh()
            with self.assertRaises(ValueError):load_checkpoint(path,expected_sha256=cert['sha256'],model=local,expected_binding=binding,expected_initial_digest=initial)
            restore_certified_initial(path,expected_sha256=cert['sha256'],model=local,binding=binding,initial_digest=initial)
            self.assertEqual(tensor_digest(local.state_dict()),initial)

    def test_final_checkpoint_cannot_impersonate_initial_and_failure_is_nonmutating(self):
        model=self.fresh();binding={'fixture':True};initial=tensor_digest(model.state_dict())
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'final.pt';cert=save_checkpoint(path,model,opt,50,binding,initial)
            for sha in (cert['sha256'],'bad'):
                local=self.fresh();before=tensor_digest(local.state_dict())
                with self.assertRaises(ValueError):restore_certified_initial(path,expected_sha256=sha,model=local,binding=binding,initial_digest=initial)
                self.assertEqual(tensor_digest(local.state_dict()),before)


if __name__=='__main__':unittest.main()
