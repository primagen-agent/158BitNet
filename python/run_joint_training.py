"""Launch exactly the two approved CG-003 arms after evidence validation."""
import argparse
import json
from pathlib import Path

from joint_optimizer import FEATURE_DIGEST, train_candidate
from joint_training_data import JointTrainingPackage
from native_memory_encoder import sha_file
from prepare_neural_memory_protocol import digest
from train_joint_memory import require_joint_launch


def run(a):
    config=json.loads(Path(a.experiment).read_text());repo=Path(__file__).resolve().parents[1]
    base=repo/'training/memory/neural-system'
    def item(path):return {'path':str(path),'sha256':sha_file(path)}
    evidence={'experiment_digest':digest(config),'snapshot':str(Path(a.snapshot).resolve()),
        'snapshot_sha256':sha_file(a.snapshot),'features':str(Path(a.features).resolve()),'gguf':a.gguf,
        'optimizer_preflight':item(a.preflight),
        'baseline':item(base/'reviews/CG-003/baseline.json'),
        'baseline_score':item(base/'reviews/CG-003/baseline-score.json')}
    require_joint_launch(config=config,launch_evidence=evidence)
    package=JointTrainingPackage(a.features,FEATURE_DIGEST)
    audit=json.loads((base/'checks/CG-003-data.json').read_text());root=Path(a.output)
    root.mkdir(parents=True,exist_ok=False)
    with (root/'launch.json').open('x') as f:json.dump(evidence,f,indent=2)
    for arm in ('joint_aux','joint_product'):
        result=train_candidate(package,config,base/'data/JB-001',audit,arm,root/arm,launch_evidence=evidence)
        print('CG003_ARM_DONE '+json.dumps({'arm':arm,'steps':result['optimizer_steps'],'checkpoints':result['checkpoints']}),flush=True)
    print('CG003_TRAINING_DONE total_optimizer_steps=100 automatic_continuation=false',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('experiment','snapshot','features','gguf','preflight','output'):p.add_argument('--'+n,required=True)
    run(p.parse_args())
