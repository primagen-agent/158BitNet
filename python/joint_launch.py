"""CG-003 evidence-bound launch checks, separate from mere approval flags."""
import json
from pathlib import Path
import socket

from native_memory_encoder import BACKBONE_SHA256, sha_file
from prepare_neural_memory_protocol import digest

APPROVAL_SHA = '36d92a69dfa153a04d0c34c56dbf3a7dcdc0dd611ca85b0ce0fdf227e5ff942a'
HOST = 'pcm-6cb311adc5a8'
BUDGET_ROOT = Path('/home/ubuntu/158bitnet-locomo/build/cg003-approved-20260919')


def require_launch(*, config=None, launch_evidence=None, **kwargs):
    if not isinstance(config,dict) or not isinstance(launch_evidence,dict):
        raise ValueError('Training blocked: bound launch evidence required')
    from joint_optimizer import validate_config, FEATURE_DIGEST
    from review_joint_generation import BASELINE_SHA
    validate_config(config)
    if config.get('training_approved') is not True or socket.gethostname()!=HOST:
        raise ValueError('Training blocked: approved budget on registered host required')
    repo=Path(__file__).resolve().parents[1]
    approval=repo/'training/memory/neural-system/experiments/CG-003-budget-approval.json'
    if sha_file(approval)!=APPROVAL_SHA:raise ValueError('Training blocked: approval changed')
    if launch_evidence.get('experiment_digest')!=digest(config):raise ValueError('Training blocked: experiment changed')
    snapshot_path=Path(launch_evidence['snapshot']);snapshot=json.loads(snapshot_path.read_text())
    if sha_file(snapshot_path)!=launch_evidence['snapshot_sha256']:raise ValueError('Training blocked: snapshot changed')
    for name,want in snapshot['source_sha256'].items():
        if sha_file(repo/name)!=want:raise ValueError('Training blocked: staged source changed: '+name)
    def report(name):
        item=launch_evidence[name];path=Path(item['path'])
        if sha_file(path)!=item['sha256']:raise ValueError('Training blocked: evidence changed: '+name)
        return json.loads(path.read_text())
    preflight=report('optimizer_preflight')
    if (preflight.get('format')!='cg003-optimizer-preflight-v1' or preflight.get('passed') is not True or
            preflight.get('real_model_optimizer_steps')!=0 or preflight.get('host')!=HOST or
            preflight.get('feature_package_digest')!=FEATURE_DIGEST or
            preflight.get('source_snapshot_sha256')!=sha_file(snapshot_path) or
            preflight.get('source_sha256')!=snapshot['source_sha256']):
        raise ValueError('Training blocked: optimizer qualification missing/stale')
    if [r['arm'] for r in preflight['real_first_batch']]!=['joint_aux','joint_product'] or not all(r['passed'] for r in preflight['real_first_batch']):
        raise ValueError('Training blocked: incomplete real-batch preflight')
    if [r['arm'] for r in preflight['tiny_loops']]!=['joint_aux','joint_product'] or not all(r['passed'] and r['steps_cpu_cuda']==[50,50] for r in preflight['tiny_loops']):
        raise ValueError('Training blocked: incomplete tiny-loop preflight')
    baseline=report('baseline');score=report('baseline_score')
    if launch_evidence['baseline']['sha256']!=BASELINE_SHA or baseline['memory_enabled'] is not False:
        raise ValueError('Training blocked: baseline changed')
    if score['baseline_sha256']!=BASELINE_SHA or score['report_sha256']!=BASELINE_SHA or score['total']!=24 or score['needs_review']!=0:
        raise ValueError('Training blocked: baseline review incomplete')
    for name,want in score['review_sha256'].items():
        if sha_file(repo/name)!=want:raise ValueError('Training blocked: reviewer evidence changed')
    native=json.loads((repo/'training/memory/neural-system/checks/CG-003-native-preflight.json').read_text())
    parity=json.loads((repo/'training/memory/neural-system/checks/CG-003-cuda-preflight.json').read_text())
    if len(native['rows'])!=12 or parity['passed']!=16 or parity['total']!=16:
        raise ValueError('Training blocked: native/CUDA certificates incomplete')
    # Qualifying source modules must match the tested versions, not just filenames.
    for name,want in native['source_sha256'].items():
        if sha_file(repo/'python'/name)!=want:raise ValueError('Training blocked: native source changed')
    if sha_file(launch_evidence['gguf'])!=BACKBONE_SHA256:raise ValueError('Training blocked: GGUF changed')
    from joint_training_data import JointTrainingPackage
    JointTrainingPackage(launch_evidence['features'],FEATURE_DIGEST)
    return {'prerequisites_complete':True,'approval_sha256':APPROVAL_SHA,'budget_root':str(BUDGET_ROOT)}
