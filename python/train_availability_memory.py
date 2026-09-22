"""CG-001 frozen-feature GPU trainer. Training is gated; preflight takes no steps."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch

from availability_supervision import ReplyTargets, supervised_loss, teacher_forcing_requests
from continuous_memory import continuous_logits
from memory_availability import AvailabilityMemoryFusion, STATES
from native_memory_encoder import NativeFeatureBank, BACKBONE_SHA256, sha_file
from native_prefix_bank import PrefixBank
from neural_memory_generation import GenerationInput, TEMPLATE_VERSION
from prepare_neural_memory_protocol import digest


SOURCE_FILES = ("python/train_availability_memory.py", "python/memory_availability.py", "python/memory_fusion.py",
                "python/availability_supervision.py", "python/continuous_memory.py", "python/native_prefix_bank.py",
                "python/native_memory_encoder.py", "python/neural_memory_generation.py")


@dataclass(frozen=True)
class ForwardFeatures:
    hidden: torch.Tensor
    base_logits: torch.Tensor
    source: torch.Tensor


def forward_example(module, features, frozen_head, scale):
    if type(features) is not ForwardFeatures: raise ValueError("label-free feature object required")
    if any(t.requires_grad for t in (features.hidden,features.base_logits,features.source,frozen_head)):
        raise ValueError("backbone/head/source inputs must be frozen")
    read=module.inspect(module.layers-1,features.hidden,module.prepare(features.source))
    logits,_=continuous_logits(features.base_logits,read.residual,frozen_head,scale)
    return logits,read.state_logits[:1]


class TrainingPackage:
    def __init__(self, root, expected_digest):
        self.root=Path(root); self.manifest=json.loads((self.root/"manifest.json").read_text())
        m=self.manifest
        if digest(m)!=expected_digest or m["format"]!="availability-training-features-v1" or m["backbone_sha256"]!=BACKBONE_SHA256 or m["template_version"]!=TEMPLATE_VERSION:
            raise ValueError("training feature identity mismatch")
        if m["reference_parity"] != [True]*4: raise ValueError("unqualified native prefix extractor")
        if sha_file(self.root/"records.json")!=m["records_sha256"] or sha_file(self.root/"output-head.npy")!=m["output_head_sha256"]:
            raise ValueError("training records/head corruption")
        self.prefix=PrefixBank(self.root/"prefixes",expected_manifest_sha256=m["prefix_manifest_sha256"])
        self.source=NativeFeatureBank(self.root/"sources",expected_encoder_id=m["encoder_identity"]["encoder_id"],expected_manifest_sha256=m["source_manifest_sha256"])
        head=np.load(self.root/"output-head.npy",allow_pickle=False,mmap_mode="r")
        if head.shape!=(73448,1024) or head.dtype!=np.float32 or not np.isfinite(head).all(): raise ValueError("invalid frozen head")
        self.head=torch.from_numpy(np.array(head,copy=True))
        self.scale=m["logit_scale"]
        self.records=json.loads((self.root/"records.json").read_text())
        if len(self.records)!=768 or len({r["id"] for r in self.records})!=768: raise ValueError("pilot record inventory mismatch")
        if {s:sum(r["split"]==s for r in self.records) for s in ("train","dev")}!={"train":512,"dev":256}:
            raise ValueError("unexpected split inventory")

    def sample(self, record, device):
        data=dict(record["targets"])
        for key in ("prompt_token_ids","completion_token_ids"): data[key]=tuple(data[key])
        target=ReplyTargets(**data)
        request=GenerationInput(tuple(record["prompt_ids"]),tuple(record["source_texts"]))
        if len(request.source_texts)>1: raise ValueError("single supplied source pilot only")
        values=[self.prefix(r.prompt_token_ids) for r in teacher_forcing_requests(request,target)]
        hidden=torch.stack([v[0] for v in values]).to(device)
        base=torch.stack([v[1] for v in values]).to(device)
        source=self.source(request.source_texts[0]).to(device) if request.source_texts else torch.empty(0,2048,device=device)
        return ForwardFeatures(hidden,base,source),target


def source_identity():
    repo=Path(__file__).resolve().parents[1]
    return {p:sha_file(repo/p) for p in SOURCE_FILES}


def preflight(package, config, device):
    torch.manual_seed(config["initial_seed"])
    cpu=AvailabilityMemoryFusion(1024,24,8)
    with torch.no_grad():
        cpu.content.layer_gain[23]=.1
        cpu.uncertainty_output.weight.normal_(std=.001)
    candidate=AvailabilityMemoryFusion(1024,24,8).to(device)
    candidate.load_state_dict(cpu.state_dict())
    head=package.head.to(device)
    records=[next(r for r in package.records if r["targets"]["state_index"]==i) for i in range(3)]
    comparisons=[]
    for record in records:
        features,target=package.sample(record,"cpu")
        transferred=ForwardFeatures(*(t.to(device) for t in (features.hidden,features.base_logits,features.source)))
        with torch.no_grad():
            expected,expected_state=forward_example(cpu,features,package.head,package.scale)
            actual,actual_state=forward_example(candidate,transferred,head,package.scale)
        # Compare state probabilities because structurally masked logits contain -inf.
        logits_match=torch.allclose(expected,actual.cpu(),atol=1e-5,rtol=1e-4)
        states_match=torch.allclose(expected_state.softmax(-1),actual_state.cpu().softmax(-1),atol=1e-5,rtol=1e-4)
        candidate.zero_grad(set_to_none=True)
        logits,state=forward_example(candidate,transferred,head,package.scale)
        loss=supervised_loss(logits,state,target,state_coefficient=config["loss"]["state_coefficient"])
        loss["weighted_total"].backward()
        grads=[p.grad for p in candidate.parameters() if p.grad is not None]
        finite=bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        uncertainty_norm=float(candidate.uncertainty_output.weight.grad.norm())
        comparisons.append({"id":record["id"],"logits_match":bool(logits_match),"states_match":bool(states_match),
                            "max_logit_error":float((expected-actual.cpu()).abs().max()),"finite_gradients":finite,
                            "uncertainty_gradient_norm":uncertainty_norm,"head_gradient_absent":head.grad is None})
    return {"device":str(device),"torch_version":torch.__version__,"cuda_device":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
            "passed":all(c["logits_match"] and c["states_match"] and c["finite_gradients"] and c["uncertainty_gradient_norm"]>0 and c["head_gradient_absent"] for c in comparisons),
            "comparisons":comparisons,"optimizer_steps":0,"source_sha256":source_identity(),"training_approved":False}


def require_training_ready(config, package_digest, preflight_report, baseline_path):
    # A status edit alone is insufficient: artifacts and current source are bound.
    if config.get("prerequisites_complete") is not True or not config.get("source_manifest") or not config.get("launch_command"):
        raise ValueError("CG-001 training blocked: registered launch/source prerequisites incomplete")
    if not preflight_report["passed"] or not preflight_report["device"].startswith("cuda"):
        raise ValueError("CG-001 requires qualified CUDA training")
    if not baseline_path: raise ValueError("fixed free-generation baseline required")
    baseline=json.loads(Path(baseline_path).read_text())
    if baseline.get("feature_package_digest")!=package_digest or baseline.get("step")!=0 or baseline.get("cached_tokens")!=0 or baseline.get("reused_tokens")!=0:
        raise ValueError("baseline provenance/cache mismatch")
    if baseline.get("template_version")!=TEMPLATE_VERSION or len(baseline.get("predictions",[]))!=16 or baseline.get("oracle_answers_used") is not False:
        raise ValueError("incomplete fixed baseline")
    manifest_path=Path(__file__).resolve().parents[1]/"training/memory/neural-system"/config["source_manifest"]
    evidence=json.loads(manifest_path.read_text())
    if evidence.get("source_sha256")!=source_identity() or evidence.get("feature_package_digest")!=package_digest:
        raise ValueError("launch source/features changed")
    if evidence.get("baseline_sha256")!=sha_file(baseline_path) or evidence.get("approved_first_stage_steps")!=50:
        raise ValueError("baseline or bounded launch evidence mismatch")
    if preflight_report.get("source_sha256")!=source_identity(): raise ValueError("stale device preflight")


def evaluate_teacher_forced(module,package,head,config,device):
    rows=[]; module.eval()
    with torch.no_grad():
        for record in package.records:
            if record["split"]!="dev": continue
            features,target=package.sample(record,device)
            logits,state=forward_example(module,features,head,package.scale)
            losses=supervised_loss(logits,state,target,state_coefficient=config["loss"]["state_coefficient"])
            rows.append({"id":record["id"],"state_target":target.state_index,"state_prediction":int(state.argmax(-1)),
                         "generation_nll":float(losses["generation"]),"state_nll":float(losses["state"])})
    return {"measurement":"teacher_forced_not_memory_accuracy","rows":rows,"total":len(rows),
            "state_correct":sum(r["state_target"]==r["state_prediction"] for r in rows),
            "mean_generation_nll":sum(r["generation_nll"] for r in rows)/len(rows)}


def train_first_stage(package,config,device,root,package_digest):
    """Exactly the first registered 50-step interval, then hand off for C evaluation."""
    torch.manual_seed(config["initial_seed"]); rng=random.Random(config["initial_seed"])
    module=AvailabilityMemoryFusion(1024,24,8).to(device)
    head=package.head.to(device); head_version=head._version
    settings=config["optimizer"]
    if (settings["name"],settings["precision"],settings["amp"],settings["tf32"],settings["batch_size"])!=("AdamW","float32",False,False,4):
        raise ValueError("unreviewed optimizer variant")
    optimizer=torch.optim.AdamW(module.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
    train=[r for r in package.records if r["split"]=="train"]
    def checkpoint(step):
        payload={"format":"availability-memory-pilot-v1","step":step,"state_dict":module.state_dict(),
                 "backbone_sha256":BACKBONE_SHA256,"template_version":TEMPLATE_VERSION,
                 "encoder_identity":package.manifest["encoder_identity"],"feature_package_digest":package_digest,
                 "source_sha256":source_identity(),"optimizer_state":optimizer.state_dict(),
                 "torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all() if device.type=="cuda" else [],
                 "python_rng_state":rng.getstate(),"experiment":config,"deployment_approved":False}
        path=root/f"step-{step:06d}.pt"
        with path.open("xb") as stream: torch.save(payload,stream)
        evaluation=evaluate_teacher_forced(module,package,head,config,device)
        with (root/f"step-{step:06d}-teacher-forced.json").open("x") as f: json.dump(evaluation,f,indent=2); f.write("\n")
        print(json.dumps({"step":step,"checkpoint":str(path),"mean_development_nll":evaluation["mean_generation_nll"],
                          "state_correct":evaluation["state_correct"],"state_total":evaluation["total"],"memory_accuracy_measured":False}),flush=True)
    checkpoint(0)
    with (root/"training.jsonl").open("x") as log:
        for step in range(1,51):
            module.train(); optimizer.zero_grad(set_to_none=True)
            batch=rng.sample(train,settings["batch_size"])
            total=0.; diagnostics=[]
            for record in batch:
                features,target=package.sample(record,device)
                logits,state=forward_example(module,features,head,package.scale)
                losses=supervised_loss(logits,state,target,state_coefficient=config["loss"]["state_coefficient"])
                objective=losses["weighted_total"]/(len(batch)*1.5)
                if not torch.isfinite(objective): raise ValueError("nonfinite loss; no optimizer update")
                objective.backward(); total+=float(objective.detach())
                diagnostics.append({"id":record["id"],"nll":float(losses["generation"].detach()),"state_nll":float(losses["state"].detach())})
            norm=torch.nn.utils.clip_grad_norm_(module.parameters(),settings["gradient_clip_norm"],error_if_nonfinite=True)
            optimizer.step()
            if head.grad is not None or head._version!=head_version: raise ValueError("frozen output head changed")
            row={"step":step,"weighted_loss":total,"gradient_norm_before_clip":float(norm),"examples":diagnostics}
            log.write(json.dumps(row)+"\n"); log.flush()
            if step%10==0: print(json.dumps({k:row[k] for k in ("step","weighted_loss","gradient_norm_before_clip")}),flush=True)
    checkpoint(50)
    with (root/"status.json").open("x") as f:
        json.dump({"optimizer_steps":50,"status":"awaiting_fixed_panel_C_generation_and_review","automatic_continuation":False,
                   "memory_accuracy_measured":False,"deployment_approved":False},f,indent=2); f.write("\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("experiment","features","feature-digest","output"): parser.add_argument("--"+name,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--preflight-only",action="store_true"); parser.add_argument("--baseline")
    args=parser.parse_args()
    config=json.loads(Path(args.experiment).read_text())
    if config["id"]!="CG-001" or config["fixed_constraints"]["backbone_sha256"]!=BACKBONE_SHA256: raise ValueError("wrong experiment")
    torch.set_num_threads(4); torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise ValueError("CUDA unavailable; do not silently train locally")
    package=TrainingPackage(args.features,args.feature_digest)
    registered=Path(__file__).resolve().parents[1]/"training/memory/neural-system"/config["data_manifest"]
    if digest(json.loads(registered.read_text()))!=package.manifest["corpus_manifest_sha256"]:
        raise ValueError("feature corpus differs from registered experiment")
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    report=preflight(package,config,device)
    report.update(feature_package_digest=args.feature_digest,experiment_sha256=sha_file(args.experiment))
    with (root/"preflight.json").open("x") as f: json.dump(report,f,indent=2); f.write("\n")
    print(json.dumps(report,indent=2),flush=True)
    if not report["passed"]: raise SystemExit(1)
    if not args.preflight_only:
        require_training_ready(config,args.feature_digest,report,args.baseline)
        train_first_stage(package,config,device,root,args.feature_digest)


if __name__=="__main__":main()
