"""CG-002 numerical C qualification, not learned recall or model selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from continuous_memory import continuous_logits
from diagnose_continuous_memory import forward
from native_continuous_generation import check_trace
from native_memory_encoder import NativeMemoryEncoder,sha_file
from train_availability_memory import TrainingPackage
from train_role_separated_memory import FEATURE_DIGEST,diagnostic_model,source_identity,qualification_records


def run(a):
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);package=TrainingPackage(a.features,FEATURE_DIGEST)
    encoder=NativeMemoryEncoder(a.gguf,a.encoder_probe)
    if encoder.identity!=package.manifest['encoder_identity']: raise ValueError('encoder identity changed')
    model=diagnostic_model(3180923); results=[]
    for i,record in enumerate(qualification_records(package)):
        state=record['targets']['state_index']
        features,_=package.sample(record,'cpu');ids=tuple(record['prompt_ids']);directory=root/f'case-{i}';directory.mkdir()
        with torch.no_grad():
            prepared=model.prepare(features.source)
            decision=model.decide(features.hidden[:1],prepared)
            content,uncertainty=model.branches(features.hidden[:1],prepared)
            selected=model.selected_residual(features.hidden[:1],decision)
        checks=[]
        for name,delta in (('zero',torch.zeros_like(content)),('content',content),('uncertainty',uncertainty),('predicted',selected)):
            value=forward(a.probe,a.gguf,ids,delta[0].numpy().copy(),directory,name)
            check_trace(directory/(name+'.log'),len(ids))
            if not np.array_equal(value['hidden'],features.hidden[0].numpy()) or not np.array_equal(value['base'],features.base_logits[0].numpy()):
                raise ValueError('C base differs from frozen training domain')
            with torch.no_grad(): expected,correction=continuous_logits(features.base_logits[:1],delta,package.head,package.scale)
            passed=all(np.allclose(value[k],v[0].numpy(),atol=1e-5,rtol=1e-4) for k,v in (('logits',expected),('correction',correction)))
            if name=='zero': passed=passed and np.array_equal(value['logits'],value['base'])
            checks.append({'branch':name,'passed':bool(passed),'max_logit_error':float(np.max(np.abs(value['logits']-expected[0].numpy())))})
        results.append({'id':record['id'],'predicted_route':int(decision.logits.argmax()),'checks':checks})
        print(json.dumps({'state_fixture':state,'passed':all(c['passed'] for c in checks)}),flush=True)
    report={'passed':all(c['passed'] for r in results for c in r['checks']),'cases':results,'optimizer_steps':0,
            'feature_package_digest':FEATURE_DIGEST,'source_sha256':source_identity(),'encoder_identity':encoder.identity,
            'artifact_sha256':{k:sha_file(getattr(a,k)) for k in ('gguf','probe','encoder_probe')},
            'raw_sha256':{str(p.relative_to(root)):sha_file(p) for p in root.rglob('*') if p.is_file()},
            'memory_accuracy_measured':False,'scope':'nonzero numerical branch fixtures, same C backbone and output projection'}
    with (root/'native-certificate.json').open('x') as f:json.dump(report,f,indent=2)
    if not report['passed']: raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','gguf','probe','encoder-probe','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
