"""Snapshot the full CG-003 source and feature package for a fresh GPU stage."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from joint_optimizer import FEATURE_DIGEST
from joint_training_data import JointTrainingPackage
from native_memory_encoder import sha_file


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    package=JointTrainingPackage(a.features,FEATURE_DIGEST)
    repo=Path(__file__).resolve().parents[1];paths=[repo/n for n in ('CMakeLists.txt','README.md','.gitignore')]
    for folder in ('src','include','python','tools','tests','scripts','cmake','training','examples','.github'):
        base=repo/folder
        if base.exists():paths.extend(f for f in base.rglob('*') if f.is_file() and not f.is_symlink() and
            '__pycache__' not in f.parts and '.git' not in f.parts and f.name!='.DS_Store' and f.suffix not in ('.pyc','.gguf','.pt','.npz','.npy'))
    paths=sorted(set(paths));features=Path(a.features)
    files=sorted(f for f in features.rglob('*') if f.is_file() and not f.is_symlink())
    snapshot={'format':'cg003-training-snapshot-v1','source_sha256':{str(f.relative_to(repo)):sha_file(f) for f in paths},
        'feature_sha256':{str(f.relative_to(features)):sha_file(f) for f in files},'feature_package_digest':FEATURE_DIGEST,
        'external_head_sha256':package.manifest['output_head_sha256']}
    with Path(a.output).open('xb') as out,tarfile.open(fileobj=out,mode='w:gz',compresslevel=1) as tar:
        for f in paths:tar.add(f,arcname='source/'+str(f.relative_to(repo)),recursive=False)
        for f in files:
            if f.name!='output-head.npy':tar.add(f,arcname='features/'+str(f.relative_to(features)),recursive=False)
        raw=(json.dumps(snapshot,indent=2)+'\n').encode();info=tarfile.TarInfo('source-snapshot.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    report={'sha256':sha_file(a.output),'bytes':Path(a.output).stat().st_size,'source_files':len(paths),'feature_files':len(files)}
    with Path(a.output+'.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)
