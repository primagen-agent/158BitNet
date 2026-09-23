"""Complete source snapshot plus small bound tensors; never include GGUF/head."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from native_memory_encoder import sha_file
from prepare_neural_memory_protocol import digest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];paths=[repo/n for n in ('CMakeLists.txt','README.md','.gitignore')]
    for folder in ('src','include','python','tools','tests','scripts','cmake','training','examples','.github'):
        base=repo/folder
        if base.exists():
            paths.extend(f for f in base.rglob('*') if f.is_file() and not f.is_symlink() and '__pycache__' not in f.parts and '.git' not in f.parts
                         and f.name!='.DS_Store' and f.suffix not in ('.pyc','.gguf','.pt','.npz','.npy'))
    paths=sorted(set(paths));fixtures=Path(a.fixtures)
    snapshot={'format':'cg003-platform-source-snapshot-v1','source_sha256':{str(f.relative_to(repo)):sha_file(f) for f in paths},
              'fixture_manifest_digest':digest(json.loads((fixtures/'manifest.json').read_text())),
              'fixture_sha256':{n:sha_file(fixtures/n) for n in ('manifest.json','records.json','data.npz')},'gguf_or_head_included':False}
    with Path(a.output).open('xb') as out,tarfile.open(fileobj=out,mode='w:gz',compresslevel=1) as tar:
        for f in paths:tar.add(f,arcname='source/'+str(f.relative_to(repo)),recursive=False)
        for n in snapshot['fixture_sha256']:tar.add(fixtures/n,arcname='fixtures/'+n,recursive=False)
        raw=(json.dumps(snapshot,indent=2)+'\n').encode();info=tarfile.TarInfo('source-snapshot.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    report={'archive':a.output,'sha256':sha_file(a.output),'bytes':Path(a.output).stat().st_size,'source_files':len(paths),
            'fixture_digest':snapshot['fixture_manifest_digest'],'snapshot_digest':digest(snapshot)}
    with Path(a.output+'.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)
