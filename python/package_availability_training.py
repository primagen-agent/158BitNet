"""Make a complete code snapshot plus required training artifacts, excluding GGUFs."""
import argparse
import io
import json
from pathlib import Path
import tarfile

from native_memory_encoder import sha_file
from prepare_neural_memory_protocol import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("features","corpus","baseline","output"): parser.add_argument("--"+name,required=True)
    parser.add_argument('--source-only',action='store_true',help='Reuse separately hash-verified input artifacts; archive the entire source snapshot only')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]; features=Path(args.features); output=Path(args.output)
    paths=[repo/"CMakeLists.txt",repo/"README.md",repo/".gitignore"]
    for folder in ("src","include","python","tools","tests","scripts","cmake","training"):
        root=repo/folder
        if root.exists():
            paths.extend(p for p in root.rglob("*") if p.is_file() and not p.is_symlink() and
                         "__pycache__" not in p.parts and p.suffix in (".py",".c",".h",".cpp",".mm",".sh",".cmake",".in",".txt",".md",".json",".jsonl",".S"))
    paths=sorted(set(paths))
    package=json.loads((features/"manifest.json").read_text())
    snapshot={"format":"cg001-source-snapshot-v1","source_sha256":{str(p.relative_to(repo)):sha_file(p) for p in paths},
              "feature_package_digest":digest(package),"baseline_sha256":sha_file(args.baseline),"gguf_included":False,
              "inputs_reused_not_included":args.source_only}
    datafiles=["manifest.json","records.json","output-head.npy","prefixes/manifest.json","sources/manifest.json","sources/features.npz"]
    datafiles += ["prefixes/"+b["path"] for b in json.loads((features/"prefixes/manifest.json").read_text())["batches"]]
    with output.open("xb") as stream, tarfile.open(fileobj=stream,mode="w:gz",compresslevel=1) as archive:
        for p in paths: archive.add(p,arcname="source/"+str(p.relative_to(repo)),recursive=False)
        blob=(json.dumps(snapshot,indent=2)+"\n").encode(); info=tarfile.TarInfo("source-snapshot.json"); info.size=len(blob); archive.addfile(info,io.BytesIO(blob))
        for name in ([] if args.source_only else datafiles):
            p=features/name
            if not p.resolve().is_relative_to(features.resolve()) or p.is_symlink(): raise ValueError("unsafe feature path")
            archive.add(p,arcname="features/"+name,recursive=False)
        for p in ([] if args.source_only else sorted(Path(args.corpus).iterdir())):
            if p.is_file() and not p.is_symlink(): archive.add(p,arcname="corpus/"+p.name,recursive=False)
        if not args.source_only: archive.add(args.baseline,arcname="baseline.json",recursive=False)
    report={"archive":str(output),"archive_sha256":sha_file(output),"archive_bytes":output.stat().st_size,
            "source_file_count":len(paths),"source_snapshot_digest":digest(snapshot),"feature_package_digest":digest(package)}
    with output.with_suffix(output.suffix+".json").open("x") as f: json.dump(report,f,indent=2); f.write("\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
