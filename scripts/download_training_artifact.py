"""Download a hash-pinned artifact over an existing authenticated tmux pane.

Temporarily pipes the pane's output to a local binary sink. Never overwrites an
existing pipe or artifact, changes TTY settings, or opens a network listener.
The raw transport is retained for audit. The remote command is read-only.
"""
import argparse
import base64
import hashlib
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid


SENDER = r'''
import base64,hashlib,socket,sys,time
path,expected,nonce,host=sys.argv[1:]
if socket.gethostname()!=host or not path.startswith('/home/ubuntu/158bitnet-locomo/build/') or '..' in path.split('/'):
    raise ValueError('wrong training source')
h=hashlib.sha256()
with open(path,'rb') as f:
    for block in iter(lambda:f.read(1048576),b''): h.update(block)
if h.hexdigest()!=expected: raise ValueError('source hash mismatch')
print('CGSTART '+nonce,flush=True)
with open(path,'rb') as f:
    sequence=0
    for block in iter(lambda:f.read(49152),b''):
        print('CGDATA '+nonce+' '+str(sequence)+' '+base64.b64encode(block).decode(),flush=True)
        sequence+=1
        time.sleep(.01)
print('CGEND '+nonce+' '+expected,flush=True)
'''


def recover(raw, target, expected):
    """Re-parse a completed raw stream, including shell bracketed-paste controls."""
    target=Path(target)
    if target.exists(): raise FileExistsError(target)
    partial=target.with_name(target.name+'.recover-partial')
    checksum=hashlib.sha256(); sequence=0; nonce=None; complete=False
    with Path(raw).open('rb') as source, partial.open('xb') as output:
        for raw_line in source:
            line=raw_line.rstrip(b'\r\n').split(b'\r')[-1]
            if nonce is None:
                match=re.fullmatch(rb'CGSTART ([0-9a-f]{32})',line)
                if match: nonce=match[1]
            elif line.startswith(b'CGDATA '+nonce+b' '):
                _,_,index,payload=line.split(b' ',3)
                if int(index)!=sequence: raise ValueError('transport sequence mismatch')
                value=base64.b64decode(payload,validate=True)
                output.write(value); checksum.update(value); sequence+=1
            elif line==b'CGEND '+nonce+b' '+expected.encode():
                complete=True; break
        if not complete or checksum.hexdigest()!=expected: raise ValueError('incomplete/corrupt transport')
        output.flush(); os.fsync(output.fileno())
    os.link(partial,target); partial.unlink()
    return sequence


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--sink':
        with open(sys.argv[2],'xb',buffering=0) as output:
            while True:
                block=os.read(0,65536)
                if not block: break
                output.write(block)
        return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--sha256',required=True)
    p.add_argument('--output',required=True); p.add_argument('--target',default='test')
    p.add_argument('--host',default='pcm-6cb311adc5a8')
    a=p.parse_args()
    if len(a.sha256)!=64 or any(c not in '0123456789abcdef' for c in a.sha256): raise ValueError('SHA256 required')
    target=Path(a.output).resolve(); partial=target.with_name(target.name+'.partial')
    raw=target.with_name(target.name+'.transport')
    if any(x.exists() for x in (target,partial,raw)): raise FileExistsError('download target exists')
    pane=subprocess.check_output(['tmux','display-message','-p','-t',a.target,'#{pane_id} #{pane_pipe}'],text=True).strip().split()
    if len(pane)!=2 or pane[1]!='0': raise ValueError('pane already has an output pipe')
    nonce=uuid.uuid4().hex; digest=hashlib.sha256(); count=0; sequence=0
    sink=shlex.join([sys.executable,str(Path(__file__).resolve()),'--sink',str(raw)])
    subprocess.run(['tmux','pipe-pane','-o','-t',pane[0],sink],check=True)
    try:
        for _ in range(100):
            if raw.exists(): break
            time.sleep(.05)
        if not raw.exists(): raise RuntimeError('transport sink did not start')
        encoded=base64.b64encode(SENDER.encode()).decode()
        command=shlex.join(['/home/ubuntu/miniforge3/envs/metis/bin/python','-c',
                           'import base64; exec(base64.b64decode('+repr(encoded)+'))',a.source,a.sha256,nonce,a.host])
        subprocess.run(['tmux','send-keys','-t',pane[0],'-l',command],check=True)
        subprocess.run(['tmux','send-keys','-t',pane[0],'Enter'],check=True)
        buffer=b''; started=False; deadline=time.monotonic()+90
        with partial.open('xb') as output,raw.open('rb') as transport:
            while True:
                block=transport.read(1048576)
                if block: buffer+=block; deadline=time.monotonic()+90
                else:
                    if time.monotonic()>deadline: raise TimeoutError('download stalled; partial transport retained')
                    time.sleep(.05); continue
                while b'\n' in buffer:
                    line,buffer=buffer.split(b'\n',1); line=line.rstrip(b'\r').split(b'\r')[-1]
                    if line==('CGSTART '+nonce).encode(): started=True; continue
                    if not started: continue
                    if line.startswith(('CGDATA '+nonce+' ').encode()):
                        _,_,index,payload=line.split(b' ',3)
                        if int(index)!=sequence: raise ValueError('transport sequence mismatch')
                        value=base64.b64decode(payload,validate=True)
                        output.write(value); digest.update(value); count+=len(value); sequence+=1
                        if sequence%256==0: print(f'downloaded {count} bytes',flush=True)
                    elif line==('CGEND '+nonce+' '+a.sha256).encode():
                        if digest.hexdigest()!=a.sha256: raise ValueError('download hash mismatch')
                        output.flush(); os.fsync(output.fileno())
                        os.link(partial,target); partial.unlink()
                        print(f'verified {count} bytes SHA256 {a.sha256}',flush=True)
                        return
    finally:
        subprocess.run(['tmux','pipe-pane','-t',pane[0]],check=True)


if __name__=='__main__': main()
