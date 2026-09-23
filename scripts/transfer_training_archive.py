"""Upload a hashed artifact through an already authenticated tmux SSH pane.

No credentials, new network listener or public storage. Receiver restores TTY
mode after completion/failure. Sender waits for every chunk acknowledgment.
"""
import argparse
import base64
import hashlib
from pathlib import Path
import shlex
import subprocess
import time
import uuid


RECEIVER = r'''
import base64,hashlib,os,select,socket,sys,termios,time,tty
dest,size,expected,nonce,host=sys.argv[1:]
size=int(size)
if socket.gethostname()!=host or not dest.startswith('/home/ubuntu/158bitnet-locomo/build/') or '..' in dest.split('/'):
    raise ValueError('wrong training destination')
if os.path.exists(dest): raise FileExistsError(dest)
stream=open(dest+'.partial','xb')
fd=sys.stdin.fileno(); old=termios.tcgetattr(fd); digest=hashlib.sha256()
def emit(text):
    sys.stdout.write(chr(13)+chr(10)+text+chr(13)+chr(10)); sys.stdout.flush()
try:
    tty.setraw(fd); emit('CGREADY '+nonce)
    left=size; sequence=0
    while left:
        count=min(left,262144); length=4*((count+2)//3); buffer=bytearray(); deadline=time.monotonic()+60
        while len(buffer)<length:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([fd],[],[],remaining)[0]: raise TimeoutError('chunk timeout')
            block=os.read(fd,min(65536,length-len(buffer)))
            if not block: raise EOFError('closed transport')
            buffer.extend(block)
        value=base64.b64decode(buffer,validate=True)
        if len(value)!=count: raise ValueError('chunk length')
        stream.write(value); digest.update(value); left-=count
        emit('CGACK '+nonce+' '+str(sequence)+' '+hashlib.sha256(value).hexdigest()); sequence+=1
    stream.flush(); os.fsync(stream.fileno()); stream.close()
    if digest.hexdigest()!=expected: raise ValueError('artifact hash mismatch')
    os.link(dest+'.partial',dest); os.unlink(dest+'.partial')
    emit('CGDONE '+nonce+' '+expected)
except Exception as error:
    emit('CGFAIL '+nonce+' '+type(error).__name__)
    raise
finally:
    stream.close(); termios.tcflush(fd,termios.TCIFLUSH); termios.tcsetattr(fd,termios.TCSADRAIN,old)
'''


def capture(target):
    return subprocess.check_output(['tmux','capture-pane','-J','-p','-t',target,'-S','-12'],text=True)


def wait_marker(target,marker,nonce,timeout=70):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        text=capture(target)
        if 'CGFAIL '+nonce in text: raise RuntimeError('receiver failed; partial artifact retained')
        if marker in text: return
        time.sleep(.05)
    raise TimeoutError('missing ACK; receiver will time out and restore terminal')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file'); parser.add_argument('--destination',required=True)
    parser.add_argument('--target',default='test'); parser.add_argument('--host',default='pcm-6cb311adc5a8')
    args=parser.parse_args()
    source=Path(args.file); digest=hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    checksum=digest.hexdigest(); nonce=uuid.uuid4().hex
    encoded=base64.b64encode(RECEIVER.encode()).decode()
    command=shlex.join(['/home/ubuntu/miniforge3/envs/metis/bin/python','-c',
                        'import base64; exec(base64.b64decode('+repr(encoded)+'))',
                        args.destination,str(source.stat().st_size),checksum,nonce,args.host])
    subprocess.run(['tmux','send-keys','-t',args.target,'-l',command],check=True)
    subprocess.run(['tmux','send-keys','-t',args.target,'Enter'],check=True)
    wait_marker(args.target,'CGREADY '+nonce,nonce)
    total=source.stat().st_size; sent=0
    with source.open('rb') as stream:
        sequence=0
        while True:
            value=stream.read(262144)
            if not value: break
            buffer='cg001-'+nonce
            subprocess.run(['tmux','load-buffer','-b',buffer,'-'],input=base64.b64encode(value),check=True)
            subprocess.run(['tmux','paste-buffer','-d','-b',buffer,'-t',args.target],check=True)
            wait_marker(args.target,'CGACK '+nonce+' '+str(sequence)+' '+hashlib.sha256(value).hexdigest(),nonce)
            sent+=len(value); sequence+=1
            if sequence%64==0 or sent==total: print(f'transferred {sent}/{total} bytes',flush=True)
    wait_marker(args.target,'CGDONE '+nonce+' '+checksum,nonce)
    print('verified SHA256 '+checksum+' destination '+args.destination,flush=True)


if __name__=='__main__':main()
