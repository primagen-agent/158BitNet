"""DG-019 TRAINING-ONLY teacher trajectories; never a recall backend.

Gold spans are used only to compile explicitly oracle teacher cursor steps.
Neural forward later receives natural input features and past-prefix C hidden,
not this object, its labels, future tokens or oracle cursor state.
"""
from dataclasses import dataclass,replace

from append_value_transport import compile_append_value
from check_fine_span_interface import positive_bytes
from fine_span_supervision import ByteTargets
from native_memory_encoder import digest
from neural_memory_generation import encode_generation_input,generation_prompt,RESERVED
from value_transport import ActivatedFact,FactPayload,PayloadSnapshot,ReplyBinding,ValueSpan,Origin,require


@dataclass(frozen=True)
class TeacherTrajectory:
    record_id: str
    input_sha256: str
    prompt_ids: tuple
    completion_ids: tuple
    phases: tuple
    idle_targets: tuple
    response: str
    static_targets: ByteTargets
    origin: str = 'teacher_forcing_oracle'

    def __post_init__(self):
        require(self.origin=='teacher_forcing_oracle' and type(self.static_targets) is ByteTargets,'explicit teacher origin required')
        for ids in (self.prompt_ids,self.completion_ids):
            require(type(ids) is tuple and bool(ids) and all(type(t) is int and t>=0 for t in ids),'immutable teacher IDs required')
        require(type(self.phases) is tuple and type(self.idle_targets) is tuple and
                len(self.completion_ids)==len(self.phases)==len(self.idle_targets),'teacher alignment mismatch')
        require(self.static_targets.idle_mode is None,'static loss must not double-count idle modes')
        for phase,label in zip(self.phases,self.idle_targets):
            require(phase in ('GENERATE','START','CONTINUE','END') and
                    label==({'GENERATE':0,'START':1,'CONTINUE':None,'END':0}[phase]),'phase/mode mismatch')
        active=self.static_targets.route==1
        require(self.phases.count('START')==int(active) and self.phases.count('END')==int(active),'missing/duplicate transition')
        if active:
            start=self.phases.index('START');end=self.phases.index('END')
            require(start<end and all(p=='CONTINUE' for p in self.phases[start+1:end]) and
                    all(p=='GENERATE' for p in self.phases[:start]+self.phases[end+1:]),'invalid cursor transition order')
        else:require(all(p=='GENERATE' for p in self.phases),'positive phase without support')

    def prefix(self,position,*,purpose):
        require(purpose=='training_diagnostic','teacher trajectories cannot be used for autonomous inference')
        require(type(position) is int and 0<=position<len(self.completion_ids),'teacher position outside trajectory')
        return self.prompt_ids+self.completion_ids[:position]

    def reference_positions(self):
        return tuple(sorted({0,len(self.phases)//2,len(self.phases)-1}|
                            {i for i,p in enumerate(self.phases) if p!='GENERATE'}))


def compile_teacher(runtime,label,meta,codec,binding,memory_model_sha256,*,purpose,capacity=128):
    require(purpose=='training_diagnostic','gold compiler is training-only')
    require(label['id']==runtime['id']==meta['id'] and label['input_sha256']==digest(runtime) and meta['split']=='train',
            'training-only bound labels required; dev/test rejected')
    require(binding.backbone_sha256==codec.backbone_sha256 and binding.tokenizer_sha256==codec.tokenizer_sha256,'codec binding mismatch')
    response=label['response'];require(type(response) is str and bool(response) and '\0' not in response and
            not any(x in response for x in RESERVED),'invalid teacher reply')
    state={'no_memory_needed':0,'supported':1,'insufficient':2}[label['state']]
    request=encode_generation_input(runtime,codec.tokenizer);initial=request.prompt_token_ids
    parts=[];phases=[]
    def append_text(text,phase):
        if not text:return
        ids=codec.encode_value(text.encode())
        require(b''.join(codec.decode_pieces(ids))==text.encode() and not any(codec.forbidden(t) for t in ids),'unsafe/nonexact teacher segment')
        parts.extend(ids);phases.extend([phase]*len(ids))
    target=ByteTargets(state,None,None,None) if state!=1 else None
    if state==1:
        require(len(runtime['episodes'])==1,'one supplied source required')
        raw=runtime['episodes'][0]['text'].encode();target=replace(positive_bytes(raw,meta),idle_mode=None)
        vs,ve=target.value_bytes;value=raw[vs:ve].decode()
        require(response.count(value)==1,'ambiguous/missing teacher value; do not drop record')
        pre,post=response.split(value);append_text(pre,'GENERATE')
        snapshot=PayloadSnapshot(binding,memory_model_sha256,'teacher',runtime['id'],0,(FactPayload('source',raw),))
        reply=ReplyBinding('teacher',runtime['id'],digest(runtime['context']))
        handle=ActivatedFact(snapshot.digest,reply,ValueSpan(0,vs,ve),1.,Origin.ORACLE_FIXTURE)
        prefix=initial+tuple(parts)
        cursor=compile_append_value(handle,snapshot,reply,prefix,generation_prompt(runtime)+pre,codec,allow_oracle=True)
        first=True
        while not cursor.done:
            token=cursor.select(codec.bos_id,snapshot,reply,prefix,allow_oracle=True)
            cursor=cursor.commit(token,snapshot,reply,prefix,allow_oracle=True)
            parts.append(token);phases.append('START' if first else 'CONTINUE');first=False;prefix+=(token,)
        require(cursor.committed_text==value,'teacher cursor changed value')
        end=len(parts);append_text(post,'GENERATE')
    else:append_text(response,'GENERATE');end=None
    stop=codec.tokenizer.eos();require(codec.decode_pieces((stop,))==(b'<|im_end|>',),'unqualified EOS')
    parts.append(stop);phases.append('GENERATE')
    if end is not None:phases[end]='END'
    require(b''.join(codec.decode_pieces(parts[:-1]))==response.encode(),'full teacher reply bytes changed')
    require(len(initial)+len(parts)<=capacity,'teacher trajectory exceeds capacity; no truncation')
    idle=tuple(None if p=='CONTINUE' else int(p=='START') for p in phases)
    return TeacherTrajectory(runtime['id'],digest(runtime),initial,tuple(parts),tuple(phases),idle,response,target)
