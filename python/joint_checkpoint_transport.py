"""Use the certified step-zero artifact across CPU math/library domains."""
import torch

from joint_optimizer import FORMAT, tensor_digest, load_checkpoint, _finite_tree
from native_memory_encoder import BACKBONE_SHA256, sha_file
from neural_memory_generation import TEMPLATE_VERSION


def restore_certified_initial(path, *, expected_sha256, model, binding, initial_digest):
    if sha_file(path)!=expected_sha256:raise ValueError('initial artifact hash mismatch')
    data=torch.load(path,map_location='cpu',weights_only=True)
    if (data['format']!=FORMAT or type(data['step']) is not int or data['step']!=0 or
            data['route_policy']!=model.route_policy or data['binding']!=binding or
            data['initial_state_digest']!=initial_digest or data['state_digest']!=initial_digest or
            data['backbone_sha256']!=BACKBONE_SHA256 or data['template_version']!=TEMPLATE_VERSION or
            data['deployment_approved'] is not False or data['resume_allowed'] is not False):
        raise ValueError('uncertified initial artifact')
    state=data['state_dict'];current=model.state_dict()
    if (state.keys()!=current.keys() or any(state[n].shape!=v.shape or state[n].dtype!=v.dtype or
        not torch.isfinite(state[n]).all() for n,v in current.items()) or tensor_digest(state)!=initial_digest):
        raise ValueError('initial tensor inventory or digest mismatch')
    if data['optimizer_parameter_names']!=[n for n,p in model.named_parameters() if p.requires_grad] or data['optimizer_state']['state']:
        raise ValueError('initial artifact has updates or different trainable parameters')
    _finite_tree(data['optimizer_state'])
    # No tolerance-based repair and no final-weight-as-initial shortcut: restore
    # exact server initialization bytes only after checking its separately pinned SHA.
    model.load_state_dict(state,strict=True)
    return load_checkpoint(path,expected_sha256=expected_sha256,model=model,
                           expected_binding=binding,expected_initial_digest=initial_digest)
