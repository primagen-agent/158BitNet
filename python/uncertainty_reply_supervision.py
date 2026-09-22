"""DG-022 training-side supervision of complete uncertainty replies.

Zero optimizer steps. Teacher targets only extend past prefixes and are read
by the loss after each label-free branch forward; they never enter any
neural forward, snapshot, controller or serving path.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from autonomous_value_controller import LiveFrame
from query_only_uncertainty import QueryOnlyUncertainty
from value_transport import require

UNCERTAINTY_ROUTE = 2


def prefix_chain(prompt_ids, target_ids):
    """Exact teacher-forced prefixes: prompt + targets[:i] for every position."""
    require(type(prompt_ids) is tuple and type(target_ids) is tuple and bool(prompt_ids) and bool(target_ids) and
            all(type(t) is int and t >= 0 for t in prompt_ids + target_ids), 'immutable teacher IDs required')
    return tuple(prompt_ids + target_ids[:i] for i in range(len(target_ids)))


@dataclass(frozen=True)
class UncertaintyTrajectory:
    """Route-2 reply trajectory binding actual frames to past-token prefixes."""
    record_id: str
    route_target: int
    prompt_ids: tuple
    target_ids: tuple
    eos_id: int
    max_new_tokens: int
    frames: tuple

    def __post_init__(self):
        require(type(self.record_id) is str and bool(self.record_id) and self.route_target == UNCERTAINTY_ROUTE,
                'uncertainty supervision accepts only ground-truth insufficient routes')
        require(type(self.eos_id) is int and self.eos_id >= 0 and type(self.max_new_tokens) is int and
                self.max_new_tokens > 0, 'explicit EOS and reply budget required')
        require(self.target_ids[-1] == self.eos_id, 'teacher trajectory must end at explicit EOS')
        require(len(self.target_ids) <= self.max_new_tokens,
                'teacher trajectory exceeds reply budget; no truncation')
        require(type(self.frames) is tuple and len(self.frames) == len(self.target_ids), 'one frame per position')
        for frame, prefix in zip(self.frames, prefix_chain(self.prompt_ids, self.target_ids)):
            require(type(frame) is LiveFrame and frame.prefix_ids == prefix,
                    'frame prefix is not the exact teacher-forced prefix')


def trajectory_losses(branch, trajectory, head, scale):
    """Per-position CE of the query-only branch along a teacher trajectory.

    Each branch forward consumes only its frozen frame and head; the target
    token of a position is read strictly after that forward.
    """
    require(type(branch) is QueryOnlyUncertainty and type(trajectory) is UncertaintyTrajectory and
            branch.binding == trajectory.frames[0].binding, 'bound uncertainty branch required')
    losses = []
    for frame, target in zip(trajectory.frames, trajectory.target_ids):
        output = branch(frame, frame.prefix_ids, head, scale)
        losses.append(F.cross_entropy(output.logits[None], torch.tensor([target])))
    return losses


def aggregate_loss(trajectories, branch, head, scale):
    """Equal-weight mean over records of each record's trajectory CE sum."""
    require(type(trajectories) is list and bool(trajectories) and
            all(type(t) is UncertaintyTrajectory for t in trajectories), 'trajectory list required')
    sums = [torch.stack(trajectory_losses(branch, t, head, scale)).sum() for t in trajectories]
    return torch.stack(sums).mean(), sums


def gradient_structure(branch):
    """Classify trajectory-backward gradients; zero-init chain rule is expected, not learning."""
    require(all(p.grad is not None for p in branch.parameters()), 'backward has not run')
    l1 = {n: float(p.grad.abs().sum()) for n, p in branch.named_parameters()}
    require(all(bool(torch.isfinite(p.grad).all()) for p in branch.parameters()), 'nonfinite trajectory gradients')
    return {'gradient_l1': l1,
            'expected_zero_init_chain_rule': l1['output.weight'] > 0 and l1['encoder.weight'] == 0 and l1['encoder.bias'] == 0,
            'connected': all(v > 0 for v in l1.values())}


def disabled_identity(trajectory, branch, head, scale):
    """Disabled branch must return base logits bitwise at every position; never calls the network."""
    for frame in trajectory.frames:
        output = branch(frame, frame.prefix_ids, head, scale, enabled=False)
        require(torch.equal(output.logits.view(torch.int32), frame.logits.view(torch.int32)) and
                int(torch.count_nonzero(output.residual)) == 0, 'disabled path changed base logits')
    return True
