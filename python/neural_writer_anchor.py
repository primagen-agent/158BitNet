"""Differentiable full-lattice source-plan protection, for fit teachers only."""

import torch


def _maxplus(left, right):
    return (left[..., :, :, None] + right[..., None, :, :]).amax(-2)


def _ordered_scan(matrices, *, suffix=False):
    """Inclusive, order-preserving matrix products in logarithmic launches.

    Prefixes retain the leading incomplete block; suffixes retain the trailing
    one. Matrix multiplication is associative but not commutative, so both
    scans compose the earlier block on the left. No word-by-word device loop
    or device-to-host scalar inspection is needed.
    """
    products = matrices
    stride = 1
    while stride < len(products):
        combined = _maxplus(products[:-stride], products[stride:])
        products = (torch.cat((combined, products[-stride:]), 0) if suffix
                    else torch.cat((products[:stride], combined), 0))
        stride *= 2
    return products


def strongest_wrong_path(model, logits, ends, target):
    """Return the largest score of any legal path different from ``target``.

    Forward max scores include each position's emission. Backward max scores
    include subsequent edges/emissions and the terminal potential. Their sum
    is the maximum full-path score constrained to that position and tag. Every
    wrong path differs from gold at some position, so the maximum non-gold
    max-marginal excludes exactly the one gold path and covers every competitor.

    The source model supplies the unchanged start, dynamic edge and terminal
    math. ``target`` is a same-device integer vector of valid source tags. Max
    ties have ordinary max subgradients; away from ties the result and gradient
    equal exhaustive path maximization. This helper never changes serving.
    """
    if (logits.ndim != 2 or logits.shape[1] != 5 or not len(logits)
            or ends.shape != (len(logits),) or target.shape != (len(logits),)
            or not logits.is_floating_point() or ends.dtype != logits.dtype
            or target.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            or ends.device != logits.device or target.device != logits.device):
        raise ValueError("source anchor requires aligned five-tag logits, ends and integer gold tags")
    start, _ = model.plan_potentials()
    terminal = model.plan_terminal_potentials(ends)
    first = start + logits[0]
    states = torch.arange(5, device=logits.device)
    if len(logits) == 1:
        return (first + terminal).masked_fill(states == target[0], -torch.inf).amax()

    edges = model.plan_edge_potentials(ends) + logits[1:, None, :]
    prefixes = _ordered_scan(edges)
    suffixes = _ordered_scan(edges, suffix=True)
    forward = torch.cat((first[None], (first[None, :, None] + prefixes).amax(-2)), 0)
    backward = torch.cat(((suffixes + terminal[None, None, :]).amax(-1), terminal[None]), 0)
    return (forward + backward).masked_fill(states[None] == target[:, None], -torch.inf).amax()
