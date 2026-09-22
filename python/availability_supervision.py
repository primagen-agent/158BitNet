"""Post-forward supervision and exact full-string C reply token alignment.

Training-only targets are never accepted by the neural forward interface.
Teacher-forced loss is not a memory-accuracy measurement.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from memory_availability import STATES
from neural_memory_generation import GenerationInput, RESERVED, encode_generation_input, generation_prompt
from prepare_neural_memory_protocol import digest


@dataclass(frozen=True)
class ReplyTargets:
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    state_index: int
    sample_weight: float
    response: str

    def __post_init__(self):
        for sequence in (self.prompt_token_ids, self.completion_token_ids):
            if type(sequence) is not tuple or not sequence or any(type(t) is not int or t < 0 for t in sequence):
                raise ValueError("immutable supervised token IDs required")
        if type(self.state_index) is not int or not 0 <= self.state_index < len(STATES): raise ValueError("invalid state target")
        if type(self.sample_weight) not in (float, int) or not math.isfinite(self.sample_weight) or self.sample_weight <= 0:
            raise ValueError("positive finite sample weight required")
        if type(self.response) is not str or not self.response.strip(): raise ValueError("nonempty reference response required")


def encode_reply_targets(runtime, label, tokenizer, *, context_capacity=128):
    if label["id"] != runtime["id"] or label["input_sha256"] != digest(runtime):
        raise ValueError("supervision belongs to different runtime input")
    state = label["state"]
    if state not in STATES: raise ValueError("unknown supervision state")
    if type(label["memory_needed"]) is not bool or type(label["evidence_missing"]) is not bool:
        raise ValueError("explicit annotation flags required")
    expected = "no_memory_needed" if not label["memory_needed"] else "insufficient" if label["evidence_missing"] else "supported"
    if state != expected or (not label["memory_needed"] and label["evidence_missing"]): raise ValueError("inconsistent state annotation")
    if state == "supported" and not runtime["episodes"]: raise ValueError("supported memory requires a source")
    response = label["response"]
    if type(response) is not str or not response.strip() or "\0" in response or any(t in response for t in RESERVED):
        raise ValueError("invalid response text")
    request = encode_generation_input(runtime, tokenizer)
    full = tuple(tokenizer.encode(generation_prompt(runtime) + response + "<|im_end|>", True))
    count = len(request.prompt_token_ids)
    if full[:count] != request.prompt_token_ids: raise ValueError("joint tokenization changed prompt boundary")
    if type(context_capacity) is not int or not count < len(full) <= context_capacity: raise ValueError("supervised context overflow or empty target")
    completion = full[count:]
    pieces = tokenizer.decode_pieces(list(completion))
    if completion[-1] != tokenizer.eos() or pieces[-1] != b"<|im_end|>": raise ValueError("unqualified stop token")
    if b"".join(pieces[:-1]) != response.encode("utf-8"): raise ValueError("reply bytes do not round-trip exactly")
    return ReplyTargets(request.prompt_token_ids, completion, STATES.index(state), label["sample_weight"], response)


def teacher_forcing_requests(request, targets):
    if type(request) is not GenerationInput or type(targets) is not ReplyTargets or request.prompt_token_ids != targets.prompt_token_ids:
        raise ValueError("separate matching forward input and targets required")
    for offset in range(len(targets.completion_token_ids)):
        # This is explicitly teacher forcing for the generation loss only.
        yield GenerationInput(request.prompt_token_ids + targets.completion_token_ids[:offset], request.source_texts)


def supervised_loss(generation_logits, initial_state_logits, targets, *, state_coefficient):
    if type(targets) is not ReplyTargets: raise ValueError("post-forward ReplyTargets required")
    if type(state_coefficient) not in (float, int) or not math.isfinite(state_coefficient) or state_coefficient <= 0:
        raise ValueError("explicit positive state coefficient required")
    if generation_logits.ndim != 2 or generation_logits.shape[0] != len(targets.completion_token_ids):
        raise ValueError("one generation row per completion token including EOS required")
    # Reject a row for every teacher-forced position: classification must use
    # only the initial user-prefill decision, before seeing any answer prefix.
    if initial_state_logits.shape != (1, len(STATES)): raise ValueError("only initial user-prefill state logits allowed")
    if not torch.isfinite(generation_logits).all() or torch.isnan(initial_state_logits).any() or torch.isposinf(initial_state_logits).any():
        raise ValueError("invalid supervised logits")
    if generation_logits.device != initial_state_logits.device: raise ValueError("mixed loss devices")
    if not torch.isfinite(initial_state_logits[0, targets.state_index]): raise ValueError("structurally impossible state target")
    ids = torch.tensor(targets.completion_token_ids, dtype=torch.long, device=generation_logits.device)
    if int(ids.max()) >= generation_logits.shape[1]: raise ValueError("target outside vocabulary")
    generation = F.cross_entropy(generation_logits, ids)
    state = F.cross_entropy(initial_state_logits, torch.tensor([targets.state_index], device=initial_state_logits.device))
    return {"generation": generation, "state": state, "weighted_total": targets.sample_weight * (generation + state_coefficient * state)}
