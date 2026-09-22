"""Label-free generation transport for the research continuous branch.

No neural backend, training or semantic scoring is implemented here. Source
messages go to the separate memory encoder, never into the decoder prompt.
"""
from dataclasses import dataclass
import math

from episode_memory_inputs import encoder_texts, validate_input


SYSTEM_TEXT = "Reply briefly in plain text. Do not use tools."
# v1 remains explicit for reproducing registered diagnostics. New research
# generation follows the exact ChatML suffix declared by the bound GGUF.
TEMPLATE_VERSION = "gguf-chatml-v2"
LEGACY_TEMPLATE_VERSION = "research-no-think-v1"
ASSISTANT_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n"
RESERVED = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")


@dataclass(frozen=True)
class GenerationInput:
    prompt_token_ids: tuple[int, ...]
    source_texts: tuple[str, ...]

    def __post_init__(self):
        if (type(self.prompt_token_ids) is not tuple or not self.prompt_token_ids or
                any(type(t) is not int or t < 0 for t in self.prompt_token_ids)):
            raise ValueError("immutable token IDs required")
        if type(self.source_texts) is not tuple or any(type(s) is not str or not s for s in self.source_texts):
            raise ValueError("immutable source texts required")


def generation_prompt(record, *, template_version=TEMPLATE_VERSION):
    validate_input(record)
    if template_version not in (TEMPLATE_VERSION, LEGACY_TEMPLATE_VERSION):
        raise ValueError("unsupported generation template identity")
    # Do not silently collapse multi-party identity into a bare ChatML role.
    for message in record["context"] + record["episodes"]:
        if message["speaker"] != message["role"] or message["role"] not in ("user", "assistant"):
            raise ValueError("unsupported speaker/role in generation protocol")
        if "\0" in message["text"] or any(marker in message["text"] for marker in RESERVED):
            raise ValueError("reserved template marker in natural text")
    return ("<|im_start|>system\n" + SYSTEM_TEXT + "<|im_end|>\n" +
            "".join("<|im_start|>" + m["role"] + "\n" + m["text"] + "<|im_end|>\n"
                    for m in record["context"]) +
            (ASSISTANT_SUFFIX if template_version == LEGACY_TEMPLATE_VERSION else "<|im_start|>assistant\n"))


def encode_generation_input(record, tokenizer, *, template_version=TEMPLATE_VERSION):
    prompt = generation_prompt(record, template_version=template_version)
    _, sources = encoder_texts(record)
    return GenerationInput(tuple(tokenizer.encode(prompt, True)), sources)


def greedy_generate(request, tokenizer, fresh_forward, *, stop_ids, max_new_tokens,
                    context_capacity, memory_enabled=True):
    """Call a stateless full-prefix backend; never supply gold or earlier answers.

    fresh_forward(prefix_ids, source_texts) returns a one-dimensional vocabulary
    logit vector. Its implementation must be separately certified as fresh/no-KV.
    This transport does not claim that a callable's signature proves that fact.
    """
    if type(request) is not GenerationInput:
        raise ValueError("label-free GenerationInput required")
    if type(memory_enabled) is not bool:
        raise ValueError("memory switch must be boolean")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("positive generation limit required")
    if type(context_capacity) is not int or context_capacity <= len(request.prompt_token_ids):
        raise ValueError("insufficient context capacity")
    if type(stop_ids) is not tuple or not stop_ids or any(type(t) is not int or t < 0 for t in stop_ids):
        raise ValueError("explicit stop token IDs required")
    prefix = request.prompt_token_ids
    sources = request.source_texts if memory_enabled else ()
    generated, visible = [], []
    reason, vocabulary = "max_new_tokens", None
    for _ in range(max_new_tokens):
        if len(prefix) >= context_capacity:
            reason = "context_limit"
            break
        logits = fresh_forward(prefix, sources)
        # Convert numpy/torch vectors without permitting a batch axis or NaNs.
        values = logits.tolist() if hasattr(logits, "tolist") else list(logits)
        if not values or any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError("finite vocabulary vector required")
        if vocabulary is not None and vocabulary != len(values):
            raise ValueError("vocabulary changed during generation")
        vocabulary = len(values)
        if any(t >= vocabulary for t in (*prefix, *stop_ids)):
            raise ValueError("token ID outside vocabulary")
        token = max(range(vocabulary), key=values.__getitem__)
        generated.append(token)
        if token in stop_ids:
            reason = "stop_token"
            break
        visible.append(token)
        prefix = prefix + (token,)
    raw = b"".join(tokenizer.decode_pieces(visible))
    # Preserve partial bytes explicitly instead of pretending a clipped UTF-8
    # sequence is a clean free-form answer.
    try:
        text, utf8_complete = raw.decode("utf-8"), True
    except UnicodeDecodeError:
        text, utf8_complete = raw.decode("utf-8", errors="replace"), False
    return {"text": text, "generated_token_ids": generated, "visible_token_ids": visible,
            "raw_text_hex": raw.hex(), "utf8_complete": utf8_complete,
            "finish_reason": reason, "truncated": reason != "stop_token",
            "memory_enabled": memory_enabled, "forward_calls": len(generated),
            "backend_cache_policy_verified": False}
