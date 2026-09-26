# 158BitNet

158BitNet is a small C inference runtime for OpenBMB BitCPM CANN GGUF models.
It focuses on low-bit BitNet-style decode on ARM CPUs, especially Apple Silicon
and Android arm64 devices. The repository includes a core `bitnet` library,
examples, an OpenAI-compatible HTTP server, LoRA loading, KV-cache reuse, Android
cross-compilation, tests, and profiling tools.

## Models

The project targets BitCPM CANN GGUF models from OpenBMB, including the public
[BitCPM-CANN-8B-gguf](https://huggingface.co/openbmb/BitCPM-CANN-8B-gguf)
release and the local `bitcpm4-{0.5b,1b,3b,8b}-tq2_0.gguf` family used during
development.

In this codebase, "1.58-bit" means BitNet-style ternary weights, not a 1.58B
parameter count. The local model sizes are named by parameter scale, for example
`0.5b`, `1b`, `3b`, and `8b`; the low-bit weight format is usually `TQ2_0`.

Typical GGUF metadata for these models:

- GGUF v3 container with OpenBMB CANN finetune metadata.
- 32K context length in the model metadata.
- LLaMA/MiniCPM-style chat template using `<|im_start|>` and `<|im_end|>`.
- Main transformer projections in `TQ2_0`.
- Embedding/output tensors may use `F16`, `Q4_K`, `Q6_K`, or tied output paths,
  depending on the model size.
- Vocabulary size is 73,448 for the local BitCPM4 models.

Place models under `models/`:

```sh
models/bitcpm4-0.5b-tq2_0.gguf
models/bitcpm4-1b-tq2_0.gguf
models/bitcpm4-3b-tq2_0.gguf
models/bitcpm4-8b-tq2_0.gguf
```

You can also inspect a model directly:

```sh
./build/gguf_inspect models/bitcpm4-1b-tq2_0.gguf
```

## Android 865 Best Known Results

Historical best results observed on the Snapdragon 865 Android device used for
this project are listed below. These numbers are direct `test_profile_decode`
short-context decode microbenchmarks after the ARM/Android optimization work.
They are not HTTP long-generation throughput numbers, and they are not measured
with LoRA enabled.

| Model | GGUF file | Best observed decode | Notes |
| --- | --- | ---: | --- |
| 0.5B | `bitcpm4-0.5b-tq2_0.gguf` | `>100 tok/s` | Best historical 0.5B Android decode result after tied-output and 865 tuning. |
| 1B | `bitcpm4-1b-tq2_0.gguf` | `~34 tok/s` | Best historical 1B Android microbenchmark; HTTP long generation was lower. |
| 3B | `bitcpm4-3b-tq2_0.gguf` | `~16 tok/s` | Best historical 3B Android result after output chunk tuning. |
| 8B | `bitcpm4-8b-tq2_0.gguf` | `~8 tok/s` | Best historical 8B Android result; this was the corrected 8B baseline. |

Use these numbers as regression guardrails. When comparing new optimizations,
keep the benchmark type, model, thread count, context length, and background
process state the same.

## Architecture

The runtime is intentionally compact:

- `src/bitnet.c`: model loading, forward pass, KV cache, LoRA application, output
  projection, and decode-facing API.
- `src/quant_tq2_0.c`: ternary `TQ2_0` kernels, including ARM NEON paths.
- `src/quant_q6k.c`, `src/quant_q4k.c`: GGUF quantization helpers used by
  embeddings and output projection.
- `src/tokenizer.c`: GGUF tokenizer support.
- `examples/minimal_generate.c`: minimal CLI generation.
- `examples/openai_server.c`: OpenAI-compatible HTTP server using Mongoose and
  cJSON from `src/third_party/`.
- `tools/train_xiaoli_lora.c`: small local LoRA generation tool used by tests.
- `tests/`: correctness tests, HTTP API tests, Android-relevant profiling, and
  decode benchmarks.

High-level decode flow:

1. Load GGUF metadata, tokenizer, tensor descriptors, and model weights.
2. Tokenize prompt text with the model tokenizer.
3. Prefill prompt tokens into a `bitnet_context_t`.
4. For each generated token, run one-token decode, sample greedily or with
   repetition penalty, append to history, and update KV cache.
5. Reuse cached KV state for multi-turn requests when `session_id` is provided.

Key runtime features:

- `TQ2_0` low-bit matrix-vector kernels for transformer projections.
- Output projection acceleration with expanded Q6K/Q8 cache paths where
  applicable.
- F32 and Q8 KV-cache modes through `bitnet_set_kv_cache_type`.
- OpenAI-compatible completions/chat API, including streaming SSE responses.
- Session KV reuse and full-history de-duplication for multi-turn HTTP calls.
- External LoRA loading with `--lora` and `--lora-scale`.

## Important Optimizations

The runtime has gone through several rounds of optimization for decode-heavy
BitCPM/BitNet inference. The most important changes are below.

### TQ2_0 Ternary Kernels

`TQ2_0` stores 256 ternary weights per block in 64 packed bytes plus one FP16
scale. The logical weight values are `{-1, 0, +1}` with one reserved 2-bit code.
This makes decode mostly a memory-layout and dot-product problem rather than a
general floating-point GEMM problem.

The implementation keeps the llama.cpp-compatible interleaved element ordering:

- 64 packed bytes encode 256 weights.
- Each byte stores four 2-bit codes.
- The four codes are interleaved by 32-element strides.
- Per-block scales are extracted once into scale caches during model loading.

The baseline scalar path is kept for correctness and non-NEON builds, but ARM
decode uses specialized int8/NEON paths.

### LUT-Based Matmul

The early optimized path builds a lookup table from the current activation
vector, then uses packed weight bytes as indices into that LUT.

For a row block:

1. Build a LUT for the current activation vector.
2. For each packed `TQ2_0` byte, look up the precomputed contribution of its four
   2-bit codes.
3. Apply the per-block scale and accumulate the row result.

This avoids repeatedly unpacking each ternary code into floats. Pair kernels such
as gate+up share one LUT for two projections, reducing repeated preprocessing.

### VTBL / `vqtbl1q_s8` Unpack

On ARM NEON with dot-product support, the runtime uses `vqtbl1q_s8` to decode
2-bit ternary nibbles. A table lookup replaces the usual chain of `AND`, `USHR`,
masking, and subtract operations.

The NEON path uses the "bsums" trick:

- Decode codes as `{0, 1, 2}` instead of `{-1, 0, +1}`.
- Compute `dot(code, activation)`.
- Subtract the precomputed activation block sum once.

This removes a subtraction from the inner unpack loop and lets the hot path
combine `vqtbl1q_s8` with `vdotq_s32`.

### `vdotq_s32` Dot Product

For ARMv8.2-A dot-product targets, the hot low-bit kernels use `vdotq_s32` over
int8 activations and unpacked ternary codes. This is used in:

- Single-row `TQ2_0` projection.
- Pair projection for gate+up.
- 4-row and 8-row grouped kernels.
- Q6K/Q8 output projection.

The kernels use dual accumulators in several places to break dependency chains
and expose more instruction-level parallelism to the CPU.

### I2_S / I2S Weight Reordering

The I2_S path reorders `TQ2_0` weights at model load time into a 4-row packed
2-bit layout. The one-time load cost buys a simpler decode layout:

- Four rows are packed together to improve activation reuse.
- Per-row/per-block scales are stored beside the reordered weights.
- Parallel APIs support single projection, paired projection, and fused Q/K/V.
- The runtime can compute Q/K/V in one dispatch and gate+up in one paired kernel.

This reduces repeated reads of the same activation vector and cuts thread-pool
dispatch overhead for common transformer projection groups.

Relevant code:

- `bitnet_tq2_0_reorder_to_i2s`
- `bitnet_tq2_0_matmul_i2s_neon_parallel`
- `bitnet_tq2_0_matmul_i2s_neon_pair_parallel`
- `bitnet_tq2_0_matmul_i2s_qkv_parallel`

### TL1 / `vqtbl1q_s8` GEMM Path

The TL1 path is another layout experiment inspired by llama.cpp/BitNet kernels.
It converts `TQ2_0` into consecutive nibble pairs and builds a compact
`vqtbl1q_s8` LUT from the activation vector.

The TL1 pipeline is:

1. Reorder `TQ2_0` interleaved weights to TL1 at load time.
2. Quantize and preprocess the current activation vector into a table.
3. Process blocked row groups while sharing the LUT across many rows.

This path is useful for comparing table-lookup GEMM behavior against I2S and the
direct VTBL/dotprod kernels.

### RMSNorm + Quantize Fusion

For I2S decode, the runtime fuses RMSNorm output production with int8 activation
quantization where possible:

- Attention RMSNorm can produce the normalized hidden state and the int8 vector
  used by Q/K/V in one pass.
- FFN RMSNorm can similarly feed gate/up projection quantization.
- FFN down uses a known max from `silu(gate) * up` to avoid an unnecessary second
  max scan.

The goal is to reduce memory bandwidth and avoid repeating normalization or
quantization work in each projection.

### Fused Projection Dispatch

Several transformer projections naturally share the same input vector. The
runtime exploits that:

- Q/K/V can be computed together through the I2S QKV parallel path.
- Gate+up can be computed through paired kernels and one shared quantized input.
- Paired LUT kernels share the activation LUT across two matrices.

This matters on mobile CPUs because thread-pool scheduling and repeated input
quantization can become visible at short decode lengths.

### Output Projection Cache

The output layer can dominate decode for small and medium models because it has
to score the full vocabulary. The runtime builds expanded output caches at model
load time when the model format allows it:

- Q6K output can be expanded into Q8 rows with compact scale storage.
- F16 tied embeddings can be converted into block-scaled Q8 rows.
- Output rows are evaluated in 4-row or 8-row NEON dot-product kernels.
- The output worker pool splits vocabulary rows into chunks and parallelizes
  logits computation.

Android uses smaller default chunks for some compact output shapes to improve
load balancing on Snapdragon-class devices. `BITNET_OUTPUT_CHUNK_ROWS` can be
used for experiments.

### Multi-Threaded Decode

The runtime uses a small global worker pool for selected hot paths instead of
creating threads per request. The default thread count is 3 and can be overridden
with:

```sh
BITNET_NUM_THREADS=4 ./build/openai_server models/bitcpm4-1b-tq2_0.gguf
```

Threaded work is used where it pays off, especially output projection and large
TQ2 projections. Small matrices can stay single-threaded to avoid scheduling
overhead.

### Q8 KV Cache

The public API exposes both F32 and Q8 KV-cache modes:

```c
bitnet_set_kv_cache_type(ctx, BITNET_KV_CACHE_Q8);
```

Q8 KV cache stores keys and values as int8 plus per-head scales. During
attention, query heads are quantized and dotted against cached int8 keys, while
cached int8 values are accumulated back into float attention outputs. This cuts
KV-cache memory bandwidth and storage versus F32 cache, which becomes more
important as context length grows.

### HTTP Session KV Reuse

The OpenAI-compatible server keeps per-session `bitnet_context_t` objects when a
request provides `session_id`.

The server reports:

- `cached_tokens`: tokens already present in the session context.
- `reused_tokens`: tokens reused from cache for this request.
- `context_tokens`: current session context length after generation.

If a client sends full conversation history on every turn, the server compares
the token prefix and skips re-evaluating the cached portion. This makes multi-turn
chat faster without requiring clients to omit history manually.

### Sparse Output LoRA

External LoRA is supported for transformer projections and the output layer. The
example Xiaoli LoRA used in tests mostly changes a sparse set of output tokens.
At load time, the runtime detects non-zero output rows in the LoRA B matrix and
only updates those logits during decode.

This keeps LoRA behavior in adapter weights while avoiding a dense full-vocab
LoRA pass for mostly sparse output adapters.

## Build

All build output should stay under `build/`.

```sh
cmake -S . -B build
cmake --build build --target openai_server minimal_generate gguf_inspect -j 8
```

Build tests and benchmark binaries:

```sh
cmake --build build -j 8
```

## Minimal CLI Usage

```sh
./build/minimal_generate models/bitcpm4-1b-tq2_0.gguf "The capital of France is" 32
```

Useful environment variables:

```sh
BITNET_NUM_THREADS=3
BITNET_MAX_CONTEXT=2048
BITNET_REPEAT_LAST_N=64
BITNET_REPEAT_PENALTY=1.1
# Set to 0 to avoid the faster, additional-memory compact Q8 output cache.
BITNET_OUTPUT_Q8_CACHE=1
```

`BITNET_NUM_THREADS` controls both the persistent pthread workers and x86
OpenMP projection kernels.  The runtime clamps it to each pool's capacity.

## OpenAI-Compatible HTTP Server

Start the server:

```sh
./build/openai_server models/bitcpm4-1b-tq2_0.gguf \
  --host 127.0.0.1 \
  --port 8080
```

Defaults:

- Model id: `bitnet`
- Default `max_tokens`: `1024`
- Request `max_tokens` cap: `4096`
- Minimum context grows to fit the configured request cap.

Chat completion:

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bitnet",
    "max_tokens": 64,
    "messages": [
      {"role": "user", "content": "The capital of France is"}
    ]
  }'
```

Text completion:

```sh
curl -sS http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bitnet",
    "prompt": "The capital of France is",
    "max_tokens": 64
  }'
```

Streaming:

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bitnet",
    "stream": true,
    "max_tokens": 64,
    "messages": [
      {"role": "user", "content": "Explain KV cache briefly."}
    ]
  }'
```

Multi-turn KV-cache reuse:

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bitnet",
    "session_id": "demo-session",
    "reset_session": true,
    "max_tokens": 64,
    "messages": [
      {"role": "user", "content": "Remember the code LAN-TQ2-42. Reply STORED."}
    ]
  }'

curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bitnet",
    "session_id": "demo-session",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "What code did I ask you to remember?"}
    ]
  }'
```

With `session_id`, the server keeps a context for the session and reports
`bitnet_session.cached_tokens`, `reused_tokens`, and `context_tokens`. If the
client sends full history again, the server de-duplicates the already cached
prefix instead of evaluating it twice.

## LoRA

Load an external LoRA adapter:

```sh
./build/openai_server models/bitcpm4-1b-tq2_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --lora build/lora/xiaoli.bnlora \
  --lora-scale 1.0
```

Generate the example Xiaoli adapter used by tests:

```sh
cmake --build build --target train_xiaoli_lora -j 8
./build/train_xiaoli_lora models/bitcpm4-1b-tq2_0.gguf build/lora/xiaoli.bnlora
```

LoRA is applied by runtime weights, not by hidden server-side persona prompts.
The HTTP server should remain a general inference framework.

## Android

Set the Android NDK path and build under `build/android-arm64-v8a`:

```sh
ANDROID_NDK=/path/to/android-ndk ./scripts/build_android.sh openai_server minimal_generate gguf_inspect
```

Push to a connected device:

```sh
adb -s 192.168.210.10:5555 push build/android-arm64-v8a/openai_server /data/local/tmp/openai_server
adb -s 192.168.210.10:5555 shell chmod 755 /data/local/tmp/openai_server
```

Run on device with models already placed in `/data/models/`:

```sh
adb -s 192.168.210.10:5555 shell \
  'cd /data/local/tmp && BITNET_NUM_THREADS=3 ./openai_server /data/models/bitcpm4-1b-tq2_0.gguf --host 127.0.0.1 --port 18187'
```

Device-side smoke request:

```sh
adb -s 192.168.210.10:5555 shell \
  "curl -sS http://127.0.0.1:18187/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"bitnet\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}]}'"
```

For decode-only regression checks on Snapdragon 865, compare against the
[best known Android results](#android-865-best-known-results) rather than HTTP
long-generation throughput.

## Tests And Profiling

Run unit tests:

```sh
ctest --test-dir build --output-on-failure
```

Cross-build Android tests by opting in explicitly:

```sh
BITNET_BUILD_TESTS=ON ANDROID_NDK=/path/to/android-ndk \
  ./scripts/build_android.sh test_ops test_i2s_correctness test_quant_tq2_0 test_q6k_layout
```

Run HTTP API tests:

```sh
python3 tests/test_openai_server_default_config.py build/openai_server models/bitcpm4-0.5b-tq2_0.gguf
python3 tests/test_openai_server_api.py build/openai_server models/bitcpm4-0.5b-tq2_0.gguf
python3 tests/test_openai_server_lora_api.py build/openai_server models/bitcpm4-1b-tq2_0.gguf build/lora/xiaoli.bnlora
python3 tests/test_openai_server_multiturn_quality.py build/openai_server models/bitcpm4-1b-tq2_0.gguf build/lora/xiaoli.bnlora
```

Decode profiling:

```sh
cd build
for t in 1 2 3 4 6; do
  echo -n "$t threads: "
  BITNET_NUM_THREADS=$t ./test_profile_decode 2>&1 | grep decode_tok_s
done
```

Multi-turn KV-cache profiling:

```sh
./build/test_profile_multiturn models/bitcpm4-1b-tq2_0.gguf
./build/test_profile_multiturn models/bitcpm4-1b-tq2_0.gguf q8
```

## Notes

- Keep generated binaries, test outputs, Android builds, and LoRA artifacts under
  `build/`.
- The local benchmark numbers depend heavily on model size, thread count,
  context length, HTTP vs direct runtime path, and whether another server process
  is still running.
- Before Android benchmarking, check for stale server processes:

```sh
adb -s 192.168.210.10:5555 shell 'ps -A | grep openai_server || true'
```

---

# Neural Memory System

The repository also contains a neural memory subsystem trained on top of the
frozen 0.5B backbone. Memory weights are trained separately from the GGUF,
exported to a single `.bnmodel` file, and loaded by a pure C server with
no Python dependency at inference time.

## Quick start

```sh
# Build
cmake -S . -B build && cmake --build build -j 8

# Plain LLM inference (no memory)
./build/c_neural_server models/bitcpm4-0.5b-tq2_0.gguf

# With neural memory
./build/c_neural_server models/bitcpm4-0.5b-tq2_0.gguf \
  --memory-model build/neural-c-weights/model.bnmodel
```

## Memory server usage

```
Caius currently lives in Aachen. Darya currently works in Bilbao.
                                   → [write] stored
Where does Caius live now?         → route=1 (supported) → value='Aachen'
Where does Kai live now?           → route=2 → natural refusal
Morgan lives in Lima.              → [write] stored
Where does Morgan live now?        → route=2 (single-fact phrasing is outside
                                     the route head's training form) → refusal
What is 3 plus 7?                  → usually route=0 (normal base chat);
                                     may misroute after a JB-form episode
                                     (measured transfer weakness, LC-002)
```

Recall activates on the JB training form (two-fact episodes with
"currently lives/works"); plain single-fact sentences and unseen
subject–attribute combinations refuse rather than guess.

### Persistence

```sh
# Save memory state
/save memory.bnstate

# Restart server, then load
/load memory.bnstate

# Or use auto-load/auto-save on startup/exit
./build/c_neural_server model.gguf \
  --memory-model weights.bnmodel \
  --load-state memory.bnstate \
  --save-state memory.bnstate
```

### Memory recall: what actually activates (measured, 2026-09-24)

The trained route head certifies only the JB training form — two-fact
episodes ("X currently lives in Y. Z currently works in W."). Measured
behavior of the current system:

| Input shape | Behavior |
| --- | --- |
| Several stored episodes + query about ANY of them | ranked recall (EC-005 joint head, holdout top-1 0.81 over unseen worlds): e.g. 3 episodes stored, query the FIRST subject → its value |
| JB-form episode + in-distribution query | neural recall (e.g. → `Aachen`) |
| Plain single-fact sentences (Morgan lives in Lima.) | stored, but query refuses (route=insufficient) — the route head never saw this form |
| Query about an absent subject | natural refusal (uncertainty branch) |
| Arithmetic after a JB-form episode | may misroute to recall (known transfer weakness, LC-002) |

The earlier "six memory types" table described the pre-LC-002 system whose
route decision was an empirical override (route=2 + logit patch → always
extract). With the trained route restored, those plain-form examples no
longer activate; the table was removed rather than left misleading.

### Architecture

```
User message
  ↓ tokenize + C backbone forward
  ↓ encoder features (2048-dim, L2-normalized hidden+lexical pair)
  ↓ route head (trained 3-class: normal/supported/insufficient)
  │  — exact port of the trained reader's span-based evidence
  │    (candidate spans + fact-probability-weighted evidence),
  │    verified by test_route_parity against Python logits
  ├─ normal      → plain base generation
  ├─ supported   → wide-head (2048-dim) value span selection on the episode
  └─ insufficient→ uncertainty residual added to the final hidden state,
                   projected through the backbone's own Q6_K output head
                   (bitnet_project_hidden_to_logits) → natural refusal
```

Both route and value heads read features computed with the training-matched
JSON framing (query = one-element array, episode = single object, escaped
text). Memory content is UNBOUNDED: the session table grows dynamically,
episodes are arbitrary-length strings, and input lines have no fixed cap.
Recall operates on a 512-token feature window per episode (long episodes
truncate at a UTF-8 boundary; storage keeps the full text). Episode feature
means are cached at write time only when a ranking head consumes them and
persist in `.bnstate` v2. There is no route-override patch: the trained
heads decide, and refusal is the trained uncertainty branch.

### Files

| File | Role |
| --- | --- |
| `src/neural_memory.c/h` | Pure C neural inference (route + wide heads + uncertainty branch, framing/span layout) |
| `src/memory_state.c/h` | Session episode store, capacity 512, `.bnstate` v1/v2 persistence |
| `tools/c_neural_server.c` | Pure C server (sessions, memory, persistence, routing) |
| `build/neural-c-weights/model.bnmodel` | Trained neural weights (v3, 43 tensors, joint ranking head) |
| `python/train_wide_live_ce.py` | Value-span training (live features + tokenizer) |
| `python/export_bnmodel.py` | `.bnmodel` writer/inspector (v2/v3) |
| `python/eval_locomo_neural.py` | LoCoMo driver (raw-turn writes, real restart, read-only queries) |
| `tests/test_memory_state.c`, `tests/test_neural_memory.c`, `tests/test_route_parity.c` | ctest-registered C units |
| `tests/test_c_neural_server_memory.py` | End-to-end REPL integration tests |
| `training/memory/neural-system/` | Experiment registrations, reviews, training data |

### Verification and training results

| Capability | Status | Established in |
| --- | --- | --- |
| Route logits parity (C == trained Python reader) | 10/10 records | `test_route_parity` (ctest) |
| Episode ranking, unseen worlds (EC-005) | holdout top-1 **0.812** | `reviews/EC-005/` |
| Multi-episode recall, non-last episode | verified | integration test |
| Uncertainty refusal (natural, EN/ZH) | 8–10/10 panels | V3-004 lineage |
| Normal chat | 4/4 panels | V3-005 onward |
| Route discrimination (dev, n=892) | 82% | V3-009 (two-stage) |
| In-distribution recall end-to-end in C (JB form) | verified | integration test (ctest-excluded, needs GGUF) |
| Persistence across restart (v2 state) | verified | `test_memory_state` + integration |
| Episode relevance head (EC-001) | **failed gate** (holdout top-1 0.292 < 0.80) — not deployed | `reviews/EC-001/` |
| LoCoMo10 baseline (LC-001, pre-fix) | F1 0.0022, cat-5 rejection 0.0 | `reviews/LC-001/` |
| LoCoMo10 post-fix (LC-002) | storage fixed (168-402/conv), cat-5 rejection 0.213 (was 0.0), F1 0.0036 | `reviews/LC-002/` |
| LoCoMo10 ranking head (LC-003) | mechanics confirmed, F1 flat: form transfer fails | `reviews/LC-003/` |
| LoCoMo10 mixed-form + tau (LC-004) | cat-5 rejection **0.946**, first exact match; recall ~0 (threshold rejects answerable cross-form queries) | `reviews/LC-004/` |
| LoCoMo10 paraphrase corpus (LC-005) | F1 0.000, rejection 0.938 — self-distillation does not bridge to real dialogue (generative-process gap) | `reviews/LC-005/` |

### Known limitations

1. **Episode selection is learned and deployed** (EC-005): a joint
   per-pair-interaction ranking head (the only architecture of five tested
   that learned subject identity) selects which stored episode answers the
   query — 0.81 top-1 on held-out semantic worlds. Its absolute scores are
   NOT calibrated (deployment tau=0.02): the head ranks, the route head
   still decides whether to answer. Span-level binding of two facts inside
   ONE episode remains in-distribution-only.
2. **Route head distribution.** Supported/insufficient decisions are only
   trained on the JB form (two-fact "currently lives/works" episodes);
   plain single-fact sentences refuse.
3. **Subject–value binding on unseen combinations** remains unsolved — an
   architectural limit of frozen 0.5B features (see PAPER.md §6).

### LoCoMo10

`python/eval_locomo_neural.py` drives the pinned official LoCoMo10
(SHA-verified) through the C server: raw-turn single-line writes with the
automatic gate, `/save`, real process restart, `/load`, then read-only
questions scored with the official category F1. Baseline (LC-001):
F1 0.0022 over 1,540 answerable questions, 100% activation including all 446
adversarial ones (rejection 0.0), storage capped at 16 episodes. The LC-002
re-run confirms the engineering fixes (full storage, real refusal path,
exact route computation) while the remaining gap is fully attributed to the
documented representation wall: value extraction still reads only the last
episode and the JB-trained route head over-accepts natural dialogue
questions (conversation 0 answers all 199 queries from the last episode's
span). Details in `reviews/LC-002/`.

## NEON optimization

The GGUF reference-parity path received NEON optimizations that recovered
~75% of the long-context decode performance gap while maintaining bit-identical
outputs across all four model sizes (0.5B/1B/3B/8B):

1. Contiguous dot-product fast path (same accumulator mapping and reduction tree)
2. Fused attention value-column pass (identical per-lane FMA chains)
3. NEON element-wise RMSNorm tail
4. Stack scratch for matmul (eliminated per-call malloc/free)
5. NEON Q6_K dequantization with vdot

Verification: 7 golden logits byte-identical, teacher-forced NLL bit-equal,
200-token generation byte-identical to pre-optimization baseline.
