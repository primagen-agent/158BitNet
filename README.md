# 158BitNet

158BitNet is a C11 inference runtime for OpenBMB BitCPM CANN GGUF models with
ternary `TQ2_0` weights. It supports Apple Silicon, Android arm64, and x86
processors with runtime SIMD dispatch.

The repository also contains a persistent-memory subsystem. Memory parameters
are trained separately from the GGUF backbone, are bound to the exact backbone
SHA-256 identity, and can be used without LoRA or KV-cache reuse.

## Features

- `TQ2_0`, `Q6_K`, and `Q4_K` tensor support
- ARM NEON and x86 AVX2/AVX-VNNI/AVX512-VNNI kernels
- runtime CPU feature detection and dispatch
- optional Apple Metal backend
- OpenAI-compatible chat/completions HTTP API
- streaming responses and reusable chat sessions
- F32 or Q8 KV cache
- native persistent neural memory using `.bnmem` and `.bnstate`
- addressed episodic memory using `.bnctrl`, `.bnptr`, and `.bnepisodic`
- CRC validation and exact-backbone identity checks for memory artifacts
- memory export/import across process restarts

Models, training checkpoints, memory data, and build artifacts belong under
`models/` or `build/` and must not be committed.

## Build

```sh
cmake -S . -B build
cmake --build build -j 8
```

Build only the user-facing tools:

```sh
cmake --build build \
  --target openai_server minimal_generate gguf_inspect -j 8
```

Enable Apple Metal:

```sh
cmake -S . -B build -DBITNET_ENABLE_METAL=ON
cmake --build build -j 8
```

Android arm64:

```sh
ANDROID_NDK=/path/to/android-ndk \
  ./scripts/build_android.sh \
  openai_server minimal_generate gguf_inspect
```

## Command-line generation

```sh
./build/minimal_generate \
  models/bitcpm4-0.5b-tq2_0.gguf \
  "Write a short introduction to ternary language models."
```

Inspect GGUF metadata and tensors:

```sh
./build/gguf_inspect models/bitcpm4-0.5b-tq2_0.gguf
```

## HTTP server

```sh
./build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx 4096 \
  --max-tokens 256
```

Chat completion:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bitnet",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
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
    "messages": [
      {"role": "user", "content": "Explain ternary weights briefly."}
    ]
  }'
```

## Persistent memory

There are two complementary memory paths.

### Design basis

The native memory path follows the main ideas in
*Native Sparse Memory for Fast, Adaptive, and Parallel
Memory-Augmented Language Models*:

- memory modules are attached to transformer layers;
- token importance controls sparse writes;
- gated-delta updates maintain dynamic memory state;
- normalized reads are fused into later transformer computation;
- memory parameters are trained while the backbone remains frozen.

The code also contains optional rank and layer gates motivated by
*Designing Compact Neural Architectures via Neuron Gating and Mixed
Activation* (arXiv:2607.26760). Compression and architecture search are
disabled in the current accuracy-first training command. The current
implementation does not claim to reproduce the complete mixed-activation
search method.

### Native neural memory

The native path attaches trainable memory modules to transformer layers:

- `.bnmem` contains the trained memory architecture and parameters.
- `.bnstate` contains one session's committed dynamic memory state.
- the GGUF backbone remains frozen during memory training.
- memory reads and writes use fresh contexts and do not require KV reuse.
- `.bnmem` can only be loaded with the exact GGUF used for training.

Run the server:

```sh
mkdir -p build/memory-states

./build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  --memory-model build/memory_05b.bnmem \
  --memory-state-dir build/memory-states \
  --host 127.0.0.1 \
  --port 8080
```

### Addressed episodic memory

The addressed path stores exact dynamic text outside the trained parameters:

- `.bnctrl` classifies write/update/delete/ignore actions and maps entries and
  questions into a shared address space.
- `.bnptr` extracts an exact answer span from a retrieved record.
- `.bnepisodic` contains session memory data and can be exported/imported.
- update replaces the nearest matching address.
- delete writes a retrievable tombstone so stale values are not returned.

Run the addressed path without a native `.bnmem`:

```sh
mkdir -p build/memory-states

./build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  --memory-state-dir build/memory-states \
  --episodic-memory \
  --memory-controller build/memory_05b.bnctrl \
  --memory-pointer build/memory_05b.bnptr \
  --episodic-top-k 5 \
  --episodic-lexical-weight 0.25 \
  --host 127.0.0.1 \
  --port 8080
```

The server response includes the resolved `memory_action`. Clients may omit
`memory_action` to use the learned router, or provide `write`, `update`,
`delete`, or `ignore` explicitly.

## Memory export and import

Commit a memory:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "reset_session": true,
    "memory_action": "write",
    "max_tokens": 1,
    "messages": [
      {"role": "user", "content": "Please remember that my preferred drink is matcha."}
    ]
  }'
```

Export:

```sh
curl http://127.0.0.1:8080/v1/memory/export \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo"}'
```

After restarting the server, import:

```sh
curl http://127.0.0.1:8080/v1/memory/import \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo"}'
```

Query without reusing KV state:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "memory_action": "ignore",
    "memory_copy": true,
    "max_tokens": 24,
    "messages": [
      {"role": "user", "content": "What is my preferred drink?"}
    ]
  }'
```

## Training

Training requires PyTorch, the C tokenizer probe, and a small GGUF reader
shim. Compile the two helper artifacts once:

```sh
gcc -O2 -fPIC -shared \
  python/ggwshim.c -I src \
  -o build/libggwshim.so

gcc -O2 \
  python/tok_probe.c -I include -I src \
  -o build/tok_probe \
  build/libbitnet.a -lm -lpthread -ldl
```

On a macOS shell running under Rosetta, prefix both compiler commands with
`arch -arm64`.

### Train native memory

```sh
GGUF=models/bitcpm4-0.5b-tq2_0.gguf \
  ./scripts/train_memory_05b.sh
```

The default output is:

```text
build/memory_05b.bnmem
build/memory_05b.final.bnmem
```

The current accuracy-first configuration:

- trains memory parameters on every backbone layer;
- uses full-rank query/key/value projections;
- uses sparse AlphaTopP token selection and gated-delta writes;
- uses no SVD compression, NAS gates, LoRA, or answer decoder;
- evaluates queries in a fresh context with KV reuse disabled;
- keeps LoCoMo outside the training set.

### Train the addressed controller

```sh
python3 python/train_addressed_memory_controller.py \
  models/bitcpm4-0.5b-tq2_0.gguf \
  build/memory_05b.bnctrl \
  --lib build/libggwshim.so \
  --tok-probe build/tok_probe \
  --cache build/memory_05b_controller_features.npz \
  --device auto \
  --rank 128 \
  --steps 1000
```

The controller jointly learns:

- write/update/delete/ignore routing;
- value-invariant entry addresses;
- query-to-entry retrieval;
- grouped first-person addresses;
- explicit and implicit memory expressions.

### Train the pointer

Train the pointer from scratch on independent synthetic write/update forms:

```sh
python3 python/finetune_memory_pointer_synthetic.py \
  models/bitcpm4-0.5b-tq2_0.gguf \
  - \
  build/memory_05b.bnptr \
  --lib build/libggwshim.so \
  --tok-probe build/tok_probe \
  --cache build/memory-pointer-features.pt \
  --device auto
```

## Validation

Build and run the C test suite:

```sh
ctest --test-dir build --output-on-failure
```

Python memory regression tests:

```sh
python3 tests/test_metis_training_formula.py
python3 tests/test_bnmem_python_roundtrip.py
```

Run the 90-case remember/update/forget persistence protocol:

```sh
python3 tests/eval_reference_protocol.py \
  build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  - \
  --protocol explicit \
  --n-per-op 30 \
  --persistence \
  --addressed \
  --auto-action \
  --controller build/memory_05b.bnctrl \
  --pointer build/memory_05b.bnptr \
  --output build/memory_reference_90.json
```

This protocol writes memory, exports it, restarts the service, imports it, and
then queries it. It rejects a result if the QA request reports non-zero cached
or reused KV tokens.

The current addressed controller and pointer reached:

| Test | Result |
| --- | ---: |
| Explicit remember/update/forget persistence | 90/90 |
| Implicit-expression pilot | 8/9 |

These compact tests validate exact storage, update, deletion, persistence, and
automatic routing. They do not establish general long-conversation memory
quality.

LoCoMo remains an evaluation-only dataset. A bounded 20-question, two-session
smoke test reached 3.46% F1 on categories 1–4, showing that open-domain event
addressing and long-conversation retrieval still require improvement. Do not
use LoCoMo conversations as memory-model training data.

Run a bounded LoCoMo evaluation:

```sh
python3 tests/eval_locomo.py \
  build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  - \
  build/locomo10.json \
  --addressed \
  --controller build/memory_05b.bnctrl \
  --oracle-write \
  --turn-level \
  --max-questions 20 \
  --output build/locomo_results.json
```

## CPU dispatch

One x86 binary supports scalar, AVX2, AVX-VNNI, and AVX512-VNNI kernels.
Selection is automatic.

Override it for testing:

```sh
BITNET_CPU_TIER=avx2 ./build/minimal_generate model.gguf "Hello"
```

Suppress startup diagnostics:

```sh
BITNET_QUIET=1 ./build/minimal_generate model.gguf "Hello"
```

## Performance profiling

```sh
cd build
for threads in 1 2 3 4 6; do
  BITNET_NUM_THREADS=$threads ./test_profile_decode
done
```

Additional sweep tools:

```sh
./scripts/perf_sweep.sh
python3 scripts/perf_summarize.py
```

## Source map

- `include/bitnet.h` — public C API
- `src/bitnet.c` — model loading, forward pass, sessions, KV cache, memory hooks
- `src/metis/metis_file.{h,c}` — `.bnmem` and `.bnstate`
- `src/metis/episodic_store.{h,c}` — `.bnepisodic`
- `src/metis/memory_controller.{h,c}` — `.bnctrl`
- `src/metis/memory_pointer.{h,c}` — `.bnptr`
- `src/x86/` — x86 SIMD kernels and dispatch registration
- `examples/openai_server.c` — HTTP server and memory endpoints
- `python/train_memory.py` — native memory trainer
- `python/train_addressed_memory_controller.py` — addressed controller trainer
- `python/memory_text_encoding.py` — addressed-memory feature encoding
- `python/memory_pointer_training.py` — shared pointer training utilities
- `python/finetune_memory_pointer_synthetic.py` — pointer fine-tuning
- `tests/eval_reference_protocol.py` — persistence accuracy evaluation
- `tests/eval_locomo.py` — bounded LoCoMo evaluation
