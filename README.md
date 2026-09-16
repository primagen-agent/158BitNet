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
- automatic typed-event memory through the normal chat API
- neural field extraction, version linking, activation, and exact-value recall
- immutable event versions with persistent `.bnepisodic` and `.bnevent` state
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

The current memory system is an automatic typed-event memory for the exact
BitCPM 0.5B backbone on which its learned artifacts were trained. Normal chat
messages can create or update persistent memories, and later questions recall
the current value through neural activation. Memory state is external to the
backbone and survives service restarts through export and import.

### Memory principles

The system follows these rules:

- **Separate behavior from data.** Learned artifacts define how to recognize,
  write, link, and activate memories. Session-specific facts are stored as
  mutable runtime data and are not baked into the model parameters.
- **Store immutable events.** A correction creates a new event that supersedes
  its predecessor. The old event remains available for audit but is removed
  from the active candidate set.
- **Activate before recall.** A question is compared neurally with active
  events. Only the selected event can expose its compiled value span.
- **Do not use RAG.** Stored text is never appended to the prompt, and recall
  does not query a lexical or vector-search index.
- **Do not rely on KV cache.** Memory tests use fresh query prefills and reject
  non-zero `cached_tokens` or `reused_tokens`.
- **Bind artifacts to one backbone.** Every learned memory artifact records the
  matching GGUF identity. A model trained with
  `bitcpm4-0.5b-tq2_0.gguf` cannot be loaded with another GGUF.
- **Prefer accuracy before compression.** The current system uses full-layer
  frozen-backbone features and does not enable SVD, NAS/NG, layer pruning,
  LoRA, or a separate answer decoder.

Memory writing and recall are complementary:

```text
declarative chat message
    |
    v
write/update decision
    |
    v
neural operation and field extraction
    |
    v
neural predecessor activation
    |
    v
deterministic immutable-event compilation
    |
    v
active typed-event memory

fresh question
    |
    v
query-to-active-event neural activation
    |
    +--> NULL: use ordinary generation
    |
    v
one activated event
    |
    v
compiled evidence pointer
    |
    v
exact current value
```

The deterministic compiler does not decide semantic similarity. It applies
version invariants after the neural stages have selected the operation,
fields, and predecessor: create a new event, supersede one exact active event,
or leave the active set unchanged.

### System architecture

The deployed path consists of five stages:

1. **Chat action gate**
   classifies a message as `ignore`, `write`, `update`, or `delete`. A loaded
   neural action controller takes priority. Without one, the server uses a
   conservative statement/question fallback so ordinary questions are not
   stored.
2. **Autonomous writer**
   predicts `assert` or `supersede`, extracts entity, predicate, value, and
   optional valid-time spans, and produces neural address anchors.
3. **Version linker**
   compares a new update with every active event, ranks possible predecessors,
   and may reject the complete set when no valid predecessor exists.
4. **Immutable event store**
   records source evidence, byte spans, typed fields, and explicit
   supersede/retract links. Only current events participate in recall.
5. **Query activator and local pointer**
   selects one active event or NULL from a fresh question. After activation,
   the pointer returns only the value bytes already compiled for that event.

The HTTP server runs these stages directly in the normal
`POST /v1/chat/completions` path. The same writer is also exposed through
`POST /v1/memory/remember` for forced writes and diagnostics.

Runtime state is stored in two files per session:

| File | Contents |
| --- | --- |
| `.bnepisodic` | Immutable source records and evidence bytes |
| `.bnevent` | Typed fields, active versions, and version links |

`POST /v1/memory/export` writes both files atomically under
`--memory-state-dir`. `POST /v1/memory/import` validates and restores them
after a process restart.

### Model structure

The learned memory system is split into four deployable artifacts:

| Artifact | Current responsibility |
| --- | --- |
| `.bntwrite` | Autonomous operation prediction, four-field localization, neural anchors, and span fusion |
| `.bntpair` | Shared token-level entity/predicate pair encoder |
| `.bntlink` | Candidate ranking and set-level predecessor-existence decision |
| `.bntqact` | Natural-query candidate scoring and candidate-set-aware NULL decision |

The current 0.5B artifacts are:

```text
build/memory_v270_05b_autonomous_writer.bntwrite
build/memory_v251_05b_normalized_set_link_pair.bntpair
build/memory_v251_05b_normalized_set_link_pair.bntlink
build/memory_v255_05b_typed_query_activator.bntqact
```

The writer, pair encoder, and query activator consume RMS-normalized hidden
states captured from all 24 backbone layers. Layers are grouped into four
bands (`0-5`, `6-11`, `12-17`, and `18-23`) with learned band fusion. Raw GGUF
token-embedding rows are used where the training objective requires lexical
identity. Learned address vectors are rank 128.

The autonomous writer contains:

- an operation head for `assert` and `supersede`;
- separate entity, predicate, value, and time localizers;
- field-specific contiguous-span taggers;
- predicate/value activation-key adapters;
- deterministic UTF-8 word-boundary and possessive normalization;
- calibrated neural-anchor/span fusion.

The pair and link models contain:

- entity and predicate address projections;
- token-level contextual and tied-embedding comparisons;
- pairwise same-entity/same-predicate features;
- a learned residual joint scorer;
- a permutation-invariant candidate-set existence head.

The query activator reuses the frozen pair encoder, then adds a
query-specific candidate residual and a separate set-aware NULL head. This
keeps write-time predecessor selection and read-time question activation as
different learned tasks.

All binary loaders validate the backbone SHA-256, tensor geometry, per-tensor
CRC, and trailing data. The query artifact is also bound to the exact pair
encoder used during its training.

### Training method

Training uses the exact C tokenizer and hidden states from the matching frozen
0.5B GGUF. LoCoMo is evaluation-only and is never read as training data.

Build the tokenizer and GGUF feature helpers once:

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

The training sequence is:

1. Generate repeated-dialogue worlds with create, update, distractor, missing
   predecessor, alias, temporal, relation, and NULL-query cases using
   `python/prepare_repeated_dialogue_curriculum.py`.
2. Keep training, validation, relation-OOD, layout-OOD, and final challenge
   relations/templates disjoint.
3. Capture all-layer frozen-backbone features with the C tokenizer through
   `python/prepare_typed_memory_features.py`.
4. Train operation and four typed fields with
   `python/train_typed_memory_writer.py`.
5. Train contiguous field spans with `python/train_typed_span_tagger.py`,
   then adapt only predicate/value activation keys with
   `python/train_typed_anchor_keys.py`.
6. Select anchor/span fusion weights on development sets with
   `python/tune_typed_anchor_span_fusion.py`; the sealed challenge split is
   used only for the final measurement.
7. Train active-event predecessor ranking and missing-predecessor rejection
   with `python/train_typed_pair_verifier.py`.
8. Compile current-value and NULL query banks with
   `python/prepare_typed_activation_curriculum.py`, freeze the pair encoder,
   and train `python/train_typed_query_activator.py`.
9. Export the four C artifacts and require Python/C parity before lifecycle
   testing.

Export commands:

```sh
python3 python/export_typed_pair_encoder.py \
  build/memory_v251_05b_normalized_set_link_pair.pt \
  build/memory_v251_05b_normalized_set_link_pair.bntpair

python3 python/export_typed_link_model.py \
  build/memory_v251_05b_normalized_set_link_pair.pt \
  build/memory_v251_05b_normalized_set_link_pair.bntlink

python3 python/export_typed_query_activator.py \
  build/memory_v255_05b_typed_query_head/activator.pt \
  build/memory_v251_05b_normalized_set_link_pair.pt \
  build/memory_v251_05b_normalized_set_link_pair.bntpair \
  build/memory_v255_05b_typed_query_activator.bntqact

python3 python/export_typed_writer_model.py \
  build/memory_v257_05b_typed_writer_all_spans/writer.pt \
  build/memory_v269_05b_expanded_span_tagger/tagger.pt \
  build/memory_v268_05b_expanded_anchor_keys/keys.pt \
  build/memory_v270_05b_autonomous_writer.bntwrite
```

Checkpoint selection is accuracy-first:

- writer: minimum performance across normal, relation-OOD, and layout-OOD
  domains, followed by one sealed challenge evaluation;
- linker: conditional predecessor accuracy plus missing-predecessor rejection;
- query activator: maximize the minimum of current-value top-1 and NULL
  accuracy;
- runtime: exact decision parity, export/restart/import stability, and zero KV
  reuse.

### Usage

Start the complete automatic-memory server:

```sh
mkdir -p build/memory-states

./build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx 4096 \
  --memory-state-dir build/memory-states \
  --episodic-memory \
  --typed-pair-model \
    build/memory_v251_05b_normalized_set_link_pair.bntpair \
  --typed-link-model \
    build/memory_v251_05b_normalized_set_link_pair.bntlink \
  --typed-query-model \
    build/memory_v255_05b_typed_query_activator.bntqact \
  --typed-writer-model \
    build/memory_v270_05b_autonomous_writer.bntwrite
```

The four learned artifacts and the GGUF must be the mutually compatible files
used during training. Typed query/writer models require both the pair and link
models. The typed writer path rejects LoRA.

Normal chat automatically writes declarative facts:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "messages": [{
      "role": "user",
      "content": "Briar conference registration is a virtual access ticket."
    }],
    "max_tokens": 32
  }'
```

The response contains `memory_auto` with the action decision, storage status,
compiled event, extracted fields, and predecessor-resolution details. Set
`"memory_auto": false` to disable implicit memory for one request.

Updates use the same endpoint:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "messages": [{
      "role": "user",
      "content": "Briar changed the conference registration to an in-person ticket."
    }],
    "max_tokens": 32
  }'
```

Questions automatically invoke neural activation and do not require
`memory_copy`:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "messages": [{
      "role": "user",
      "content": "What is Briar current conference registration?"
    }],
    "max_tokens": 32
  }'
```

If the query activator returns NULL, the server falls back to ordinary
generation. A successful memory result reports
`neural_typed_query_activation_then_compiled_pointer`.

Force a raw-text write for diagnostics:

```sh
curl http://127.0.0.1:8080/v1/memory/remember \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "memory_record": "Briar conference registration is an in-person ticket."
  }'
```

Export and restore the session:

```sh
curl http://127.0.0.1:8080/v1/memory/export \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo"}'

curl http://127.0.0.1:8080/v1/memory/import \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo"}'
```

Run the complete normal-chat lifecycle regression:

```sh
python3 tests/test_openai_server_typed_chat_auto_memory.py \
  build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  build/memory_v251_05b_normalized_set_link_pair.bntpair \
  build/memory_v251_05b_normalized_set_link_pair.bntlink \
  build/memory_v255_05b_typed_query_activator.bntqact \
  build/memory_v270_05b_autonomous_writer.bntwrite
```

### Test results

Current measurements as of September 15, 2026 use the matching 0.5B backbone,
fresh query prefills, no KV reuse, no LoRA, no RAG, and no LoCoMo training
data.

Autonomous writer on the sealed 2,048-event challenge:

| Metric | Result |
| --- | ---: |
| Operation exact | **98.63%** |
| Entity compiled evidence exact | **92.77%** |
| Predicate neural anchor inside the gold field | **94.73%** |
| Value compiled evidence exact | **90.82%** |
| Time compiled evidence exact | **100%** |
| Functional writer joint | **82.03%** |
| All four surface spans exact | 66.02% |

`Functional writer joint` requires the operation, entity evidence, value
evidence, time evidence, and predicate neural activation to be correct in the
same example. Predicate surface-span byte equality is not required because
version identity uses the neural address.

Write-time predecessor linking on the 2,048-event relation-OOD split:

| Metric | Result |
| --- | ---: |
| Candidate pairs | 36,044 |
| Conditional version-link accuracy | **64.941%** |
| Update predecessor-link accuracy | 63.664% |
| Predecessor rank accuracy | 72.723% |
| True-predecessor acceptance | 78.492% |
| Missing-predecessor rejection | 48.229% |
| Existence balanced accuracy | 63.360% |

Read-time query activation on the held-out relation-OOD lifecycle:

| Metric | Python | C runtime |
| --- | ---: | ---: |
| Current-value exact recall | 12/16 = **75.0%** | 12/16 = **75.0%** |
| NULL-query rejection | 6/8 = **75.0%** | 6/8 = **75.0%** |
| Combined | 18/24 = **75.0%** | 18/24 = **75.0%** |
| Candidate-only positive rank | 13/16 = 81.25% | Same decisions |
| Export/restart/import | — | Stable |

The complete normal-chat C lifecycle passes locally and on the GPU server:

- ordinary questions are not written;
- declarative create and update messages are written automatically;
- an unrelated distractor does not replace the target memory;
- natural chat works without source/date prefixes;
- streaming writes are persisted;
- `memory_auto:false` prevents implicit writes;
- questions recall through neural activation without `memory_copy`;
- export, process restart, import, and repeated recall are stable;
- `cached_tokens` and `reused_tokens` remain zero;
- no LoRA, RAG, or retrieved-context injection is used.

The repository regression suites pass 19/19 CTest tests and 63/63 Python
unit tests.

The complete typed-event system does not yet have a valid full LoCoMo score.
Component scores above must not be multiplied into an end-to-end estimate.
The main measured limitations are relation-OOD predecessor selection, NULL
calibration, the rule-based fallback used when no neural action controller is
loaded, and the absence of a fully trained autonomous typed retraction path.

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
- `src/bitnet.c` — model loading, forward pass, sessions, and runtime hooks
- `src/quant_tq2_0.c`, `src/quant_q6k.c`, `src/quant_q4k.c` — quantized
  kernels and tensor helpers
- `src/ops.c` — normalization, activation, RoPE, residual, and softmax ops
- `src/gguf.c`, `src/tensor.c`, `src/tokenizer.c`, `src/sampler.c` — model
  format and generation support
- `src/x86/` — x86 SIMD kernels and dispatch registration
- `src/bitnet_metal.mm` — optional Apple Metal backend
- `examples/openai_server.c` — OpenAI-compatible HTTP server
- `examples/minimal_generate.c` — minimal command-line generation
- `tools/gguf_inspect.c` — GGUF metadata and tensor inspection
- `tests/` — correctness, API, lifecycle, and performance tests
