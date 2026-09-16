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
   stored. This gate decides whether to invoke the writer; its heuristic
   write/update label does not override the writer's predicted operation.
   Only an explicit request `memory_action` overrides that prediction.
   If a loaded action controller fails, automatic writes are skipped rather
   than authorized by a heuristic fallback.
2. **Autonomous writer**
   predicts `assert` or `supersede`, extracts entity, predicate, value, and
   optional valid-time spans, and produces neural address anchors.
   A predicted change is not proof that an earlier version is stored. If no
   predecessor is activated, automatic memory records a new assertion without
   deactivating another event. An explicit `memory_action:"update"` still fails
   when no predecessor is found; inference errors are never treated as absence.
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

Each snapshot uses two immutable content-addressed data files and one manifest:

| File | Contents |
| --- | --- |
| `.bnepisodic` | Immutable source records and evidence bytes |
| `.bnevent` | Typed fields, active versions, and version links |
| `.bnsnapshot` | Atomic manifest selecting the two complete files by SHA-256 |

`POST /v1/memory/export` writes and flushes both new files, then atomically
replaces the session manifest under `--memory-state-dir`. A partial write
cannot replace the previously committed snapshot. Import checks whole-file
hashes, internal validation, and source/value consistency before changing
live state. Use one server writer per state directory. Committed snapshot files
are retained; automatic snapshot garbage collection is not implemented.

Repeated identical inputs whose event is still active are idempotent. Source
deduplication retains the actual original record index. Failed event
compilation rolls back newly appended evidence.

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
build/memory_v273_05b_context_writer.bntwrite
build/memory_v251_05b_normalized_set_link_pair.bntpair
build/memory_v272_05b_absolute_link/link.bntlink
build/memory_v255_05b_typed_query_activator.bntqact
```

The writer, pair encoder, and query activator consume RMS-normalized hidden
states captured from all 24 backbone layers. Layers are grouped into four
bands (`0-5`, `6-11`, `12-17`, and `18-23`) with learned band fusion. Raw GGUF
token-embedding rows are used where the training objective requires lexical
identity. Learned address vectors are rank 128.

The autonomous writer contains:

- a full-message operation head for `assert` and `supersede`: mean and final
  token features feed a `2048 -> 128 -> 2` MLP, independently of field extraction;
- separate entity, predicate, value, and time localizers;
- field-specific contiguous-span taggers;
- predicate/value activation-key adapters;
- deterministic UTF-8/Latin word completion, entity possessive normalization,
  and a learned-anchor fallback when the entity span is malformed;
- optional neural-anchor/span fusion, disabled in the deployed configuration.

The pair and link models contain:

- entity and predicate address projections;
- token-level contextual and tied-embedding comparisons;
- pairwise same-entity/same-predicate features;
- a learned residual joint scorer;
- a permutation-invariant existence head with both absolute and normalized scores.

The query activator reuses the frozen pair encoder, then adds a
query-specific candidate residual and a separate set-aware NULL head. This
keeps write-time predecessor selection and read-time question activation as
different learned tasks.

All binary loaders validate the backbone SHA-256, tensor geometry, per-tensor
CRC, and trailing data. The query artifact is also bound to the exact pair
encoder used during its training.

### Training method

Training and serving use the exact same 0.5B GGUF identity. The backbone is
frozen, with all 24 layers retained and no LoRA, SVD, NAS/NG, or KV reuse.

The current writer curriculum is generated by
`python/prepare_natural_memory_curriculum.py`: 192 training worlds
(1,056 events), 24 development worlds (132 events), and 24 final-test worlds
(132 events). Create and update examples are balanced. Most messages are
ordinary text without source wrappers or dates; approximately 20% contain
an explicit date. Missing time is supervised at the BOS position and
compiled as unknown, never replaced with an invented date. The loader
rejects overlong messages instead of silently truncating labeled fields.

Entities and relations are disjoint across splits. Surface templates are
shared, so this evaluation measures entity/relation generalization, not
unseen-template or unrestricted conversational generalization. LoCoMo and
the independent chat fixtures are never used for training or selection.

The deployment is trained in stages:

1. Capture frozen-backbone features using the exact C tokenizer.
   Field labels are aligned to those tokens. If standalone field encoding has
   different boundary tokens, alignment uses the original tokens' decoded bytes;
   ambiguous matches or spans containing extra text are rejected.
2. Fine-tune the writer's operation, anchors, and field boundaries. Select
   on development create/update accuracy and field localization.
3. Freeze the writer and train contiguous field-span taggers with balanced
   create/update sampling. The deployed maximum field span is 16 tokens.
   Anchor/span fusion is disabled for this configuration.
   Then train the separate full-message operation head while keeping every
   field parameter frozen. It averages the four hidden-state bands, pools
   non-BOS tokens, and combines that mean with the final token's representation.
   Both vectors are L2-normalized. This retains sentence-level change cues
   that may be absent from the extracted value's causal representation.
4. Keep the deployed pair encoder and joint candidate scorer frozen. Train
   only the predecessor-existence head on natural-message candidate banks,
   including missing predecessors and one-candidate positive/negative sets.
   Its seven inputs preserve both normalized ranking information and the
   absolute top two scores. Normalized scores alone cannot distinguish a
   strong singleton match from a weak one.
5. Verify that re-exporting the pair encoder produces the exact original
   binary hash. This permits reuse of the existing query activator, which
   remains bound to that encoder.
6. Require Python/C operator parity, then choose the deployed combination
   on development chat lifecycles. Only afterward run the frozen final
   test through normal chat, export, process restart, import, and fresh recall.

Run the reproducible pipeline on the logged-in GPU server after syncing the
source. Use a new output directory for each run:

```sh
bash scripts/train_natural_memory.sh \
  models/bitcpm4-0.5b-tq2_0.gguf \
  build/memory_v257_05b_typed_writer_all_spans/writer.pt \
  build/natural_memory \
  build/memory_v251_05b_normalized_set_link_pair.pt \
  build/memory_v251_05b_normalized_set_link_pair.bntpair

bash scripts/train_context_memory.sh \
  models/bitcpm4-0.5b-tq2_0.gguf \
  build/natural_memory build/context_memory
```

Pass `-` as the second argument to initialize a fresh writer directly from
the GGUF geometry. The reported model uses compatible pretrained writer
initialization. The pipeline builds `tok_probe` with CMake so runtime
link dependencies, including OpenMP on Linux, are preserved.

Training limits are 800 steps for the writer, 600 for the span taggers,
800 for the predecessor head, and 1,000 for the full-message operation head.
Early stopping and checkpoint selection use development metrics, not final
test results. Component accuracy is not end-to-end memory accuracy.

The final writer is `writer.bntwrite` in the context-operation run directory;
the link artifact is `link/link.bntlink` in the natural-memory run directory.
The writer binary includes the full-message operation tensors. The corresponding
`.pt` files are training checkpoints, not session memory. The compatible pair
and query binaries are also required for serving.

Checkpoints record configurations and data fingerprints. Training and
validation worlds must not overlap, feature caches must match their source
and backbone, and evaluation-only records are rejected. Final test outcomes
must not be used to tune thresholds or select checkpoints.

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
    build/memory_v272_05b_absolute_link/link.bntlink \
  --typed-query-model \
    build/memory_v255_05b_typed_query_activator.bntqact \
  --typed-writer-model \
    build/memory_v273_05b_context_writer.bntwrite
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

Run the final raw-chat lifecycle evaluation after freezing the model selection:

```sh
python3 python/prepare_natural_memory_curriculum.py \
  build/natural_memory/final_eval --test-only --seed 2730916 --valid-worlds 24

python3 python/prepare_natural_memory_eval.py \
  build/natural_memory/final_eval/test.jsonl \
  build/natural_memory/final_eval/final_chat.json --role final

python3 tests/eval_memory_chat_holdout.py \
  build/openai_server models/bitcpm4-0.5b-tq2_0.gguf \
  build/memory_v251_05b_normalized_set_link_pair.bntpair \
  build/memory_v272_05b_absolute_link/link.bntlink \
  build/memory_v255_05b_typed_query_activator.bntqact \
  build/memory_v273_05b_context_writer.bntwrite \
  --fixture build/natural_memory/final_eval/final_chat.json \
  --output build/natural_memory/final_results.json
```

### Test results

The current combination was selected on development data and then evaluated
on a 24-world, 90-question synthetic test (seed 2730916) on September 16, 2026.
The final split uses different entities and relations but shares the
training template family. It was not used for training, calibration, or
checkpoint selection.
The table reports the documented artifact combination's C-runtime regression
results. Reusing this fixture is not a new independent test.

| C-runtime regression metric | Result |
| --- | ---: |
| Current facts, end-to-end exact recall | **53/66 = 80.30%** |
| Unknown questions, activation rejection | **24/24 = 100%** |
| Original chat messages submitted for writing | 132 |
| Write requests accepted (not extraction accuracy) | 132/132 |
| Creates rejected as unresolved updates | 0/66 |
| Updates rejected for missing activated predecessor | 0/66 |
| Predicted changes stored as assertions without a predecessor | 2 |
| JSON/SSE activation and memory-answer parity | Passed |
| Export, restart, import, zero KV reuse | Passed |

Every fact is written through normal `POST /v1/chat/completions`.
Gold entities, predicates, value spans, and answers are never submitted.
Sessions are exported, the server is restarted, and fresh queries are sent
after import. Pre-import checks verify that the new process does not already
have the session's memories. Successful answers must come from neural
activation followed by the compiled pointer; ordinary generation cannot
count as a memory success.

The NULL metric measures activation rejection, not whether an ordinary
generated reply correctly expresses uncertainty. The two metrics should
not be conflated. This synthetic result is not a LoCoMo score, and it does
not establish unrestricted dialogue-memory generalization.

Current limitations are explicit: accepted events can still have incorrect
entity or value spans. Among 13 failed current-fact questions, 6 lacked the
correct entity/value pair in the final active state; 7 had that pair present
but still failed recall. This diagnostic is not itself a query-ranking metric.
Unknown-question rejection is not uniformly solved: on the development set,
the same deployed combination rejected 22/24 unknown questions, with two false
activations. The final set's 24/24 result is not a guarantee of safe abstention.
The active-only single-value reader does not implement historical or multi-fact
recall. Retraction, polarity, and modality are not fully trained.

Natural dialogue remains unreliable: facts can be misclassified as changes,
fields can be extracted incorrectly, and unrelated facts can be linked as
versions of one property. The synthetic curriculum uses six create and six
update templates shared across splits, so its scores do not measure unseen
conversational expression. General memory accuracy on the full LoCoMo test set
has not been established. LoCoMo is evaluation-only.

The documented serving configuration uses the statement/question fallback,
not an experimental learned write gate. It can miss facts embedded in questions
and can mistake quoted or hypothetical content for facts. Research checkpoints
are not part of the validated four-artifact serving configuration.

The writer and seven-input predecessor head pass Python/C parity checks.
The predecessor tests include 1, 2, and 17 candidates; the pair encoder's
binary hash is verified unchanged. The local regression suites pass
20/20 CTest tests and 101/101 Python unit tests. The real-model HTTP cold-start
regression passes six checks, including strict explicit updates, unrelated
memory preservation, and no-KV recall after restart/import. Run it with
`tests/test_openai_server_typed_cold_start.py` and the same six positional
artifact arguments as the chat evaluator. Runtime safety regressions
cover capacity-boundary updates, source deduplication, failed snapshot
writes, malformed imports, and zero-KV streaming.

Evaluation reports under `build/` record each write and query, fixture and
artifact SHA-256 hashes, and the source revision/dirty status. Final-test
failures are diagnostic evidence, not additional training examples.
Create a new sealed test if later development is tuned to this test's
specific examples.

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
- `src/metis/typed_*_model.c` — learned writer, pair, link, and query runtime
- `src/metis/episodic_store.c`, `src/metis/event_store.c`,
  `src/metis/memory_snapshot.c` — evidence, event versions, and atomic snapshots
- `scripts/train_natural_memory.sh`, `scripts/train_context_memory.sh` —
  current staged memory-training and Python/C parity pipeline
- `python/prepare_natural_memory_curriculum.py`,
  `python/prepare_natural_memory_eval.py` — synthetic worlds and raw-chat fixtures
- `tests/eval_memory_chat_holdout.py` — no-KV HTTP write/restart/recall evaluation
- `tests/` — correctness, API, lifecycle, and performance tests
