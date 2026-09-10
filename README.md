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
- addressed episodic memory using `.bnctrl`, `.bnptr5`, and `.bnepisodic`
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

The memory subsystem separates learned memory behavior from mutable memory
data. Learned artifacts decide what operation to perform, where a fact belongs,
and which exact bytes form its value. Session facts remain external so that
they can be updated, deleted, exported, and imported without retraining.

There are three deliberately separated paths:

- **Native neural memory** is the only path whose result is called memory
  recall. Query hidden states activate committed matrix state inside memory
  layers, and the activated result changes normal transformer computation.
- **Neural episode activation plus local pointer** is the current
  accuracy-first design. A neural controller must activate one stored episode
  first; only then may a query-conditioned pointer select an answer span
  inside that episode. It never searches across records and never appends
  retrieved text to the generation prompt.
- **Addressed episodic retrieval** combines a controller, byte pointer,
  external records, and optional retrieval. It is useful for write-policy
  research, persistence diagnostics, and an explicitly labelled RAG baseline,
  but its retrieved-text or direct-copy results are not `.bnmem` recall.

The current artifact responsibilities are:

| Artifact | Responsibility |
| --- | --- |
| `.bnctrl` | Backbone-bound action router and semantic address projector |
| `.bnptr5` | Backbone-bound write/update byte-span extractor |
| `.bnret1` | RAG-baseline natural-evidence retriever; excluded from native recall |
| `.bneptr` | Backbone-bound neural span pointer over one already activated episode |
| `.bnepisodic` | External source records for audit and retrieval baselines |
| `.bnevent` | Typed event versions and explicit supersede/retract links |
| `.bnmem` | Backbone-bound native per-layer memory architecture and parameters |
| `.bnstate` | Mutable native M/S state for one session |

`.bnanswer` and `.bnrouter` are experimental Python artifacts. They are not
part of the deployable C memory path and must not be reported as `.bnmem`
runtime accuracy.

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

The current design conclusion is to establish correct memory behavior before
compression. Native training therefore uses all backbone layers, full-rank
memory projections, a frozen backbone, and no LoRA. SVD, NAS, neuron gates, and
layer pruning remain optional experiments and are not enabled in the current
accuracy baseline.

### Current accuracy-first design

The latest experiments isolate recall into two learned stages:

```text
fresh query, no KV reuse
    |
    v
neural episode activation
    |
    v
one activated episode only
    |
    v
query-conditioned neural start/end pointer
    |
    v
selected episode-local token span
    |
    v
normal answer generation or exact value return
```

This is deliberately not RAG. The pointer receives hidden states for the
episode selected by neural activation; it cannot rank or inspect other
episodes, and source text is not injected into the prompt. Source and query
are encoded independently by the exact frozen 0.5B backbone.

The C runtime now contains the episode-local pointer operator and `.bneptr`
loader. The file stores full-precision query/source start/end projections,
local boundary heads, the maximum span, tensor CRCs, and the exact backbone
SHA-256. Loading against another GGUF is rejected.

The current C integration intentionally stops at the component boundary:
`metis_episode_pointer_select()` consumes the hidden states of one already
activated episode and returns an inclusive token span. The unified serving
path that trains and executes neural episode activation and this pointer
together is still incomplete. Until that exists, oracle-episode pointer
accuracy must not be described as end-to-end memory recall or LoCoMo
accuracy.

### Native recall contract

```text
fresh query (no KV/session reuse)
    |
    v
backbone hidden activations at each configured memory layer
    |
    v
.bnmem learned query/read/gate parameters
    |
    v
activate committed M/S from imported .bnstate
    |
    v
inject the memory contribution into the attention branch
    |
    v
normal backbone generation
```

No retrieved record, evidence text, hidden prompt, or copied value may be
inserted into this path. Read-only queries discard their captured rows instead
of committing the question or generated answer back into M/S. A valid result
must report zero `cached_tokens` and `reused_tokens`, a positive
`memory_activation_reads`, and a positive `memory_activation_l2`.

### Historical addressed retrieval baseline

The addressed path is a structured persistence and RAG comparison pipeline:

```text
user request
    |
    v
fresh backbone prefill
    |
    +--> immutable raw episode (`memory_record`)
    |
    v
V104d durable-fact gate + open semantic addresses (research)
    |
    v
candidate predicate retrieval
    |
    v
V104l length-invariant support + conditional evidence span (research)
    |
    v
deterministic event compiler
    |  subject = source speaker
    |  same value = no-op
    |  explicit functional-property change = supersede exact target
    |  set/event value = append
    `  explicit deletion + exact target = retract
    |
    v
.bnepisodic v4 evidence + optional .bnevent projection
    |
    v
rebuildable BNRET1 token/global index
    |
    v
lexical score + bounded BNRET1 semantic tie-breaker
    |
    +--> exact stored record for memory_copy
    |
    `--> retrieved context for normal generation
```

The final two branches above are query-based retrieval. They must be enabled
explicitly and must never be reported as native memory recall.

The historical addressed research components are:

- **V104d natural writer**: a research-stage writer trained on independent
  human multi-session dialogue, not LoCoMo. It sees the preceding turn and
  current turn, predicts whether durable memory exists and how many atomic
  facts are present, and maps each fact to a broad family plus an open-domain
  semantic address. The exact current turn remains the source evidence;
  crowd-written summaries are supervision, not deployable replacement text.
- **V104l evidence grounder**: the current research-stage extractive
  component. Given the current natural turn and one candidate predicate
  description, it predicts candidate support with a length-invariant binary
  head and trains the exact byte span only when evidence is present. This
  replaces the retired null-versus-all-spans normalization, whose evidence
  mass grew with the quadratic number of legal spans and shifted calibration
  across domains. Candidate and dialogue encodings remain separate and share
  one trainable projection over the frozen exact-0.5B backbone. Training
  samples a positive predicate and a wrong predicate from the same dialogue
  turn, applies a contrastive ordering loss, and balances MultiWOZ and SGD
  sources. Exact token-internal byte offsets recover the source surface.
  Candidate data uses word-boundary-aware evidence matching and preserves
  complete positive/negative groups during limiting. The target value is
  never included in the input. LoRA and KV reuse are not used.
- **V102 controller**: the V101 query/entry address heads combined with the
  V91f action MLP. It classifies `ignore`, `write`, `update`, and `delete`, and
  projects requests and queries into a shared normalized address space.
- **V103 pointer**: a `BNPTR5` byte pointer trained across payload lengths
  1–8. It handles token-boundary changes and exact compound values containing
  hyphens, underscores, slashes, colons, version-like strings, and spaces.
- **Episodic store v4**: exact records plus persisted semantic keys and an
  optional bounded salience value. Update
  replaces the closest active address above the configured threshold; if no
  address matches, it appends a new record. Delete writes a retrievable
  tombstone so an old value is not silently returned.
- **V105c natural retriever**: four full-rank query/entry projections provide
  global and late-interaction token scores. Its contribution is bounded to
  `[-0.49, +0.49]`, so it may resolve lexical ties but cannot overturn a
  candidate with one additional exact lexical match.

Salience is query-independent: it describes whether a turn is generally worth
remembering, not whether it answers a particular question. The runtime can
persist it and use it as a bounded lexical-tie residual, but its default
ranking weight is `0`. The first LoCoMo ablation improved rank 1 slightly while
hurting top-5 retrieval and answer F1, so it is not enabled in the official
path.

The retrieval index is derived data. Writes update the affected entry
incrementally; reset clears it; importing `.bnepisodic` rebuilds it from the
stored source records. The exported memory data therefore remains independent
of cached model activations.

Typed events are optional verified projections over raw episodes. The v2
schema stores a stable `event_id`, source episode, open canonical predicate,
value, `property`/`set`/`event` cardinality, polarity, modality, valid time,
operation, raw-record index, and exact half-open UTF-8 byte spans for subject
and value evidence. A `supersede` or `retract` must name an active
`target_event_id` with the same entity, predicate, and memory kind. Only a
single-valued property may be superseded. Invalid or ambiguous evidence and
targets are rejected instead of updating the nearest vector. Existing v1
`.bnevent` files remain importable; their missing fields receive conservative
property/positive/actual defaults and unknown evidence spans.

Event relation resolution is deliberately not a neural classifier once an
exact structured candidate is already known. A rule baseline reproduced 100%
of the retired V104e training and validation labels because the candidate slot
and value fields already determined the answer; its trained classifier was
therefore redundant and less accurate. The compiler instead uses predicate
cardinality and explicit language evidence: repeated normalized values are
no-ops, different set/event values append, a single-valued property supersedes
only with explicit change evidence, and retraction always requires explicit
deletion evidence plus an exact active target.

Memory action and address features are computed from fresh backbone prefills.
A successful addressed write, update, or delete is committed after request
processing. Retrieval remains off during normal generation unless the server
is started with `--episodic-context-injection`; that option is a RAG baseline.

Every `.bnctrl`, `.bnptr5`, `.bnret1`, and `.bnmem` is bound to the SHA-256
identity and hidden geometry of the exact GGUF used for training. A memory
artifact trained with `bitcpm4-0.5b-tq2_0.gguf` cannot be used with a different
GGUF or a different backbone size.

Run the addressed RAG baseline:

```sh
mkdir -p build/memory-states

./build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  --memory-state-dir build/memory-states \
  --episodic-memory \
  --memory-controller build/memory_v102_05b_combined_controller.bnctrl \
  --memory-pointer build/memory_v103_05b_compound_pointer.bnptr5 \
  --memory-retriever build/memory_v105c_05b_natural_lexical_residual.bnret1 \
  --episodic-context-injection \
  --episodic-top-k 5 \
  --episodic-lexical-weight 0.25 \
  --host 127.0.0.1 \
  --port 8080
```

The response contains the resolved `memory_action`. A client may omit
`memory_action` to use the learned router, or explicitly supply `write`,
`update`, `delete`, or `ignore`.

### Native neural-memory research path

The native path attaches trainable memory modules to transformer layers:

- `.bnmem` contains the trained memory architecture and parameters.
- `.bnstate` contains one session's committed dynamic memory state.
- the GGUF backbone remains frozen during memory training.
- memory reads and writes use fresh contexts and do not require KV reuse.
- `.bnmem` can only be loaded with the exact GGUF used for training.
- recall requests use `"reset_context": true` and
  `"memory_commit": false`; this preserves imported M/S while preventing KV
  reuse and query/answer contamination.

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

Run the activation-only LoCoMo protocol:

```sh
python3 tests/eval_locomo.py \
  build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  build/memory_05b.bnmem \
  build/locomo10.json \
  --mode memory \
  --output build/locomo_native_results.json
```

`--mode memory` rejects addressed retrieval, `.bnret1`, pointer answers,
writer plans, and oracle episodic writes. It also restarts and imports state,
uses a fresh transient context for every question, prevents query/answer
commits, and fails immediately if the C runtime reports no non-zero
memory-layer activation.

This path remains useful for studying learned sparse write/read dynamics, but
the current full LoCoMo result shows that it has not yet learned reliable
open-domain long-conversation memory. It is not the source of the high
controller or pointer component scores reported below.

## Memory export and import

With `--memory-state-dir` enabled, export/import persists the committed memory
state only. It does not serialize KV cache or pending capture rows. The native
path writes `.bnstate`; the addressed path writes the matching `.bnepisodic`
file.

Typed events are stored in the matching `.bnevent` file. Apply an asserted
event only after its source episode has been stored:

```sh
curl -s http://127.0.0.1:8080/v1/memory/event \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "event_id": "alice-residence-v1",
    "episode_id": "episode-1",
    "source_id": "D1:1",
    "entity": "Alice",
    "predicate": "lives_in",
    "value": "Kyoto",
    "valid_time": "2026-09-10",
    "operation": "assert",
    "memory_kind": "property",
    "polarity": "positive",
    "modality": "actual",
    "raw_record_index": 0
  }'
```

When `entity` and `value` occur exactly once in the source record, the server
infers their evidence spans. Otherwise the client must also provide
`subject_text`, `value_text`, `subject_start`, `subject_end`, `value_start`,
and `value_end`. Offsets are UTF-8 bytes, not Unicode character indices.

Query the newest non-superseded, non-retracted value:

```sh
curl -s http://127.0.0.1:8080/v1/memory/event/current \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "entity": "Alice",
    "predicate": "lives_in"
  }'
```

For set-valued or append-only event predicates, query every active value:

```sh
curl -s http://127.0.0.1:8080/v1/memory/event/active \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "entity": "Alice",
    "predicate": "likes"
  }'
```

Commit a memory using automatic routing:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "reset_session": true,
    "max_tokens": 1,
    "messages": [
      {"role": "user", "content": "My persistent project codename is amber-orchid."}
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
    "memory_copy": true,
    "max_tokens": 24,
    "messages": [
      {"role": "user", "content": "What is my project codename?"}
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

### Train the current episode-local neural pointer

The best current extractive recall component uses the synthetic five-task
memory curriculum:

- reconstruction after a memory boundary;
- remember, update, delete, and selective-forget operations;
- operation examples with unrelated distractors;
- multiple facts and entities in one trajectory;
- ordinary post-memory questions that must not be answered from memory.

The generated curriculum contains 50,000 training samples and 2,000 held-out
validation samples, split evenly across the five task files. LoCoMo is not
read by the generator or trainer. The current run samples 2,000 training
trajectories and 1,000 validation trajectories; exact token-span filtering
produces 820 training rows and 386 validation rows. Non-extractive answers and
rows without unambiguous evidence are excluded from this pointer experiment.

Set `DATA_DIR` to the generated five-task curriculum and run:

```sh
GGUF=models/bitcpm4-0.5b-tq2_0.gguf \
DATA_DIR=build/memory_v87_paper_curriculum \
  ./scripts/train_memory_05b_episode_pointer.sh
```

The recipe uses the exact C tokenizer, independently encoded source/query
hidden states from the frozen matching 0.5B backbone, rank-128 normalized
token-to-token start/end matching, local boundary heads, a 160-token maximum
span, AdamW at `1.5e-4`, and early stopping on held-out exact text accuracy.
It uses no KV cache, LoRA, retrieved-text prompt injection, or LoCoMo
training data.

Outputs:

```text
build/memory_05b_episode_pointer.pt
build/memory_05b_episode_pointer.bneptr
```

The `.pt` file is the research checkpoint. The `.bneptr` file is the
CRC-checked C artifact bound to the exact GGUF SHA-256.

### Train the historical addressed system

This comparison pipeline requires a CUDA-capable training machine.

```sh
GGUF=models/bitcpm4-0.5b-tq2_0.gguf \
  ./scripts/train_memory_v102_05b_memory_system.sh
```

The orchestration trains the action router, address heads, and V103 pointer,
then combines the controller heads. Its deployable outputs are:

```text
build/memory_v102_05b_combined_controller.bnctrl
build/memory_v103_05b_compound_pointer.bnptr5
```

The addressed training path:

- uses the exact C runtime tokenizer and frozen GGUF hidden states;
- does not use LoRA or KV-cache reuse;
- keeps training, ID validation, and strict-OOD payloads disjoint;
- trains the pointer on 1–8-token values;
- includes exact compound-value boundaries;
- raises alignment `max_span` to 16 so context-dependent BPE boundary changes
  are not silently filtered from 8-token data;
- never reads LoCoMo as training data.

Train only the current V103 pointer:

```sh
GGUF=models/bitcpm4-0.5b-tq2_0.gguf \
  ./scripts/train_memory_v103_05b_compound_pointer.sh
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
python3 tests/test_episode_pointer.py
python3 tests/test_memory_v91_action_router.py
python3 tests/test_memory_v92_payload_pointer.py
```

Evaluate an exported pointer with the native C operator on the same held-out
feature cache:

```sh
gcc -O2 -fPIC -shared \
  src/metis/episode_pointer.c src/sha256.c \
  -I src -o build/libepisode_pointer.so -lm

PYTHONPATH=python python3 tests/eval_episode_pointer_c.py \
  build/libepisode_pointer.so \
  build/memory_05b_episode_pointer.bneptr \
  build/memory_05b_episode_pointer.pt \
  build/memory_05b_episode_pointer_valid_features.pt \
  models/bitcpm4-0.5b-tq2_0.gguf \
  build/tok_probe
```

Run the current automatic addressed-memory lifecycle regression:

```sh
python3 tests/test_openai_server_memory_auto_route.py \
  build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  build/memory_v102_05b_combined_controller.bnctrl \
  build/memory_v103_05b_compound_pointer.bnptr5
```

This regression checks automatic ignore/write/update/delete routing, exact
compound-value storage, one-record update semantics, export, server restart,
import, exact recall, and deletion tombstones. It also rejects any response
with non-zero cached or reused KV tokens.

### Current measured results — September 13, 2026

The best current neural activation and local-pointer measurements are:

| Test | Result |
| --- | ---: |
| Short opaque held-out neural episode activation | 341/341, 100% |
| Short opaque held-out final exact text after local copy | 341/341, 100% |
| Same short opaque questions with memory disabled | 0% |
| V167 oracle-episode pointer training rows | 820 |
| V167 held-out pointer rows | 386 |
| Python held-out exact span/text | 282/386, 73.06% |
| C held-out exact span/text | 282/386, 73.06% |
| Python/C selected-span agreement | 386/386, 100% |
| KV cache / LoRA / RAG used by V167 | no / no / no |
| Full local CTest suite | 23/23 passed |

The 100% short-opaque result establishes that neural activation followed by
episode-local copying can work when the synthetic distribution is tightly
controlled. It does not establish open-domain memory quality. V167 is the
stronger component check: it trains on the synthetic paper-five-task
curriculum, supplies the correct episode only to isolate read capacity, and
evaluates disjoint held-out entities and values. The identical C and Python
results show that export and C inference preserve the trained pointer
behavior.

The remaining 26.94% pointer errors are mainly boundary failures: the start
head sometimes includes the relation or source template, and the end head
sometimes drops the last subtoken. The next model change should separate
relation matching from value-boundary prediction and add local
neighbor/boundary features. More training steps alone are not justified by
the current error pattern.

These results are component measurements, not a completed LoCoMo result.
End-to-end evaluation must wait for the neural episode activator and
episode-local pointer to be trained and executed together, followed by
export, restart, import, and fresh-query evaluation.

The historical 0.5B addressed components reached:

| Test | Result |
| --- | ---: |
| V91f strict-OOD action routing | 99.19% |
| V103 pointer ID exact bytes, 2,400 samples | 99.17% |
| V103 pointer strict-OOD exact bytes, 2,400 samples | 98.42% |
| V103 explicit-write strict OOD | 99.38% |
| V103 implicit-write strict OOD | 97.63% |
| V103 update strict OOD | 98.25% |
| V103 hyphen payloads | 99.19% |
| V103 slash/underscore/colon payloads | 100% |
| `cobalt-cedar` 5-token OOD regression | exact |
| C write/update/export/restart/import/recall/delete lifecycle | PASS |
| BNRET1 C load, CRC and exact-backbone validation | PASS |
| BNRET1 write/search/export/restart/import ordering | PASS |

These are component and bounded lifecycle measurements. They show that action
routing and exact value extraction are learnable, but they do not establish
general long-conversation memory quality.

The historical full V73 native-memory LoCoMo run used 10 conversations and
1,986 questions. It restarted and exported/imported memory per conversation,
with KV cache disabled. It predates the mandatory activation counters and
read-only-query guard, so its score is diagnostic and must be rerun before it
is accepted under the current protocol:

| LoCoMo result | Score |
| --- | ---: |
| Official categories 1–4 questions | 1,540 |
| Official micro F1, categories 1–4 | 2.49% |
| Official macro F1, categories 1–4 | 3.83% |
| Category 1 mean F1 | 2.35% |
| Category 2 mean F1 | 0.95% |
| Category 3 mean F1 | 9.72% |
| Category 4 mean F1 | 2.30% |
| Category 5 mean F1 | 60.54% |

The result is a failure for general neural memory, despite much higher
synthetic component scores. For the native path, the unresolved questions are
whether durable information is encoded into M/S, whether a fresh query
activates the correct state strongly enough, whether layer fusion preserves
the recalled signal, and whether training credit flows through the complete
write-to-read trajectory. Candidate retrieval is a separate baseline problem.

Therefore:

- LoCoMo remains evaluation-only and must not be used as memory data to
  memorize;
- V103 fixes a concrete write-value boundary problem but does not by itself
  improve LoCoMo reasoning or retrieval;
- backbone parameter count is not assumed to be the primary source of memory
  ability, although every learned artifact remains technically bound to the
  exact training backbone;
- SVD and NAS/neuron-gate compression stay deferred until the uncompressed
  system demonstrates reliable end-to-end memory accuracy.

### Next design target

The accuracy-first target is a jointly trained activation-to-read path:

1. store immutable episodes and train a neural controller to activate the
   correct episode without lexical or vector-search fallback;
2. condition the episode-local reader on the activated episode only;
3. split relation matching from value-boundary prediction and add local
   neighbor features to address V167's over-copy and truncated-end failures;
4. keep the complete differentiable activation-to-pointer trajectory during
   training instead of training only with an oracle episode;
5. package the activator and pointer with exact-backbone identity checks, then
   evaluate after export, process restart, and state import;
6. prohibit KV reuse, cross-episode search, retrieved-text injection, and
   query/answer commits during evaluation;
7. require activation-on versus activation-off and shuffled-episode ablations
   to prove that answers depend on the learned memory activation.

The event writer and episodic store remain available for write-policy and
audit experiments. Their deterministic retrieval, V105 reranking, V106
reader, and pointer output are comparison systems, not substitutes for this
native activation target.

### V104–V106 writer and retrieval findings

These measurements describe writer, grounding, retrieval, and retrieved-text
answering experiments. They do not measure `.bnmem + .bnstate` neural recall:

- The original V104 synthetic structured writer learned its generated
  templates but did not transfer reliably to natural dialogue.
- V104d corrects the objective: it trains on 24,000 independent natural
  dialogue turns and separates memory-worthiness from update resolution. On
  the untouched 4,000-turn natural test split, write precision/recall/F1 is
  87.76%/81.66%/84.60%, versus 80.38% F1 for writing every turn. At the
  validation-selected 0.98 threshold it reaches 95.09% precision with 36.72%
  positive coverage.
- V104d's open semantic address reaches Recall@1/5/10 of
  13.55%/29.50%/38.55% over 3,831 held-out fact summaries. The corresponding
  untrained backbone representation reaches 0.34%/1.07%/2.32%.
- With a recall-first 0.8 threshold fixed from the natural-domain curve, V104d
  selects 33.19% of 5,882 LoCoMo turns while retaining 80.00% of unique gold
  evidence turns. It retains at least one gold evidence turn for 77.84% of
  questions and all gold evidence turns for 68.53%. LoCoMo remains
  evaluation-only.
- Using that threshold as a hard storage filter is rejected. On LoCoMo
  conversation 0 it retained 128/419 records, reduced evidence Recall@5 from
  47.46% to 32.99%, and reduced categories 1–4 answer F1 from 7.46% to 5.00%.
- Keeping all 419 records and adding V104d salience with weight 0.25 is also
  not a deployment win: Recall@1 improved from 25.25% to 26.52%, but Recall@5
  fell from 47.46% to 46.57% and answer F1 fell from 7.46% to 6.92%.
  Salience persistence remains available for future joint training, with
  runtime ranking weight disabled by default.
- A candidate-aware V104e state-relation classifier was rejected after a
  deterministic rule using its exposed candidate slot/value fields reproduced
  100% of both generated splits. The neural classifier's 97.75% validation
  accuracy did not demonstrate language understanding and the obsolete
  training path was removed.
- V104f replaces that target with a non-tautological grounding task: decide
  whether a candidate predicate is expressed in natural dialogue and copy the
  exact value from source evidence. Its training input never contains the
  answer value, and five complete domain-slot pairs are absent from all
  training candidates. Every negative is a wrong candidate paired with the
  same utterance as a positive example; empty-turn negatives are excluded so
  the model cannot solve the task with a generic memory-present detector.
  Candidate and dialogue encodings are independent, exact bytes are recovered
  in two stages (token span, then boundary-token byte offsets), and the
  presence decision must agree with candidate-conditioned token evidence.
  Official development is used for ID thresholding and OOD checkpoint
  selection; official test was used once for the frozen V104f
  predicate-OOD result and is now sealed.
- V104f checkpoint selection now uses the harmonic mean of exact positive
  event writes and negative-candidate rejection. Positive-only accuracy was
  rejected because an all-write policy can appear strong while corrupting the
  memory store. Thresholds are selected on ID validation, checkpoints on
  OOD-development, and OOD-test is evaluated once after selection.
- On the now-retired internal dialogue holdout, the first clean hierarchical
  dual-tower run reached 27.00% exact positive writes, 48.13% negative
  rejection, and 34.59% event harmonic mean. Adding balanced cross-story CoQA
  hard negatives improved negative rejection to 66.25%, but reduced positive
  writes to 24.25% (35.50% harmonic mean).
- The current best V104f research run combines those hard negatives with a
  lower `5e-5` task learning rate and 25% broad-task replay. On that same
  internal holdout it reached 39.00% exact positive writes, 37.13% negative
  rejection, 38.04% event harmonic mean, and 38.06% end-to-end exact. A
  query-prefixed cross-encoder probe regressed to 32.14% harmonic mean and was
  rejected without a full-data run. These internal values guided development
  and therefore are not reported as final unbiased test estimates.
- After the architecture was frozen, the same checkpoint was evaluated once
  on the previously unused official MultiWOZ 2.2 test split: 35.63% exact
  positive writes, 40.75% negative rejection, 38.02% event harmonic mean,
  38.19% end-to-end exact, and 47.00% positive span exact. This official split
  is now sealed from further model selection.
- V104g introduced token-level candidate/evidence cross-attention and a
  joint null-versus-span likelihood. It reached 86.60%/86.70% exact positive
  writes/negative rejection on ID development, but only 37.75%/67.00% on
  predicate-OOD development. Later analysis found that summing probability
  over every legal span makes the evidence score depend on the quadratic
  number of spans, so calibration shifts with utterance length. The objective
  is retired rather than treated as the corrected baseline.
- V104h tied the candidate and evidence projections. It reached
  86.65%/86.95% on ID development and 40.00%/61.13% on predicate-OOD
  development. The marginal change did not fix the objective.
- V104i added 48,000 training rows from 26 SGD services while retaining
  MultiWOZ and CoQA. It reached 85.25%/85.20% on MultiWOZ ID and
  49.63%/55.63% on MultiWOZ predicate-OOD. On eight unseen SGD development
  services it reached 58.50%/89.95%, with 76.50% positive span exact. This
  showed useful cross-service representations but a large domain-dependent
  threshold shift.
- V104j separated a length-invariant support logit from conditional span
  extraction, sampled MultiWOZ and SGD equally, and added a same-turn
  positive-versus-wrong-predicate ordering loss. It reached 88.10%/81.25% on
  MultiWOZ ID, 52.63%/52.75% on MultiWOZ predicate-OOD, and
  57.20%/88.80% on unseen SGD services. Calibration became balanced, but
  MultiWOZ OOD span exact remained 62.38%.
- V104k added reverse attention and candidate-generated start/end query
  vectors. Its best minimum development score improved by only 0.75 points,
  from 52.63% to 53.38%, and MultiWOZ OOD span exact improved by only 0.75
  points. The added structure is rejected; semantic supervision and paired
  data are the larger bottlenecks.
- V104l keeps the simpler V104j structure and fixes the data pipeline:
  literal evidence uses word-boundary-aware matching, balanced limiting keeps
  complete same-turn positive/negative pairs, and all newly encoded rows carry
  explicit source/group identity. Complete training groups increased from
  4,469 to 10,213 for MultiWOZ and from 8,700 to 21,782 for SGD. The current
  best checkpoint reaches 89.90%/84.80% on MultiWOZ ID and
  58.13%/59.50% on MultiWOZ predicate-OOD, with 69.25% positive span exact
  and 58.80% event harmonic mean. It reaches 83.40%/88.55% on SGD ID and
  62.25%/88.50% on eight unseen SGD services, with 78.95% positive span
  exact. An OOD-fitted threshold does not improve the MultiWOZ result.
- Predicate-level V104l accuracy is still uneven. Positive exact writes are
  97.32% for `hotel-area` and 87.32% for `taxi-leaveat`, but only 25.00% for
  `restaurant-food`. `train-destination` reaches 68.80% positive exact while
  rejecting only 47.52% of wrong candidates. The next justified experiment is
  schema-description paraphrase consistency, not more steps or a larger
  pointer.
- V104l remains a Python research checkpoint rather than a `.bnmem`, and
  automatic C event compilation stays disabled. The accuracy-first gate
  requires at least 80% on both exact positive writes and negative rejection
  on an untouched predicate-OOD test, followed by open-domain
  entity/predicate tests.
- V105c was trained on natural evidence QA, never LoCoMo. On held-out natural
  QA it improved Recall@1 from 45.0% to 49.0% and MRR from 0.561 to 0.595.
- A V105d continuation mixed CoQA question/evidence pairs with MSC
  summary/source paraphrases. It improved its MSC validation slice but did not
  improve LoCoMo conversation 0: Recall@1/5 and MRR were
  25.51%/47.59%/37.85%, below V105c's 26.52%/49.24%/39.4%. The dedicated
  V105d recipe was removed; V105c remains the serving retriever.
- On LoCoMo conversation 0, the C runtime ingested 419 turns, exported state,
  restarted, imported it, rebuilt the index, and answered 199 questions with
  no KV reuse. Evidence Recall@1/5/10 was 25.25%/47.46%/52.28%; categories
  1–4 answer F1 was 7.46%.
- V106 free generation, linear/MLP copy pointers, and the contextual reader
  all underperformed the unmodified 0.5B backbone when fed the same retrieved
  evidence. On the same 50-question slice, the backbone scored 15.88% F1,
  the MLP query pointer 3.81%, and the contextual reader 5.43%. They remain
  research artifacts and are not loaded by the serving path.

The measured bottleneck is now split cleanly. Natural memory-worthiness is
learnable with the 0.5B backbone, so the immediate writer bottleneck is no
longer the write gate. It is candidate generation, open-domain predicate
canonicalization, exact value grounding, cardinality-aware event compilation,
and temporal extraction. Candidate relation labels must not be learned when
they are already determined by structured fields. Retrieval still misses multi-evidence
questions, and answer generation loses additional accuracy even when relevant
answer tokens are present. Any routed copy/boolean/temporal/composition module
must beat the frozen backbone on held-out natural QA before C integration.

Run the explicitly labelled addressed/RAG LoCoMo baseline:

```sh
python3 tests/eval_locomo.py \
  build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  - \
  build/locomo10.json \
  --mode retrieval-baseline \
  --addressed \
  --controller build/memory_v102_05b_combined_controller.bnctrl \
  --pointer build/memory_v103_05b_compound_pointer.bnptr5 \
  --retriever build/memory_v105c_05b_natural_lexical_residual.bnret1 \
  --oracle-write \
  --turn-level \
  --output build/locomo_results.json
```

The following commands reproduce the rejected hard-filter ablation; they are
diagnostic, not the recommended serving configuration:

```sh
PYTHONPATH=python python3 python/eval_memory_v104d_locomo_writer.py \
  models/bitcpm4-0.5b-tq2_0.gguf \
  build/memory_v104d_05b_msc_writer_full.pt \
  build/locomo10.json \
  --lib build/libggwshim.so \
  --tok-probe build/tok_probe \
  --confidence 0.8 \
  --output build/locomo_v104d_writer_full_t080.json

python3 tests/eval_locomo.py \
  build/openai_server \
  models/bitcpm4-0.5b-tq2_0.gguf \
  - \
  build/locomo10.json \
  --mode retrieval-baseline \
  --addressed \
  --retriever build/memory_v105c_05b_natural_lexical_residual.bnret1 \
  --write-plan build/locomo_v104d_writer_full_t080.json \
  --turn-level \
  --diagnostics \
  --output build/locomo_v104d_results.json
```

Use `--priority-plan PLAN --priority-weight WEIGHT` to run the all-record
salience ablation. `WEIGHT` defaults to `0`; values must remain below `0.5`,
and the combined neural residual is clamped so it cannot overturn an
additional exact lexical match.

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
- `src/metis/event_store.{h,c}` — `.bnevent` event versions
- `src/metis/memory_controller.{h,c}` — `.bnctrl`
- `src/metis/memory_pointer.{h,c}` — `.bnptr` and `.bnptr5`
- `src/metis/episode_pointer.{h,c}` — `.bneptr` loader and episode-local
  neural span selection
- `src/metis/memory_retriever.{h,c}` — `.bnret1`
- `src/metis/retrieval_index.{h,c}` — rebuildable retrieval features
- `src/x86/` — x86 SIMD kernels and dispatch registration
- `examples/openai_server.c` — HTTP server and memory endpoints
- `python/train_memory.py` — native memory trainer
- `python/train_episode_pointer.py` — frozen-backbone episode-pointer trainer
  and `.bneptr` exporter
- `python/train_addressed_memory_controller.py` — addressed controller trainer
- `python/memory_text_encoding.py` — addressed-memory feature encoding
- `python/prepare_memory_v91_action_router.py` — action-router curriculum
- `python/prepare_memory_v92_payload_pointer.py` — exact payload curriculum
- `python/train_memory_v89_copy_pointer.py` — current `BNPTR5` trainer
- `python/merge_memory_controller_heads.py` — V102 controller assembly
- `scripts/train_memory_v102_05b_memory_system.sh` — current addressed pipeline
- `scripts/train_memory_v103_05b_compound_pointer.sh` — current pointer pipeline
- `scripts/train_memory_05b_episode_pointer.sh` — current episode-local pointer
  training and C export recipe
- `tests/test_openai_server_memory_auto_route.py` — C lifecycle regression
- `tests/eval_memory_payload_pointer.py` — pointer metrics by operation, length,
  and payload family
- `tests/test_episode_pointer.py` — pointer math and binary export regression
- `tests/test_episode_pointer.c` — C loader, backbone binding, and span
  selection regression
- `tests/eval_episode_pointer_c.py` — held-out C/Python parity and accuracy
- `tests/eval_locomo.py` — LoCoMo end-to-end evaluation
