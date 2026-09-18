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

The selected resident model bundle is tracked in this repository. Supply the
matching 0.5B GGUF backbone separately at
`models/bitcpm4-0.5b-tq2_0.gguf`. Session memory, generated training data,
and build artifacts belong under `build/` and are not committed.

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

The bundled memory setup uses the exact BitCPM 0.5B GGUF whose SHA-256 is
`44cb4e0db8374d4247bba391b3d3b7c0b3ee815c36cf2e20d0d1a3570677bd95`.
Provide that GGUF separately at `models/bitcpm4-0.5b-tq2_0.gguf`; it is not
committed. The selected memory artifacts are in
`models/memory/resident-0.5b/`, with file hashes in `manifest.json`.
The resident path is experimental: its measured accuracy is below a useful
general-purpose memory target.

### Architecture

A normal chat message follows this path:

1. The server's statement/question gate decides whether to attempt a write.
   This gate still uses rules unless a separate action controller is supplied.
2. The trained writer reads a fresh prefill of all 24 backbone layers and
   predicts the operation and entity, predicate, value, and optional time
   spans. Invalid extraction fails the write.
3. The server stores the source bytes and compiled spans as an immutable
   event. In resident mode, every nonduplicate observation is retained; the reader
   learns whether a question asks for current or historical facts. The
   writer's operation prediction is reported, but resident mode records each
   new observation as an assertion instead of using the old predecessor linker.
4. A separate fresh prefill supplies final hidden states and input embeddings
   to `resident.bnresid`. It writes token-level semantic and identity
   addresses into the session's resident state.

At recall, the question receives its own fresh prefill. The resident model
activates stored addresses using learned token attention, identity
compatibility, version links, history scope, and a count head. It selects
zero to four events. The server then copies the selected values from their
validated source spans. Stored text is not searched by query or inserted
into the generation prompt. A zero-event decision returns empty content.

The writer and resident reader are trained separately. The GGUF backbone is
frozen; this configuration uses no LoRA, SVD, or NAS/NG compression.
Memory evaluation checks that `cached_tokens` and `reused_tokens` are zero.

### Files and model roles

| File | Resident-mode role |
| --- | --- |
| `models/bitcpm4-0.5b-tq2_0.gguf` | Frozen 0.5B backbone; supply separately with the exact hash above |
| `models/memory/resident-0.5b/writer.bntwrite` | Trained automatic operation and field extraction; used for writes |
| `models/memory/resident-0.5b/resident.bnresid` | Trained token-address write, neural activation, version scope, and answer-count parameters |
| `models/memory/resident-0.5b/pair.bntpair` | Loaded for feature geometry and compatibility; its pair scorer is bypassed |
| `models/memory/resident-0.5b/link.bntlink` | Loaded for current server compatibility; its predecessor scorer is bypassed |
| `models/memory/resident-0.5b/query.bntqact` | Loaded for current server compatibility; its query scorer is bypassed |
| `models/memory/resident-0.5b/resident.pt` | Selected research checkpoint used to export `.bnresid`; not loaded by C |

The current server requires all four typed artifacts together with
`--episodic-memory`, even when `--resident-model` selects the new reader.
It rejects a different GGUF or LoRA for this configuration. Do not substitute
another 0.5B, 1B, or 3B GGUF merely because its architecture is similar.

Session facts are data, not model weights. `POST /v1/memory/export` saves a
content-addressed `.bnepisodic` source file, `.bnevent` event file, and
`.bnresident` address file under `--memory-state-dir`. A `.bnsnapshot`
manifest commits their hashes atomically. Import verifies the files and
restores the resident tensors without re-encoding source messages; a failed
import leaves live memory unchanged. Keep one server writer per state
directory.

### Start and use

Build as described above, provide the matching GGUF, then start the server:

```sh
mkdir -p build/memory-states

./build/openai_server models/bitcpm4-0.5b-tq2_0.gguf \
  --host 127.0.0.1 --port 8080 --ctx 4096 \
  --memory-state-dir build/memory-states --episodic-memory \
  --typed-writer-model models/memory/resident-0.5b/writer.bntwrite \
  --typed-pair-model models/memory/resident-0.5b/pair.bntpair \
  --typed-link-model models/memory/resident-0.5b/link.bntlink \
  --typed-query-model models/memory/resident-0.5b/query.bntqact \
  --resident-model models/memory/resident-0.5b/resident.bnresid
```

Use the same `session_id` for writes and questions. The ordinary chat route
attempts automatic memory unless `"memory_auto": false` is sent:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","messages":[{"role":"user","content":"Morgan\u0027s home city is Lima."}],"max_tokens":16}'

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","messages":[{"role":"user","content":"Which home city does Morgan have now?"}],"max_tokens":16}'
```

Inspect `memory_auto.stored` on the write response. A memory answer includes
`memory_copy.mode = neural_resident_activation_then_compiled_pointer` and
`memory_copy.selected_event_indices`. The answer is only as accurate as the
writer's extracted span and the resident model's selected events. For a
diagnostic forced write, send raw text with `POST /v1/memory/remember`.

Export the session, stop the server, restart it with the same model paths and
state directory, then import before querying:

```sh
curl http://127.0.0.1:8080/v1/memory/export \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo"}'

# Restart the server here with the command above.

curl http://127.0.0.1:8080/v1/memory/import \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo"}'
```

Resident mode supports at most 32 events per session and 128 input tokens
including BOS per write or question. It can return at most four events.
Deletion and the typed-event mutation/query endpoints are unsupported in
this mode; delete requests are rejected.

### Training and export

The resident curriculum is generated by
`python/prepare_memory_set_curriculum.py --diverse-train` with 128 training
worlds and 24 development worlds. The selected checkpoint uses a
128-dimensional factorized resident baseline, then a trainable identity
channel over frozen baseline parameters. Identity training uses synthetic
entity-role labels; those labels are never passed to inference. Both stages
use the exact C tokenizer and the same frozen 0.5B GGUF. The selected
identity run uses seed `2810917`; it trained for 1,000 steps and selected
checkpoint step `800` on development metrics.
The baseline and identity trainers are
`python/train_resident_memory_set.py` and
`python/train_resident_identity.py`; the latter requires a matching
baseline checkpoint and backbone feature cache. The automatic writer is
trained separately on natural-message field labels with
`scripts/train_natural_memory.sh` and `scripts/train_context_memory.sh`.
LoCoMo and the sealed split are not training inputs.

The committed `.pt` checkpoint contains the selected baseline and identity
weights. Re-export the C artifact without retraining:

```sh
python3 python/export_resident_identity.py \
  models/memory/resident-0.5b/resident.pt \
  build/resident-reexport.bnresid
```

The final training package is `training/memory/resident-0.5b/`. It records
the data, selected checkpoints, training entry points and hashes for the
models loaded by resident serving:

| Model | Training data | Training method and script |
| --- | --- | --- |
| `resident.bnresid` | `corpora/resident/{train,valid}.jsonl` and training-only `train_supervision.json` | Factorized resident baseline with `python/train_resident_memory_set.py`, followed by an identity channel over its frozen weights with `python/train_resident_identity.py`. Re-run with `retrain_resident_and_writer.sh`. |
| `writer.bntwrite` | `corpora/writer/{train,valid}.jsonl` | Natural-message write, field-span and context-operation supervision with `scripts/train_natural_memory.sh` and `scripts/train_context_memory.sh`. Re-run with `retrain_resident_and_writer.sh`. |
| `pair.bntpair` | Selected `pair.pt` and `pair_init.pt` are retained; the original raw pair corpus is not hash-bound in the checkpoint | Entity/predicate pairing and version-link supervision with `python/train_typed_pair_verifier.py`; loaded for compatibility, not used by resident recall. |
| `link.bntlink` | Retained writer corpus, frozen `pair.pt` and regenerated backbone features | Predecessor-existence head with `python/train_natural_link_head.py`; loaded for compatibility, not used by resident recall. |
| `query.bntqact` | Exact supervised `feature_data/query_{train,valid}.pt`; matching-ID raw examples in `corpora/query/` | Candidate/NULL activation head with `python/train_typed_query_activator.py`; retraining helper `retrain_query_from_features.py`. Loaded for compatibility, not used by resident recall. |

The original query raw-source bytes are not asserted to match the regenerated
matching-ID examples. The retained supervised feature rows are the exact
query-head training inputs. The pair checkpoint likewise cannot prove its
original raw-corpus identity; the package does not claim a bitwise full
retraining of that legacy compatibility component. See the package
`README.md` for commands, initializers, selection steps and provenance.
Check every retained input and reproduce all five deployed binaries with:

```sh
python3 training/memory/resident-0.5b/verify.py \
  models/bitcpm4-0.5b-tq2_0.gguf
```

### Tests and measured results

Run the C regression suite and the local HTTP development evaluation
(requires the matching GGUF and a fresh output directory):

```sh
ctest --test-dir build --output-on-failure

python3 tests/eval_resident_http.py \
  models/bitcpm4-0.5b-tq2_0.gguf \
  models/memory/resident-0.5b/resident.bnresid \
  build/resident-http-run
```

The HTTP evaluator sends ordinary chat messages to write facts. It exports
sessions, restarts the process, imports state, and asks fresh questions. It
checks zero session KV reuse, unchanged answers after restart, rejection of
corrupt state, and absence of raw-event re-encoding on recall. It supplies
no gold field spans or targets to the server. Results are written to
`build/resident-http-run/summary.json`.

| 0.5B C/HTTP development metric | Result |
| --- | ---: |
| Facts stored | 192/192 |
| Values extracted exactly | 191/192 |
| Current-fact answer sets | 54/96 (56.25%) |
| Multiple-fact answer sets | 57/96 (59.38%) |
| Historical answer sets | 29/48 (60.42%) |
| Correct empty-answer decisions | 22/48 (45.83%) |
| All answer sets | **162/288 (56.25%)** |

The native C operator and Python agree on the same resident states and
C backbone features for all 288 selected event sets. The Python formula
with its training-side features selected correct evidence on 174/288
questions; with C backbone features that fell to 164/288. The writer's
value error reduced complete answers to 162/288. These are development
diagnostics, not LoCoMo results or an independent final-test score.
The resident model still has weak generalization and empty-answer
accuracy. The local C regression suite passed 21/21 tests.

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
- `src/metis/typed_writer_model.c`, `src/metis/typed_pair_encoder.c`,
  `src/metis/typed_link_model.c`, `src/metis/typed_query_activator.c` —
  typed artifact loading and automatic writing
- `src/metis/resident_identity.c`, `python/export_resident_identity.py` —
  experimental native resident activation and checkpoint export
- `tests/eval_resident_http.py`, `tests/diagnose_resident_c.py`,
  `tools/resident_probe.c` — resident HTTP lifecycle and feature-domain diagnostics
- `src/metis/episodic_store.c`, `src/metis/event_store.c`,
  `src/metis/memory_snapshot.c` — evidence, event versions, and atomic snapshots
- `scripts/train_natural_memory.sh`, `scripts/train_context_memory.sh` —
  training stages for the automatic writer
- `python/prepare_memory_set_curriculum.py`,
  `python/train_resident_memory_set.py`, `python/train_resident_identity.py` —
  resident curriculum, baseline, and identity training
- `tests/` — correctness, API, lifecycle, and performance tests
