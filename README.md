# 158BitNet

C11 inference runtime for BitCPM GGUF models, with ARM NEON and runtime x86 SIMD dispatch. It provides a CLI, an OpenAI API server, sparse LoRA loading, optional Apple Metal acceleration, and a learned persistent memory network.

## Build and run

```sh
cmake -S . -B build
cmake --build build -j 8
ctest --test-dir build --output-on-failure

./build/minimal_generate models/bitcpm4-1b-tq2_0.gguf "Hello"
./build/openai_server models/bitcpm4-1b-tq2_0.gguf --host 127.0.0.1 --port 8080
```

The server supports `/v1/models`, `/v1/completions`, `/v1/chat/completions`, SSE streaming and session KV reuse. `BITNET_NUM_THREADS` sets CPU threads; `BITNET_CPU_TIER=avx2` can select an available x86 tier. Metal is enabled with `-DBITNET_ENABLE_METAL=ON`.

Android builds use the NDK:

```sh
ANDROID_NDK=/path/to/android-ndk ./scripts/build_android.sh openai_server minimal_generate gguf_inspect
```

## Learned memory

The model selects original fact spans, write/update/delete operations, subject references and relevant memories. Independent read and update/delete query encoders each produce four relation attention heads, attending to every token of each memory. Their pooling and attention concentrations are separate, so read refinement can preserve update and forget behavior. A nonlinear learned matcher combines raw and projected subject identity. Subject identity and temporal intent remain separate learned channels. A learned identity update gate anchors later read rounds to the model's original subject prediction while allowing evidence to update it. Read relevance and update/delete scope have separate supervision: an update must also find earlier values of the same property. The runtime validates byte ranges and addresses, applies transactions, and saves state. Stored provenance contains the selected fragment and its byte offsets in the original observation. Activation scans the entire memory bank with the learned network and appends selected evidence to the generation prompt. Each iteration scores all records again and replaces the evidence set, so earlier false activations can be withdrawn; convergence requires the same full set. There is no keyword lookup or nearest-neighbor index. A record joins the set when the reader's learned joint probability exceeds `NS_ACTIVATION_PROBABILITY` (0.1): the authored development bank activates 76% of positive questions at the neutral 0.5 point and 80% at 0.1, with zero false activation on its null questions at both. Later rounds and the served prompt carry at most `NS_EVIDENCE_BUDGET` (8) highest-scoring records, so the augmented context and the prompt stay bounded independently of bank size. An unresolved reference does not suppress activation; the reader then scores the bank from the query alone, so the decision remains with the model.

`models/memory-current/model.bnsmodel` contains the current network parameters; `model.pt` supports continued training. A session snapshot uses `.bnslots` version 2 and contains committed memories, provenance, encoded tokens and each slot's 16-wide context representation. The model is bound to its backbone SHA-256, and snapshots are bound to the exact memory model. Import validates the complete snapshot before replacing existing state. Only the current model and snapshot format are supported; snapshots contain no KV cache or pending data.

```sh
./build/openai_server models/bitcpm4-1b-tq2_0.gguf \
  --memory-network models/memory-current/model.bnsmodel \
  --read-head models/memory-current/read-head.bin \
  --memory-state-dir build/memory-state --host 127.0.0.1 --port 8080
```

`--read-head` enables the learned read-decision head, which replaces the reader
logit for activation while keeping the same bank scan and convergence rounds.
The artifact is bound to the memory model's backbone SHA-256.

A request may supply literal provenance in `memory_observation`:

```json
{
  "session_id": "example",
  "messages": [{"role": "user", "content": "I keep my spare tools in a blue box."}],
  "memory_observation": {
    "speaker": "Aileen",
    "timestamp": "2034-02-17 10:30 UTC",
    "source_id": "turn-01"
  }
}
```

Use `memory_read_only: true` for questions that must not change memory. `/v1/memory/export` and `/v1/memory/import` operate on the session identified by `session_id`. Memory requests use fresh backbone contexts; persistent KV reuse is disabled.

Set `BITNET_NS_FEATURE_TRACE=1` when measuring native feature evaluation. Each encoding writes a JSON event to stderr with its process ID, actual KV positions before and after evaluation, evaluated hidden rows and success status. Tracing defaults to off and omits text and token IDs. `python/neural_feature_trace.py` validates a complete child log against the launched PID; also check the child exit status and every operation result. Missing or failed measurements do not establish zero reuse. Attention to tokens within the same fresh encoding is distinct from reusing KV across encodings.

All memory development, training and tests use the 1B backbone. Its complete 2048-wide hidden states and 2048-wide token embeddings form a 4096-wide input; each memory token is 64-wide, and the query has four 64-wide relation heads plus a 64-wide intent channel (320 values). Training starts with new memory parameters and identity-bound C features. Current acceptance results belong in `models/memory-current/validation.json`.

The network contains 91 tensors and 4,988,911 parameters; its exported model is 19,956,428 bytes. A source boundary head reads the previous, current and next source representation to predict begin and end scores. A bias-free 12288-to-2 projection also reads all three original 1B word features, with hidden and lexical channels normalized separately for each word and zero padding outside the observation. A separate 12352-to-1 projection learns fact-versus-background membership from the unchanged 64-wide source representation followed by those normalized raw word features. The source representation includes the complete observation and interaction context, plus bidirectional word cues; it retains its original scale. Membership adds the same score to WRITE, UPDATE, DELETE and inside-source tags, preserving their relative action scores. The end score adjusts source continuation, source termination and the final CRF state. Source representations and actor-reference heads stay fixed during residual training. Memory selection still uses learned full-bank activation; source decisions use learned scores without punctuation or keyword gates.

A learned source-context network compares this full-context source representation with the same encoder's body-only view. The latter retains the complete body summary, word projection, encoder bias and bidirectional word context; only the historical summary is zero. For each source word, the network reads 2176 values: the 64-wide body view, the 64-wide full-minus-body difference and the normalized 2048-wide original hidden state. A 32-unit tanh layer feeds an unbounded scalar output, added equally to the four source tags. References always retain the original full-context representation. The output layer starts at zero, preserving initial source decisions while allowing first-step output-layer gradients. The network uses existing C backbone features and requires no additional backbone pass.

A separate source-kind residual reads 4096 values: the last 2048-wide backbone hidden state followed by the mean hidden state of the complete fact text, with one L2 normalization over the concatenation. Its three logits supplement the existing classifier on the 64-wide mean memory representation. This preserves independent temporal cues from the 1B backbone without changing memory-token or query geometry, and uses the same C features without another backbone pass. Source-text kind remains separate from physical current, superseded or tentative record state.

A query-conditioned context branch supplements the read matcher without changing its four token attention heads, pooled token values or pair inputs. The complete fact's last and mean 2048-wide backbone hidden states form one jointly normalized 4096-wide summary. A bias-free projection U maps it to 16 values followed by tanh; each slot stores this representation. A separate bias-free projection V maps the query's 256 relation values to 16 values followed by tanh. Their elementwise product passes through the 16-to-64 projection B and is added before the existing matcher's hidden tanh. The read logit adds the existing output weights applied to the difference between corrected and original hidden states. Zero B reproduces the original matcher. The frozen output bias plus or minus the sum of absolute output weights bounds its mathematical logit range; this necessary envelope does not prove pair-separation capacity. Scope uses its separate unchanged matcher. Identity and temporal parameters and original-subject anchoring remain unchanged; different selected evidence can still change feedback queries and gate outputs. The branch requires no additional backbone pass.

The query binder has its own learned raw-feature residual. The last and mean complete 4096-wide backbone features of the query body form one normalized 8192-wide summary; the interaction context produces another. A bias-free 16384-to-64 projection followed by tanh is added to the original mean writer representation for query need, source cue and origin predictions. Write spans and write references continue using the original representation. Zero initialization exactly preserves the parent. C computes the residual before clearing the context scratch space for the body-only write view. The model still learns all semantic decisions, including unresolved references; no source is forced when a query is ambiguous.

The current model remains experimental. Component and serving status is recorded in `models/memory-current/validation.json`. Qualification still requires both an 80% strict correct-fact fraction and 80% write F1, plus 70% positive and overall complete-support accuracy after real new-process import and read-only queries with zero KV reuse. The LoCoMo run below meets the 80% activation-recall target on category 1–4 questions; it does not establish write, evidence or answer quality.

## LoCoMo evaluation

LoCoMo is evaluation-only: it is never used for training or checkpoint selection. The evaluator reads the pinned `data/evaluation/locomo10.json` (upstream `snap-research/locomo` revision `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`, dataset SHA-256 `79fa87e9…`). Every conversation is written automatically by the learned writer, exported, and imported into a new server process through an empty-bank probe; questions are answered read-only with the snapshot unchanged. All 1,986 questions of the full run verified `read_only` and zero persistent KV positions.

Full 10-conversation result (`build/recall-full10/`, 8,291.7 s, zero request errors) with the current activation floor, compared with the earlier full-set run (`build/locomo-full-clean44/`) of the same writer:

| Metric | Current | Earlier |
| --- | --- | --- |
| Category 1–4 activation recall | **1,319/1,540 (85.65%)** | 117/1,540 (7.60%) |
| Category 5 records activated (of 446) | 352 | 29 |
| Category 5 null rejection | 21.08% | 93.50% |
| Annotated evidence recall (per question) | 40.56% | 3.42% |
| Annotated evidence coverage (any / all turns) | 45.96% / 35.61% | 3.65% / 3.26% |
| Annotated evidence F1 | 7.72% | 3.44% |
| Strict annotated support-set exact | 1.37% | 3.13% |
| Answer F1 (categories 1–4) | 8.81% | 4.60% |
| Normalized exact (categories 1–4) | 0.52% | 0.07% |

Category activation recall is 240/282 (1), 278/321 (2), 81/96 (3) and 720/841 (4).

The gain comes from the activation operating point, not a new retriever. Of 1,540 category 1–4 questions, 641 have a write-side reference and 899 do not. Activation rises from 117/641 (18.25%) to 557/641 (86.90%) for resolved rows and from 0/899 to 762/899 (84.76%) for unresolved rows, because an unresolved reference no longer suppresses activation: the reader scores the bank from the query alone. `NS_ACTIVATION_PROBABILITY` (0.1) and the 8-record `NS_EVIDENCE_BUDGET` keep the augmented context and prompt bounded independently of bank size.

The trade-off is explicit. The lower floor also activates category 5 questions, so their null rejection falls from 93.50% to 21.08%. Active sets are large (mean 39.7 records, median 8, p90 117, maximum 915), 277 questions exhausted the round budget, and only the 8 highest-scoring records enter the prompt. Evidence F1 7.72% and strict support-set exact 1.37% show that activation recall is not evidence precision: correct turns activate together with many unrelated records. Answer F1 8.81% is not answer qualification. Report activation recall, evidence F1 and answer F1 separately, as above.

Generation requests the specific detail asked for and anchors relative time to each source message time. An empty activation explicitly says no stored background is supplied. Relative-date arithmetic and unsupported personal dates require independent answer evaluation; manually supplied evidence does not qualify automatic memory accuracy.

Provenance: the measurement used the experimental writer export `writer-conversation-clean44` (SHA-256 `eec53c03…`), which the development guard rejected for adoption; it is committed as `models/memory-locomo-full/model.bnsmodel` for reproduction only. The retained serving model is `models/memory-current/model.bnsmodel` (SHA-256 `fd68c6dc…`). The clean44 writer covers 1,425/1,425 annotated evidence turns, while the retained writer covers 95.8%, so this activation number is an upper bound for the retained model's bank. The reported run reused the earlier run's write artifacts (the evaluator skips an existing `written.json`; new runs record `writes_reused` per conversation) and re-ran every question with the new activation configuration. The comparison is not a controlled ablation: the earlier run also used the learned read-decision head, batch-write ingestion and three workers. The activation floor was authored from the independent development bank, not selected on LoCoMo.

Run the full configuration (all ten conversations; omit the pilot limits):

```sh
python3 python/eval_locomo_slots.py build/openai_server \
  models/bitcpm4-1b-tq2_0.gguf models/memory-locomo-full/model.bnsmodel \
  --output-dir build/locomo-full
```

Each conversation uses a fresh server process; use a new output directory for every evaluation. `--ingester` selects the native batch writer instead of raw-turn HTTP writes, and `--read-head` enables the learned read-decision head. The manifest pins the dataset, binaries, model and source hashes.

## Training and evaluation

Standard writer/reference training supervises NEED for every gold write span. BIO and origin losses apply only when the gold span needs a reference. Self-contained writes leave these unused outputs unsupervised; genuinely unresolved references keep `need=True` with empty sources and still teach all-O BIO and zero origins. This mask follows gold supervision during training and adds no serving rule.

Authored native episode evaluation keeps the scorer-only canonical subject in `span.entity`, separately from an emitted external reference. Self-contained facts use `need=False` and empty `references`; resolved facts use `need=True` and one reference from the context or causal body prefix. Explicit reference ranges must address the original UTF-8 bytes before the selected body span. Gold fields never enter the native writer's five-field observation transport.

Query subject binding is trained by `train_neural_dialogue.py`. Its `--train-heads query` mode trains all thirteen query-reference tensors (1,193,032 parameters); `query-context` trains only the new raw projection (1,048,576 parameters); `query-gate` trains only the two need-gate tensors (65 parameters); `query-source` freezes that gate and the raw query projection, and trains the ten source-selection/BIO/origin tensors (144,391 parameters). Context, gate and source modes require a trained `--resume`. Source encoding, writing, memory encoding, full-bank activation, scope, identity and time tensors stay fixed in all four modes. Source-only fit data must include queries with used-source supervision. These modes are distinct from the reader's three-tensor `query` mode below.

The reader's `--train-heads identity-match` mode trains only the four existing identity matcher tensors (33 parameters), freezing the other 87 tensors. It requires `--resume --freeze-memory-encoder --semantic-topic-weight 0` with zero feature dropout and anchor/functional weights; identity-diversity, expansion and evidence-cues are rejected. Ordinary recall and scope/lifecycle development groups must separately preserve their initial results before a candidate can be selected. Raw fit provenance is checked before conversion and again at the loss boundary, and diagnostic checkpoints save the current mode's constraints. Query references come from the frozen writer, but fact banks still use authored supervision; this mode's component scores do not qualify automatic native writing or memory accuracy.

`--query-binding-generalization` adds 108 independent fit questions and a separate 54-question development group. They cover explicit subjects requiring no external reference, resolvable speaker/history references, and genuinely ambiguous references. Anonymous first-person questions retain the actual `User` speaker role. Fit batches give each available class eight examples, divided equally between new and replay examples when available. `--query-binding-anchor-weight 1 --query-binding-anchor-margin 1` preserves capped component margins of an independent frozen thirteen-tensor parent on parent-correct fit need/BIO decisions. Parent logits use the parent's own raw projection from the unchanged original candidate, including when resuming a nonzero projection. Supervision averages nonempty class means; the worst need and BIO margin violations supply the anchor. Development and sealed rows are rejected from fit sampling and supervision before feature extraction. Authored spans and teacher masks never enter serving, and LoCoMo is excluded from fitting and checkpoint selection.

Query-binding selection ranks the combined original and added development groups by worst class accuracy, mean class accuracy and complete row accuracy. Every original, boundary, narrative and added group separately retains source quality, complete references, need decisions, per-class binding accuracy, ambiguity rejection and wrong-subject floors. Actual terminal and selected states are reported separately, and the initial state remains eligible. Soft fit anchors and finite development floors do not guarantee unseen accuracy.

`--matcher-corrections` is an experimental fit-only objective for resumed `matcher` training with a frozen, undropped memory encoder, zero topic weight and full-parent retention. It reads the independent parent's actual identity/temporal log gate, avoiding numerical reconstruction at the gate=0.5 boundary. Reachable parent errors learn a finite signed semantic margin. Every reachable correct pair retains its capped margin, including correct evidence inside an incomplete support set. Wrong-positive and wrong-negative losses separately average each eligible query's worst violation. Correct-pair protection uses the largest capped violation across the entire batch, so adding unchanged correct queries cannot dilute a newly lost correct margin. This is a training preference; it can concentrate gradients on an outlier or limit difficult corrections and does not establish held-out accuracy. Teacher eligibility remains frozen and no teacher mask enters serving. Main training replaces both activation BCE and relation auxiliary supervision; replay replaces its BCE while retaining full-parent margin and probability preservation. Reports and selected checkpoints retain this run's initial ordinary views, initial evidence views and initial selection score separately from the selected results. These baselines are captured before refinement and replace inherited resume metadata, including when a later checkpoint is selected.

Reader refinement and writer query-binding refinement save `last-step.pt` before restoring the selected checkpoint. It contains the actual final training state and is marked diagnostic-only and ineligible for selection. `selected.pt` remains eligible even when refinement resumes from a diagnostic checkpoint; it retains the selected state and this run's freshly captured initial metrics. This separates failed-refinement diagnosis from model selection.

```sh
python3 -m pip install -r requirements.txt
python3 python/prepare_neural_colloquial.py --output build/curriculum
python3 python/augment_neural_colloquial.py \
  --curriculum build/curriculum/curriculum.json --output build/augmented

python3 python/train_neural_dialogue.py \
  --gguf models/bitcpm4-1b-tq2_0.gguf --probe build/neural_slot_probe \
  --curriculum build/augmented/curriculum.json --output build/writer \
  --replay-dialogue --replay-episodes --boundary-generalization \
  --narrative-diversity --source-context-generalization \
  --steps 1800 --learning-rate 0.001

python3 python/train_neural_dialogue_reader.py \
  --gguf models/bitcpm4-1b-tq2_0.gguf --probe build/neural_slot_probe \
  --planner-checkpoint build/writer/selected.pt --features build/writer/features.pt \
  --curriculum build/augmented/curriculum.json --output build/reader \
  --replay-episodes --evidence-cues --identity-diversity --expansion \
  --fit-paraphrases --feature-dropout 0.1 --steps 1200 --learning-rate 0.0002
```

Set `--device cuda` on the GPU server, `--device mps` on Apple GPUs, or use CPU. Writer training can start without `--resume`. Reader training can start without `--resume` when a trained writer is supplied with `--planner-checkpoint`; the writer parameters remain fixed. `--replay-dialogue`, `--replay-episodes`, `--expansion`, `--fit-paraphrases`, `--identity-diversity` and `--evidence-cues` add independent synthetic training cases. Fit-only support corrections preserve held-out data. Paired expansion questions explicitly distinguish their exact source from same-subject, same-property alternatives. Evidence augmentation is eligible only when every stated cue belongs to the sampled fact bank; complete multi-cue sets remain intact. Semantic auxiliary losses transfer complete supporting content across synthetic actors while final activation labels retain the correct actor, and distinguish wrong relations and specifically annotated alternative events. Complete content, property and original source kind define a transferred source; specific object alternatives must match the query's requested source kinds. Physical current/history/tentative state remains a separate activation teacher. Evidence augmentation mixes one to four supporting and distracting facts and keeps the original question and its model-predicted reference. Checkpoint selection uses development data and keeps the initial checkpoint when training degrades.

With `--expansion`, the five material-preference sources describe different making projects, share original source kind 0, and may coexist with zero, one or two sources per actor. Other actor/property groups retain exactly one alternative source, avoiding conflicting current-home or current-job assertions. `--expansion-material-max-sources 2` is the default zero/one/two-source curriculum; `--expansion-material-max-sources 1` supplies a fixed single-source control. The original 480 facts, 1,440 specific questions and 64 null questions remain unchanged; `--evidence-cues` retains the same 400 authored cue additions before filtering them to each actual bank. Eight broad questions ask for the materials and reasons across the five declared uses, and eight unsupported questions ask about a garden hammock. Broad supports bind to all sources actually present in the sampled world, including an explicit empty set; absent specific questions retain their paired author source while final activation labels use actual bank membership. All evidence cues must exist in that bank. Each expansion batch reserves 24 source-present, 8 original null, 4 source-absent, 8 broad and 8 hammock questions. Reports separate initial, actual last-step and selected results for fixed zero/one/two-source fit worlds; these diagnostics never select checkpoints. Source kind describes the original text and remains separate from physical slot state and insertion order.

Continued reader training supports `--train-heads identity`, `read`, `scope`, `semantic`, `matcher`, `activation`, `query`, `context-query`, `context`, `temporal` or `all` (the default for a fresh reader). `matcher` updates only the read classifier and requires `--semantic-topic-weight 0`. `temporal` requires a trained resumed reader and zero topic weight; without prefit it updates ten base/context kind, temporal intent and version-gate tensors. With `--kind-context-prefit`, contextual-kind calibration precedes main training, both kind classifiers freeze, and only six intent/version tensors train. Text kind uses the annotated source kind, while temporal intent permits multiple physical record states. Query binding, semantic matching, identity and scope parameter tensors remain frozen in temporal mode; time changes can still affect feedback query mixing. Scope refinement trains its classifier and dedicated query channel. Recall losses normalize supporting, rejecting and hard-negative positions within each question before averaging eligible questions, so repeated source variants do not dominate single-support questions. Scope selection uses independently authored requests and subjects. Lifecycle training retains physical insertion order, excludes deleted records, and replays superseded, current and tentative record states even when their text is unchanged. Recall selection includes fixed zero-, one- and four-evidence challenges and ranks their worst score first. Questions are stratified by relation, temporal kind and query form, with every nonempty relation covered; cue sampling is fixed independently of teacher labels and query order. Semantic, matcher, activation, query, context-query, context and temporal refinements must also preserve every ordinary development view's strict positive support accuracy, false matches and null rejection. `--qualification-curriculum` excludes the reserved native question observation IDs from checkpoint selection. These component challenges use supervised banks and do not replace native acceptance.

In `train_neural_dialogue_reader.py`, `--train-heads query` refines only `query_encode.weight`, `query_encode.bias` and `query_pool.weight` (262,592 parameters). It requires a resumed reader, `--freeze-memory-encoder`, zero feature dropout and topic weight, and positive `--reader-functional-weight`. All other 88 tensors, including the 46 source-planner/reference tensors, memory encoder, read classifier, three context projections, attention scales and scope channel, remain frozen. The query changes can still alter the frozen feedback identity/time gates' inputs; full-parent replay therefore recomputes both parent and student traces. The original model-predicted subject remains available to the learned identity update gate. Fixed-gate matcher corrections and matcher anchors cannot be combined with query refinement. Ordinary component and scope guards apply.

`--train-heads context-query` refines only V (`read_context_query.weight`, 4,096 parameters), freezing the other 90 tensors. `--train-heads context` refines U, V and B (`read_context_memory.weight`, `read_context_query.weight` and `read_context_delta.weight`, 70,656 parameters), freezing the other 88 tensors. Both require `--resume`, `--freeze-memory-encoder`, `--feature-dropout 0`, `--semantic-topic-weight 0`, `--reader-anchor-weight 0` and a positive `--reader-functional-weight`; fixed-gate matcher corrections are disallowed. Memory-token encoding, the read classifier, query encoding and pooling, attention scales, kind, identity, temporal, scope and all 46 source-planner/reference tensors stay frozen. The current nonzero B permits U/V gradients immediately; zero B provides no U/V gradient, so V-only refinement cannot learn from that initialization.

An independent frozen complete 90-tensor parent and the student recompute their own U slot contexts live from shared immutable original-hidden last/mean C features; they do not share projected sidecars. Feedback queries are recomputed independently, retaining the original model-predicted subject for the identity update gate. Query geometry remains 320 values, and changed evidence can still alter the frozen gates' inputs and outputs. Changing U requires regenerating stored slot contexts from original features or fact text. Any changed model artifact requires freshly prepared slots and model-bound snapshots; an existing snapshot cannot be rebound to different weights. The current 90-tensor model and snapshot-v2 formats stay unchanged, with no historical-format loading. Ordinary component and scope guards still apply, and failed refinement retains the initial eligible checkpoint.

Matcher refinement can additionally use `--reader-anchor-weight 1 --reader-anchor-margin 1 --freeze-memory-encoder --feature-dropout 0`. Detached parent matcher parameters protect complete, correctly predicted fit support sets using capped margins under the unchanged identity and temporal gates. Training computes the actual identity/time log gate directly; rounded final and semantic outputs never reconstruct it. Gates at or below one half cannot activate a record and receive no semantic preservation gradient. Ineligible pairs are masked before squaring to avoid irrelevant loss overflow. The largest lost margin per question determines preservation loss; parent and student equality starts at exactly zero loss and gradient. This training protection does not guarantee development accuracy. Public component and scope guards still select the checkpoint, with checks controlled by `--eval-every`.

`--reader-functional-weight 1 --reader-functional-margin 1` supports resumed `semantic`, `matcher`, `activation`, `query`, `context-query`, `context` and `temporal` refinement with frozen memory-token encoding and no feature dropout or matcher anchor. An independent frozen copy of the complete parent recomputes its own query, bank, context projections, identity, time and feedback outputs from shared immutable C inputs. Old fit banks replay gold support sets alongside capped final-activation margins and probability distillation for parent-correct complete sets. Incorrect parent sets are not distilled. Sampling includes zero-, one- and four-evidence views; development, new binding and reserved native queries cannot enter protection. These fit measurements do not select checkpoints or qualify automatic memory accuracy.

`--train-heads activation` jointly refines semantic matching, the three context projections and the existing intent/version heads, requiring a resumed reader, full-parent retention, frozen undropped memory-token encoding and zero topic weight. It trains 19 tensors and freezes the other 71, including source planning, identity, both source-kind classifiers and scope. Final-activation loss retains the time gradient path; multi-state intent supervision distinguishes current, superseded and tentative records. Evaluation reports the positive supports blocked by the current identity/time gates with saturated semantic scores. This diagnostic never masks training targets or serving records and does not affect checkpoint selection.

In `temporal` mode, full-parent retention enables a dedicated fit curriculum. Four of every five main steps use the complete lifecycle bank with eight zero-evidence questions each for current, history, tentative and all versions. The fifth uses the complete old fit bank with eight positives per state and eight null questions. A separate classification batch contains twelve original source-kind examples per class; it does not shrink the recall bank or inject gold metadata into scoring. Main, kind and replay sampling use independent seeds. Time correction trains positive supports only when the frozen parent's semantic/identity probability permits activation, and negative same-subject/property records only in explicitly disallowed physical states. It applies to main and replay correction loss; complete-parent margin/distillation, full-bank evaluation and serving remain unmasked. All queries retain intent supervision, including supports that time alone cannot repair.

`--kind-context-prefit` enables fixed CPU float64 ridge calibration of the 4096-to-3 contextual-kind residual on all original fit facts. Its target is `3 * one_hot(source_kind) - (base_logits - base_logits.mean(-1, keepdim=True))`, with class-balanced samples, ridge 0.001 and an unpenalized intercept. The 64-to-3 base classifier stays fixed. Calibration checks finite parameters and every fit classification's positive margin after conversion to the real device and precision, then freezes all four kind tensors for the six-tensor intent/version phase. Calibration failures preserve the existing residual. Every development bank and original kind class must preserve its initial correct count; a failed floor stops before main training, and step zero remains eligible. This boolean option requires focused full-parent temporal retention. Original kind labels enter training only; serving computes both classifier outputs from native features.

`--relation-bindings` adds a separate fit bank of 36 independent facts and 96 questions. Each actor's same device has checking, storage and care facts, so object nouns alone cannot identify the correct property. Questions include one- and two-fact supports plus absent-property nulls; original development groups remain unchanged. The frozen writer predicts references from ordinary original observations before feature extraction. Semantic, matcher, activation, query, context-query or context training with zero topic weight samples this bank every fourth step. `--extra-features build/reader-binding-1b/delta-features.pt` reuses its fresh, identity-bound 1B C features alongside the base reader cache. Initial, last-trained and selected fit metrics distinguish fitting from checkpoint acceptance and do not affect selection. These supervised-bank examples and their labels never enter serving.

Relation refinement can use `--train-heads semantic --freeze-memory-encoder --semantic-topic-weight 0.05 --attention-learning-rate 0.01`. It freezes memory-token encoding, kind predictions and the separate scope channel while learning the read query, classifier, feedback gate, four attention scales and three context projections. All three context projections train in `all`, `read`, `semantic`, `activation` and `context` modes; `context-query` trains V only. They remain frozen in `matcher`, `query`, `temporal`, `scope` and `identity` modes. Changing U requires regenerating slot contexts, and every changed model artifact requires fresh snapshots bound to its exact identity. A fixed scope development guard rejects regression in strict matching, false matches or null activation. The 33 source planner tensors and 13 query-reference tensors are synchronized with `--planner-checkpoint` and frozen even when resuming, so training query references and exported source parameters agree. Reference-only binding skips the boundary head. Feature caches validate their exact text set, backbone and probe identity. Supply multiple cache paths with one `--extra-features path1 path2` option or repeat the option; all paths are retained in order and an empty supplied option is rejected. The 35 current corrected expansion inputs are in `build/memory-1b/reader/current-source-features.pt`; they are immutable C features bound to the current 1B backbone and fixed probe. Independent object negatives and hierarchical relation/question/evidence-family sampling reduce template shortcuts; metadata scoring gradients are detached.

`--episode-relations` adds independent fit content variations and broad questions requiring every supporting fact of a person's relation. `--relation-paraphrases` requires that option and adds two authored question families per each of the existing 48 object-specific fit facts: 96 additional questions about supported causes, frequency, conditions or details. It preserves all facts, original questions, complete supports and same-person alternative-object negatives. Family IDs participate in balanced sampling; duplicates and changed source/target definitions are rejected. Development and acceptance remain unchanged.

Source training uses `--train-heads writer-edge` for all five boundary tensors, or `writer-raw` to update only the residual and freeze the other 90 tensors. `writer-context` freezes the residual's current-word columns as well, learning only its two neighbor branches with zero optimizer decay. `--boundary-generalization` supplies separate synthetic fit and development examples with different subjects, facts and wrappers. These residual modes use complete-path CRF and endpoint contrast losses. On incorrect fit plans, auxiliary begin/end supervision targets only the parent's misplaced boundaries, averaging each nonempty target class equally. Unchanged positions preserve parent scores only when parent and gold agree on fact membership; words outside the gold fact receive no end-label supervision. `--boundary-anchor-weight` preserves outputs of a frozen copy of the complete resumed parent on already-correct fit examples. `--boundary-path-anchor-weight` protects that parent's advantage over every incorrect complete source plan, including inserted facts, omitted facts and boundary shifts; `--boundary-path-anchor-margin` caps the preserved advantage. Forward/backward max-marginals find the strongest incorrect path with ordered parallel scans. Parent parameters and fit-only teacher structures are immutable and cached; initial protection loss and gradient are zero. `--eval-every` controls development checks. Selection requires improving the attached development F1 score while preserving complete-plan, source, action, fact F1, actor-reference accuracy and ambiguous-reference safety on every attached group. `--narrative-diversity` adds forty independent multi-fact fit turns with interleaved questions and fictional controls, plus twelve public development observations. Selection preserves each group's complete plans, source/actions, references and fact F1, and improves the mean of the added development F1 scores. The initial checkpoint remains eligible. LoCoMo and sealed native evaluations are excluded from training and selection.

`writer-membership` updates only the fact/background projection; `writer-content` jointly updates it and the five boundary tensors. `--membership-weight` controls fit-only correction of the frozen parent's missing and spurious source words. Missing words use their actual gold source-tag score against O; spurious words reject every source tag against O. Each present error class receives equal weight, and membership-matching positions preserve the parent's residual. These masks are immutable parent/gold comparisons, never masks from the changing student or inputs to serving. Correct observations retain full-path and complete residual anchors. The backbone, source encoder, action differences, actor/query references and read/scope parameters remain fixed. Sparse correction prevents one missed tail from being diluted by a long sequence of already-correct source words. This objective does not guarantee held-out accuracy; all development floors still apply.

`writer-context-residual` trains only the four source-context network tensors, freezing the other 87 tensors. Residual anchors snapshot the complete parent boundary, membership and source-context network parameters. Correct observations and matching word positions preserve total membership, including a nonzero parent network output. `writer-content` trains six tensors and keeps the source-context network fixed. `writer-context-content` jointly trains those six tensors and the four source-context tensors while retaining complete-parent anchors and development quality floors.

`writer-context-binding` requires a trained resumed writer and trains those ten content tensors plus the two write-reference need-gate tensors. Gold span means retain the original full-context encoding and enter training only. Need supervision corrects frozen-parent gate errors; capped signed margins preserve its correct gates, including observations whose source plan is already correct. `--write-need-weight` and `--write-need-anchor-margin` control this objective. Other write-reference heads and all query binding remain frozen; complete-plan and reference development floors still apply.

`--factorized-source-corrections --source-correction-margin 1` separates source-word membership from source structure when resuming anchored `writer-context-content` or `writer-context-binding`. Membership compares the strongest legal path with the correct occupied words against every path with different occupancy, using detached boundaries. Structure fixes gold occupancy and compares complete tag paths using boundary parameters, with common membership omitted exactly. Frozen-parent eligibility and capped gaps protect already-correct components; fixed action-only limitations are reported. On initially wrong source plans this replaces the CRF, endpoint and absolute boundary/membership objectives and uses `--structured-weight` for both corrections. Correct complete source plans retain their original MSE/full-path anchors, and need-gate supervision is unchanged. Reports measure full fit before training and before restoring the selected checkpoint, with explicit fit-only scope. The option changes training only, with no new runtime rule or tensor geometry.

`--assertion-generalization` adds 64 independent fit observations: actual quoted notes, passive reports and factual premises in questions, plus fictional, conditional, courtesy and pure-question controls. Original development groups remain unchanged. Fit sampling reserves positive and negative assertion examples while retaining rare update and delete cases. Fresh texts must be extracted through the same exact 1B C probe; this option supplies training labels and adds no runtime semantic rule.

`--source-error-balance` applies to source-context residual/content/binding training with parent anchors. It averages each nonempty frozen-parent missing, spurious or other incorrect fit-row class, then averages those class means separately from the correct-row anchor mean. A row with both missing and spurious words contributes to both error groups. This prevents the many already-correct fit rows from diluting rare corrections. Classes come from the immutable fit-only parent teacher; the sampler and development selection guards remain unchanged.

`--source-context-generalization` adds empty and neutral-history views only to the authored boundary/narrative fit worlds. It preserves original bodies, provenance, actions, spans and speaker/body reference ranges, rejects references into the changed history, and leaves every development group untouched. These views address the observed correlation between wrappers and history availability. They are suitable for the self-contained authored facts and body-defined fictional/question controls, not facts whose meaning or subject depends on history. The option needs either `--boundary-generalization` or `--narrative-diversity`; it adds no serving rule.

Continue source-context training from the current checkpoint and retained synthetic feature cache:

```sh
python3 python/train_neural_dialogue.py \
  --gguf models/bitcpm4-1b-tq2_0.gguf --probe build/neural_slot_probe \
  --resume models/memory-current/model.pt \
  --curriculum build/memory-1b/augmented/curriculum.json \
  --features build/memory-1b/writer/features.pt --output build/writer-next \
  --train-heads writer-context-residual --replay-dialogue --replay-episodes \
  --boundary-generalization --narrative-diversity --source-context-generalization \
  --source-error-balance --membership-weight 0.5 \
  --boundary-anchor-weight 1 --boundary-path-anchor-weight 10 \
  --boundary-path-anchor-margin 1000 --steps 800 --eval-every 40 \
  --learning-rate 0.001 --device mps
```

Native acceptance uses `tests/eval_neural_episodes_native.py`: automatic writes, a durable snapshot, a new reader process, full-bank activation and a read-only snapshot check. Gold spans and support labels are used only for training or external scoring. Add `--trace-activation` to write `activation-trace.jsonl` from the actual C scoring passes without changing activation or binary output:

```sh
python3 tests/eval_neural_episodes_native.py \
  --ingester build/neural_slot_ingest --checker build/neural_recall_check \
  --gguf models/bitcpm4-1b-tq2_0.gguf \
  --artifact models/memory-current/model.bnsmodel \
  --curriculum build/memory-1b/native-pilot-curriculum.json \
  --split development --trace-activation --output build/native-trace
```

Use a new output directory for each evaluation. The lower-level checker accepts `neural_recall_check MEMORY_MODEL BACKBONE INPUT OUTPUT [ROUNDS] [--trace JSONL]`. JSONL records the actual semantic, identity, version and intent factors, selected IDs, and current/original query and subject fingerprints. `set_converged` means stable activation IDs, including an initially empty set; it does not mean complete support coverage. Unresolved binding emits a query-end record without round factors. Trace data is for evaluation diagnosis and must not enter training or checkpoint selection.

The native report also includes `stage_metrics` and each question's `first_failed_stage`. Positive questions are checked in order: complete source availability, reference resolution, then complete support activation. Complete availability requires the full annotated source fragment and its expected reference; a useful truncated prefix is still incomplete under this strict metric. `conditional_activation_exact_accuracy` covers only positive questions with complete writes and resolved references, returns `null` for an empty denominator, and excludes null/ambiguous questions. Keep the end-to-end score alongside it: neither a conditional score nor activation-set convergence establishes memory quality.

For continued read refinement, use the original augmented observation curriculum. These commands extract fresh features with the current probe. Optional `--features` and `--extra-features` caches must match both the backbone and the exact probe binary; rebuilding a probe does not authorize changing a cache's identity. Diagnostics go under `build/`; selected, terminal and exported models go under `models/`. Both output directories must be new.

```sh
python3 python/train_neural_dialogue_reader.py \
  --gguf models/bitcpm4-1b-tq2_0.gguf --probe build/neural_slot_probe \
  --resume models/memory-current/model.pt \
  --planner-checkpoint models/memory-current/model.pt \
  --curriculum build/memory-1b/augmented/curriculum.json \
  --output build/reader-next --model-output models/reader-next \
  --qualification-curriculum build/memory-1b/native-pilot-curriculum.json \
  --train-heads semantic --freeze-memory-encoder --semantic-topic-weight 0.05 \
  --replay-episodes --evidence-cues --identity-diversity --expansion \
  --fit-paraphrases --episode-relations --relation-paraphrases \
  --steps 600 --learning-rate 0.0002 \
  --attention-learning-rate 0.01 --device mps
```

For contextual-kind calibration followed by focused time refinement, keep the same original curriculum:

```sh
python3 python/train_neural_dialogue_reader.py \
  --gguf models/bitcpm4-1b-tq2_0.gguf --probe build/neural_slot_probe \
  --resume models/memory-current/model.pt \
  --planner-checkpoint models/memory-current/model.pt \
  --curriculum build/memory-1b/augmented/curriculum.json \
  --output build/temporal-next --model-output models/temporal-next \
  --qualification-curriculum build/memory-1b/native-pilot-curriculum.json \
  --train-heads temporal --kind-context-prefit --freeze-memory-encoder \
  --semantic-topic-weight 0 --reader-functional-weight 1 --reader-functional-margin 1 \
  --replay-episodes --evidence-cues --identity-diversity --expansion \
  --fit-paraphrases --episode-relations --relation-paraphrases \
  --steps 600 --eval-every 40 --learning-rate 0.001 --device mps
```

NN operator benchmarks use fixed model weights and deterministic mathematical inputs. On the local arm64 machine, four-output GEMV blocks improve encoding/query operators about 3.1x; 96-unit MPS sequence-loss forward/backward improves from 7.87 ms to 2.54 ms. The C floating-point maximum difference is 4.77e-7, with unchanged parity tolerances. These are NN operator measurements and do not measure backbone throughput or memory accuracy. The CRF partition uses an ordered parallel log-semiring reduction; fit teacher caches and bulk CPU Viterbi decoding avoid repeated accelerator synchronization.

```sh
./build/neural_ops_bench models/memory-current/model.bnsmodel \
  models/bitcpm4-1b-tq2_0.gguf build/nn-ops.json build/nn-ops.f32
python3 tests/test_neural_writer_boundaries.py --bench --device mps \
  --iterations 20 --output build/writer-ops.json
```

## Checks

```sh
python3 tests/test_neural_slots_model.py
python3 tests/test_neural_kind_context.py
python3 tests/test_neural_reader_context.py
python3 tests/test_neural_reader_expansion_worlds.py
python3 tests/test_neural_expansion_training.py
python3 tests/test_neural_dialogue_training.py
python3 tests/test_neural_stream_http.py build/openai_server \
  models/bitcpm4-1b-tq2_0.gguf
python3 tests/test_locomo_evaluation.py
python3 tests/test_locomo_slots_evaluation.py
python3 tests/test_neural_dialogue_http.py build/openai_server \
  models/bitcpm4-1b-tq2_0.gguf models/memory-current/model.bnsmodel \
  --output build/memory-http.json
python3 tests/test_openai_server_api.py build/openai_server models/bitcpm4-1b-tq2_0.gguf
```

Core regression tests cover kernels, dispatch, GGUF, tokenizer, generation, KV cache, LoRA, concurrent evaluation and memory storage. Decode profiling is available through `test_profile_decode`, `test_profile_multiturn`, `test_i2s_bench` and `bench_e2e`. Optional differential validation against llama.cpp uses `BITNET_BUILD_LLAMA_REFERENCE` and `tests/test_llama_reference.py`.

## Files

- `include/bitnet.h`, `src/bitnet.c`: public inference API and forward/decode path.
- `src/quant_*`, `src/ops.c`, `src/x86/`, `src/bitnet_dispatch.c`: kernels and dispatch.
- `src/gguf.c`, `src/tokenizer.c`, `src/sampler.c`: model, token and sampling support.
- `src/neural_slots.c`: current memory network, observation planner, activation and snapshots.
- `examples/`: CLI and HTTP server.
- `python/`: current network, independent curricula, training and LoCoMo scoring.
- `tools/`: inspection, native feature extraction, memory ingestion/recall and parity probes.
- `tests/`: current correctness, integration, parity and evaluation checks.

Generated data and build outputs belong in `build/`. `models/` holds the untracked 1B backbone and the tracked trained memory artifacts: `models/memory-current/` (serving parameters, training checkpoint, read head and validation report) and `models/memory-locomo-full/` (evaluation-only writer used for the LoCoMo result). Datasets and other models remain untracked.
