# Memory-to-generation architecture gate

The governing design and implementation order are now documented in
`../neural-system/DESIGN.md` and `../neural-system/PLAN.md`. This directory
retains component diagnostics; it does not authorize skipping their gates.

This experiment checks whether persistent token representations can condition
natural autoregressive replies from the exact frozen BitCPM 0.5B backbone.
It is a component experiment with supplied episode boundaries, not automatic
memory, a LoCoMo score, or a replacement for the resident C bundle.

## Structure

`GatedMemoryFusion` learns full-rank projections from contextual hidden states
and lexical embeddings into a shared memory bank. Each source is encoded in
a fresh context; BOS is excluded. Every backbone layer has a separate scalar
residual gain, initialized to zero, and can cross-attend to the same bank after
self-attention. Projection parameters are shared across layers. A learned
NULL key has a zero value. A token-conditioned use gate controls the residual.
The gates are ordinary memory-fusion controls, not NAS or neuron pruning.
The evidence-conditioned variant instead gates on the current hidden vector,
the attended memory output, their product and their absolute difference.
Its auxiliary usability supervision is applied only at the query boundary,
before any teacher-forced response token. The first three response tokens
receive higher loss weight to expose the reply/abstain decision and subject
spelling. These labels and weights are training-only, never runtime inputs.

The backbone remains frozen, but downstream backbone computations retain the
gradient path into the new modules. There is no LoRA, SVD, backbone KV reuse,
retrieval service, source-text prompt insertion, or direct answer copying.
Empty memory bypasses the module exactly, including after training. During
decoding the memory is read-only and the entire response prefix is recomputed.

## Data and training

`python/prepare_memory_fusion_curriculum.py` creates 32 training worlds and
eight development worlds. Names and question renderers differ across splits;
the two relation types and value vocabulary are shared. Each world has four
conditions: original memory, replaced value with the identical question,
removed target with a different person's memory, and added distractor.
No LoCoMo or sealed test data is used. Sources have supplied boundaries and
are encoded automatically; selection/write-policy learning is not evaluated.
The optional `--diverse-entities` changes only the training entity pool.
Development JSONL bytes remain identical and a regression verifies this.

Training uses response-token cross entropy, balanced condition sampling,
AdamW and gradient clipping. The control recipe selects on development loss.
`--select-by-generation` selects on the worst condition's subject-and-value
accuracy, then paired-swap accuracy, total correct responses, and finally NLL.
Source features are saved/reloaded and checked for tensor equality.
The report includes the backbone/corpus/checkpoint hashes and runtime version.

From the repository root on the CUDA host:

```sh
bash scripts/train_memory_fusion.sh \
  /path/to/bitcpm4-0.5b-tq2_0.gguf build/fusion-gate1
```

`python/train_memory_fusion.py` accepts explicit step, seed, learning-rate and
generation limits. The matching backbone SHA-256 is enforced. Outputs are
`model/selected.pt`, `model/source_features.pt`, and `model/summary.json`.
The checkpoint is research-only and has no C exporter.

The entity-diversity intervention uses the same model initialization and fixed
development set, with 256 training worlds, 600 steps, and accumulation eight:

```sh
FUSION_DIVERSE_ENTITIES=1 FUSION_TRAIN_WORLDS=256 FUSION_STEPS=600 \
FUSION_ACCUMULATION=8 FUSION_EVAL_EVERY=100 \
bash scripts/train_memory_fusion.sh \
  /path/to/bitcpm4-0.5b-tq2_0.gguf build/fusion-diverse
```

To test evidence-conditioned gating with the same diverse curriculum, add
`FUSION_EVIDENCE_GATE=1` and use a different output directory. This enables
auxiliary gate loss weight 1, first-three-response-token weight 4, and free
generation checkpoint selection. It does not change the development corpus.

`python/diagnose_memory_fusion.py` evaluates training replay, changed names
with the training question form, and changed question forms with training
names. It verifies checkpoint and corpus hashes and never updates weights.
Training replay is reported separately and is not generalization accuracy.
`--decision-only` isolates first-response-token accuracy without supplying the
correct response prefix. `python/rescore_memory_fusion.py` can audit saved
responses with the stricter subject-aware checker; it verifies the exact
development file and inputs, records the original report hash, and never
regenerates outputs or changes model weights.

The C tokenizer must preserve registered ChatML delimiters as atomic tokens;
`tests/test_tokenizer.c` checks this prerequisite. Tokenization of ordinary
source text is unchanged. Response training terminates on the actual EOS ID,
not the individual tokens produced by an incorrectly split delimiter.

## Predeclared gate

Greedy generation, not teacher-forced loss, decides the gate. At least 90% of
examples in each condition must produce the expected value (or uncertainty
when absent), name the correct subject, exclude competing values, and avoid
memory/tool commentary. Subject correctness is a stricter additional guard:
value-only scoring is retained separately and never treated as complete
answer accuracy. These controlled targets name the subject explicitly;
the checker is not a general-purpose evaluator of arbitrary paraphrases.
Both original and replaced-value answers must pass in at least 90% of paired
worlds. Empty-memory logits must equal the frozen backbone exactly. Full
sentence exact matches are also reported. These are small development probes,
not a claim of general linguistic quality or independent-test accuracy.

Failure blocks expanding to autonomous writes or production C integration.
Inspect the retained generated replies and determine whether the failed
component is content transfer, identity activation, or no-memory behavior.
Do not lower the threshold after observing the run or relabel a teacher-loss
improvement as successful memory-conditioned generation.

Run architecture regressions locally:

```sh
python3 -m unittest discover -s tests -p 'test_memory_fusion.py'
```

## Measured decision

The checked results in `results.json` use the same 32 development examples
and subject-aware scoring for every candidate. The two control reports were
rescored from their saved replies, without rerunning generation. Value-only
successes that misspell the subject are not counted as correct answers.

| Candidate | Subject-and-value correct | Both members of a value-swap pair correct |
| --- | ---: | ---: |
| Fusion control, 32 training worlds | 14/32 | 1/8 |
| Entity diversity control, 256 training worlds | 4/32 | 1/8 |
| Evidence-conditioned gate and generation selection | 6/32 | 0/8 |

All fail the architecture gate. The evidence-conditioned candidate was selected
at step 400 of 600. All runs preserve empty-memory logits and persisted source
features exactly. Twelve architecture/audit regressions pass, but these checks
do not establish useful memory accuracy.

The control checkpoint answers both value variants correctly on eight sampled
training worlds; that drops to two paired worlds when only names change and
seven when only question wording changes. This is training replay/diagnostic
evidence, not a final-test result. The diverse control's removed-target group
has mean teacher-forced NLL 0.224 but first-response-token NLL 2.833 and 0/8
correct first decisions, illustrating why teacher-forced loss is insufficient.

The next unresolved architectural requirements are explicit episode-level
subject/content binding, reliable rejection of unrelated evidence, and exact
subject preservation in generation. The present token-level fusion candidate
is not approved for autonomous-memory training or C deployment. No production
memory model was replaced.
