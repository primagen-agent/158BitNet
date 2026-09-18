# Selected 0.5B memory training package

This directory retains the training inputs and selected checkpoints for the
model bundle in `models/memory/resident-0.5b/`. `manifest.json` records the
SHA-256 of every retained input. The GGUF backbone is supplied separately and
must match the SHA-256 in both manifests. No LoCoMo conversation, sealed test
record, user session state, or GGUF is part of the package.

## Data and trained components

| Component | Selected training source | Checkpoints |
| --- | --- | --- |
| Resident baseline | `corpora/resident/train.jsonl`, `valid.jsonl`; generated with `--diverse-train`, 128/24 worlds | `baseline.pt` (factorized 128-wide, seed 2810917, selected at step 900) |
| Resident identity | Same worlds plus training-only `train_supervision.json`; exact C tokenizer and frozen-backbone features | Deployed `models/memory/resident-0.5b/resident.pt` (frozen baseline, identity selected at step 800 of 1,000) |
| Writer and context operation | `corpora/writer/train.jsonl`, `valid.jsonl`; 192/24 natural-message worlds | `writer_init.pt`, `writer.pt`, `tagger.pt`, `keys.pt`, `operation.pt` |
| Typed pair | Existing pair curriculum and frozen writer initialization | `pair_init.pt`, `pair.pt` (selected at step 500) |
| Typed link | Writer corpus, frozen pair model; seven-input predecessor-existence head | `link.pt` (selected at step 100) |
| Typed query | Frozen pair's candidate features with positive and NULL labels | `feature_data/query_train.pt`, `query_valid.pt`, `query.pt` |

The writer uses all 24 frozen backbone layers in four six-layer bands. The
resident model uses normalized final hidden states and input embeddings from
the same backbone. Neither path uses LoRA, SVD, or session KV reuse during
memory evaluation. The pair, link, and query binaries are loaded for current
server compatibility; their scorers are bypassed in resident mode.

The resident train/validation JSONL hashes match those embedded in both the
selected baseline and identity checkpoints. The writer JSONL hashes match a
fresh generation using `python/prepare_natural_memory_curriculum.py` and seed
2710916. Training annotations in `train_supervision.json` never enter runtime
write or read calls. The held-out test split is not retained here.

`pair_init.pt` and `writer_init.pt` are retained starting points. The earlier
data used to train those initializers is not claimed as part of this selected
training package. The pair checkpoint does not record the hash of its raw
training corpus, so exact training of that earlier stage from scratch cannot
be verified from these files. The final pair binary can nevertheless be
re-exported exactly from `pair.pt`.

## Verify and re-export the selected artifacts

From the repository root, with Python and PyTorch installed:

```sh
python3 training/memory/resident-0.5b/verify.py \
  models/bitcpm4-0.5b-tq2_0.gguf
```

The verifier checks the backbone, training input and checkpoint hashes,
parent-checkpoint bindings, and re-exports all five model binaries into a
temporary directory. Every re-export must match the checked-in binary
byte-for-byte. A successful check verifies what was selected and retained;
it is separate from measuring memory accuracy.

## Repeat the trainable resident path

On the CUDA training host, use a new output directory:

```sh
bash training/memory/resident-0.5b/retrain_resident_and_writer.sh \
  models/bitcpm4-0.5b-tq2_0.gguf build/memory-retrain
```

This runs `python/train_resident_memory_set.py` with the factorized baseline
configuration, then `python/train_resident_identity.py` with the retained
selected baseline. It runs the natural writer and context-operation stages
through `scripts/train_natural_memory.sh` and
`scripts/train_context_memory.sh`. The C tokenizer supplies token IDs, and
the feature cache is regenerated from the frozen backbone. New training
outputs remain under `build/` and do not replace the checked-in models.
Random seeds, step limits, corpora and initialization are recorded here and
in the checkpoints; bitwise equality of a new GPU training run is not
guaranteed across GPU/PyTorch environments.

The optional compatibility query head can be trained directly on retained
candidate feature rows:

```sh
python3 training/memory/resident-0.5b/retrain_query_from_features.py \
  build/new-query.pt --device cuda
```

`python/train_natural_link_head.py` trains the link head from `pair.pt` and
writer feature caches. `python/train_typed_pair_verifier.py` is the pair-stage
trainer. The archived `link.pt`, `pair.pt`, and `query.pt` are sufficient to
re-export the exact compatibility binaries used by this serving bundle.
