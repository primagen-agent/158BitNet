#!/usr/bin/env python3
"""Compose compatible development-selected writer/reader parameter sets.

Never mixes cached memory representations: the resulting identity requires fresh
native writes. Only explicitly trained reader parameters may replace the writer.
"""

import argparse
import json
from pathlib import Path
import torch
from neural_slot_model import export
from train_neural_dialogue_reader import load_model, READER_HEADS
from neural_training_common import sha


def compose(writer, reader):
    if writer["backbone_sha256"] != reader["backbone_sha256"]:
        raise ValueError("component backbone mismatch")
    allowed = tuple(reader["reader_heads"])
    supported = set(READER_HEADS)
    if (not allowed or not all(isinstance(head, str) for head in allowed)
            or len(allowed) != len(set(allowed)) or not set(allowed) <= supported):
        raise ValueError("unexpected reader parameter selection")
    states = load_model(writer).state_dict()
    reader_states = load_model(reader).state_dict()
    if states.keys() != reader_states.keys():
        raise ValueError("component tensor geometry mismatch")
    result = dict(states)
    for key, value in reader_states.items():
        if value.shape != states[key].shape:
            raise ValueError("component shape mismatch")
        if key.startswith(allowed):
            result[key] = value
    return dict(
        writer,
        state_dict=result,
        reader_heads=allowed,
        reader_step=reader["reader_step"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("writer", "reader", "gguf", "output"):
        p.add_argument("--" + key, required=True)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    writer = torch.load(args.writer, map_location="cpu", weights_only=False)
    reader = torch.load(args.reader, map_location="cpu", weights_only=False)
    combined = compose(writer, reader)
    if combined["backbone_sha256"] != sha(args.gguf):
        raise ValueError("deployment backbone mismatch")
    combined.update(
        writer_checkpoint_sha256=sha(args.writer),
        reader_checkpoint_sha256=sha(args.reader),
        deployment_approved=False,
        locomo_used=False,
    )
    torch.save(combined, root / "selected.pt")
    digest = export(
        load_model(combined), combined["backbone_sha256"], root / "model.bnsmodel"
    )
    report = dict(
        model_sha256=digest,
        writer_step=combined.get("selected_step", 0),
        reader_step=combined["reader_step"],
        reader_heads=combined["reader_heads"],
        checkpoint_selection="development splits only",
        native_qualified=False,
        locomo_used=False,
    )
    (root / "composition.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
