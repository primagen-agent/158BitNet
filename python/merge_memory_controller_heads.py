#!/usr/bin/env python3
"""Combine an addressed controller with a separately validated action head."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_addressed_memory_controller import (
    AddressedController,
    save_bnctrl,
)


ACTION_KEYS = (
    "action_hidden.weight",
    "action_hidden.bias",
    "action.weight",
    "action.bias",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("address_checkpoint")
    parser.add_argument("action_checkpoint")
    parser.add_argument("gguf")
    parser.add_argument("output")
    args = parser.parse_args()
    address = torch.load(
        args.address_checkpoint,
        map_location="cpu", weights_only=False)
    action = torch.load(
        args.action_checkpoint,
        map_location="cpu", weights_only=False)
    for field in (
        "hidden", "rank", "action_rank", "pooling",
        "temperature", "backbone_sha256",
    ):
        if address[field] != action[field]:
            raise ValueError(f"controller mismatch for {field}")
    state = {
        key: value.detach().cpu().clone()
        for key, value in address["state_dict"].items()
    }
    for key in ACTION_KEYS:
        if key not in action["state_dict"]:
            raise ValueError(f"action checkpoint lacks {key}")
        state[key] = action["state_dict"][key].detach().cpu().clone()
    model = AddressedController(
        address["hidden"], address["rank"], address["action_rank"])
    model.load_state_dict(state)
    threshold = address["metrics"]["address_threshold"]
    save_bnctrl(
        args.output, model, args.gguf,
        address["hidden"], address["rank"],
        address["temperature"], address["pooling"], threshold)
    combined = dict(address)
    combined["state_dict"] = model.state_dict()
    combined["metrics"] = {
        **address["metrics"],
        **{
            key: value
            for key, value in action["metrics"].items()
            if key.startswith("action_")
        },
    }
    checkpoint_path = str(Path(args.output).with_suffix(".pt"))
    torch.save(combined, checkpoint_path)
    print({
        "output": args.output,
        "checkpoint": checkpoint_path,
        "address_threshold": threshold,
        "action_accuracy": combined["metrics"]["action_accuracy"],
    })


if __name__ == "__main__":
    main()
