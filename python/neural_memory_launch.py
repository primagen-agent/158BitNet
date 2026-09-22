"""Fail-closed interlock for the unreviewed episode trainer.

This deliberately does not implement an approved launch path yet. P1 evidence,
P2A draft review, source/data hashes and a registered command must be checked in
that future path. Editing a JSON status alone must not launch today's draft.
"""
import json
from pathlib import Path


def require_episode_training_ready(experiment_path=None):
    path = (Path(experiment_path) if experiment_path is not None else
            Path(__file__).resolve().parents[1] / "training/memory/neural-system/experiments/EP-001.json")
    try:
        experiment = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("Training blocked: missing or invalid experiment registration") from exc
    if not isinstance(experiment, dict):
        raise ValueError("Training blocked: missing or invalid experiment registration")
    if experiment.get("prerequisites_complete") is not True:
        raise ValueError("Training blocked: P1.1-P1.7 are not complete; see neural-system/PLAN.md")
    raise ValueError("Training blocked: episode trainer is an unreviewed draft; no approved launch path exists")
