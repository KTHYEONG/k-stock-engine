# mypy: ignore-errors
"""Parser sections, horizon parsing, direct-data/training request construction."""
from __future__ import annotations

import argparse


def _build_training_request(args: argparse.Namespace) -> dict[str, object]:
    artifact_id = getattr(args, "artifact_id", None)
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError("artifact_id must be a non-empty string")
    return {"artifact_id": artifact_id.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("--artifact-id", required=True)
    return parser
__all__ = ["_build_training_request", "build_parser"]
