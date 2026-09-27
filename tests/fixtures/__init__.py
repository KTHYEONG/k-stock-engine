"""Shared test fixtures replacing per-file helper copies."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.fixtures.catalog_fixtures import blob_for, seed_corp_code_bridge, seed_receipts

__all__ = [
    "blob_for",
    "fake_transport",
    "publish_fixture_dataset",
    "scope_runtime",
    "seed_corp_code_bridge",
    "seed_receipts",
]


def fake_transport(*, responses: list[Any] | None = None, fail: Exception | None = None):
    """Return a scripted transport callable with an attempt log and fake clock.

    The callable mimics the ``request_json(endpoint, params)`` seam used by
    provider clients: each call records ``(endpoint, params)`` in ``calls``
    and replays the next scripted response. ``now`` advances one second per
    attempt to emulate pacing without network access.
    """
    from datetime import UTC, datetime, timedelta

    script = list(responses or [])
    calls: list[tuple[str, dict[str, Any]]] = []
    state = {"now": datetime(2026, 1, 2, tzinfo=UTC)}

    def _request(endpoint: str, params: dict[str, Any]) -> Any:
        calls.append((endpoint, dict(params)))
        state["now"] = state["now"] + timedelta(seconds=1)
        if fail is not None:
            raise fail
        if script:
            action = script[min(len(calls) - 1, len(script) - 1)]
            if isinstance(action, Exception):
                raise action
            return action
        return {"status": "000", "list": []}

    def _now() -> Any:
        return state["now"]

    _request.calls = calls  # type: ignore[attr-defined]
    _request.now = _now  # type: ignore[attr-defined]
    return _request


def scope_runtime(tmp_path: Path):
    """Return a minimal scoped workspace with a registry for fixture tests."""
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(
        scope_config=Path("config/research/kr_swing_2019_v1.toml"),
        data_root=tmp_path / "data",
    )
    runtime.workspace.initialize()
    return runtime


def publish_fixture_dataset(layer_root: Path, kind: str, partitions: dict[str, Any], *, layer: str = "silver"):
    """Wrap :func:`src.data.datasets.publish_dataset` for fixture datasets."""
    import polars as pl

    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    frames = {path: value for path, value in partitions.items() if isinstance(value, pl.DataFrame)}
    identity = DatasetIdentity(
        kind=kind,
        layer=DatasetLayer.GOLD if layer == "gold" else DatasetLayer.SILVER,
        policy_version=f"{kind}-fixture-v1",
        inputs={},
        params={},
    )
    return publish_dataset(layer_root=layer_root, identity=identity, partitions=frames)
