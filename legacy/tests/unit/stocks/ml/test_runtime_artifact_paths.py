"""Runtime artifact paths: replay and OOF fixtures never touch mem: or scratch/."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

TERMINAL_OBS_01_NO_MEM_OR_SCRATCH_SIDE_EFFECT = (
    "TERMINAL_OBS_01_NO_MEM_OR_SCRATCH_SIDE_EFFECT"
)


def _market_frame() -> pl.DataFrame:
    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(6)]
    rows = []
    for session in sessions:
        for t in range(2):
            price = 100.0 + t
            rows.append(
                {
                    "instrument_id": f"KRX:{t + 1:05d}",
                    "session": session,
                    "open": price,
                    "close": price * 1.01,
                    "volume": 1e6,
                    "trading_value": price * 1e6,
                }
            )
    return pl.DataFrame(rows)


def _manifest(frame: pl.DataFrame):
    from src.core.datasets import DatasetManifest
    from src.core.instruments import AssetKind

    return DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="h",
        provider_version="p",
        universe_policy_version="u",
        universe_policy_hash="u",
        feature_set="stock_net_alpha_v1",
        feature_set_hash="f",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 1, 6, tzinfo=UTC),
        generated_time=datetime(2024, 1, 6, tzinfo=UTC),
        row_count=frame.height,
    )


def test_terminal_obs_01_no_mem_or_scratch_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TERMINAL_OBS_01_NO_MEM_OR_SCRATCH_SIDE_EFFECT.

    Replay contexts bind the caller-owned tmp registry, OOF spill lives only
    below the run-scoped temporary root and is removed on close; no
    repository-relative ``mem:`` directory or ``scratch/`` path is created.
    """
    monkeypatch.chdir(tmp_path)
    import legacy.stocks.ml.fitting as fitting_module
    from legacy.stocks.ml.contracts import NetAlphaTrainingRequest
    from legacy.stocks.ml.fitting import OofCache
    from legacy.stocks.ml.fitting import default_oof_cache_base
    from legacy.stocks.ml.training import _default_oof_cache_base as training_cache_base
    from legacy.stocks.ml.training import _execution_replay_context, _OofCache
    from legacy.stocks.research.artifacts import ModelArtifactRegistry

    assert str(training_cache_base()).startswith(str(Path.cwd()) + "/") or "tmp" in str(
        training_cache_base()
    )

    # Replay fixture: caller-owned registry under tmp_path only.
    frame = _market_frame()
    request = NetAlphaTrainingRequest(
        artifact_id="terminal_obs_01", candidate_horizon_sessions=(10,)
    )
    registry = ModelArtifactRegistry(tmp_path / "registry")
    context = _execution_replay_context(
        registry,
        request,
        _manifest(frame),
        frame,
        request.policy_profiles[0],
        seed=7,
        horizon_sessions=10,
        rebalance_frequency_sessions=5,
        top_k=12,
    )
    assert context.registry.root == tmp_path / "registry"
    assert not context.registry.root.exists() or context.registry.root.is_relative_to(
        tmp_path
    )

    # OOF fixture: one run-scoped TemporaryDirectory below the tmp base.
    monkeypatch.setattr(fitting_module, "PROJECT_ROOT", tmp_path)
    cache: OofCache = _OofCache(default_oof_cache_base())
    spill_root = cache.root
    assert spill_root.is_relative_to(tmp_path)
    scores = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "session": [datetime(2024, 1, 2, tzinfo=UTC)],
            "predicted_net_alpha": [0.01],
        }
    )
    labels = scores.clone()
    oof_path, labels_path = cache.store(10, scores, labels)
    assert oof_path.is_relative_to(spill_root)
    assert labels_path.is_relative_to(spill_root)
    cache.close()
    assert not spill_root.exists()
    assert not list((tmp_path / "tmp" / "training").glob("oof-*"))

    # Neither a repository-relative mem: directory nor scratch/ was created.
    assert not Path("mem:").exists()
    assert not (tmp_path / "mem:").exists()
    assert not Path("scratch").exists()
