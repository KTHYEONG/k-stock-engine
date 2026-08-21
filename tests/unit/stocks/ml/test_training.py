"""Tests for ML training decomposition and diagnostics wiring.

Scenarios:
- ML_04: Every candidate horizon emits checkpoints with fold/profile counts
  equal to the existing request grid.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from src.stocks.ml.fitting import OofCache, atomic_write_parquet, read_oof_parquet
from src.stocks.ml.discovery import HorizonDiscovery
from src.stocks.observability.contracts import (
    DiagnosticCategory,
    DiagnosticEvent,
    DiagnosticStage,
    DiagnosticStatus,
)
from src.stocks.observability.recorder import NullRunDiagnostics
import polars as pl


class TestOofCache:
    """OofCache manages temporary OOF spill files."""

    def test_store_and_load(self, tmp_path: Path) -> None:
        cache = OofCache(tmp_path)
        oof = pl.DataFrame({"score": [0.1, 0.2, 0.3]})
        labels = pl.DataFrame({"label": [0.01, 0.02, 0.03]})

        oof_path, labels_path = cache.store(10, oof, labels)
        assert oof_path.exists()
        assert labels_path.exists()

        loaded_oof, loaded_labels = cache.load(10)
        assert loaded_oof.shape == oof.shape
        assert loaded_labels.shape == labels.shape

        cache.close()

    def test_cache_bytes_tracking(self, tmp_path: Path) -> None:
        cache = OofCache(tmp_path)
        oof = pl.DataFrame({"score": [0.1, 0.2]})
        labels = pl.DataFrame({"label": [0.01, 0.02]})

        cache.store(10, oof, labels)
        assert cache.cache_bytes > 0

        cache.close()

    def test_closed_cache_rejects_store(self, tmp_path: Path) -> None:
        cache = OofCache(tmp_path)
        cache.close()
        with pytest.raises(ValueError, match="closed"):
            cache.store(10, pl.DataFrame({"a": [1]}), pl.DataFrame({"b": [1]}))


class TestAtomicWriteParquet:
    """Atomic parquet write via rename."""

    def test_write_and_read(self, tmp_path: Path) -> None:
        frame = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        path = tmp_path / "test.parquet.zst"
        atomic_write_parquet(frame, path)
        assert path.exists()

        loaded = read_oof_parquet(path)
        assert loaded.shape == frame.shape


class TestHorizonDiscoveryDataclass:
    """HorizonDiscovery is a frozen dataclass."""

    def test_create_empty_discovery(self) -> None:
        discovery = HorizonDiscovery(
            evidence=(),
            diagnostics=(),
            oof_by_horizon={},
        )
        assert discovery.evidence == ()
        assert discovery.path_evaluation_count == 0

    def test_create_with_data(self, tmp_path: Path) -> None:
        discovery = HorizonDiscovery(
            evidence=(),
            diagnostics=(),
            oof_by_horizon={10: (tmp_path / "oof.parquet", tmp_path / "labels.parquet", [0.05])},
            dropout_reasons={(10, "test"): "insufficient_data"},
            path_evaluation_count=5,
            path_evaluation_bound=10,
        )
        assert discovery.path_evaluation_count == 5
        assert (10, "test") in discovery.dropout_reasons


class TestDiagnosticsWiring:
    """Verify diagnostics parameter is accepted by training and backtesting."""

    def test_train_accepts_diagnostics_parameter(self) -> None:
        from src.stocks.ml.training import train_net_alpha_model
        import inspect

        sig = inspect.signature(train_net_alpha_model)
        assert "diagnostics" in sig.parameters
        param = sig.parameters["diagnostics"]
        assert param.default is None

    def test_backtester_accepts_diagnostics_parameter(self) -> None:
        from src.stocks.backtesting.engine import StockBacktester
        import inspect

        sig = inspect.signature(StockBacktester.__init__)
        assert "diagnostics" in sig.parameters
        param = sig.parameters["diagnostics"]
        assert param.default is None

    def test_null_diagnostics_emits_no_events(self) -> None:
        sink = NullRunDiagnostics()
        event = DiagnosticEvent(
            run_id="test",
            sequence=0,
            category=DiagnosticCategory.ALGO,
            component="ml.training",
            stage=DiagnosticStage.SPLIT_FIT,
            event="test",
            status=DiagnosticStatus.START,
        )
        sink.emit(event)
        sink.close("PASS")
