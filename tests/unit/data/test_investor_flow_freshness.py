"""Investor-flow freshness coverage and fail-closed threshold invariants."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from src.config.live_gate import LiveGatePolicy
from src.core.time import KRX_TZ
from src.data.investor_flow_freshness import (
    StaleInvestorFlowError,
    check_investor_flow_freshness,
    investor_flow_freshness,
)

SESSION = date(2026, 10, 7)
OTHER = date(2026, 10, 6)
SEAL = datetime(2026, 10, 8, 8, 54, tzinfo=KRX_TZ)


def _targets(session: date, n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {"session": [session] * n, "ticker": [f"{i:06d}" for i in range(n)]},
        schema={"session": pl.Date, "ticker": pl.String},
    )


def _flow(session: date, tickers: list[str], hashes: dict[str, str] | None = None) -> pl.LazyFrame:
    hashes = hashes or {}
    return pl.DataFrame(
        {
            "session": [session] * len(tickers),
            "ticker": list(tickers),
            "source_hash": [hashes.get(ticker, f"hash-{ticker}") for ticker in tickers],
        },
        schema={"session": pl.Date, "ticker": pl.String, "source_hash": pl.String},
    ).lazy()


def _retrieved(tickers: list[str], when: datetime = SEAL - timedelta(hours=12)) -> dict[str, datetime]:
    return {f"hash-{ticker}": when for ticker in tickers}


def test_all_present_passes() -> None:
    tickers = [f"{i:06d}" for i in range(4)]
    (result,) = investor_flow_freshness(
        targets=_targets(SESSION, 4),
        flow=_flow(SESSION, tickers),
        retrieved_at=_retrieved(tickers),
        source_sessions=[SESSION],
        seal_time=SEAL,
    )
    assert (result.required_cells, result.present_cells, result.late_cells, result.missing_cells) == (4, 4, 0, 0)
    assert result.missing_fraction == 0.0


def test_missing_cells_counted() -> None:
    tickers = [f"{i:06d}" for i in range(100)]
    (result,) = investor_flow_freshness(
        targets=_targets(SESSION, 100),
        flow=_flow(SESSION, tickers[:97]),
        retrieved_at=_retrieved(tickers[:97]),
        source_sessions=[SESSION],
        seal_time=SEAL,
    )
    assert result.missing_cells == 3
    assert result.missing_fraction == pytest.approx(0.03)


def test_late_retrieval_counts_as_missing() -> None:
    tickers = ["000000", "000001"]
    retrieved = _retrieved(tickers)
    retrieved["hash-000000"] = SEAL + timedelta(microseconds=1)
    (result,) = investor_flow_freshness(
        targets=_targets(SESSION, 2),
        flow=_flow(SESSION, tickers),
        retrieved_at=retrieved,
        source_sessions=[SESSION],
        seal_time=SEAL,
    )
    assert result.late_cells == 1
    assert result.present_cells == 1
    assert result.missing_cells == 1


def test_unknown_hash_counts_as_missing() -> None:
    (result,) = investor_flow_freshness(
        targets=_targets(SESSION, 1),
        flow=_flow(SESSION, ["000000"]),
        retrieved_at={},
        source_sessions=[SESSION],
        seal_time=SEAL,
    )
    assert result.late_cells == 1
    assert result.missing_cells == 1


def test_duplicate_rows_count_once() -> None:
    frame = pl.DataFrame(
        {
            "session": [SESSION, SESSION],
            "ticker": ["000000", "000000"],
            "source_hash": ["hash-a", "hash-b"],
        },
        schema={"session": pl.Date, "ticker": pl.String, "source_hash": pl.String},
    ).lazy()
    (result,) = investor_flow_freshness(
        targets=_targets(SESSION, 1),
        flow=frame,
        retrieved_at={"hash-a": SEAL - timedelta(hours=1), "hash-b": SEAL + timedelta(hours=1)},
        source_sessions=[SESSION],
        seal_time=SEAL,
    )
    assert result.present_cells == 1
    assert result.missing_cells == 0


def test_zero_required_is_not_stale() -> None:
    (result,) = investor_flow_freshness(
        targets=_targets(OTHER, 3),
        flow=_flow(OTHER, ["000000", "000001", "000002"]),
        retrieved_at=_retrieved(["000000", "000001", "000002"]),
        source_sessions=[SESSION],
        seal_time=SEAL,
    )
    assert result.required_cells == 0
    assert result.missing_fraction == 0.0


def _stub_io(
    monkeypatch: pytest.MonkeyPatch, targets: pl.DataFrame, flow: pl.LazyFrame, retrieved: dict[str, datetime]
) -> None:
    import src.data.investor_flow_freshness as freshness_module

    collected = flow.collect()
    monkeypatch.setattr(
        freshness_module, "investor_flow_targets", lambda runtime: SimpleNamespace(cells=targets)
    )
    monkeypatch.setattr(freshness_module, "read_dataset", lambda path, columns=None: collected.lazy())

    class _Catalog:
        def __init__(self, root: Path) -> None:
            self._root = root

        def blobs(self, *, source: str) -> Any:
            assert source in ("ls_investor_flow", "kis_investor_flow")
            extra = {"hash-unreferenced": SEAL - timedelta(hours=1)}
            return (
                SimpleNamespace(content_hash=content_hash, retrieved_at=when)
                for content_hash, when in {**retrieved, **extra}.items()
            )

    monkeypatch.setattr(freshness_module, "ReceiptCatalog", _Catalog)


def test_threshold_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    tickers = [f"{i:06d}" for i in range(100)]
    policy = LiveGatePolicy(max_missing_flow_fraction=0.03)
    runtime = SimpleNamespace(workspace=SimpleNamespace(bronze_root=tmp_path))
    on_time = SEAL - timedelta(hours=12)

    _stub_io(monkeypatch, _targets(SESSION, 100), _flow(SESSION, tickers[:97]), _retrieved(tickers[:97], on_time))
    with caplog.at_level(logging.INFO):
        passed = check_investor_flow_freshness(
            runtime=runtime,  # type: ignore[arg-type]
            flow_dataset_dir=tmp_path / "flow",
            source_sessions=[SESSION],
            seal_time=SEAL,
            policy=policy,
        )
    assert len(passed) == 1
    assert passed[0].missing_fraction == pytest.approx(0.03)
    assert any("gate=investor_flow" in record.message for record in caplog.records)

    _stub_io(monkeypatch, _targets(SESSION, 100), _flow(SESSION, tickers[:96]), _retrieved(tickers[:96], on_time))
    with pytest.raises(StaleInvestorFlowError) as exc_info:
        check_investor_flow_freshness(
            runtime=runtime,  # type: ignore[arg-type]
            flow_dataset_dir=tmp_path / "flow",
            source_sessions=[SESSION],
            seal_time=SEAL,
            policy=policy,
        )
    assert exc_info.value.reason == "stale_investor_flow"
    assert len(exc_info.value.freshness) == 1
    assert exc_info.value.freshness[0].missing_fraction == pytest.approx(0.04)
    assert SESSION.isoformat() in str(exc_info.value)


def test_naive_seal_time_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        investor_flow_freshness(
            targets=_targets(SESSION, 1),
            flow=_flow(SESSION, ["000000"]),
            retrieved_at=_retrieved(["000000"]),
            source_sessions=[SESSION],
            seal_time=datetime(2026, 10, 8, 8, 54),
        )


def test_other_sessions_ignored() -> None:
    tickers = ["000000", "000001"]
    results = investor_flow_freshness(
        targets=pl.concat([_targets(SESSION, 2), _targets(OTHER, 2)]),
        flow=_flow(OTHER, tickers),
        retrieved_at=_retrieved(tickers),
        source_sessions=[SESSION, OTHER],
        seal_time=SEAL,
    )
    assert [item.source_session for item in results] == [SESSION, OTHER]
    assert results[0].missing_cells == 2
    assert results[1].missing_cells == 0

    (only,) = investor_flow_freshness(
        targets=pl.concat([_targets(SESSION, 2), _targets(OTHER, 2)]),
        flow=_flow(OTHER, tickers),
        retrieved_at=_retrieved(tickers),
        source_sessions=[OTHER],
        seal_time=SEAL,
    )
    assert only.missing_cells == 0
