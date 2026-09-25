"""Backtest CLI registry and dividend fail-closed contracts."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from src.backtest.cli import main
from src.data.dataset_registry import DatasetRegistry
from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

_SCOPE = Path("config/research/kr_swing_2019_v1.toml")
_ENGINE = Path("config/backtest/default_engine.toml")


def _panel_frame() -> pl.DataFrame:
    rows = [
        {
            "session": session,
            "instrument_id": "KRX:005930",
            "market": "KOSPI",
            "open": 100,
            "high": 100,
            "low": 100,
            "close": 100,
            "base_price": 100,
            "tick_size": 1,
            "upper_limit": 200,
            "lower_limit": 1,
            "volume": 1000,
            "sell_tax_rate": 0.0,
            "adtv20": 1_000_000.0,
            "ret_vol60": 0.0,
            "share_factor": 1.0,
            "eligible": True,
            "open_at_upper": False,
            "open_at_lower": False,
        }
        for session in (date(2024, 1, 2), date(2024, 1, 3))
    ]
    return pl.DataFrame(rows)


def _publish(root: Path, kind: str, layer: DatasetLayer, frame: pl.DataFrame, params: dict[str, object] | None = None):
    identity = DatasetIdentity(
        kind=kind,
        layer=layer,
        policy_version=f"{kind}-fixture-v1",
        inputs={},
        params=params or {},
    )
    published = publish_dataset(
        layer_root=root / ("gold" if layer is DatasetLayer.GOLD else "silver") / "kr_swing_2019_v1",
        identity=identity,
        partitions={"part.parquet": frame},
    )
    DatasetRegistry(root / "state" / "kr_swing_2019_v1").register(kind, published.dataset_id)
    return published


def _strategy_config(tmp_path: Path) -> Path:
    path = tmp_path / "strategy.toml"
    path.write_text("min_adtv20_krw = 0\nmax_names = 1\n", encoding="utf-8")
    return path


def _args(tmp_path: Path, strategy: Path, *, dividends: str | None = None, no_dividends: bool = False) -> list[str]:
    args = [
        "--scope-config", str(_SCOPE),
        "--data-root", str(tmp_path / "data"),
        "--strategy", "equal_weight_liquid",
        "--strategy-config", str(strategy),
        "--engine-config", str(_ENGINE),
        "--capital", "1000000",
        "--start", "2024-01-02",
        "--end", "2024-01-03",
    ]
    if dividends is not None:
        args.extend(["--dividends-dataset-id", dividends])
    if no_dividends:
        args.append("--no-dividends")
    return args


def _setup(tmp_path: Path) -> tuple[Path, object]:
    data_root = tmp_path / "data"
    panel = _publish(data_root, "market_panel", DatasetLayer.GOLD, _panel_frame())
    strategy = _strategy_config(tmp_path)
    return strategy, panel


def test_cli_resolves_dividends_from_silver_registry_and_records_ids(tmp_path: Path, capsys) -> None:
    strategy, panel = _setup(tmp_path)
    dividends = _publish(
        tmp_path / "data",
        "dividend_events",
        DatasetLayer.SILVER,
        pl.DataFrame(
            {
                "instrument_id": ["KRX:005930"],
                "ex_session": [date(2024, 1, 3)],
                "pay_session": [date(2024, 1, 3)],
                "dps_krw": [100],
            }
        ),
    )

    assert main(_args(tmp_path, strategy)) == 0
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 2
    run_dirs = sorted((tmp_path / "data" / "state" / "kr_swing_2019_v1" / "backtests").iterdir())
    manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8")) for path in run_dirs if path.is_dir()]
    assert manifests
    assert {manifest["inputs"]["market_panel"] for manifest in manifests} == {panel.dataset_id}
    assert {manifest["inputs"]["dividends"] for manifest in manifests} == {dividends.dataset_id}


def test_cli_missing_dividend_fails_before_simulation(tmp_path: Path, monkeypatch, capsys) -> None:
    strategy, _panel = _setup(tmp_path)
    import src.backtest.engine as engine

    def _must_not_run(*args, **kwargs):
        raise AssertionError("simulation started before dividend resolution")

    monkeypatch.setattr(engine, "run_backtest", _must_not_run)
    assert main(_args(tmp_path, strategy)) == 1
    assert "not registered" in json.loads(capsys.readouterr().out)["error"]


def test_cli_no_dividends_bypasses_registry(tmp_path: Path, capsys) -> None:
    strategy, _panel = _setup(tmp_path)

    assert main(_args(tmp_path, strategy, no_dividends=True)) == 0
    run_dirs = sorted((tmp_path / "data" / "state" / "kr_swing_2019_v1" / "backtests").iterdir())
    manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8")) for path in run_dirs if path.is_dir()]
    assert manifests
    assert {manifest["inputs"]["dividends"] for manifest in manifests} == {"none"}


def test_cli_rejects_conflicting_dividend_flags_before_resolution(tmp_path: Path, capsys) -> None:
    strategy, _panel = _setup(tmp_path)
    args = _args(tmp_path, strategy, dividends="dividend_events_0123456789abcdef", no_dividends=True)

    assert main(args) == 1
    assert "cannot be combined" in json.loads(capsys.readouterr().out)["error"]


def test_cli_rejects_non_gold_panel_dataset(tmp_path: Path, capsys) -> None:
    strategy, _panel = _setup(tmp_path)
    wrong = _publish(
        tmp_path / "data",
        "daily_market",
        DatasetLayer.GOLD,
        _panel_frame(),
    )
    args = _args(tmp_path, strategy, no_dividends=True)
    args.extend(["--panel-dataset-id", wrong.dataset_id])

    assert main(args) == 1
    assert "not a verified Gold market_panel" in json.loads(capsys.readouterr().out)["error"]


def test_cli_rejects_non_silver_dividend_dataset(tmp_path: Path, capsys) -> None:
    strategy, _panel = _setup(tmp_path)
    wrong = _publish(
        tmp_path / "data",
        "daily_market",
        DatasetLayer.SILVER,
        pl.DataFrame({"instrument_id": ["KRX:005930"]}),
    )
    args = _args(tmp_path, strategy, dividends=wrong.dataset_id)

    assert main(args) == 1
    assert "not a verified Silver dividend_events" in json.loads(capsys.readouterr().out)["error"]
