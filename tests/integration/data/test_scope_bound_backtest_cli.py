from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.data.cli import _parse_args, main

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
SESSIONS = [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5), date(2023, 1, 6)]
CLOSES = [100.0, 110.0, 120.0, 130.0, 140.0]
A = "KRX:005930"


def _write_scope_config(tmp_path: Path, *, scope_id: str = "kr_swing_2019_v1", hold_end: str = "2025-12-31", forward: str = "2026-01-01") -> Path:
    config_path = tmp_path / f"{scope_id}.toml"
    config_path.write_text(
        "\n".join([
            f'scope_id = "{scope_id}"',
            'evidence_start = "2019-01-01"',
            'development_start = "2020-01-01"',
            'development_end = "2022-12-31"',
            'validation_start = "2023-01-01"',
            'validation_end = "2023-12-31"',
            'holdout_start = "2024-01-01"',
            f'holdout_end = "{hold_end}"',
            f'forward_start = "{forward}"',
            "[features]",
            "price_lookback_sessions = 252",
            "fundamental_lookback_quarters = 5",
            'fundamental_fiscal_start = "2019Q1"',
            "investor_flow_enabled = false",
            "industry_enabled = false",
            "[collection]",
            "dart_daily_budget = 16000",
            "dart_batch_identities = 500",
        ]),
        encoding="utf-8",
    )
    return config_path


def _seed(tmp_path: Path, *, scope_config: Path | None = None, sessions: list[date] | None = None) -> tuple[Path, Path]:
    from src.data.gold import create_scope_bound_gold_release
    from src.data.runtime import load_data_runtime
    from src.data.scope_coverage import CoverageRequirement, ScopeCoverageReport

    days = sessions or SESSIONS
    runtime = load_data_runtime(scope_config=scope_config or SCOPE_CONFIG, data_root=tmp_path / "data")
    bars_dir = runtime.workspace.silver_root / "daily_market" / "bars-v1"
    bars_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([
        {"session": day, "instrument_id": A, "close": CLOSES[SESSIONS.index(day)] if day in SESSIONS else 100.0}
        for day in days
    ]).write_parquet(bars_dir / "bars.parquet")
    (bars_dir / "dataset_manifest.json").write_text(
        json.dumps({"scope_hash": runtime.scope.content_hash, "dataset_id": "bars-v1"}), encoding="utf-8"
    )
    coverage = ScopeCoverageReport(
        scope_hash=runtime.scope.content_hash,
        fulfilled=(
            CoverageRequirement(source="krx_daily_market", natural_key=days[0].isoformat(), as_of=days[0], fiscal_period=None, required=True),
            CoverageRequirement(source="financial_facts", natural_key="00126380:2023:11013", as_of=date(2023, 5, 15), fiscal_period="2023Q1", required=True),
        ),
        missing=(),
        unresolved=(),
    )
    create_scope_bound_gold_release(
        runtime=runtime, dataset_id="gold-v1", silver_dataset_ids={"daily_market": "bars-v1"},
        universe_policy_hash="u", feature_policy_hash="f", coverage_report=coverage,
    )
    release_dir = runtime.workspace.gold_root / "releases" / "gold-v1"
    pl.DataFrame([
        {"session": day, "instrument_id": A, "eligible": True, "verified": True, "common": True} for day in days
    ]).write_parquet(release_dir / "universe.parquet")
    strategy_policy = tmp_path / "strategy.json"
    strategy_policy.write_text(json.dumps({"strategy_id": "equal-weight", "kind": "equal_weight", "target_weight": 1.0, "max_positions": 5}), encoding="utf-8")
    execution_policy = tmp_path / "execution.json"
    execution_policy.write_text(json.dumps({"commission_rate": 0.001, "tax_rate": 0.0025}), encoding="utf-8")
    universe_policy = tmp_path / "universe.json"
    universe_policy.write_text(json.dumps({"universe": "common-stock-v1"}), encoding="utf-8")
    return runtime, strategy_policy, execution_policy, universe_policy


def _backtest_args(tmp_path: Path, scope_config: Path, strategy_policy: Path, execution_policy: Path, universe_policy: Path, extra: list[str] | None = None) -> list[str]:
    return [
        "backtest", "--scope-config", str(scope_config), "--data-root", str(tmp_path / "data"),
        "--segment", "validation", "--silver-dataset-id", "daily_market=bars-v1",
        "--gold-dataset-id", "gold-v1", "--strategy-id", "equal-weight",
        "--strategy-policy", str(strategy_policy), "--execution-policy", str(execution_policy),
        "--universe-policy", str(universe_policy), *(extra or []),
    ]


def test_backtest_command_has_no_date_fallback() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["backtest", "--scope-config", "s.toml", "--data-root", "d"])
    with pytest.raises(SystemExit):
        _parse_args([
            "backtest", "--scope-config", "s.toml", "--data-root", "d", "--segment", "validation",
            "--validation-start", "2016-01-04",
        ])
    with pytest.raises(SystemExit):
        _parse_args([
            "backtest", "--scope-config", "s.toml", "--data-root", "d", "--segment", "forward",
            "--gold-dataset-id", "g", "--strategy-id", "s",
        ])


def test_backtest_command_writes_immutable_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runtime, strategy_policy, execution_policy, universe_policy = _seed(tmp_path)

    assert main(_backtest_args(tmp_path, SCOPE_CONFIG, strategy_policy, execution_policy, universe_policy)) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    run_dir = Path(out["run_dir"])
    assert run_dir.parent.name == "backtests"
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "result.json").is_file()
    assert out["manifest_hash"] == run_dir.name
    assert out["metrics"]["trade_count"] > 0
    assert runtime.workspace.runs_root in run_dir.parents


def test_backtest_command_reuses_only_same_identity(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed(tmp_path)
    args = _backtest_args(tmp_path, SCOPE_CONFIG, tmp_path / "strategy.json", tmp_path / "execution.json", tmp_path / "universe.json")

    assert main(args) == 0
    first = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    before = (Path(first["run_dir"]) / "result.json").read_bytes()
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert second["run_dir"] == first["run_dir"]
    assert second["manifest_hash"] == first["manifest_hash"]
    assert (Path(second["run_dir"]) / "result.json").read_bytes() == before


def test_backtest_command_rejects_post_2025_segments(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scope_config = _write_scope_config(tmp_path, scope_id="future_scope", hold_end="2026-12-31", forward="2027-01-01")
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    runtime, strategy_policy, execution_policy, universe_policy = _seed(tmp_path, scope_config=scope_config, sessions=days)

    args = [
        "backtest", "--scope-config", str(scope_config), "--data-root", str(tmp_path / "data"),
        "--segment", "holdout", "--silver-dataset-id", "daily_market=bars-v1",
        "--gold-dataset-id", "gold-v1", "--strategy-id", "equal-weight",
        "--strategy-policy", str(strategy_policy), "--execution-policy", str(execution_policy),
        "--universe-policy", str(universe_policy),
    ]
    assert main(args) == 1
    assert "2026" in capsys.readouterr().out


def test_backtest_command_rejects_malformed_inputs(tmp_path: Path) -> None:
    _seed(tmp_path)
    base = _backtest_args(tmp_path, SCOPE_CONFIG, tmp_path / "strategy.json", tmp_path / "execution.json", tmp_path / "universe.json")

    def _without(flag: str) -> list[str]:
        out = list(base)
        index = out.index(flag)
        del out[index:index + 2]
        return out

    assert main(_without("--silver-dataset-id")) == 1
    malformed = [token if token != "daily_market=bars-v1" else "daily_market" for token in base]
    assert main(malformed) == 1
    duplicated = [*base, "--silver-dataset-id", "daily_market=bars-v1"]
    assert main(duplicated) == 1
    unknown = [token if token != "daily_market=bars-v1" else "nope=bars-v1" for token in base]
    assert main(unknown) == 1
    assert main([token if token != "equal-weight" else "nope" for token in base]) == 1
    bad_policy = tmp_path / "bad-policy.json"
    bad_policy.write_text("[1]", encoding="utf-8")
    assert main([token if token != str(tmp_path / "strategy.json") else str(bad_policy) for token in base]) == 1
