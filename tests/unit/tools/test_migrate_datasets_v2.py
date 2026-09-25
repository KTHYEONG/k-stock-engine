"""Migration copy/transaction and lineage invariants."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from tools.migrate_datasets_v2 import migrate_datasets_v2

_SCOPE = Path("config/research/kr_swing_2019_v1.toml")
_RULES = Path("config/market/krx_market_rules.toml")
_D0 = date(2020, 1, 6)
_D1 = date(2020, 1, 7)


def _write_legacy(root: Path, dataset_id: str, frame: pl.DataFrame, extra: dict[str, object] | None = None) -> Path:
    dataset = root / dataset_id
    partitions: list[dict[str, object]] = []
    if "session" in frame.columns and frame["session"].n_unique() > 1:
        for session in sorted(frame["session"].unique().to_list()):
            part_frame = frame.filter(pl.col("session") == session)
            relative = f"session={session.isoformat()}/part.parquet"
            partition = dataset / relative
            partition.parent.mkdir(parents=True, exist_ok=True)
            part_frame.write_parquet(partition)
            partitions.append({
                "path": relative,
                "row_count": part_frame.height,
                "parquet_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
            })
    else:
        partition = dataset / "part-00000.parquet"
        partition.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(partition)
        partitions.append({
            "path": partition.name,
            "row_count": frame.height,
            "parquet_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
        })
    manifest = {
        "dataset_id": dataset_id,
        "policy_version": "fixture-v1",
        "partitions": partitions,
        **(extra or {}),
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return dataset


def _daily_row(day: date, ticker: str) -> dict[str, object]:
    return {
        "session": day,
        "instrument_id": f"KRX:{ticker}",
        "ticker": ticker,
        "market": "KOSPI",
        "open": 100,
        "high": 100,
        "low": 100,
        "close": 100,
        "change": 0,
        "base_price": 100,
        "volume": 1000,
        "trading_value": 100000,
        "market_cap": 1000000,
        "listed_shares": 1000,
        "price_state": "tradable",
        "invalid_reason": None,
        "available_at": datetime(day.year, day.month, day.day, 18, tzinfo=__import__("src.core.time", fromlist=["KRX_TZ"]).KRX_TZ),
        "source_hash": "d" * 64,
        "policy_version": "krx-daily-market-v1",
    }


def _universe_row(day: date, ticker: str) -> dict[str, object]:
    return {
        "session": day,
        "instrument_id": f"KRX:{ticker}",
        "ticker": ticker,
        "eligible": True,
        "exclusion_reason": "eligible",
    }


def test_migration_stages_copy_and_registers_rebuilt_gold(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    silver = data_root / "silver" / "kr_swing_2019_v1"
    gold = data_root / "gold" / "kr_swing_2019_v1"
    universe_id = "ordinary_universe_" + "a" * 16
    daily_id = "daily_market_" + "b" * 16
    panel_id = "market_panel_" + "c" * 16
    universe_frame = pl.DataFrame([_universe_row(_D0, "005930"), _universe_row(_D1, "005930")])
    daily_frame = pl.DataFrame([_daily_row(_D0, "005930"), _daily_row(_D1, "005930")])
    panel_frame = pl.DataFrame([
        {**_daily_row(day, "005930"), "eligible": True, "exclusion_reason": "eligible", "gap_before": False,
         "ret_price": None if day == _D0 else 0.0, "share_factor": None if day == _D0 else 1.0,
         "tick_size": 1, "upper_limit": 200, "lower_limit": 1, "limits_applicable": True,
         "open_at_upper": False, "open_at_lower": False, "close_at_upper": False, "close_at_lower": False,
         "adtv20": 100000.0, "adtv60": None, "ret_vol60": None}
        for day in (_D0, _D1)
    ])
    old_universe = _write_legacy(silver, universe_id, universe_frame, {"source_hashes": ["u" * 64]})
    old_daily = _write_legacy(silver, daily_id, daily_frame, {"universe_dataset_id": universe_id, "source_hashes": ["d" * 64]})
    old_panel = _write_legacy(gold, panel_id, panel_frame, {
        "daily_market_dataset_id": daily_id,
        "universe_dataset_id": universe_id,
        "policy_version": "krx-market-panel-v2",
        "adtv_short_sessions": 20,
        "adtv_long_sessions": 60,
        "return_vol_sessions": 60,
    })
    assert old_universe.is_dir()
    assert old_daily.is_dir()
    assert old_panel.is_dir()

    results = migrate_datasets_v2(
        scope_config=_SCOPE,
        data_root=data_root,
        apply=False,
        rules_path=_RULES,
    )

    assert {result.kind for result in results} == {"ordinary_universe", "daily_market", "market_panel"}
    assert old_universe.is_dir()
    assert old_daily.is_dir()
    assert old_panel.is_dir()
    registry = json.loads((data_root / "state" / "kr_swing_2019_v1" / "datasets.json").read_text(encoding="utf-8"))
    assert set(registry["current"]) == {"ordinary_universe", "daily_market", "market_panel"}
    for result in results:
        assert result.path.is_dir()
        assert (result.path / "manifest.json").is_file()


def test_discovery_includes_nested_financial_generation(tmp_path: Path) -> None:
    from tools.migrate_datasets_v2 import _discover

    data_root = tmp_path / "data"
    generation = data_root / "silver" / "kr_swing_2019_v1" / "financial_facts" / ("f" * 64)
    generation.mkdir(parents=True)
    partition = generation / "part.parquet"
    pl.DataFrame({"value": [1]}).write_parquet(partition)
    (generation / "content_manifest.json").write_text(json.dumps({
        "partitions": [{"path": partition.name, "sha256": hashlib.sha256(partition.read_bytes()).hexdigest()}]
    }), encoding="utf-8")
    (generation / "dataset_manifest.json").write_text(json.dumps({"time_end": "2020-01-01T00:00:00+00:00"}), encoding="utf-8")

    found = _discover(
        silver_root=data_root / "silver" / "kr_swing_2019_v1",
        gold_root=data_root / "gold" / "kr_swing_2019_v1",
    )

    assert [dataset.dataset_id for dataset in found] == ["financial_facts_" + "f" * 16]


def test_migration_rebuilds_financial_facts_and_quality_with_row_equality(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from src.core.krx_calendar import xkrx_session_calendar
    from src.data.financial_quality import build_financial_quality_events
    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.datasets import read_dataset

    data_root = tmp_path / "data"
    bronze = data_root / "bronze" / "kr_swing_2019_v1"
    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    payload = json.dumps(
        {
            "records": [
                {
                    "ticker": "005930",
                    "corp_code": "00126380",
                    "fiscal_period": "2015Q3",
                    "filing_id": "F1",
                    "fact": "sales",
                    "published_at": "2015-11-16T00:00:00+00:00",
                    "value": 10.0,
                    "unit": "KRW",
                }
            ]
        },
        sort_keys=True,
    ).encode()
    receipt_dir = bronze / "financial_facts" / "fixture"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_bytes(payload)
    (receipt_dir / "receipt.json").write_text(
        json.dumps(
            {
                "kind": "financial_facts",
                "content_hash": hashlib.sha256(payload).hexdigest(),
                "source_path": "fixture",
                "retrieved_at": "2016-01-01T00:00:00+00:00",
                "ingested_at": "2016-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    seed = refresh_dart_financial_facts(
        bronze_root=bronze,
        silver_root=tmp_path / "seed-silver",
        artifact_root=tmp_path / "seed-artifacts",
        decision_time=decision_time,
        calendar=xkrx_session_calendar(),
    )
    facts_frame = read_dataset(Path(seed.dataset_path)).collect()
    quality_frame = build_financial_quality_events(
        facts_frame, unresolved_events=(), decision_time=decision_time
    )

    def _write_nested(table: str, generation: str, frame: pl.DataFrame) -> Path:
        generation_dir = data_root / "silver" / "kr_swing_2019_v1" / table / generation
        generation_dir.mkdir(parents=True)
        partition = generation_dir / "part.parquet"
        frame.write_parquet(partition)
        (generation_dir / "content_manifest.json").write_text(
            json.dumps(
                {
                    "partitions": [
                        {"path": partition.name, "sha256": hashlib.sha256(partition.read_bytes()).hexdigest()}
                    ]
                }
            ),
            encoding="utf-8",
        )
        (generation_dir / "dataset_manifest.json").write_text(
            json.dumps({"time_end": decision_time.isoformat()}), encoding="utf-8"
        )
        return generation_dir

    old_facts = _write_nested("financial_facts", "a" * 64, facts_frame)
    old_quality = _write_nested("financial_quality", "b" * 64, quality_frame)

    results = migrate_datasets_v2(
        scope_config=_SCOPE,
        data_root=data_root,
        apply=False,
        rules_path=_RULES,
    )

    assert {result.kind for result in results} == {"financial_facts", "financial_quality"}
    assert old_facts.is_dir()
    assert old_quality.is_dir()
    rebuilt = {result.kind: read_dataset(result.path).collect() for result in results}
    assert rebuilt["financial_facts"].equals(facts_frame)
    assert rebuilt["financial_quality"].equals(quality_frame)
    registry = json.loads(
        (data_root / "state" / "kr_swing_2019_v1" / "datasets.json").read_text(encoding="utf-8")
    )
    assert set(registry["current"]) == {"financial_facts", "financial_quality"}


def test_migration_retires_dangling_ls_universe_with_exact_reason(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    silver = data_root / "silver" / "kr_swing_2019_v1"
    flow_id = "investor_flow_1111111111111111"
    old_universe_id = "ordinary_universe_eeb927f1c7b84081"
    frame = pl.DataFrame(
        {
            "session": [_D0],
            "ticker": ["005930"],
            "individual_net_shares": [1],
            "foreign_net_shares": [0],
            "institution_net_shares": [0],
            "other_net_shares": [0],
        }
    )
    old_flow = _write_legacy(silver, flow_id, frame, {"universe_dataset_id": old_universe_id})

    results = migrate_datasets_v2(scope_config=_SCOPE, data_root=data_root, apply=False, rules_path=_RULES)

    assert [result.kind for result in results] == ["investor_flow_ls"]
    assert old_flow.is_dir()
    registry = json.loads(
        (data_root / "state" / "kr_swing_2019_v1" / "datasets.json").read_text(encoding="utf-8")
    )
    assert registry["retired"][old_universe_id] == (
        "universe generation superseded before hash-verified lineage; LS flow kept for union lineage"
    )


def test_migration_abort_removes_staged_targets_and_preserves_legacy_scope(tmp_path: Path, monkeypatch) -> None:
    import tools.migrate_datasets_v2 as migration

    data_root = tmp_path / "data"
    silver = data_root / "silver" / "kr_swing_2019_v1"
    old_id = "ordinary_universe_" + "a" * 16
    old = _write_legacy(
        silver,
        old_id,
        pl.DataFrame([_universe_row(_D0, "005930")]),
        {"source_hashes": ["a" * 64]},
    )

    def _fail_registry(*_args, **_kwargs):
        raise OSError("registry commit failed")

    monkeypatch.setattr(migration, "_write_registry_document", _fail_registry)
    with pytest.raises(OSError, match="registry commit failed"):
        migration.migrate_datasets_v2(
            scope_config=_SCOPE, data_root=data_root, apply=False, rules_path=_RULES
        )

    assert old.is_dir()
    assert not any(path.name.startswith("ordinary_universe_") and path.name != old_id for path in silver.iterdir())
    assert not (data_root / "state" / "kr_swing_2019_v1" / "datasets.json").exists()
