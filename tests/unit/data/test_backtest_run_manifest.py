from pathlib import Path


def test_backtest_run_manifest_round_trip_is_content_addressed(tmp_path) -> None:
    from datetime import date

    from src.data.backtest_run_manifest import (
        build_legacy_backtest_run_manifest,
        load_legacy_backtest_run_manifest,
        write_legacy_backtest_run_manifest,
    )
    from src.data.schemas import SilverTable

    ids = {table: (table.value + '-hash') for table in SilverTable}
    manifest = build_legacy_backtest_run_manifest(
        silver_root=tmp_path / 'silver', gold_root=tmp_path / 'gold',
        silver_dataset_ids=ids, gold_dataset_id='gold-hash',
        validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29),
        strategy_id='core-v1', policy_versions={'market_inputs': 'korean-equity-market-inputs-v2'},
    )

    first = write_legacy_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / 'artifacts')
    second = write_legacy_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / 'artifacts')
    loaded = load_legacy_backtest_run_manifest(first)

    assert first == second
    assert first.name == f'{manifest.content_hash}.json'
    assert loaded == manifest
    assert loaded.silver_dataset_ids['daily_market'] == 'daily_market-hash'


def test_backtest_run_manifest_rejects_tampered_hash_and_incomplete_tables(tmp_path) -> None:
    import json
    from datetime import date

    import pytest

    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest, load_legacy_backtest_run_manifest
    from src.data.schemas import PITDataError, SilverTable

    incomplete = {SilverTable.CALENDAR: 'calendar-hash'}
    with pytest.raises(PITDataError, match='exactly'):
        build_legacy_backtest_run_manifest(
            silver_root=tmp_path / 'silver', gold_root=tmp_path / 'gold', silver_dataset_ids=incomplete,
            gold_dataset_id='gold-hash', validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29),
            strategy_id='core-v1', policy_versions={'market_inputs': 'korean-equity-market-inputs-v2'},
        )

    path = tmp_path / 'tampered.json'
    path.write_text(json.dumps({'schema_version': 'backtest-run-v1', 'content_hash': '0' * 64}), encoding='utf-8')
    with pytest.raises(PITDataError, match='manifest'):
        load_legacy_backtest_run_manifest(path)


def _manifest_kwargs(tmp_path, **overrides):
    from datetime import date

    from src.data.schemas import SilverTable

    base = {
        "silver_root": tmp_path / "silver",
        "gold_root": tmp_path / "gold",
        "silver_dataset_ids": {table: f"{table.value}-hash" for table in SilverTable},
        "gold_dataset_id": "gold-hash",
        "validation_start": date(2016, 1, 4),
        "validation_end": date(2016, 12, 29),
        "strategy_id": "core-v1",
        "policy_versions": {"market_inputs": "korean-equity-market-inputs-v2"},
    }
    base.update(overrides)
    return base


def test_backtest_run_manifest_rejects_invalid_dataset_ids(tmp_path) -> None:
    import pytest

    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest
    from src.data.schemas import PITDataError, SilverTable

    for bad in ("a/b", "a\\b", ".", "..", "   ", "", " padded "):
        ids = {table: (bad if table == SilverTable.CALENDAR else f"{table.value}-hash") for table in SilverTable}
        with pytest.raises(PITDataError, match="dataset"):
            build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path, silver_dataset_ids=ids))
    with pytest.raises(PITDataError, match="dataset"):
        build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path, gold_dataset_id="a/b"))
    with pytest.raises(PITDataError, match="dataset"):
        build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path, gold_dataset_id=123))


def test_backtest_run_manifest_rejects_bad_tables_policies_and_range(tmp_path) -> None:
    from datetime import date

    import pytest

    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="exactly"):
        build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path, silver_dataset_ids={"nope": "x"}))
    with pytest.raises(PITDataError, match="exactly"):
        build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path, silver_dataset_ids=[("calendar", "x")]))
    with pytest.raises(PITDataError, match="exactly"):
        build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path, silver_dataset_ids={42: "x"}))
    with pytest.raises(PITDataError, match="polic"):
        build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path, policy_versions=[("a", "b")]))
    with pytest.raises(PITDataError, match="validation_start"):
        build_legacy_backtest_run_manifest(
            **_manifest_kwargs(
                tmp_path, validation_start=date(2016, 12, 29), validation_end=date(2016, 1, 4)
            )
        )
    with pytest.raises(PITDataError, match="strategy_id"):
        build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path, strategy_id=""))
    for policies, _match in (({}, "policy"), ({"": "v"}, "policy"), ({"k": ""}, "policy")):
        with pytest.raises(PITDataError, match="polic"):
            build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path, policy_versions=policies))


def test_backtest_run_manifest_load_rejects_corrupt_files(tmp_path) -> None:
    import json

    import pytest

    from src.data.backtest_run_manifest import (
        build_legacy_backtest_run_manifest,
        load_legacy_backtest_run_manifest,
        write_legacy_backtest_run_manifest,
    )
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="manifest"):
        load_legacy_backtest_run_manifest(tmp_path / "absent.json")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{nope", encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_legacy_backtest_run_manifest(bad_json)
    not_dict = tmp_path / "list.json"
    not_dict.write_text("[1]", encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_legacy_backtest_run_manifest(not_dict)
    manifest = build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path))
    good = write_legacy_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / "artifacts")
    raw = json.loads(good.read_text(encoding="utf-8"))
    for field in ("content_hash", "silver_dataset_ids"):
        mutated = dict(raw)
        mutated[field] = 42
        path = tmp_path / f"wrong-{field}.json"
        path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(PITDataError, match="manifest"):
            load_legacy_backtest_run_manifest(path)
    mutated = dict(raw)
    mutated["schema_version"] = "other-v9"
    path = tmp_path / "wrong-version.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_legacy_backtest_run_manifest(path)
    mutated = dict(raw)
    mutated["extra"] = 1
    path = tmp_path / "extra-field.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_legacy_backtest_run_manifest(path)
    mutated = dict(raw)
    mutated["validation_start"] = "not-a-date"
    path = tmp_path / "bad-date.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_legacy_backtest_run_manifest(path)
    mutated = dict(raw)
    mutated["content_hash"] = "0" * 64
    path = tmp_path / "hash-mismatch.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_legacy_backtest_run_manifest(path)


def test_backtest_run_manifest_write_rejects_tampered_existing_bytes(tmp_path) -> None:
    import json

    import pytest

    from src.data.backtest_run_manifest import (
        _manifest_to_json,
        build_legacy_backtest_run_manifest,
        write_legacy_backtest_run_manifest,
    )
    from src.data.schemas import PITDataError

    manifest = build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path))
    first = write_legacy_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / "artifacts")
    first.write_text("{tampered", encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        write_legacy_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / "artifacts")
    forged = build_legacy_backtest_run_manifest(
        **_manifest_kwargs(tmp_path, strategy_id="tampered")
    )
    object.__setattr__(forged, "content_hash", manifest.content_hash)
    object.__setattr__(forged, "schema_version", "other")
    with pytest.raises(PITDataError, match="manifest"):
        write_legacy_backtest_run_manifest(manifest=forged, artifact_root=tmp_path / "other-artifacts")
    wrong_hash = build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path))
    object.__setattr__(wrong_hash, "content_hash", "1" * 64)
    with pytest.raises(PITDataError, match="manifest"):
        write_legacy_backtest_run_manifest(manifest=wrong_hash, artifact_root=tmp_path / "other-artifacts")
    different = build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path, strategy_id="other"))
    first.write_text(json.dumps(_manifest_to_json(different)), encoding="utf-8")
    with pytest.raises(PITDataError, match="differs"):
        write_legacy_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / "artifacts")


def test_backtest_run_manifest_treats_lifecycle_events_as_optional(tmp_path) -> None:
    from datetime import date

    import pytest

    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest
    from src.data.schemas import PITDataError, SilverTable

    base = {
        "silver_root": tmp_path / "silver",
        "gold_root": tmp_path / "gold",
        "gold_dataset_id": "gold-hash",
        "validation_start": date(2016, 1, 4),
        "validation_end": date(2016, 12, 29),
        "strategy_id": "core-v1",
        "policy_versions": {"market_inputs": "korean-equity-market-inputs-v2"},
    }
    full = {table: f"{table.value}-hash" for table in SilverTable}
    manifest = build_legacy_backtest_run_manifest(silver_dataset_ids=full, **base)
    assert manifest.silver_dataset_ids[SilverTable.LIFECYCLE_EVENTS.value] == "lifecycle_events-hash"
    without_lifecycle = {
        table: f"{table.value}-hash" for table in SilverTable if table is not SilverTable.LIFECYCLE_EVENTS
    }
    with pytest.raises(PITDataError, match="exactly every SilverTable"):
        build_legacy_backtest_run_manifest(silver_dataset_ids=without_lifecycle, **base)


from datetime import date

import pytest

from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest
from src.data.schemas import PITDataError, SilverTable

def test_manifest_v2_rejects_missing_lifecycle_dataset(tmp_path):
    ids = {table: 'dataset' for table in SilverTable if table is not SilverTable.LIFECYCLE_EVENTS}
    with pytest.raises(PITDataError, match='exactly every SilverTable'):
        build_legacy_backtest_run_manifest(silver_root=tmp_path / 'silver', gold_root=tmp_path / 'gold', silver_dataset_ids=ids, gold_dataset_id='gold', validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29), strategy_id='core-v1', policy_versions={'pit': 'v1'})


def test_manifest_v1_fails_with_rebuild_required(tmp_path):
    import json
    from datetime import date
    import pytest
    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest, load_legacy_backtest_run_manifest, write_legacy_backtest_run_manifest
    from src.data.schemas import SilverTable
    ids = {table: f"{table.value}-hash" for table in SilverTable}
    manifest = build_legacy_backtest_run_manifest(silver_root=tmp_path / "silver", gold_root=tmp_path / "gold", silver_dataset_ids=ids, gold_dataset_id="gold", validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29), strategy_id="core-v1", policy_versions={"pit": "v1"})
    path = write_legacy_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / "artifacts")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = "backtest-run-v1"
    v1_path = tmp_path / "v1.json"
    v1_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(Exception, match="rebuild-required"):
        load_legacy_backtest_run_manifest(v1_path)


def test_backtest_run_manifest_rejects_compounding_v2_before_fundamental_floor(tmp_path) -> None:
    from datetime import date

    import pytest

    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest
    from src.data.schemas import PITDataError

    # When/Then: QVEF lookback cannot be served before the FY2015 floor.
    with pytest.raises(PITDataError, match="compounding-v2 validation_start"):
        build_legacy_backtest_run_manifest(
            **_manifest_kwargs(
                tmp_path, strategy_id="compounding-v2",
                validation_start=date(2016, 5, 15), validation_end=date(2016, 12, 29),
            )
        )

    # And: the floor date itself is accepted.
    accepted = build_legacy_backtest_run_manifest(
        **_manifest_kwargs(
            tmp_path, strategy_id="compounding-v2",
            validation_start=date(2016, 5, 16), validation_end=date(2016, 12, 29),
        )
    )
    assert accepted.validation_start == date(2016, 5, 16)

    # And: strategies that do not consume QVEF keep their earlier windows.
    assert build_legacy_backtest_run_manifest(**_manifest_kwargs(tmp_path)).strategy_id == "core-v1"


SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")


def _scope_runtime(tmp_path: Path, *, scope_id: str = "kr_swing_2019_v1", filename: str = "scope.toml",
                   evidence: str = "2019-01-01", dev: tuple[str, str] = ("2020-01-01", "2022-12-31"),
                   val: tuple[str, str] = ("2023-01-01", "2023-12-31"),
                   hold: tuple[str, str] = ("2024-01-01", "2025-12-31"), forward: str = "2026-01-01"):
    from src.data.runtime import load_data_runtime

    config_path = tmp_path / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join([
            f'scope_id = "{scope_id}"',
            f'evidence_start = "{evidence}"',
            f'development_start = "{dev[0]}"',
            f'development_end = "{dev[1]}"',
            f'validation_start = "{val[0]}"',
            f'validation_end = "{val[1]}"',
            f'holdout_start = "{hold[0]}"',
            f'holdout_end = "{hold[1]}"',
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
    return load_data_runtime(scope_config=config_path, data_root=tmp_path / "data")


def _stamp_silver(runtime, *, table: str = "daily_market", dataset_id: str = "bars-v1", scope_hash: str | None = None) -> None:
    import json

    target = runtime.workspace.silver_root / table / dataset_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "dataset_manifest.json").write_text(
        json.dumps({"scope_hash": scope_hash or runtime.scope.content_hash, "dataset_id": dataset_id}),
        encoding="utf-8",
    )


def _coverage(runtime, *, missing: tuple = (), unresolved: tuple = ()):
    from datetime import date

    from src.data.scope_coverage import CoverageRequirement, ScopeCoverageReport

    fulfilled = (
        CoverageRequirement(source="krx_daily_market", natural_key="2023-01-02", as_of=date(2023, 1, 2), fiscal_period=None, required=True),
        CoverageRequirement(source="financial_facts", natural_key="00126380:2023:11013", as_of=date(2023, 5, 15), fiscal_period="2023Q1", required=True),
    )
    return ScopeCoverageReport(scope_hash=runtime.scope.content_hash, fulfilled=fulfilled, missing=tuple(missing), unresolved=tuple(unresolved))


def _stamp_release(runtime, *, dataset_id: str = "gold-v1", silver_ids: dict | None = None):
    from src.data.gold import create_scope_bound_gold_release

    return create_scope_bound_gold_release(
        runtime=runtime,
        dataset_id=dataset_id,
        silver_dataset_ids=silver_ids or {"daily_market": "bars-v1"},
        universe_policy_hash="u",
        feature_policy_hash="f",
        coverage_report=_coverage(runtime),
    )


def _scope_fixture(tmp_path: Path):
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    _stamp_silver(runtime)
    _stamp_release(runtime)
    return runtime


def _build_scope_manifest(runtime, **overrides):
    from src.data.backtest_run_manifest import build_backtest_run_manifest

    params = {
        "segment": "validation",
        "silver_dataset_ids": {"daily_market": "bars-v1"},
        "gold_dataset_id": "gold-v1",
        "strategy_id": "equal-weight",
        "strategy_policy_hash": "s",
        "execution_policy_hash": "e",
        "universe_policy_hash": "u",
    }
    params.update(overrides)
    return build_backtest_run_manifest(runtime=runtime, **params)


def test_build_backtest_run_manifest_derives_segment_dates(tmp_path) -> None:
    from datetime import date

    runtime = _scope_fixture(tmp_path)

    validation = _build_scope_manifest(runtime, segment="validation")
    assert (validation.period_start, validation.period_end) == (date(2023, 1, 1), date(2023, 12, 31))
    development = _build_scope_manifest(runtime, segment="development")
    assert (development.period_start, development.period_end) == (date(2020, 1, 1), date(2022, 12, 31))
    holdout = _build_scope_manifest(runtime, segment="holdout")
    assert (holdout.period_start, holdout.period_end) == (date(2024, 1, 1), date(2025, 12, 31))
    assert len({validation.content_hash, development.content_hash, holdout.content_hash}) == 3

    again = _build_scope_manifest(runtime, segment="validation")
    assert again.content_hash == validation.content_hash
    manifest_path = runtime.workspace.runs_root / "backtests" / validation.content_hash / "manifest.json"
    assert manifest_path.is_file()


def test_build_backtest_run_manifest_binds_scope_hash(tmp_path) -> None:
    first = _scope_fixture(tmp_path / "a")
    second_runtime = _scope_runtime(tmp_path / "b", scope_id="kr_swing_2019_v2", filename="scope.toml")
    _stamp_silver(second_runtime)
    _stamp_release(second_runtime)

    first_manifest = _build_scope_manifest(first)
    second_manifest = _build_scope_manifest(second_runtime)

    assert first_manifest.scope_hash != second_manifest.scope_hash
    assert first_manifest.content_hash != second_manifest.content_hash


def test_build_backtest_run_manifest_rejects_legacy_period(tmp_path) -> None:
    import pytest

    from src.data.backtest_run_manifest import build_backtest_run_manifest
    from src.data.schemas import PITDataError

    runtime = _scope_runtime(
        tmp_path, scope_id="legacy_2016", filename="legacy.toml", evidence="2016-01-01",
        dev=("2016-01-04", "2016-12-30"), val=("2016-12-31", "2017-12-30"),
        hold=("2017-12-31", "2018-12-30"), forward="2018-12-31",
    )
    with pytest.raises(PITDataError, match="outside the completed scope"):
        build_backtest_run_manifest(
            runtime=runtime, segment="validation", silver_dataset_ids={"daily_market": "bars-v1"},
            gold_dataset_id="gold-v1", strategy_id="equal-weight",
            strategy_policy_hash="s", execution_policy_hash="e", universe_policy_hash="u",
        )


def test_build_backtest_run_manifest_rejects_forward_segment(tmp_path) -> None:
    import pytest

    from src.data.backtest_run_manifest import build_backtest_run_manifest
    from src.data.schemas import PITDataError

    runtime = _scope_fixture(tmp_path)
    with pytest.raises(PITDataError, match="unknown backtest segment"):
        build_backtest_run_manifest(
            runtime=runtime, segment="forward", silver_dataset_ids={"daily_market": "bars-v1"},
            gold_dataset_id="gold-v1", strategy_id="equal-weight",
            strategy_policy_hash="s", execution_policy_hash="e", universe_policy_hash="u",
        )


def test_build_backtest_run_manifest_rejects_foreign_dataset(tmp_path) -> None:
    import json

    import pytest

    from src.data.schemas import PITDataError

    runtime = _scope_fixture(tmp_path)

    _stamp_silver(runtime, dataset_id="foreign-bars", scope_hash="0" * 64)
    with pytest.raises(PITDataError, match="foreign Scope hash"):
        _build_scope_manifest(runtime, silver_dataset_ids={"daily_market": "foreign-bars"})
    with pytest.raises(PITDataError, match="foreign Scope hash"):
        _build_scope_manifest(runtime, silver_dataset_ids={"daily_market": "absent-bars"})
    with pytest.raises(PITDataError, match="single path component"):
        _build_scope_manifest(runtime, silver_dataset_ids={"daily_market": "../evil"})
    with pytest.raises(PITDataError, match="single path component"):
        _build_scope_manifest(runtime, silver_dataset_ids={"daily_market": ""})

    release_path = runtime.workspace.gold_root / "releases" / "gold-v1" / "release.json"
    raw = json.loads(release_path.read_text(encoding="utf-8"))
    raw["scope_hash"] = "0" * 64
    release_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PITDataError, match="foreign Scope hash"):
        _build_scope_manifest(runtime)
    release_path.unlink()
    with pytest.raises(PITDataError, match="foreign Scope hash"):
        _build_scope_manifest(runtime)

    _stamp_release(runtime, dataset_id="gold-now")
    now_path = runtime.workspace.gold_root / "releases" / "gold-now" / "release.json"
    now_raw = json.loads(now_path.read_text(encoding="utf-8"))
    now_raw["decision_time"] = "2026-05-01T00:00:00+00:00"
    now_path.write_text(json.dumps(now_raw), encoding="utf-8")
    with pytest.raises(PITDataError, match="outside the completed scope"):
        _build_scope_manifest(runtime, gold_dataset_id="gold-now")
    now_raw["decision_time"] = "not-a-date"
    now_path.write_text(json.dumps(now_raw), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid release metadata"):
        _build_scope_manifest(runtime, gold_dataset_id="gold-now")

    import shutil

    shutil.rmtree(runtime.workspace.silver_root / "daily_market")
    (tmp_path / "outside").mkdir()
    (runtime.workspace.silver_root / "daily_market").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(PITDataError, match="outside the workspace"):
        _build_scope_manifest(runtime)


def test_build_backtest_run_manifest_policy_change_invalidates_identity(tmp_path) -> None:
    runtime = _scope_fixture(tmp_path)

    assert _build_scope_manifest(runtime).content_hash != _build_scope_manifest(runtime, execution_policy_hash="e2").content_hash


def test_build_backtest_run_manifest_rejects_malformed_requests(tmp_path) -> None:
    import json

    import pytest

    from src.data.schemas import PITDataError

    runtime = _scope_fixture(tmp_path)

    with pytest.raises(PITDataError, match="non-empty mapping"):
        _build_scope_manifest(runtime, silver_dataset_ids={})
    with pytest.raises(PITDataError, match="non-empty mapping"):
        _build_scope_manifest(runtime, silver_dataset_ids=[("daily_market", "bars-v1")])
    _stamp_silver(runtime, table="calendar", dataset_id="cal-v1")
    with pytest.raises(PITDataError, match="daily_market bars"):
        _build_scope_manifest(runtime, silver_dataset_ids={"calendar": "cal-v1"})
    with pytest.raises(PITDataError, match="table name"):
        _build_scope_manifest(runtime, silver_dataset_ids={123: "bars-v1"})
    with pytest.raises(PITDataError, match="strategy_id"):
        _build_scope_manifest(runtime, **{"strategy_id": "  "})
    with pytest.raises(PITDataError, match="execution_policy_hash"):
        _build_scope_manifest(runtime, execution_policy_hash="")
    with pytest.raises(PITDataError, match="gold_dataset_id"):
        _build_scope_manifest(runtime, gold_dataset_id="../evil")

    manifest = _build_scope_manifest(runtime)
    manifest_path = runtime.workspace.runs_root / "backtests" / manifest.content_hash / "manifest.json"
    manifest_path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(PITDataError, match="differs from requested"):
        _build_scope_manifest(runtime)
    manifest_path.write_text("[1]", encoding="utf-8")
    with pytest.raises(PITDataError, match="unreadable"):
        _build_scope_manifest(runtime)
