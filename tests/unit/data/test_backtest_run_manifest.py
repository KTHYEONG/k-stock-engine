def test_backtest_run_manifest_round_trip_is_content_addressed(tmp_path) -> None:
    from datetime import date

    from src.data.backtest_run_manifest import (
        build_backtest_run_manifest,
        load_backtest_run_manifest,
        write_backtest_run_manifest,
    )
    from src.data.schemas import SilverTable

    ids = {table: (table.value + '-hash') for table in SilverTable}
    manifest = build_backtest_run_manifest(
        silver_root=tmp_path / 'silver', gold_root=tmp_path / 'gold',
        silver_dataset_ids=ids, gold_dataset_id='gold-hash',
        validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29),
        strategy_id='core-v1', policy_versions={'market_inputs': 'korean-equity-market-inputs-v2'},
    )

    first = write_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / 'artifacts')
    second = write_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / 'artifacts')
    loaded = load_backtest_run_manifest(first)

    assert first == second
    assert first.name == f'{manifest.content_hash}.json'
    assert loaded == manifest
    assert loaded.silver_dataset_ids['daily_market'] == 'daily_market-hash'


def test_backtest_run_manifest_rejects_tampered_hash_and_incomplete_tables(tmp_path) -> None:
    import json
    from datetime import date

    import pytest

    from src.data.backtest_run_manifest import build_backtest_run_manifest, load_backtest_run_manifest
    from src.data.schemas import PITDataError, SilverTable

    incomplete = {SilverTable.CALENDAR: 'calendar-hash'}
    with pytest.raises(PITDataError, match='exactly'):
        build_backtest_run_manifest(
            silver_root=tmp_path / 'silver', gold_root=tmp_path / 'gold', silver_dataset_ids=incomplete,
            gold_dataset_id='gold-hash', validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29),
            strategy_id='core-v1', policy_versions={'market_inputs': 'korean-equity-market-inputs-v2'},
        )

    path = tmp_path / 'tampered.json'
    path.write_text(json.dumps({'schema_version': 'backtest-run-v1', 'content_hash': '0' * 64}), encoding='utf-8')
    with pytest.raises(PITDataError, match='manifest'):
        load_backtest_run_manifest(path)


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

    from src.data.backtest_run_manifest import build_backtest_run_manifest
    from src.data.schemas import PITDataError, SilverTable

    for bad in ("a/b", "a\\b", ".", "..", "   ", "", " padded "):
        ids = {table: (bad if table == SilverTable.CALENDAR else f"{table.value}-hash") for table in SilverTable}
        with pytest.raises(PITDataError, match="dataset"):
            build_backtest_run_manifest(**_manifest_kwargs(tmp_path, silver_dataset_ids=ids))
    with pytest.raises(PITDataError, match="dataset"):
        build_backtest_run_manifest(**_manifest_kwargs(tmp_path, gold_dataset_id="a/b"))
    with pytest.raises(PITDataError, match="dataset"):
        build_backtest_run_manifest(**_manifest_kwargs(tmp_path, gold_dataset_id=123))


def test_backtest_run_manifest_rejects_bad_tables_policies_and_range(tmp_path) -> None:
    from datetime import date

    import pytest

    from src.data.backtest_run_manifest import build_backtest_run_manifest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="exactly"):
        build_backtest_run_manifest(**_manifest_kwargs(tmp_path, silver_dataset_ids={"nope": "x"}))
    with pytest.raises(PITDataError, match="exactly"):
        build_backtest_run_manifest(**_manifest_kwargs(tmp_path, silver_dataset_ids=[("calendar", "x")]))
    with pytest.raises(PITDataError, match="exactly"):
        build_backtest_run_manifest(**_manifest_kwargs(tmp_path, silver_dataset_ids={42: "x"}))
    with pytest.raises(PITDataError, match="polic"):
        build_backtest_run_manifest(**_manifest_kwargs(tmp_path, policy_versions=[("a", "b")]))
    with pytest.raises(PITDataError, match="validation_start"):
        build_backtest_run_manifest(
            **_manifest_kwargs(
                tmp_path, validation_start=date(2016, 12, 29), validation_end=date(2016, 1, 4)
            )
        )
    with pytest.raises(PITDataError, match="strategy_id"):
        build_backtest_run_manifest(**_manifest_kwargs(tmp_path, strategy_id=""))
    for policies, _match in (({}, "policy"), ({"": "v"}, "policy"), ({"k": ""}, "policy")):
        with pytest.raises(PITDataError, match="polic"):
            build_backtest_run_manifest(**_manifest_kwargs(tmp_path, policy_versions=policies))


def test_backtest_run_manifest_load_rejects_corrupt_files(tmp_path) -> None:
    import json

    import pytest

    from src.data.backtest_run_manifest import (
        build_backtest_run_manifest,
        load_backtest_run_manifest,
        write_backtest_run_manifest,
    )
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="manifest"):
        load_backtest_run_manifest(tmp_path / "absent.json")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{nope", encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_backtest_run_manifest(bad_json)
    not_dict = tmp_path / "list.json"
    not_dict.write_text("[1]", encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_backtest_run_manifest(not_dict)
    manifest = build_backtest_run_manifest(**_manifest_kwargs(tmp_path))
    good = write_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / "artifacts")
    raw = json.loads(good.read_text(encoding="utf-8"))
    for field in ("content_hash", "silver_dataset_ids"):
        mutated = dict(raw)
        mutated[field] = 42
        path = tmp_path / f"wrong-{field}.json"
        path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(PITDataError, match="manifest"):
            load_backtest_run_manifest(path)
    mutated = dict(raw)
    mutated["schema_version"] = "other-v9"
    path = tmp_path / "wrong-version.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_backtest_run_manifest(path)
    mutated = dict(raw)
    mutated["extra"] = 1
    path = tmp_path / "extra-field.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_backtest_run_manifest(path)
    mutated = dict(raw)
    mutated["validation_start"] = "not-a-date"
    path = tmp_path / "bad-date.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_backtest_run_manifest(path)
    mutated = dict(raw)
    mutated["content_hash"] = "0" * 64
    path = tmp_path / "hash-mismatch.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        load_backtest_run_manifest(path)


def test_backtest_run_manifest_write_rejects_tampered_existing_bytes(tmp_path) -> None:
    import json

    import pytest

    from src.data.backtest_run_manifest import (
        _manifest_to_json,
        build_backtest_run_manifest,
        write_backtest_run_manifest,
    )
    from src.data.schemas import PITDataError

    manifest = build_backtest_run_manifest(**_manifest_kwargs(tmp_path))
    first = write_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / "artifacts")
    first.write_text("{tampered", encoding="utf-8")
    with pytest.raises(PITDataError, match="manifest"):
        write_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / "artifacts")
    forged = build_backtest_run_manifest(
        **_manifest_kwargs(tmp_path, strategy_id="tampered")
    )
    object.__setattr__(forged, "content_hash", manifest.content_hash)
    object.__setattr__(forged, "schema_version", "other")
    with pytest.raises(PITDataError, match="manifest"):
        write_backtest_run_manifest(manifest=forged, artifact_root=tmp_path / "other-artifacts")
    wrong_hash = build_backtest_run_manifest(**_manifest_kwargs(tmp_path))
    object.__setattr__(wrong_hash, "content_hash", "1" * 64)
    with pytest.raises(PITDataError, match="manifest"):
        write_backtest_run_manifest(manifest=wrong_hash, artifact_root=tmp_path / "other-artifacts")
    different = build_backtest_run_manifest(**_manifest_kwargs(tmp_path, strategy_id="other"))
    first.write_text(json.dumps(_manifest_to_json(different)), encoding="utf-8")
    with pytest.raises(PITDataError, match="differs"):
        write_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / "artifacts")


def test_backtest_run_manifest_treats_lifecycle_events_as_optional(tmp_path) -> None:
    from datetime import date

    import pytest

    from src.data.backtest_run_manifest import build_backtest_run_manifest
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
    manifest = build_backtest_run_manifest(silver_dataset_ids=full, **base)
    assert manifest.silver_dataset_ids[SilverTable.LIFECYCLE_EVENTS.value] == "lifecycle_events-hash"
    without_lifecycle = {
        table: f"{table.value}-hash" for table in SilverTable if table is not SilverTable.LIFECYCLE_EVENTS
    }
    with pytest.raises(PITDataError, match="exactly every SilverTable"):
        build_backtest_run_manifest(silver_dataset_ids=without_lifecycle, **base)


from datetime import date

import pytest

from src.data.backtest_run_manifest import build_backtest_run_manifest
from src.data.schemas import PITDataError, SilverTable

def test_manifest_v2_rejects_missing_lifecycle_dataset(tmp_path):
    ids = {table: 'dataset' for table in SilverTable if table is not SilverTable.LIFECYCLE_EVENTS}
    with pytest.raises(PITDataError, match='exactly every SilverTable'):
        build_backtest_run_manifest(silver_root=tmp_path / 'silver', gold_root=tmp_path / 'gold', silver_dataset_ids=ids, gold_dataset_id='gold', validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29), strategy_id='core-v1', policy_versions={'pit': 'v1'})


def test_manifest_v1_fails_with_rebuild_required(tmp_path):
    import json
    from datetime import date
    import pytest
    from src.data.backtest_run_manifest import build_backtest_run_manifest, load_backtest_run_manifest, write_backtest_run_manifest
    from src.data.schemas import SilverTable
    ids = {table: f"{table.value}-hash" for table in SilverTable}
    manifest = build_backtest_run_manifest(silver_root=tmp_path / "silver", gold_root=tmp_path / "gold", silver_dataset_ids=ids, gold_dataset_id="gold", validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29), strategy_id="core-v1", policy_versions={"pit": "v1"})
    path = write_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / "artifacts")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = "backtest-run-v1"
    v1_path = tmp_path / "v1.json"
    v1_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(Exception, match="rebuild-required"):
        load_backtest_run_manifest(v1_path)


def test_backtest_run_manifest_rejects_compounding_v2_before_fundamental_floor(tmp_path) -> None:
    from datetime import date

    import pytest

    from src.data.backtest_run_manifest import build_backtest_run_manifest
    from src.data.schemas import PITDataError

    # When/Then: QVEF lookback cannot be served before the FY2015 floor.
    with pytest.raises(PITDataError, match="compounding-v2 validation_start"):
        build_backtest_run_manifest(
            **_manifest_kwargs(
                tmp_path, strategy_id="compounding-v2",
                validation_start=date(2016, 5, 15), validation_end=date(2016, 12, 29),
            )
        )

    # And: the floor date itself is accepted.
    accepted = build_backtest_run_manifest(
        **_manifest_kwargs(
            tmp_path, strategy_id="compounding-v2",
            validation_start=date(2016, 5, 16), validation_end=date(2016, 12, 29),
        )
    )
    assert accepted.validation_start == date(2016, 5, 16)

    # And: strategies that do not consume QVEF keep their earlier windows.
    assert build_backtest_run_manifest(**_manifest_kwargs(tmp_path)).strategy_id == "core-v1"
