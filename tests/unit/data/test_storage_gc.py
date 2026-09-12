"""Silver/Gold storage-root retention planning unit tests."""


def test_find_orphaned_staging_paths_detects_legacy_and_current_naming(tmp_path) -> None:
    from src.data.storage_gc import find_orphaned_staging_paths

    base = tmp_path / "gold" / "stocks"
    current = base / ".d8a2c2fbc08af8d2cd0cf5c92e43d075.6ca26589abcd.staging"
    legacy = base / ".staging-d8a2c2fbc08af8d2cd0cf5c92e43d075-6ca26589"
    published = base / "universe" / "070b2df1671b409ea07d3d4f818f037a"
    current.mkdir(parents=True)
    legacy.mkdir(parents=True)
    published.mkdir(parents=True)

    found = find_orphaned_staging_paths(base)

    assert found == tuple(sorted([str(current), str(legacy)]))


def test_find_orphaned_staging_paths_returns_empty_for_missing_base(tmp_path) -> None:
    from src.data.storage_gc import find_orphaned_staging_paths

    assert find_orphaned_staging_paths(tmp_path / "missing") == ()


def test_plan_storage_root_retention_keeps_canonical_and_referenced_roots(tmp_path) -> None:
    import json

    from src.data.storage_gc import plan_storage_root_retention

    silver_base = tmp_path / "silver"
    gold_base = tmp_path / "gold"
    for name in ("stocks", "stocks_provenance_20260911", "stocks_prepared_20260910_v5"):
        (silver_base / name).mkdir(parents=True)
    for name in ("stocks", "stocks_research_annual_provenance_20260911"):
        (gold_base / name).mkdir(parents=True)
    artifact_runs = tmp_path / "artifacts" / "runs"
    artifact_runs.mkdir(parents=True)
    (artifact_runs / "run1.json").write_text(
        json.dumps({
            "silver_root": "data/silver/stocks_provenance_20260911",
            "gold_root": "data/gold/stocks_research_annual_provenance_20260911",
        }),
        encoding="utf-8",
    )

    plan = plan_storage_root_retention(silver_base=silver_base, gold_base=gold_base, artifact_root=tmp_path / "artifacts")

    assert plan.retained_silver_roots == ("stocks", "stocks_provenance_20260911")
    assert plan.reclaimable_silver_roots == ("stocks_prepared_20260910_v5",)
    assert plan.retained_gold_roots == ("stocks", "stocks_research_annual_provenance_20260911")
    assert plan.reclaimable_gold_roots == ()
    assert plan.deletion_eligible is True
    assert plan.blocking_reasons == ()


def test_plan_storage_root_retention_blocks_deletion_on_unreadable_artifact(tmp_path) -> None:
    from src.data.storage_gc import plan_storage_root_retention

    silver_base = tmp_path / "silver"
    (silver_base / "stocks_prepared_20260910_v5").mkdir(parents=True)
    gold_base = tmp_path / "gold"
    gold_base.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True)
    (artifact_root / "broken.json").symlink_to(artifact_root / "missing-target.json")

    plan = plan_storage_root_retention(silver_base=silver_base, gold_base=gold_base, artifact_root=artifact_root)

    assert plan.deletion_eligible is False
    assert any(reason.startswith("unreadable_artifact:") for reason in plan.blocking_reasons)


def test_plan_storage_root_retention_handles_missing_bases_and_no_candidates(tmp_path) -> None:
    from src.data.storage_gc import plan_storage_root_retention

    plan = plan_storage_root_retention(
        silver_base=tmp_path / "missing_silver",
        gold_base=tmp_path / "missing_gold",
        artifact_root=tmp_path / "missing_artifacts",
    )

    assert plan.retained_silver_roots == ()
    assert plan.reclaimable_silver_roots == ()
    assert plan.retained_gold_roots == ()
    assert plan.reclaimable_gold_roots == ()
    assert plan.deletion_eligible is True
    assert plan.blocking_reasons == ()


def test_plan_storage_root_retention_blocks_deletion_on_malformed_artifact_read(tmp_path) -> None:
    from src.data.storage_gc import plan_storage_root_retention

    silver_base = tmp_path / "silver"
    (silver_base / "stocks_prepared_20260910_v5").mkdir(parents=True)
    gold_base = tmp_path / "gold"
    gold_base.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True)
    (artifact_root / "malformed.json").write_bytes(b"\xff")

    plan = plan_storage_root_retention(silver_base=silver_base, gold_base=gold_base, artifact_root=artifact_root, read_size=1)

    assert plan.deletion_eligible is False
    assert any(reason.startswith("unreadable_artifact:") for reason in plan.blocking_reasons)


def test_plan_storage_root_retention_rejects_invalid_read_size(tmp_path) -> None:
    import pytest

    from src.data.storage_gc import plan_storage_root_retention

    with pytest.raises(ValueError, match="read_size"):
        plan_storage_root_retention(
            silver_base=tmp_path / "silver", gold_base=tmp_path / "gold",
            artifact_root=tmp_path / "artifacts", read_size=0,
        )
    with pytest.raises(ValueError, match="read_size"):
        plan_storage_root_retention(
            silver_base=tmp_path / "silver", gold_base=tmp_path / "gold",
            artifact_root=tmp_path / "artifacts", read_size=True,
        )
