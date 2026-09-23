import json
from pathlib import Path

import pytest

from src.data.schemas import PITDataError
from src.data.storage_retention import (
    apply_catalog_revision_retention,
    apply_table_generation_retention,
    discover_generation_tables,
    plan_catalog_revision_retention,
    plan_table_generation_retention,
)


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_catalog(root: Path, *, live: str, others: tuple[str, ...], stray: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in (live, *others):
        (root / f"{name}.json").write_text(json.dumps([{"source": "x", "natural_key": name}]), encoding="utf-8")
    (root / "latest.json").write_text(json.dumps({"revision": f"{live}.json"}), encoding="utf-8")
    if stray:
        (root / "notes.txt").write_text("not a revision", encoding="utf-8")


def _write_generation(table_root: Path, name: str, *, generated_time: str, content_hash: str = "") -> Path:
    generation = table_root / name
    generation.mkdir(parents=True, exist_ok=True)
    (generation / "dataset_manifest.json").write_text(
        json.dumps({"generated_time": generated_time, "content_hash": content_hash or name}), encoding="utf-8"
    )
    (generation / "part.parquet").write_bytes(b"0" * 128)
    return generation


def test_plan_catalog_revision_retention_keeps_pointed_revision_live(tmp_path: Path) -> None:
    a, b, c = _hash("a"), _hash("b"), _hash("c")
    _write_catalog(tmp_path / "catalog", live=b, others=(a, c))
    plan = plan_catalog_revision_retention(tmp_path / "catalog")
    assert plan.live_revision == f"{b}.json"
    assert plan.reclaimable_revisions == tuple(sorted((f"{a}.json", f"{c}.json")))
    assert plan.deletion_eligible is True
    assert plan.reclaimable_bytes > 0


def test_plan_catalog_revision_retention_ignores_stray_files(tmp_path: Path) -> None:
    a, b = _hash("a"), _hash("b")
    _write_catalog(tmp_path / "catalog", live=b, others=(a,), stray=True)
    plan = plan_catalog_revision_retention(tmp_path / "catalog")
    assert f"{a}.json" in plan.reclaimable_revisions
    assert "notes.txt" not in plan.reclaimable_revisions


def test_plan_catalog_revision_retention_blocks_on_missing_pointer(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    (root / f"{_hash('a')}.json").write_text("[]", encoding="utf-8")
    plan = plan_catalog_revision_retention(root)
    assert plan.deletion_eligible is False
    assert plan.reclaimable_revisions == ()
    assert "missing_or_invalid_pointer" in plan.blocking_reasons


def test_plan_catalog_revision_retention_blocks_on_unreadable_live_revision(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    (root / "latest.json").write_text(json.dumps({"revision": "missing.json"}), encoding="utf-8")
    plan = plan_catalog_revision_retention(root)
    assert plan.deletion_eligible is False
    assert "unreadable_live_revision" in plan.blocking_reasons


def test_apply_catalog_revision_retention_deletes_only_reclaimable(tmp_path: Path) -> None:
    a, b, c = _hash("a"), _hash("b"), _hash("c")
    _write_catalog(tmp_path / "catalog", live=b, others=(a, c))
    freed = apply_catalog_revision_retention(tmp_path / "catalog")
    root = tmp_path / "catalog"
    assert not (root / f"{a}.json").exists()
    assert not (root / f"{c}.json").exists()
    assert (root / f"{b}.json").exists()
    assert (root / "latest.json").exists()
    assert freed > 0


def test_apply_catalog_revision_retention_reverifies_against_current_state(tmp_path: Path) -> None:
    a, b, c = _hash("a"), _hash("b"), _hash("c")
    root = tmp_path / "catalog"
    _write_catalog(root, live=b, others=(a,))
    # 계획을 다시 계산하지 않고 외부 상태가 바뀐 뒤 apply를 호출해도, 최신 pointer를 다시 읽어야 한다.
    (root / f"{c}.json").write_text(json.dumps([{"source": "x"}]), encoding="utf-8")
    (root / "latest.json").write_text(json.dumps({"revision": f"{c}.json"}), encoding="utf-8")
    apply_catalog_revision_retention(root)
    assert (root / f"{c}.json").exists()
    assert not (root / f"{a}.json").exists()
    assert not (root / f"{b}.json").exists()


def test_apply_catalog_revision_retention_raises_when_not_eligible(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    with pytest.raises(PITDataError):
        apply_catalog_revision_retention(root)


def test_plan_table_generation_retention_selects_latest_by_generated_time(tmp_path: Path) -> None:
    table_root = tmp_path / "financial_facts"
    _write_generation(table_root, "g1", generated_time="2026-01-01T00:00:00+00:00")
    _write_generation(table_root, "g2", generated_time="2026-03-01T00:00:00+00:00")
    _write_generation(table_root, "g3", generated_time="2026-02-01T00:00:00+00:00")
    plan = plan_table_generation_retention(table_root)
    assert plan.live_generation == "g2"
    assert plan.reclaimable_generations == ("g1", "g3")
    assert plan.deletion_eligible is True
    assert plan.reclaimable_bytes > 0


def test_plan_table_generation_retention_tie_breaks_by_content_hash(tmp_path: Path) -> None:
    table_root = tmp_path / "financial_facts"
    same_time = "2026-01-01T00:00:00+00:00"
    _write_generation(table_root, "low", generated_time=same_time, content_hash="aaa")
    _write_generation(table_root, "high", generated_time=same_time, content_hash="zzz")
    plan = plan_table_generation_retention(table_root)
    assert plan.live_generation == "high"
    assert plan.reclaimable_generations == ("low",)


def test_plan_table_generation_retention_blocks_on_unreadable_manifest(tmp_path: Path) -> None:
    table_root = tmp_path / "financial_facts"
    _write_generation(table_root, "g1", generated_time="2026-01-01T00:00:00+00:00")
    bad = table_root / "g2"
    bad.mkdir()
    (bad / "dataset_manifest.json").write_text("not json", encoding="utf-8")
    plan = plan_table_generation_retention(table_root)
    assert plan.deletion_eligible is False
    assert plan.reclaimable_generations == ()
    assert any(r.startswith("unreadable_manifest:g2") for r in plan.blocking_reasons)


def test_plan_table_generation_retention_blocks_on_naive_generated_time(tmp_path: Path) -> None:
    table_root = tmp_path / "financial_facts"
    generation = table_root / "g1"
    generation.mkdir(parents=True)
    (generation / "dataset_manifest.json").write_text(
        json.dumps({"generated_time": "2026-01-01T00:00:00", "content_hash": "g1"}), encoding="utf-8"
    )
    plan = plan_table_generation_retention(table_root)
    assert plan.deletion_eligible is False


def test_apply_table_generation_retention_deletes_only_reclaimable(tmp_path: Path) -> None:
    table_root = tmp_path / "financial_facts"
    _write_generation(table_root, "g1", generated_time="2026-01-01T00:00:00+00:00")
    _write_generation(table_root, "g2", generated_time="2026-03-01T00:00:00+00:00")
    live_bytes = (table_root / "g2" / "part.parquet").read_bytes()
    freed = apply_table_generation_retention(table_root)
    assert not (table_root / "g1").exists()
    assert (table_root / "g2").exists()
    assert (table_root / "g2" / "part.parquet").read_bytes() == live_bytes
    assert freed > 0


def test_apply_table_generation_retention_raises_when_not_eligible(tmp_path: Path) -> None:
    table_root = tmp_path / "financial_facts"
    bad = table_root / "g1"
    bad.mkdir(parents=True)
    (bad / "dataset_manifest.json").write_text("not json", encoding="utf-8")
    with pytest.raises(PITDataError):
        apply_table_generation_retention(table_root)


def test_discover_generation_tables_finds_only_multi_generation_tables(tmp_path: Path) -> None:
    silver_root = tmp_path / "silver"
    _write_generation(silver_root / "financial_facts", "g1", generated_time="2026-01-01T00:00:00+00:00")
    flat = silver_root / "daily_market_abc123"
    flat.mkdir(parents=True)
    (flat / "manifest.json").write_text(json.dumps({"dataset_id": flat.name}), encoding="utf-8")
    session_dir = flat / "session=2026-03-04"
    session_dir.mkdir()
    (session_dir / "part.parquet").write_bytes(b"0")
    tables = discover_generation_tables(silver_root)
    assert tables == (silver_root / "financial_facts",)


def test_discover_generation_tables_includes_single_generation_table(tmp_path: Path) -> None:
    silver_root = tmp_path / "silver"
    _write_generation(silver_root / "financial_quality", "g1", generated_time="2026-01-01T00:00:00+00:00")
    tables = discover_generation_tables(silver_root)
    assert tables == (silver_root / "financial_quality",)
    plan = plan_table_generation_retention(tables[0])
    assert plan.deletion_eligible is True
    assert plan.reclaimable_generations == ()


def test_apply_catalog_revention_retention_is_idempotent_when_nothing_reclaimable(tmp_path: Path) -> None:
    a = _hash("a")
    _write_catalog(tmp_path / "catalog", live=a, others=())
    assert apply_catalog_revision_retention(tmp_path / "catalog") == 0


def test_apply_table_generation_retention_is_idempotent_when_nothing_reclaimable(tmp_path: Path) -> None:
    table_root = tmp_path / "financial_facts"
    _write_generation(table_root, "g1", generated_time="2026-01-01T00:00:00+00:00")
    assert apply_table_generation_retention(table_root) == 0


def test_discover_generation_tables_handles_missing_silver_root(tmp_path: Path) -> None:
    assert discover_generation_tables(tmp_path / "does-not-exist") == ()


def test_plan_table_generation_retention_blocks_on_non_dict_manifest(tmp_path: Path) -> None:
    table_root = tmp_path / "financial_facts"
    generation = table_root / "g1"
    generation.mkdir(parents=True)
    (generation / "dataset_manifest.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    plan = plan_table_generation_retention(table_root)
    assert plan.deletion_eligible is False
    assert any(r.startswith("unreadable_manifest:g1") for r in plan.blocking_reasons)


def test_plan_table_generation_retention_blocks_on_non_string_generated_time(tmp_path: Path) -> None:
    table_root = tmp_path / "financial_facts"
    generation = table_root / "g1"
    generation.mkdir(parents=True)
    (generation / "dataset_manifest.json").write_text(
        json.dumps({"generated_time": 12345, "content_hash": "g1"}), encoding="utf-8"
    )
    plan = plan_table_generation_retention(table_root)
    assert plan.deletion_eligible is False


def test_plan_table_generation_retention_blocks_on_unparsable_generated_time(tmp_path: Path) -> None:
    table_root = tmp_path / "financial_facts"
    generation = table_root / "g1"
    generation.mkdir(parents=True)
    (generation / "dataset_manifest.json").write_text(
        json.dumps({"generated_time": "not-a-date", "content_hash": "g1"}), encoding="utf-8"
    )
    plan = plan_table_generation_retention(table_root)
    assert plan.deletion_eligible is False


def test_plan_catalog_revision_retention_blocks_on_empty_revision(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    (root / "latest.json").write_text(json.dumps({"revision": ""}), encoding="utf-8")
    plan = plan_catalog_revision_retention(root)
    assert plan.deletion_eligible is False
    assert "missing_or_invalid_pointer" in plan.blocking_reasons


def test_plan_catalog_revision_retention_blocks_on_live_revision_not_a_list(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    a = _hash("a")
    (root / f"{a}.json").write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")
    (root / "latest.json").write_text(json.dumps({"revision": f"{a}.json"}), encoding="utf-8")
    plan = plan_catalog_revision_retention(root)
    assert plan.deletion_eligible is False
    assert "unreadable_live_revision" in plan.blocking_reasons


def test_apply_catalog_revision_retention_skips_a_path_that_fails_to_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, c = _hash("a"), _hash("b"), _hash("c")
    root = tmp_path / "catalog"
    _write_catalog(root, live=b, others=(a, c))
    real_unlink = Path.unlink

    def _flaky_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name == f"{a}.json":
            raise OSError("simulated concurrent deletion")
        real_unlink(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)
    freed = apply_catalog_revision_retention(root)
    assert (root / f"{a}.json").exists()
    assert not (root / f"{c}.json").exists()
    assert freed > 0
