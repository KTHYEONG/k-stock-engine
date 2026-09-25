from datetime import date
import json






def _scoped_plan_payload() -> dict[str, object]:
    return {
        "plan_id": "plan-scoped",
        "content_hash": "d" * 64,
        "coverage_start": "2026-03-06",
        "coverage_end": "2026-03-07",
        "chunk_size": 1,
        "chunks": [
            {"chunk_id": "c-0001", "symbol": "005930", "sessions": ["2026-03-07", "2026-03-06"]},
            {"chunk_id": "c-0002", "symbol": "000660", "sessions": ["2026-03-06"]},
        ],
    }


def test_load_collection_plan_path_preserves_identity(tmp_path) -> None:
    from src.data.collection_plan import load_collection_plan_path

    receipt = tmp_path / "scoped" / "plan.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(_scoped_plan_payload()), encoding="utf-8")
    plan = load_collection_plan_path(receipt)
    assert plan.plan_id == "plan-scoped"
    assert plan.content_hash == "d" * 64
    assert plan.coverage_start == date(2026, 3, 6)
    assert plan.coverage_end == date(2026, 3, 7)
    assert plan.chunk_size == 1
    assert [chunk.chunk_id for chunk in plan.chunks] == ["c-0001", "c-0002"]
    assert plan.chunks[0].sessions == (date(2026, 3, 7), date(2026, 3, 6))
    assert plan.chunks[1].symbol == "000660"


def test_load_collection_plan_path_rejects_duplicate_chunk_id(tmp_path) -> None:
    import pytest

    from src.data.collection_plan import load_collection_plan_path
    from src.data.schemas import PITDataError

    payload = _scoped_plan_payload()
    raw_chunks = payload["chunks"]
    assert isinstance(raw_chunks, list)
    raw_chunks.append({"chunk_id": "c-0001", "symbol": "005930", "sessions": ["2026-03-06"]})
    receipt = tmp_path / "plan.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PITDataError):
        load_collection_plan_path(receipt)


def test_load_collection_plan_path_rejects_out_of_window_session(tmp_path) -> None:
    import pytest

    from src.data.collection_plan import load_collection_plan_path
    from src.data.schemas import PITDataError

    payload = _scoped_plan_payload()
    payload["chunks"] = [{"chunk_id": "c-0001", "symbol": "005930", "sessions": ["2026-03-08"]}]
    receipt = tmp_path / "plan.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PITDataError):
        load_collection_plan_path(receipt)


def test_load_collection_plan_by_id_matches_path_loader(tmp_path) -> None:
    from src.data.collection_plan import load_collection_plan, load_collection_plan_path

    root = tmp_path / "plans"
    root.mkdir()
    (root / "plan-scoped.json").write_text(json.dumps(_scoped_plan_payload()), encoding="utf-8")
    by_id = load_collection_plan("plan-scoped", artifact_root=root)
    by_path = load_collection_plan_path(root / "plan-scoped.json")
    assert by_id == by_path


def test_load_collection_plan_path_rejects_invalid_receipts(tmp_path) -> None:
    import pytest

    from src.data.collection_plan import load_collection_plan, load_collection_plan_path
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError):
        load_collection_plan_path(tmp_path / "missing.json")
    (tmp_path / "adir").mkdir()
    with pytest.raises(PITDataError):
        load_collection_plan_path(tmp_path / "adir")
    with pytest.raises(PITDataError):
        load_collection_plan_path(None)  # type: ignore[arg-type]
    with pytest.raises(PITDataError):
        load_collection_plan("unknown-id", artifact_root=tmp_path)

    cases: list[object] = [
        "{not json",
        "[1, 2]",
        {**_scoped_plan_payload(), "plan_id": "  "},
        {**_scoped_plan_payload(), "content_hash": ""},
        {**_scoped_plan_payload(), "coverage_start": "bad-date"},
        {**_scoped_plan_payload(), "coverage_start": "2026-03-07", "coverage_end": "2026-03-06"},
        {**_scoped_plan_payload(), "chunk_size": 0},
        {**_scoped_plan_payload(), "chunk_size": True},
        {**_scoped_plan_payload(), "chunk_size": "1"},
        {**_scoped_plan_payload(), "chunks": []},
        {**_scoped_plan_payload(), "chunks": {}},
        {**_scoped_plan_payload(), "chunks": ["nope"]},
        {"plan_id": "p", "content_hash": "h", "coverage_start": "2026-03-06", "coverage_end": "2026-03-06", "chunk_size": 1,
         "chunks": [{"chunk_id": "", "symbol": "005930", "sessions": ["2026-03-06"]}]},
        {"plan_id": "p", "content_hash": "h", "coverage_start": "2026-03-06", "coverage_end": "2026-03-06", "chunk_size": 1,
         "chunks": [{"chunk_id": "c", "symbol": "  ", "sessions": ["2026-03-06"]}]},
        {"plan_id": "p", "content_hash": "h", "coverage_start": "2026-03-06", "coverage_end": "2026-03-06", "chunk_size": 1,
         "chunks": [{"chunk_id": "c", "symbol": "005930", "sessions": []}]},
        {"plan_id": "p", "content_hash": "h", "coverage_start": "2026-03-06", "coverage_end": "2026-03-06", "chunk_size": 1,
         "chunks": [{"chunk_id": "c", "symbol": "005930", "sessions": ["bad"]}]},
    ]
    for index, payload in enumerate(cases):
        receipt = tmp_path / f"bad-{index}.json"
        receipt.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
        with pytest.raises(PITDataError):
            load_collection_plan_path(receipt)
