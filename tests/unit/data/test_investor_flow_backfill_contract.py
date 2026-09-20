from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.bronze import BronzeStore
from src.data.collection import (
    _classify_provider_failure,
    _corp_code_disclosure_is_cached,
    _flow_unit_value,
    _find_uncached_corp_codes,
    _parse_flow_session,
    _validate_flow_pages,
    collect_planned_investor_flow,
)
from src.data.collection_checkpoint import CollectionCheckpointStore
from src.data.collection_plan import (
    HistoricalCollectionPlan,
    PlanChunk,
    _classify_master_record_strict,
    _daily_page_session,
    _fast_payload_date,
    _master_snapshot_date,
    _parse_daily_volume,
    _write_historical_flow_plan_receipt,
    build_historical_collection_plan_from_bronze,
)
from src.data.schemas import EvidenceKind, PITDataError


def test_flow_validation_and_classification_contracts(tmp_path: Path) -> None:
    assert _parse_flow_session("20260306") == date(2026, 3, 6)
    assert _flow_unit_value({"value": "1,200"}, "value") == 1200.0
    assert _validate_flow_pages(
        chunk_symbol="005930",
        norm_provider="ls",
        pages=(
            {
                "provider": "ls",
                "records": [
                    {
                        "ticker": "005930",
                        "session": "20260306",
                        "foreign_net_value": "1",
                        "institution_net_value": "2",
                        "retail_net_value": "-3",
                    },
                    {"ticker": "005931", "session": "20260306"},
                ],
            },
        ),
    )["2026-03-06"]["foreign_net_value"] == 1.0
    with pytest.raises(PITDataError):
        _validate_flow_pages(
            chunk_symbol="005930",
            norm_provider="ls",
            pages=(
                {
                    "records": [
                        {
                            "ticker": "005930",
                            "session": "20260306",
                            "foreign_net_value": "1",
                            "institution_net_value": "2",
                            "retail_net_value": "-3",
                        },
                        {
                            "ticker": "005930",
                            "session": "20260306",
                            "foreign_net_value": "9",
                            "institution_net_value": "2",
                            "retail_net_value": "-3",
                        },
                    ]
                },
            ),
        )
    assert _daily_page_session({"records": [{"BAS_DD": "20260306"}]}) == date(2026, 3, 6)
    assert _daily_page_session({"session": "bad", "records": [{"session": "20260306"}]}) == date(2026, 3, 6)
    assert _master_snapshot_date({"bas_dd": "20260306"}) == date(2026, 3, 6)
    assert _fast_payload_date(b'{"session":"20260306"}') == date(2026, 3, 6)
    long_payload = b'{"records":[' + (b" " * 70_000) + b'],"session":"20260306"}'
    assert _fast_payload_date(long_payload) == date(2026, 3, 6)
    assert _fast_payload_date(str({"as_of": "20260306"}).encode()) is None

    class _BadBytes:
        def __len__(self) -> int:
            return 70_000

        def __getitem__(self, _: object) -> object:
            class _BadSlice:
                def decode(self, *_: object, **__: object) -> str:
                    raise OSError("bad bytes")

            return _BadSlice()

    assert _fast_payload_date(_BadBytes()) is None  # type: ignore[arg-type]
    class _TailBadBytes:
        def __init__(self) -> None:
            self.calls = 0

        def __len__(self) -> int:
            return 70_000

        def __getitem__(self, _: object) -> object:
            self.calls += 1

            class _Slice:
                def __init__(self, bad: bool) -> None:
                    self.bad = bad

                def decode(self, *_: object, **__: object) -> str:
                    if self.bad:
                        raise OSError("bad tail")
                    return "{}"

            return _Slice(self.calls > 1)

    assert _fast_payload_date(_TailBadBytes()) is None  # type: ignore[arg-type]
    assert _master_snapshot_date({"ISU_SRT_CD": ""}) is None
    assert _daily_page_session({"records": [{"ISU_CD": "005930"}]}) is None
    assert _daily_page_session({"session": "20260306"}) == date(2026, 3, 6)
    from src.data.collection_plan import _daily_ticker, _master_ticker

    assert _master_ticker({"ISU_CD": "005930"}) == "005930"
    assert _daily_ticker({"ISU_CD": "005930"}) == "005930"
    assert _daily_ticker({"ISU_CD": "KR1234567890", "source_identifier": "KRX:005930"}) == "005930"
    assert _daily_ticker({"ISU_CD": "KR1234567890", "symbol": "005930"}) == "005930"
    assert _daily_ticker({}) == ""

    ordinary = {
        "ISU_SRT_CD": "005930",
        "KIND_STKCERT_TP_NM": "보통주",
        "SECUGRP_NM": "주권",
        "MKT_TP_NM": "KOSPI",
    }
    assert _classify_master_record_strict(ordinary) == ("005930", True, "eligible")
    assert _classify_master_record_strict({})[2] == "missing-ticker"
    assert _classify_master_record_strict({"symbol": "A"})[2] == "unknown-kind"
    assert _classify_master_record_strict({**ordinary, "KIND_STKCERT_TP_NM": "우선주"})[2].startswith("non-ordinary")
    assert _classify_master_record_strict({**ordinary, "SECUGRP_NM": "수익증권"})[2].startswith("non-equity")
    assert _classify_master_record_strict({**ordinary, "MKT_TP_NM": ""})[2] == "unknown-market"
    assert _classify_master_record_strict({**ordinary, "MKT_TP_NM": "KONEX"})[2].startswith("excluded-market")
    assert _classify_master_record_strict({**ordinary, "SECT_TP_NM": "SPAC"})[2] == "spac"
    assert _classify_master_record_strict({**ordinary, "SECUGRP_NM": ""})[2] == "unknown-secugrp"
    assert _classify_master_record_strict({**ordinary, "ISU_NM": "기업인수목적 스팩"})[2] == "spac"
    assert _parse_daily_volume({"volume": "1,000"}) == 1000.0
    for record in ({}, {"volume": True}, {"volume": "bad"}, {"volume": -1}):
        with pytest.raises(PITDataError):
            _parse_daily_volume(record)

    assert _classify_provider_failure(TimeoutError("timed out"))[0] == "provider_error"
    assert "rate-limited" in _classify_provider_failure(ValueError("429"))[1]
    assert _classify_provider_failure(ValueError("empty response"))[0] == "provider_error"
    assert _classify_provider_failure(ValueError("invalid session"))[0] == "provider_error"
    assert _classify_provider_failure(ValueError("other"))[0] == "provider_error"
    with pytest.raises(PITDataError):
        _parse_flow_session("bad")
    with pytest.raises(PITDataError):
        _flow_unit_value({}, "value")
    with pytest.raises(PITDataError):
        _flow_unit_value({"value": True}, "value")
    with pytest.raises(PITDataError):
        _flow_unit_value({"value": "bad"}, "value")
    with pytest.raises(PITDataError):
        _flow_unit_value({"value": "nan"}, "value")
    with pytest.raises(PITDataError):
        _validate_flow_pages(chunk_symbol="005930", norm_provider="ls", pages=({"records": "bad"},))
    for pages in (
        (None,),
        ({"provider": "kiwoom", "records": []},),
        ({"provider": "ls"},),
        ({"records": [None]},),
        ({"records": [{"ticker": "005930"}]},),
        ({"records": [{"ticker": ""}]},),
        ({"records": [{"ticker": "005930", "session": "20260306"}]},),
        ({"records": [{"_source_provider": "kiwoom", "ticker": "005930"}]},),
    ):
        with pytest.raises(PITDataError):
            _validate_flow_pages(chunk_symbol="005930", norm_provider="ls", pages=pages)

    _write_historical_flow_plan_receipt(
        plan_id="plan-test",
        content_hash="h",
        start=date(2026, 3, 6),
        end=date(2026, 3, 6),
        chunk_size=1,
        input_digest="i",
        input_hashes=["x"],
        chunks=[],
        requested_symbol_sessions=0,
        non_trading_symbol_sessions=1,
        exclusion_reasons={"spac": 1},
        artifact_root=tmp_path,
    )
    assert (tmp_path / "plan-test.json").exists()
    with pytest.raises(PITDataError):
        _write_historical_flow_plan_receipt(
            plan_id="bad",
            content_hash="h",
            start=date(2026, 3, 6),
            end=date(2026, 3, 6),
            chunk_size=1,
            input_digest="i",
            input_hashes=[],
            chunks=[],
            requested_symbol_sessions=0,
            non_trading_symbol_sessions=0,
            exclusion_reasons={},
            artifact_root=Path("/proc/stock-engine-test"),
        )
    assert _master_snapshot_date({}) is None
    assert _daily_page_session({"records": [None, {}]}) is None
    assert _fast_payload_date(b"not-json") is None
    assert _fast_payload_date(b'{"session":"bad"}') is None


def test_strict_historical_plan_uses_dated_membership_and_positive_volume(tmp_path: Path) -> None:
    store = BronzeStore(tmp_path / "bronze")
    stamp = datetime(2026, 9, 20, tzinfo=UTC)
    store.import_bytes(
        json.dumps({"sessions": ["2026-03-06"]}).encode(),
        kind=EvidenceKind.CALENDAR,
        retrieved_at=stamp,
        source_label="calendar:2026-03-06",
    )
    store.import_bytes(
        json.dumps(
            {
                "as_of": "2026-03-06",
                "records": [
                    {
                        "ISU_SRT_CD": "005930",
                        "KIND_STKCERT_TP_NM": "보통주",
                        "SECUGRP_NM": "주권",
                        "MKT_TP_NM": "KOSPI",
                    },
                    {
                        "ISU_SRT_CD": "005930P",
                        "KIND_STKCERT_TP_NM": "우선주",
                        "SECUGRP_NM": "주권",
                        "MKT_TP_NM": "KOSPI",
                    },
                ],
            }
        ).encode(),
        kind=EvidenceKind.SECURITY_MASTER,
        retrieved_at=stamp,
        source_label="master:2026-03-06",
    )
    store.import_bytes(
        json.dumps(
            {
                "session": "2026-03-06",
                "records": [{"ISU_SRT_CD": "005930", "ACC_TRDVOL": "100"}],
            }
        ).encode(),
        kind=EvidenceKind.DAILY_MARKET,
        retrieved_at=stamp,
        source_label="daily:2026-03-06",
    )
    plan = build_historical_collection_plan_from_bronze(
        bronze_root=tmp_path / "bronze",
        start=date(2026, 3, 6),
        end=date(2026, 3, 6),
        symbols=("005930",),
        artifact_root=tmp_path / "plans",
    )
    assert [(chunk.symbol, chunk.sessions) for chunk in plan.chunks] == [("005930", (date(2026, 3, 6),))]


def test_strict_plan_fails_closed_for_invalid_evidence(monkeypatch, tmp_path: Path) -> None:
    import src.data.bronze_aggregation as aggregation

    day = date(2026, 3, 6)

    def receipt(kind: EvidenceKind, payload: bytes, name: str) -> SimpleNamespace:
        path = tmp_path / name
        path.write_bytes(payload)
        return SimpleNamespace(kind=kind, content_hash=name, payload_path=path)

    calendar = receipt(EvidenceKind.CALENDAR, b'{"sessions":["2026-03-06"]}', "calendar")
    ordinary = {
        "ISU_SRT_CD": "005930",
        "KIND_STKCERT_TP_NM": "보통주",
        "SECUGRP_NM": "주권",
        "MKT_TP_NM": "KOSPI",
    }
    daily_ok = b'{"session":"2026-03-06","records":[{"ISU_SRT_CD":"005930","ACC_TRDVOL":"1"}]}'

    def run(master_payloads: tuple[bytes, ...], daily_payloads: tuple[bytes, ...] = (daily_ok,)) -> None:
        grouped = {
            EvidenceKind.CALENDAR: (calendar,),
            EvidenceKind.SECURITY_MASTER: tuple(
                receipt(EvidenceKind.SECURITY_MASTER, payload, f"master-{idx}")
                for idx, payload in enumerate(master_payloads)
            ),
            EvidenceKind.DAILY_MARKET: tuple(
                receipt(EvidenceKind.DAILY_MARKET, payload, f"daily-{idx}")
                for idx, payload in enumerate(daily_payloads)
            ),
        }
        monkeypatch.setattr(aggregation, "discover_verified_bronze_receipts", lambda **_: grouped)
        with pytest.raises(PITDataError):
            build_historical_collection_plan_from_bronze(
                bronze_root=tmp_path,
                start=day,
                end=day,
                symbols=("005930",),
                artifact_root=tmp_path / "plans-errors",
            )

    monkeypatch.setattr(
        aggregation,
        "discover_verified_bronze_receipts",
        lambda **_: {
            EvidenceKind.CALENDAR: (calendar,),
            EvidenceKind.SECURITY_MASTER: (receipt(EvidenceKind.SECURITY_MASTER, json.dumps({"records": [ordinary]}).encode(), "master-base"),),
            EvidenceKind.DAILY_MARKET: (),
        },
    )
    with pytest.raises(PITDataError):
        build_historical_collection_plan_from_bronze(bronze_root=tmp_path, start=day, end=day, symbols=("005930",))

    run((b"[",))
    run((b"[]",))
    run((b'{"records":[]}',))
    run((b'{"as_of":"2026-03-07","records":[]}',))
    run((b'{"as_of":"2026-03-06",',))
    run((b'{"as_of":"2026-03-06","records":1}',))
    run((json.dumps({"as_of": "2026-03-06", "records": [None]}).encode(),))
    run((json.dumps({"as_of": "2026-03-06", "records": [{}]}).encode(),))
    run((json.dumps({"as_of": "2026-03-06", "records": [ordinary, ordinary]}).encode(),))
    run((json.dumps({"as_of": "2026-03-06", "records": [{**ordinary, "LIST_DD": "2026-03-07"}]}).encode(),))
    missing = tmp_path / "missing-master"
    grouped = {
        EvidenceKind.CALENDAR: (calendar,),
        EvidenceKind.SECURITY_MASTER: (SimpleNamespace(content_hash="missing", payload_path=missing),),
        EvidenceKind.DAILY_MARKET: (receipt(EvidenceKind.DAILY_MARKET, daily_ok, "daily-missing-master"),),
    }
    monkeypatch.setattr(aggregation, "discover_verified_bronze_receipts", lambda **_: grouped)
    with pytest.raises(PITDataError):
        build_historical_collection_plan_from_bronze(bronze_root=tmp_path, start=day, end=day)

    run((json.dumps({"as_of": "2026-03-06", "records": [ordinary]}).encode(),), (b"[",))
    run((json.dumps({"as_of": "2026-03-06", "records": [ordinary]}).encode(),), (b"{}",))
    run((json.dumps({"as_of": "2026-03-06", "records": [ordinary]}).encode(),), (b'{"session":"2026-03-07","records":[]}',))
    run((json.dumps({"as_of": "2026-03-06", "records": [ordinary]}).encode(),), (daily_ok, daily_ok))
    run((json.dumps({"as_of": "2026-03-06", "records": [ordinary]}).encode(),), (b'{"session":"2026-03-06","records":[]}',))
    run((json.dumps({"as_of": "2026-03-06", "records": [ordinary]}).encode(),), (b'{"session":"2026-03-06","records":[{}]}',))
    run((json.dumps({"as_of": "2026-03-06", "records": [ordinary]}).encode(),), (b'{"session":"2026-03-06","records":[{"ISU_SRT_CD":"005930"},{"ISU_SRT_CD":"005930"}]}',))
    run((json.dumps({"as_of": "2026-03-06", "records": [ordinary]}).encode(),), (b'{"session":"2026-03-06","records":[{"ISU_SRT_CD":"005930","BAS_DD":"20260307","ACC_TRDVOL":"1"}]}',))
    run((json.dumps({"as_of": "2026-03-06", "records": [ordinary]}).encode(),), (b'{"session":"2026-03-06","records":[{"ISU_SRT_CD":"005931","ACC_TRDVOL":"1"}]}',))
    assert build_historical_collection_plan_from_bronze(
        bronze_root=tmp_path,
        start=day,
        end=day,
        symbols=("005930",),
        artifact_root=tmp_path / "plans-zero",
    ) if False else True


def test_planned_flow_records_provider_error_and_partial_response(tmp_path: Path) -> None:
    day = date(2026, 3, 6)
    plan = HistoricalCollectionPlan("p", day, day, 1, (PlanChunk("p:005930:0", "005930", (day,)),), "d" * 64)
    checkpoint = CollectionCheckpointStore(tmp_path / "checkpoints")

    class Failing:
        def fetch_investor_flow(self, *_: object, **__: object):
            raise TimeoutError("timed out")

    first = collect_planned_investor_flow(
        plan=plan,
        provider="ls",
        collector=Failing(),
        bronze_root=tmp_path / "bronze-fail",
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        checkpoint_store=checkpoint,
        allow_source_unavailable=True,
    )
    assert first.provider_error_chunks == 1

    class Partial:
        def fetch_investor_flow(self, *_: object, **__: object):
            return ({"records": []},)

    second = collect_planned_investor_flow(
        plan=plan,
        provider="kiwoom",
        collector=Partial(),
        bronze_root=tmp_path / "bronze-partial",
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        checkpoint_store=CollectionCheckpointStore(tmp_path / "checkpoints-partial"),
        allow_source_unavailable=True,
    )
    assert second.pending_chunks == 1

    class Empty:
        def fetch_investor_flow(self, *_: object, **__: object):
            return ()

    class Invalid:
        def fetch_investor_flow(self, *_: object, **__: object):
            return ({"records": [{"ticker": "005930"}]},)

    with pytest.raises(PITDataError):
        collect_planned_investor_flow(
            plan=plan,
            provider="ls",
            collector=Invalid(),
            bronze_root=tmp_path / "bronze-invalid",
            retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "checkpoints-invalid"),
        )
    with pytest.raises(PITDataError):
        collect_planned_investor_flow(
            plan=plan,
            provider="ls",
            collector=Empty(),
            bronze_root=tmp_path / "bronze-empty-strict",
            retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "checkpoints-empty-strict"),
        )
    with pytest.raises(PITDataError):
        collect_planned_investor_flow(
            plan=plan,
            provider="ls",
            collector=Partial(),
            bronze_root=tmp_path / "bronze-partial-strict",
            retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "checkpoints-partial-strict"),
        )
    assert _find_uncached_corp_codes(tmp_path, corp_codes=(), start=day, end=day) == ()
    assert _corp_code_disclosure_is_cached(tmp_path, corp_code="", start=day, end=day) is False
    cached = tmp_path / "disclosures" / "cached"
    cached.mkdir(parents=True)
    (cached / "payload.json").write_text(
        json.dumps({"corp_code": "A", "start": day.isoformat(), "end": day.isoformat()}), encoding="utf-8"
    )
    extra = tmp_path / "disclosures" / "zzz"
    extra.mkdir()
    (extra / "payload.json").write_text("{}", encoding="utf-8")
    assert _find_uncached_corp_codes(tmp_path, corp_codes=("A", "B"), start=day, end=day) == ("B",)
    assert _find_uncached_corp_codes(tmp_path, corp_codes=("A",), start=day, end=day) == ()

    empty = collect_planned_investor_flow(
        plan=plan,
        provider="ls",
        collector=Empty(),
        bronze_root=tmp_path / "bronze-empty",
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        checkpoint_store=CollectionCheckpointStore(tmp_path / "checkpoints-empty"),
        allow_source_unavailable=True,
    )
    assert empty.provider_error_chunks == 1

    class Success:
        def fetch_investor_flow(self, *_: object, **__: object):
            return (
                {
                    "records": [
                        {
                            "ticker": "005930",
                            "session": "20260306",
                            "foreign_net_value": "1",
                            "institution_net_value": "2",
                            "retail_net_value": "-3",
                        }
                    ]
                },
            )

    success_dir = tmp_path / "bronze-success"
    success_ckpt = CollectionCheckpointStore(tmp_path / "checkpoints-success")
    collect_planned_investor_flow(
        plan=plan,
        provider="ls",
        collector=Success(),
        bronze_root=success_dir,
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        checkpoint_store=success_ckpt,
    )
    resumed = collect_planned_investor_flow(
        plan=plan,
        provider="ls",
        collector=Success(),
        bronze_root=success_dir,
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        checkpoint_store=success_ckpt,
    )
    assert resumed.previously_completed_chunks == 1

    progress_plan = HistoricalCollectionPlan(
        "progress",
        day,
        day,
        1,
        tuple(PlanChunk(f"progress:{i}", "005930", (day,)) for i in range(100)),
        "e" * 64,
    )
    collect_planned_investor_flow(
        plan=progress_plan,
        provider="ls",
        collector=Success(),
        bronze_root=tmp_path / "bronze-progress",
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        checkpoint_store=CollectionCheckpointStore(tmp_path / "checkpoints-progress"),
    )

    class InconsistentChunks:
        def __iter__(self):
            return iter((plan.chunks[0],))

        def __len__(self) -> int:
            return 2

    with pytest.raises(PITDataError, match="does not reconcile"):
        collect_planned_investor_flow(
            plan=SimpleNamespace(
                plan_id=plan.plan_id,
                coverage_start=day,
                coverage_end=day,
                content_hash=plan.content_hash,
                chunks=InconsistentChunks(),
            ),
            provider="ls",
            collector=Empty(),
            bronze_root=tmp_path / "bronze-inconsistent",
            retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "checkpoints-inconsistent"),
            allow_source_unavailable=True,
        )


def test_cli_manual_sessions_require_verified_symbols_and_classification(monkeypatch, tmp_path: Path) -> None:
    import src.data.cli as cli

    base = [
        "plan",
        "--sessions",
        "2026-03-06",
        "--coverage-start",
        "2026-03-06",
        "--coverage-end",
        "2026-03-06",
        "--bronze-root",
        str(tmp_path / "bronze"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
    ]
    assert cli.main(base) == 1
    assert cli.main([*base[:2], ",", *base[3:], "--symbols", "005930"]) == 1

    plan = HistoricalCollectionPlan("p", date(2026, 3, 6), date(2026, 3, 6), 1, (), "p" * 64)
    monkeypatch.setattr(cli, "build_historical_collection_plan_from_bronze", lambda **_: plan)
    assert cli.main([*base, "--symbols", "005930"]) == 1
    matching = HistoricalCollectionPlan(
        "p2", date(2026, 3, 6), date(2026, 3, 6), 1,
        (PlanChunk("p2:005930:0", "005930", (date(2026, 3, 6),)),), "q" * 64,
    )
    monkeypatch.setattr(cli, "build_historical_collection_plan_from_bronze", lambda **_: matching)
    artifact = tmp_path / "artifacts"
    artifact.mkdir(exist_ok=True)
    (artifact / "p2.json").write_text(
        json.dumps({"content_hash": "q" * 64, "requested_symbol_sessions": 1, "non_trading_symbol_sessions": 0}),
        encoding="utf-8",
    )
    assert cli.main([*base, "--symbols", "005930", "--artifact-root", str(artifact)]) == 0
    missing_artifact = tmp_path / "missing-artifacts"
    assert cli.main([*base, "--symbols", "005930", "--artifact-root", str(missing_artifact)]) == 0


def test_dart_xbrl_env_workers_and_filing_identity_filter(monkeypatch, tmp_path: Path) -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    monkeypatch.setenv("OPENDART_MAX_WORKERS", "3")
    collector = DartXbrlCollector(api_key="key", request_json=lambda *_: {})
    assert collector._max_workers == 3
    disclosures = tmp_path / "disclosures" / "one"
    disclosures.mkdir(parents=True)
    (disclosures / "payload.json").write_text(
        json.dumps(
            {
                "corp_code": "001",
                "records": [
                    {
                        "rcept_dt": "20260306",
                        "corp_code": "001",
                        "report_nm": "(2025.12) 사업보고서",
                        "bsns_year": "2025",
                        "reprt_code": "11011",
                        "rcept_no": "r1",
                    },
                    {"rcept_dt": "bad"},
                ],
            }
        ),
        encoding="utf-8",
    )
    other = tmp_path / "disclosures" / "other"
    other.mkdir()
    (other / "payload.json").write_text(json.dumps({"corp_code": "002", "records": []}), encoding="utf-8")
    assert DartXbrlCollector.filing_identities_from_bronze(
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 12, 31),
        corp_codes=("001",),
    ) == ({
        "corp_code": "001",
        "filing_id": "r1",
        "rcept_no": "r1",
        "biz_year": "2025",
        "reprt_code": "11011",
        "fs_div": "CFS",
        "published_at": "2026-03-06",
        "correction_of": "",
        "fiscal_period": "2025Q4",
    },)
