"""Exchange market-action classification and publication tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.core.pit import PITDataError
from src.core.time import KRX_TZ, SessionCalendar
from src.data.datasets import load_manifest, read_dataset, verify_dataset
from src.data.market_actions import MarketActionKind, classify_market_action_title, materialize_market_actions

from tests.fixtures.kind_html import kind_row_html, kind_search_html

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
RETRIEVED_AT = datetime(2024, 6, 1, tzinfo=UTC)
CORP = "00126380"
TICKER = "005930"


def _runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _catalog(runtime):  # type: ignore[no-untyped-def]
    from src.data.receipt_catalog import ReceiptCatalog

    return ReceiptCatalog(runtime.workspace.bronze_root / "catalog")


def _publish_window(runtime, *, key: str, records: list) -> None:  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.bronze import BronzeStore
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    detail, window = key.split(":", 1)
    start, end = window.split("..")
    body = {"detail_type": detail, "start": start, "end": end, "records": records}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    store = BronzeStore(runtime.workspace.bronze_root)
    receipt = store.import_bytes(raw, kind=EvidenceKind.DISCLOSURES, retrieved_at=RETRIEVED_AT, source_label="test")
    catalog = _catalog(runtime)
    catalog.publish(
        [
            ReceiptIndexEntry(
                source="dart_disclosure_windows", natural_key=key, as_of=date.fromisoformat(end),
                fiscal_period=None, status=EvidenceStatus.SUCCESS if records else EvidenceStatus.EMPTY,
                content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at,
                payload_path=receipt.payload_path,
            )
        ],
        blobs=[
            BlobEntry(
                content_hash=receipt.content_hash, kind=EvidenceKind.DISCLOSURES,
                source="dart_disclosure_windows", usable=True, unusable_reason=None,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path,
            )
        ],
    )


def _publish_daily(runtime, *, session: date, records: list) -> None:  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.bronze import BronzeStore
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    raw = json.dumps({"session": session.isoformat(), "records": records}, sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )
    store = BronzeStore(runtime.workspace.bronze_root)
    receipt = store.import_bytes(raw, kind=EvidenceKind.DAILY_MARKET, retrieved_at=RETRIEVED_AT, source_label="test")
    catalog = _catalog(runtime)
    catalog.publish(
        [
            ReceiptIndexEntry(
                source="krx_daily_market", natural_key=session.isoformat(), as_of=session,
                fiscal_period=None, status=EvidenceStatus.SUCCESS,
                content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at,
                payload_path=receipt.payload_path,
            )
        ],
        blobs=[
            BlobEntry(
                content_hash=receipt.content_hash, kind=EvidenceKind.DAILY_MARKET,
                source="krx_daily_market", usable=True, unusable_reason=None,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path,
            )
        ],
    )


def _calendar(*days: date) -> SessionCalendar:
    return SessionCalendar(
        tuple(datetime(day.year, day.month, day.day, 9, 0, tzinfo=KRX_TZ) for day in days)
    )


def _record(rcept_no: str, *, corp: str = CORP, rcept_dt: str = "20240102", report_nm: str) -> dict:
    return {"corp_code": corp, "rcept_no": rcept_no, "rcept_dt": rcept_dt, "report_nm": report_nm, "rm": ""}


def _bridge() -> dict[str, str]:
    return {CORP: TICKER}


def test_classify_market_action_title_vocabulary() -> None:
    assert classify_market_action_title("상장폐지결정") is MarketActionKind.DELISTING_DECIDED
    assert classify_market_action_title("[기재정정]상장폐지결정") is MarketActionKind.DELISTING_DECIDED
    assert classify_market_action_title("상장폐지에 따른 정리매매개시") is MarketActionKind.LIQUIDATION_TRADING
    assert classify_market_action_title("관리종목 지정") is MarketActionKind.ADMINISTRATIVE_DESIGNATED
    assert classify_market_action_title("투자주의환기종목 지정") is MarketActionKind.ADMINISTRATIVE_DESIGNATED
    assert classify_market_action_title("관리종목 지정해제") is MarketActionKind.ADMINISTRATIVE_RELEASED
    assert classify_market_action_title("매매거래정지") is MarketActionKind.TRADING_HALTED
    assert classify_market_action_title("매매거래정지해제") is MarketActionKind.TRADING_RESUMED
    assert classify_market_action_title("상장적격성 실질심사 사유발생") is None
    assert classify_market_action_title("상장폐지 관련 조회공시 답변") is None
    assert classify_market_action_title("현금배당 결정") is None
    assert classify_market_action_title("상장폐지결정 공시번복") is None
    assert classify_market_action_title("상장폐지결정취소") is None
    assert classify_market_action_title("관리종목 지정 공시번복") is None
    assert classify_market_action_title("단기과열 완화를 위한 거래재개") is MarketActionKind.TRADING_RESUMED
    assert classify_market_action_title("") is None


def test_materialize_availability_is_next_session_open(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _publish_window(
        runtime, key="I003:2024-01-01..2024-01-31",
        records=[
            _record("20240102000001", rcept_dt="20240102", report_nm="상장폐지결정"),
            _record("20240102000002", rcept_dt="20240102", report_nm="현금배당 결정"),
        ],
    )
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
        bridge=_bridge(),
        kind_keywords=(), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    assert verify_dataset(path, known_ids=lambda _dataset_id: True).passed
    frame = read_dataset(path).collect().sort("rcept_no")
    assert frame.height == 1
    row = frame.to_dicts()[0]
    assert row["instrument_id"] == f"KRX:{TICKER}"
    assert row["kind"] == MarketActionKind.DELISTING_DECIDED.value
    assert row["announced_on"] == date(2024, 1, 2)
    assert row["available_at"] == datetime(2024, 1, 3, 9, 0, tzinfo=KRX_TZ)
    assert row["cancellation"] is False
    assert row["effective_start"] is None
    manifest = load_manifest(path)
    assert manifest.kind == "market_actions"
    assert manifest.details["actions"] == 1
    assert manifest.details["unmapped_rows"] == 0


def test_materialize_rebuild_is_idempotent(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _publish_window(
        runtime, key="I003:2024-01-01..2024-01-31",
        records=[_record("20240102000001", rcept_dt="20240102", report_nm="정리매매개시")],
    )
    kwargs = {
        "catalog": _catalog(runtime),
        "silver_root": runtime.workspace.silver_root,
        "calendar": _calendar(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
        "bridge": _bridge(),
        "kind_keywords": (),
        "kind_max_stale_sessions": 5, "kind_coverage_start": date(2024, 1, 1),
    }
    first = materialize_market_actions(**kwargs)
    before = (first / "manifest.json").read_bytes()
    second = materialize_market_actions(**kwargs)
    assert first == second
    assert (first / "manifest.json").read_bytes() == before


def test_withdrawn_delisting_cancels_panel_block(tmp_path: Path) -> None:
    from src.core.market_rules import load_krx_market_rules
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.market_panel import materialize_market_panel

    runtime = _runtime(tmp_path)
    _publish_window(
        runtime, key="I003:2024-01-01..2024-01-31",
        records=[
            _record("20240102000001", rcept_dt="20240102", report_nm="상장폐지결정"),
            _record("20240103000002", rcept_dt="20240103", report_nm="상장폐지결정 공시번복"),
        ],
    )
    sessions = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    actions = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(*sessions, date(2024, 1, 8)),
        bridge=_bridge(),
        kind_keywords=(), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 1),
    )
    frame = read_dataset(actions).collect()
    assert frame.height == 2
    assert frame.filter(pl.col("cancellation"))["rcept_no"].to_list() == ["20240103000002"]

    def _drow(session: date) -> dict:
        return {
            "session": session, "instrument_id": f"KRX:{TICKER}", "ticker": TICKER, "market": "KOSPI",
            "open": 10000, "high": 10100, "low": 9900, "close": 10000, "change": 0,
            "base_price": 10000, "volume": 1000, "trading_value": 10000000,
            "market_cap": 100000000, "listed_shares": 10000, "price_state": "tradable",
            "invalid_reason": None,
            "available_at": datetime(session.year, session.month, session.day, 18, 0, tzinfo=KRX_TZ),
            "source_hash": "a" * 64, "policy_version": "krx-daily-market-v1",
        }

    silver = tmp_path / "panel-silver"
    daily_partitions = {
        f"session={day.isoformat()}/part.parquet": pl.DataFrame([_drow(day)]) for day in sessions
    }
    daily_path = publish_dataset(
        layer_root=silver,
        identity=DatasetIdentity(kind="daily_market", layer=DatasetLayer.SILVER, policy_version="t", inputs={}, params={}),
        partitions=daily_partitions,
    ).path
    universe_partitions = {
        f"session={day.isoformat()}/part.parquet": pl.DataFrame(
            [{"instrument_id": f"KRX:{TICKER}", "ticker": TICKER, "eligible": True, "exclusion_reason": "eligible"}]
        )
        for day in sessions
    }
    universe_path = publish_dataset(
        layer_root=silver,
        identity=DatasetIdentity(kind="ordinary_universe", layer=DatasetLayer.SILVER, policy_version="t", inputs={}, params={}),
        partitions=universe_partitions,
    ).path
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path,
        rules=load_krx_market_rules(Path("config/market/krx_market_rules.toml")),
        gold_root=tmp_path / "gold", market_actions_path=actions,
    )
    panel = pl.read_parquet(result.dataset_path / "year=2024" / "part.parquet").sort("session")
    assert panel["session"].to_list() == sessions
    assert panel["entry_blocked"].to_list() == [False, True, False, False]
    assert panel["entry_block_reason"].to_list() == ["", "delisting_decided", "", ""]
    assert panel["eligible"].to_list() == [True, True, True, True]


def test_administrative_flag_reads_the_short_code_of_real_daily_pages(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    # 실제 일별 시세 페이지는 종목코드를 6자리 ISU_CD로만 싣는다.
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_CD": "000660", "SECT_TP_NM": ""}])
    _publish_daily(runtime, session=date(2024, 1, 3), records=[{"ISU_CD": "000660", "SECT_TP_NM": "관리종목(소속부없음)"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
        bridge=_bridge(),
        kind_keywords=(), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    frame = read_dataset(path).collect()
    assert frame["kind"].to_list() == [MarketActionKind.ADMINISTRATIVE_DESIGNATED.value]
    assert frame["instrument_id"].to_list() == ["KRX:000660"]


def test_kosdaq_administrative_flag_transitions(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    flagged = {"ISU_SRT_CD": "000660", "SECT_TP_NM": "관리종목(소속부없음)"}
    plain = {"ISU_SRT_CD": "000660", "SECT_TP_NM": ""}
    other = {"ISU_SRT_CD": "005930", "SECT_TP_NM": ""}
    untracked = {"SECT_TP_NM": "관리종목(소속부없음)"}
    _publish_daily(runtime, session=date(2024, 1, 2), records=[plain, other])
    _publish_daily(runtime, session=date(2024, 1, 3), records=[flagged, other, untracked])
    _publish_daily(runtime, session=date(2024, 1, 4), records=[flagged, other])
    _publish_daily(runtime, session=date(2024, 1, 5), records=[plain, other])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(
            date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)
        ),
        bridge=_bridge(),
        kind_keywords=(), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 1),
    )
    frame = read_dataset(path).collect().sort("announced_on")
    assert frame.height == 2
    assert frame["kind"].to_list() == [
        MarketActionKind.ADMINISTRATIVE_DESIGNATED.value,
        MarketActionKind.ADMINISTRATIVE_RELEASED.value,
    ]
    assert frame["instrument_id"].to_list() == ["KRX:000660", "KRX:000660"]
    assert frame["announced_on"].to_list() == [date(2024, 1, 3), date(2024, 1, 5)]
    assert frame["available_at"].to_list() == [
        datetime(2024, 1, 4, 9, 0, tzinfo=KRX_TZ),
        datetime(2024, 1, 8, 9, 0, tzinfo=KRX_TZ),
    ]
    manifest = load_manifest(path)
    assert manifest.details["administrative_designations"] == 1
    assert manifest.details["administrative_releases"] == 1


def test_unmapped_corp_code_is_skipped_and_counted(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _publish_window(
        runtime, key="I003:2024-01-01..2024-01-31",
        records=[
            _record("20240102000001", corp="00999999", rcept_dt="20240102", report_nm="상장폐지결정"),
            _record("20240102000002", rcept_dt="20240102", report_nm="상장폐지결정"),
        ],
    )
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
        bridge=_bridge(),
        kind_keywords=(), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    frame = read_dataset(path).collect()
    assert frame.height == 1
    assert frame["ticker"].to_list() == [TICKER]
    assert load_manifest(path).details["unmapped_rows"] == 1


def test_materialize_requires_a_later_session(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _publish_window(
        runtime, key="I003:2024-01-01..2024-01-31",
        records=[_record("20240104000001", rcept_dt="20240104", report_nm="상장폐지결정")],
    )
    with pytest.raises(PITDataError, match="no certified session open"):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(date(2024, 1, 2), date(2024, 1, 4)),
            bridge=_bridge(),
            kind_keywords=(), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
        )


def test_materialize_empty_catalog_publishes_no_rows(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(),
        kind_keywords=(), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    assert verify_dataset(path, known_ids=lambda _dataset_id: True).passed
    assert read_dataset(path).collect().height == 0
    manifest = load_manifest(path)
    assert manifest.details["actions"] == 0
    assert manifest.details["unmapped_rows"] == 0


def test_materialize_requires_two_sessions(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(PITDataError, match="at least two certified sessions"):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(date(2024, 1, 2)),
            bridge=_bridge(),
            kind_keywords=(), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
        )


def _publish_raw_daily(runtime, *, session: date, raw: bytes) -> Path:  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.bronze import BronzeStore
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    store = BronzeStore(runtime.workspace.bronze_root)
    receipt = store.import_bytes(raw, kind=EvidenceKind.DAILY_MARKET, retrieved_at=RETRIEVED_AT, source_label="test")
    catalog = _catalog(runtime)
    catalog.publish(
        [
            ReceiptIndexEntry(
                source="krx_daily_market", natural_key=session.isoformat(), as_of=session,
                fiscal_period=None, status=EvidenceStatus.SUCCESS,
                content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at,
                payload_path=receipt.payload_path,
            )
        ],
        blobs=[
            BlobEntry(
                content_hash=receipt.content_hash, kind=EvidenceKind.DAILY_MARKET,
                source="krx_daily_market", usable=True, unusable_reason=None,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path,
            )
        ],
    )
    return Path(receipt.payload_path)


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[1, 2]",
        b'{"session": "2024-01-02"}',
        b'{"session": "2024-01-02", "records": [1]}',
    ],
)
def test_materialize_rejects_malformed_daily_page(tmp_path: Path, raw: bytes) -> None:
    runtime = _runtime(tmp_path)
    _publish_window(
        runtime, key="I003:2024-01-01..2024-01-31",
        records=[_record("20240102000001", rcept_dt="20240102", report_nm="상장폐지결정")],
    )
    _publish_raw_daily(runtime, session=date(2024, 1, 2), raw=raw)
    with pytest.raises(PITDataError):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
            bridge=_bridge(),
            kind_keywords=(), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
        )


def test_materialize_rejects_tampered_daily_page(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    catalog = _catalog(runtime)
    entry = catalog.latest(source="krx_daily_market", natural_keys={"2024-01-02"})["2024-01-02"]
    Path(entry.payload_path).write_bytes(b"tampered")
    with pytest.raises(PITDataError, match="hash mismatch"):
        materialize_market_actions(
            catalog=catalog, silver_root=runtime.workspace.silver_root,
            calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
            bridge=_bridge(),
            kind_keywords=(), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
        )


def test_materialize_rejects_missing_daily_payload(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    payload = _publish_raw_daily(
        runtime, session=date(2024, 1, 2),
        raw=json.dumps({"session": "2024-01-02", "records": []}, sort_keys=True).encode("utf-8"),
    )
    payload.unlink()
    with pytest.raises(PITDataError, match="unreadable"):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
            bridge=_bridge(),
            kind_keywords=(), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
        )


def test_classify_market_action_title_tentative_notices_never_block() -> None:
    from src.data.market_actions import MarketActionKind, classify_market_action_title

    # DART I003 2023-03 표본 제목: 확정되지 않은 안내는 상장폐지·관리종목 조치가 아니다.
    assert classify_market_action_title("기타시장안내(관리종목지정우려종목)") is None
    assert classify_market_action_title("기타시장안내(개선기간 종료에 따른 상장폐지 여부 결정 안내)") is None
    assert classify_market_action_title("기타시장안내(상장폐지 관련)") is None
    assert classify_market_action_title("기타시장안내(감사의견 거절 관련 상장폐지 절차 미진행)") is None
    assert classify_market_action_title("기타시장안내(상장적격성 실질심사 대상 제외 결정)") is None
    assert (
        classify_market_action_title("주권매매거래정지(관리종목지정우려)") is MarketActionKind.TRADING_HALTED
    )
    assert (
        classify_market_action_title("주권매매거래정지해제(관리종목지정우려 해소)")
        is MarketActionKind.TRADING_RESUMED
    )


def test_classify_market_action_title_confirmed_liquidation_notices() -> None:
    from src.data.market_actions import MarketActionKind, classify_market_action_title

    assert (
        classify_market_action_title("주권매매거래정지해제(상장폐지에 따른 정리매매 개시)")
        is MarketActionKind.LIQUIDATION_TRADING
    )
    assert (
        classify_market_action_title(
            "기타시장안내(상장폐지결정 효력정지 가처분 신청 기각결정에 따른 정리매매절차 재개)"
        )
        is MarketActionKind.LIQUIDATION_TRADING
    )


def test_classify_market_action_title_non_krx_delistings_are_not_actions() -> None:
    from src.data.market_actions import classify_market_action_title

    # DART 실제 제목: 해외 예탁증서 상장폐지, 코스닥 이전상장, 주총 안건 상정, 가처분 신청은 조치가 아니다.
    assert classify_market_action_title("주요사항보고서(해외증권시장주권등상장폐지결정)") is None
    assert classify_market_action_title("상장폐지결정(코스닥시장 이전상장)") is None
    assert classify_market_action_title("상장폐지승인을위한의안상정결정") is None
    assert classify_market_action_title("기타경영사항(자율공시)(상장폐지결정등 효력정지 가처분신청의 건)") is None


def test_delisting_injunction_granted_cancels_the_block() -> None:
    from src.data.market_actions import _base_title, _is_delisting_cancellation_title, classify_market_action_title

    title = "기타경영사항(자율공시)(상장폐지결정 등 효력정지가처분 신청결과 인용결정)"
    assert classify_market_action_title(title) is None
    assert _is_delisting_cancellation_title(_base_title(title)) is True
    assert _is_delisting_cancellation_title(_base_title("기타시장안내(상장폐지 관련)")) is False


def test_withdrawn_or_voided_delisting_is_a_cancellation_and_subsidiary_notice_is_not_an_action() -> None:
    from src.data.market_actions import _base_title, _is_delisting_cancellation_title, classify_market_action_title

    assert classify_market_action_title("상장폐지결정(자회사의 주요경영사항)") is None
    assert classify_market_action_title("[기재정정]상장폐지결정(자회사의 주요경영사항)") is None
    for title in (
        "기타주요경영사항(상장폐지결정 철회)",
        "주권매매거래정지해제(대법원의 상장폐지결정무효 판결에 따른 주권 매매거래정지해제)",
    ):
        assert classify_market_action_title(title) is None
        assert _is_delisting_cancellation_title(_base_title(title)) is True


def test_combined_halt_and_release_notice_is_a_halt() -> None:
    from src.data.market_actions import MarketActionKind, classify_market_action_title

    assert classify_market_action_title("매매거래정지및정지해제(중요내용공시)") is MarketActionKind.TRADING_HALTED
    assert classify_market_action_title("매매거래정지 및 정지해제(풍문 등 조회공시)") is MarketActionKind.TRADING_HALTED


def test_plain_release_stays_a_resumption() -> None:
    from src.data.market_actions import MarketActionKind, classify_market_action_title

    assert classify_market_action_title("주권매매거래정지해제") is MarketActionKind.TRADING_RESUMED
    assert classify_market_action_title("매매거래정지해제") is MarketActionKind.TRADING_RESUMED


def test_liquidation_precedence_survives() -> None:
    from src.data.market_actions import MarketActionKind, classify_market_action_title

    assert (
        classify_market_action_title("주권매매거래정지해제(상장폐지에 따른 정리매매 개시)")
        is MarketActionKind.LIQUIDATION_TRADING
    )


KIND_DELISTING_HTML = """<html><body><table>
<tr><td>단축코드</td><td>A005980</td></tr>
<tr><td>정리매매 허용기간 시작일</td><td>2018-09-19</td></tr>
<tr><td>종료일</td><td>2018-10-02</td></tr>
<tr><td>상장폐지일</td><td>2018-10-04</td></tr>
</table></body></html>"""

KIND_ADMIN_COMMON_HTML = """<html><body><table>
<tr><td>종목명</td><td>삼성전자보통주</td></tr>
<tr><td>지정일</td><td>2024년 01월 03일</td></tr>
</table></body></html>"""

KIND_ADMIN_PREFERRED_HTML = """<html><body><table>
<tr><td>종목명</td><td>대상홀딩스1우선주</td></tr>
<tr><td>지정일</td><td>2024년 01월 03일</td></tr>
</table></body></html>"""






def _publish_kind_search(runtime, *, keyword: str, start: str, end: str, rows: list, total: int | None = None):  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.bronze import BronzeStore
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    reported = len(rows) if total is None else total
    body = json.dumps(
        {"keyword": keyword, "start": start, "end": end,
         "reported_total": reported, "pages": [kind_search_html(rows, reported)]},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    store = BronzeStore(runtime.workspace.bronze_root)
    receipt = store.import_bytes(body, kind=EvidenceKind.DISCLOSURES, retrieved_at=RETRIEVED_AT, source_label="test")
    _catalog(runtime).publish(
        [
            ReceiptIndexEntry(
                source="kind_notice_search", natural_key=f"{keyword}:{start}..{end}",
                as_of=date.fromisoformat(end), fiscal_period=None,
                status=EvidenceStatus.EMPTY if reported == 0 else EvidenceStatus.SUCCESS,
                content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at,
                payload_path=receipt.payload_path,
            )
        ],
        blobs=[
            BlobEntry(
                content_hash=receipt.content_hash, kind=EvidenceKind.DISCLOSURES,
                source="kind_notice_search", usable=True, unusable_reason=None,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path,
            )
        ],
    )


def _publish_kind_document(runtime, *, acptno: str, html: str, disclosed_at: str):  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.bronze import BronzeStore
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    body = json.dumps(
        {"acptno": acptno, "doc_no": "1234567", "template": "68051.htm",
         "html": html, "disclosed_at": disclosed_at},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    store = BronzeStore(runtime.workspace.bronze_root)
    receipt = store.import_bytes(body, kind=EvidenceKind.DISCLOSURES, retrieved_at=RETRIEVED_AT, source_label="test")
    _catalog(runtime).publish(
        [
            ReceiptIndexEntry(
                source="kind_notice_documents", natural_key=acptno,
                as_of=date.fromisoformat(disclosed_at[:10]), fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at,
                payload_path=receipt.payload_path,
            )
        ],
        blobs=[
            BlobEntry(
                content_hash=receipt.content_hash, kind=EvidenceKind.DISCLOSURES,
                source="kind_notice_documents", usable=True, unusable_reason=None,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path,
            )
        ],
    )


def test_kind_delisting_form_becomes_bounded_delisting_action(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20180913000001", "2018-09-13 18:59", "상장폐지", "00598", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="상장폐지", start="2018-09-13", end="2018-09-28", rows=[row])
    _publish_kind_document(runtime, acptno="20180913000001", html=KIND_DELISTING_HTML,
                           disclosed_at="2018-09-13T18:59:00+09:00")
    _publish_daily(runtime, session=date(2018, 9, 13), records=[{"ISU_SRT_CD": "005980"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2018, 9, 13), date(2018, 9, 14), date(2018, 9, 17)),
        bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2018, 9, 13),
    )
    frame = read_dataset(path).collect()
    kind_rows = frame.filter(pl.col("source") == "kind")
    assert kind_rows.height == 1
    row_out = kind_rows.to_dicts()[0]
    assert row_out["kind"] == MarketActionKind.DELISTING_DECIDED.value
    assert row_out["source"] == "kind"
    assert row_out["available_at"] == datetime(2018, 9, 14, 9, 0, tzinfo=KRX_TZ)
    assert row_out["effective_start"] == date(2018, 9, 19)
    assert row_out["effective_end"] == date(2018, 10, 4)
    assert row_out["rcept_no"] == "KIND-20180913000001"


def test_kind_pre_open_notice_is_available_the_same_day(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    title = "기타시장안내(상장폐지결정 효력정지 가처분 신청 기각결정에 따른 정리매매절차 재개)"
    row = kind_row_html("20240102000001", "2024-01-02 07:30", title, "00593", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    frame = read_dataset(path).collect().filter(pl.col("source") == "kind")
    assert frame.height == 1
    assert frame["available_at"].to_list() == [datetime(2024, 1, 2, 9, 0, tzinfo=KRX_TZ)]


def test_kind_missing_form_body_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240102000001", "2024-01-02 15:30", "상장폐지", "00593", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    with pytest.raises(PITDataError, match="20240102000001"):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
            bridge=_bridge(), kind_keywords=("상장폐지",),
            kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
        )


def test_kind_preferred_only_designation_never_touches_common_stock(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240102000001", "2024-01-02 15:30", "관리종목 지정", "00593", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="관리종목", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_kind_document(runtime, acptno="20240102000001", html=KIND_ADMIN_PREFERRED_HTML,
                           disclosed_at="2024-01-02T15:30:00+09:00")
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("관리종목",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    frame = read_dataset(path).collect()
    assert frame.height == 0
    assert load_manifest(path).details["kind_other_share_class"] == 1


def test_kind_unlisted_resolved_ticker_is_skipped_and_counted(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    title = "기타시장안내(상장폐지결정 효력정지 가처분 신청 기각결정에 따른 정리매매절차 재개)"
    row = kind_row_html("20240102000001", "2024-01-02 15:30", title, "12345", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    frame = read_dataset(path).collect()
    assert frame.height == 0
    assert load_manifest(path).details["kind_unresolved_rows"] == 1


def test_kind_company_filed_notice_is_ignored(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240102000001", "2024-01-02 15:30", "상장폐지", "00593", "삼성전자")
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    assert read_dataset(path).collect().height == 0


def test_kind_title_only_liquidation_notice_classifies_like_dart(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    title = "기타시장안내(상장폐지결정 효력정지 가처분 신청 기각결정에 따른 정리매매절차 재개)"
    row = kind_row_html("20240102000001", "2024-01-02 15:30", title, "00593", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    frame = read_dataset(path).collect()
    assert frame.height == 1
    out = frame.to_dicts()[0]
    assert out["kind"] == MarketActionKind.LIQUIDATION_TRADING.value
    assert out["source"] == "kind"
    assert out["effective_start"] is None
    assert out["effective_end"] is None


def test_kind_coverage_gap_aborts_the_build(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(PITDataError):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(date(2024, 1, 2), date(2024, 4, 2)),
            bridge=_bridge(), kind_keywords=("상장폐지",),
            kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
        )


def test_kind_identity_changes_with_kind_bronze(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    kwargs = {
        "catalog": _catalog(runtime),
        "silver_root": runtime.workspace.silver_root,
        "calendar": _calendar(date(2024, 1, 2), date(2024, 1, 3)),
        "bridge": _bridge(),
        "kind_keywords": (),
        "kind_max_stale_sessions": 5, "kind_coverage_start": date(2024, 1, 1),
    }
    first = materialize_market_actions(**kwargs)
    row = kind_row_html("20240102000001", "2024-01-02 07:30", "현금배당 결정", "00593", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-02", end="2024-01-05", rows=[row])
    second = materialize_market_actions(**kwargs)
    assert first != second


def test_kind_common_designation_becomes_administrative_action(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240102000001", "2024-01-02 15:30", "관리종목 지정", "00593", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="관리종목", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_kind_document(runtime, acptno="20240102000001", html=KIND_ADMIN_COMMON_HTML,
                           disclosed_at="2024-01-02T15:30:00+09:00")
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("관리종목",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    frame = read_dataset(path).collect()
    assert frame.height == 1
    out = frame.to_dicts()[0]
    assert out["kind"] == MarketActionKind.ADMINISTRATIVE_DESIGNATED.value
    assert out["source"] == "kind"
    assert out["ticker"] == "005930"
    assert out["effective_start"] == date(2024, 1, 3)
    assert out["effective_end"] is None


def test_kind_delisting_form_with_unlisted_ticker_is_skipped_and_counted(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240102000001", "2024-01-02 15:30", "상장폐지", "00593", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_kind_document(runtime, acptno="20240102000001", html=KIND_DELISTING_HTML,
                           disclosed_at="2024-01-02T15:30:00+09:00")
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    frame = read_dataset(path).collect()
    assert frame.height == 0
    assert load_manifest(path).details["kind_unresolved_rows"] == 1


def test_kind_administrative_missing_body_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240102000001", "2024-01-02 15:30", "관리종목 지정", "00593", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="관리종목", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    with pytest.raises(PITDataError, match="20240102000001"):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
            bridge=_bridge(), kind_keywords=("관리종목",),
            kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
        )


def test_kind_notice_after_last_session_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    title = "기타시장안내(상장폐지결정 효력정지 가처분 신청 기각결정에 따른 정리매매절차 재개)"
    row = kind_row_html("20240103000001", "2024-01-03 18:00", title, "00593", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    with pytest.raises(PITDataError, match="no certified session open"):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
            bridge=_bridge(), kind_keywords=("상장폐지",),
            kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
        )


def test_kind_notice_without_covering_daily_page_is_skipped_and_counted(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    title = "기타시장안내(상장폐지결정 효력정지 가처분 신청 기각결정에 따른 정리매매절차 재개)"
    row = kind_row_html("20240102000001", "2024-01-02 15:30", title, "00593", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_daily(runtime, session=date(2024, 1, 3), records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    frame = read_dataset(path).collect()
    assert frame.height == 0
    assert load_manifest(path).details["kind_unresolved_rows"] == 1


def test_kind_common_admin_with_unlisted_ticker_is_skipped_and_counted(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240102000001", "2024-01-02 15:30", "관리종목 지정", "12345", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="관리종목", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_kind_document(runtime, acptno="20240102000001", html=KIND_ADMIN_COMMON_HTML,
                           disclosed_at="2024-01-02T15:30:00+09:00")
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("관리종목",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    frame = read_dataset(path).collect()
    assert frame.height == 0
    assert load_manifest(path).details["kind_unresolved_rows"] == 1


def test_kind_withdrawn_delisting_is_a_cancellation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240102000001", "2024-01-02 15:30", "상장폐지결정 취소", "00593", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    frame = read_dataset(path).collect()
    assert frame.height == 1
    out = frame.to_dicts()[0]
    assert out["kind"] == MarketActionKind.DELISTING_DECIDED.value
    assert out["source"] == "kind"
    assert out["cancellation"] is True
    manifest = load_manifest(path)
    assert manifest.details["kind_cancellations"] == 1
    assert manifest.details["kind_actions"] == 0


def test_kind_coverage_uses_the_collector_window_anchor_not_the_first_session(tmp_path: Path) -> None:
    # 수집기는 scope의 evidence_start(1월 1일)부터 창을 만들고 첫 세션은 1월 4일이다.
    runtime = _runtime(tmp_path)
    _publish_kind_search(runtime, keyword="상장폐지", start="2016-01-01", end="2016-03-31", rows=[])
    _publish_daily(runtime, session=date(2016, 1, 4), records=[{"ISU_SRT_CD": "005930"}])
    _publish_daily(runtime, session=date(2016, 4, 4), records=[{"ISU_SRT_CD": "005930"}])
    calendar = _calendar(date(2016, 1, 4), date(2016, 4, 1), date(2016, 4, 4))
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=calendar, bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2016, 1, 1),
    )
    assert path.exists()

    with pytest.raises(PITDataError, match=r"상장폐지:2016-01-04\.\.2016-03-31"):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=calendar, bridge=_bridge(), kind_keywords=("상장폐지",),
            kind_max_stale_sessions=5, kind_coverage_start=date(2016, 1, 4),
        )


def test_kind_coverage_horizon_is_the_last_observed_session_not_the_calendar_end(tmp_path: Path) -> None:
    # 실제 달력은 다음 해 말까지 이어진다. 관측이 끝난 뒤의 분기 창까지 요구하면 안 된다.
    runtime = _runtime(tmp_path)
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-01", end="2024-03-31", rows=[])
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    _publish_daily(runtime, session=date(2024, 3, 29), records=[{"ISU_SRT_CD": "005930"}])
    calendar = _calendar(date(2024, 1, 2), date(2024, 3, 29), date(2025, 12, 30))

    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=calendar, bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 1),
    )

    assert path.exists()


def test_kind_channel_without_any_daily_page_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(PITDataError, match="daily-market page"):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)), bridge=_bridge(),
            kind_keywords=("상장폐지",), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 1),
        )


def test_kind_reposted_delisting_form_with_a_past_schedule_becomes_open_ended(tmp_path: Path) -> None:
    # 가처분 취소로 양식이 재게시되면 이미 지난 옛 일정이 실려 온다(실측: 065160, 2019-04-26).
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240102000001", "2024-01-02 18:00", "상장폐지", "00598", "유가증권시장본부")
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-01", end="2024-01-31", rows=[row])
    _publish_kind_document(runtime, acptno="20240102000001", html=KIND_DELISTING_HTML,
                           disclosed_at="2024-01-02T18:00:00+09:00")
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005980"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("상장폐지",), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 1),
    )
    (out,) = read_dataset(path).collect().filter(pl.col("source") == "kind").to_dicts()
    assert out["kind"] == MarketActionKind.DELISTING_DECIDED.value
    assert out["effective_start"] is None
    assert out["effective_end"] is None


def test_partial_cause_release_keeps_the_administrative_status() -> None:
    # 실측: 관리종목지정사유일부해제 265건 등은 사유 일부만 해소되어 지위가 유지된다.
    for title in (
        "관리종목지정사유일부해제(회생절차 개시신청 사유 해소)",
        "관리종목지정사유추가및일부해제(반기검토의견 의견거절 추가)",
        "관리종목지정사유일부해제",
    ):
        assert classify_market_action_title(title) is None
    assert classify_market_action_title("관리종목해제(자본잠식률 50% 미만으로 회복 등)") is (
        MarketActionKind.ADMINISTRATIVE_RELEASED
    )
    assert classify_market_action_title("관리종목지정해제") is MarketActionKind.ADMINISTRATIVE_RELEASED


def test_liquidation_resumption_after_a_reversed_order_is_not_a_withdrawal() -> None:
    # 실측(065160, 2019-04-26): 가처분 결정 취소로 정리매매가 재개된다. 제목의 `취소`는 철회가 아니다.
    assert classify_market_action_title(
        "주권매매거래정지해제(상장폐지금지 가처분 결정 취소에 따른 정리매매 재개)"
    ) is MarketActionKind.LIQUIDATION_TRADING
    assert classify_market_action_title(
        "기타시장안내(상장폐지무효확인 청구의 소 기각에 따른 정리매매절차 재개)"
    ) is MarketActionKind.LIQUIDATION_TRADING
    assert classify_market_action_title("기타 주요경영사항(상장폐지결정 철회)") is None


def _admin_kind_run(tmp_path: Path, *, desk: str, title: str):  # type: ignore[no-untyped-def]
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240102000001", "2024-01-02 15:30", title, "00593", desk)
    _publish_kind_search(runtime, keyword="관리종목", start="2024-01-02", end="2024-01-05", rows=[row])
    _publish_kind_document(runtime, acptno="20240102000001", html=KIND_ADMIN_COMMON_HTML,
                           disclosed_at="2024-01-02T15:30:00+09:00")
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(), kind_keywords=("관리종목",), kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 2),
    )
    return path, read_dataset(path).collect().filter(pl.col("source") == "kind")


def test_kosdaq_kind_release_is_ignored_because_the_daily_flag_is_the_authority(tmp_path: Path) -> None:
    # 실측: KOSDAQ 종목의 `관리종목 해제(…사유 해소)`는 사유별로 나오며 해제 329건 중 92건이 플래그와 어긋났다.
    path, frame = _admin_kind_run(tmp_path, desk="코스닥시장본부", title="관리종목 해제(자본잠식률 회복)")

    assert frame.height == 0
    assert load_manifest(path).details["kind_kosdaq_release_ignored"] == 1


def test_kospi_kind_release_still_releases_because_no_flag_exists(tmp_path: Path) -> None:
    path, frame = _admin_kind_run(tmp_path, desk="유가증권시장본부", title="관리종목 해제(자본잠식률 회복)")

    assert frame["kind"].to_list() == [MarketActionKind.ADMINISTRATIVE_RELEASED.value]
    assert load_manifest(path).details["kind_kosdaq_release_ignored"] == 0


def test_kosdaq_kind_designation_is_still_applied(tmp_path: Path) -> None:
    _, frame = _admin_kind_run(tmp_path, desk="코스닥시장본부", title="관리종목 지정(자본잠식률 50% 이상)")

    assert frame["kind"].to_list() == [MarketActionKind.ADMINISTRATIVE_DESIGNATED.value]


def test_kosdaq_release_ignored_on_both_kind_title_paths(tmp_path: Path) -> None:
    # `관리종목지정해제`(본문을 읽는 경로)와 `관리종목 해제(…)`(제목 분류 경로) 모두 KOSDAQ에서는 무시된다.
    for index, title in enumerate(("관리종목지정해제", "관리종목 해제(사유 해소)")):
        path, frame = _admin_kind_run(tmp_path / str(index), desk="코스닥시장본부", title=title)
        assert frame.height == 0, title
        assert load_manifest(path).details["kind_kosdaq_release_ignored"] == 1, title


def _actions_fingerprint(path: Path) -> object:
    import json

    manifest = load_manifest(path)
    return (
        manifest.dataset_id,
        tuple((partition.path, partition.sha256) for partition in manifest.partitions),
        json.dumps(manifest.details, sort_keys=True, default=str),
    )


def test_flag_cache_hit_and_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.market_actions as actions_module

    runtime = _runtime(tmp_path)
    plain = {"ISU_SRT_CD": "000660", "SECT_TP_NM": ""}
    flagged = {"ISU_SRT_CD": "000660", "SECT_TP_NM": "관리종목(소속부없음)"}
    _publish_daily(runtime, session=date(2024, 1, 2), records=[plain])
    _publish_daily(runtime, session=date(2024, 1, 3), records=[flagged])
    cache = tmp_path / "flags.parquet"

    def _kwargs():  # type: ignore[no-untyped-def]
        return {
            "catalog": _catalog(runtime),
            "silver_root": runtime.workspace.silver_root,
            "calendar": _calendar(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
            "bridge": _bridge(),
            "kind_keywords": (),
            "kind_max_stale_sessions": 5, "kind_coverage_start": date(2024, 1, 2),
        }

    materialize_market_actions(**_kwargs(), flag_cache=cache)
    assert cache.is_file()
    cached_sessions = set(pl.read_parquet(cache).get_column("session").to_list())
    assert cached_sessions == {date(2024, 1, 2), date(2024, 1, 3)}
    _publish_daily(runtime, session=date(2024, 1, 4), records=[flagged])
    calls = 0
    real_read = actions_module.read_payload

    def _counted(path):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real_read(path)

    monkeypatch.setattr(actions_module, "read_payload", _counted)
    kwargs = _kwargs()
    kwargs["calendar"] = _calendar(
        date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)
    )
    kwargs["silver_root"] = tmp_path / "silver-cached"
    cached_path = materialize_market_actions(**kwargs, flag_cache=cache)
    assert calls == 1
    uncached_path = materialize_market_actions(
        **{**kwargs, "silver_root": tmp_path / "silver-full", "flag_cache": None}
    )
    assert _actions_fingerprint(cached_path) == _actions_fingerprint(uncached_path)
    assert set(pl.read_parquet(cache).get_column("session").to_list()) == {
        date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),
    }


def test_packed_bronze_builds_identical_market_actions(tmp_path: Path) -> None:
    from src.data.payload_store import pack_loose

    runtime = _runtime(tmp_path)
    _publish_window(
        runtime, key="I003:2024-01-01..2024-01-31",
        records=[_record("20240102000001", rcept_dt="20240102", report_nm="상장폐지결정")],
    )
    _publish_daily(runtime, session=date(2024, 1, 2), records=[{"ISU_CD": "000660", "SECT_TP_NM": ""}])
    kwargs = {
        "catalog": _catalog(runtime),
        "silver_root": runtime.workspace.silver_root,
        "calendar": _calendar(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
        "bridge": _bridge(),
        "kind_keywords": (),
        "kind_max_stale_sessions": 5, "kind_coverage_start": date(2024, 1, 2),
    }
    first = materialize_market_actions(**kwargs)
    pack_loose(
        runtime.workspace.bronze_root, kinds=["daily_market", "disclosures"],
        offsite_paths=None, remove=True,
    )
    kwargs["catalog"] = _catalog(runtime)
    kwargs["silver_root"] = tmp_path / "silver-packed"
    second = materialize_market_actions(**kwargs)
    assert _actions_fingerprint(first) == _actions_fingerprint(second)


def test_unreadable_or_inconsistent_flag_cache_is_ignored(tmp_path: Path) -> None:
    from src.data.market_actions import _load_daily_flag_rows

    runtime = _runtime(tmp_path)
    plain = {"ISU_SRT_CD": "000660", "SECT_TP_NM": ""}
    _publish_daily(runtime, session=date(2024, 1, 2), records=[plain])
    calendar = _calendar(date(2024, 1, 2), date(2024, 1, 3))
    catalog = _catalog(runtime)
    missing = tmp_path / "no-cache.parquet"
    assert _load_daily_flag_rows(catalog, calendar, flag_cache=missing) == {
        date(2024, 1, 2): {"000660": False}
    }
    assert missing.is_file()
    garbage = tmp_path / "garbage.parquet"
    garbage.write_bytes(b"not parquet")
    assert _load_daily_flag_rows(catalog, calendar, flag_cache=garbage) == {
        date(2024, 1, 2): {"000660": False}
    }
    wrong_schema = tmp_path / "wrong.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(wrong_schema)
    assert _load_daily_flag_rows(catalog, calendar, flag_cache=wrong_schema) == {
        date(2024, 1, 2): {"000660": False}
    }
    bad_row = tmp_path / "bad-row.parquet"
    pl.DataFrame(
        [{"session": "yesterday", "source_hash": "s", "ticker": None, "flagged": False}],
        schema={"session": pl.String, "source_hash": pl.String, "ticker": pl.String, "flagged": pl.Boolean},
    ).write_parquet(bad_row)
    assert _load_daily_flag_rows(catalog, calendar, flag_cache=bad_row) == {
        date(2024, 1, 2): {"000660": False}
    }
    conflicted = tmp_path / "conflict.parquet"
    pl.DataFrame(
        [
            {"session": date(2024, 1, 2), "source_hash": "s1", "ticker": None, "flagged": False},
            {"session": date(2024, 1, 2), "source_hash": "s2", "ticker": None, "flagged": False},
        ],
        schema={"session": pl.Date, "source_hash": pl.String, "ticker": pl.String, "flagged": pl.Boolean},
    ).write_parquet(conflicted)
    assert _load_daily_flag_rows(catalog, calendar, flag_cache=conflicted) == {
        date(2024, 1, 2): {"000660": False}
    }


def test_unwritable_flag_cache_fails_closed(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.market_actions import _write_flag_cache

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        with pytest.raises(PITDataError, match="cannot be written"):
            _write_flag_cache(locked / "flags.parquet", {date(2024, 1, 2): ("s" * 64, {"000660": False})})
    finally:
        locked.chmod(0o755)


def test_flag_cache_without_session_sentinel_is_rebuilt(tmp_path: Path) -> None:
    from src.data.market_actions import _load_daily_flag_rows

    runtime = _runtime(tmp_path)
    day = date(2024, 1, 2)
    _publish_daily(runtime, session=day, records=[{"ISU_CD": "000660", "SECT_TP_NM": ""}])
    catalog = _catalog(runtime)
    entry = catalog.latest(source="krx_daily_market", natural_keys={day.isoformat()})[day.isoformat()]
    cache = tmp_path / "partial.parquet"
    pl.DataFrame(
        {"session": [day], "source_hash": [entry.content_hash], "ticker": ["000660"], "flagged": [True]},
        schema={"session": pl.Date, "source_hash": pl.String, "ticker": pl.String, "flagged": pl.Boolean},
    ).write_parquet(cache)
    assert _load_daily_flag_rows(catalog, _calendar(day, date(2024, 1, 3)), flag_cache=cache) == {
        day: {"000660": False}
    }
    assert pl.read_parquet(cache).filter(pl.col("ticker").is_null()).height == 1


def _publish_stale_setup(runtime, *, sessions: list) -> None:  # type: ignore[no-untyped-def]
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-01", end="2024-03-31", rows=[])
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-04-01", end="2024-04-02", rows=[])
    for session in sessions:
        _publish_daily(runtime, session=session, records=[{"ISU_SRT_CD": "005930"}])


def test_within_budget_builds_and_records_provenance(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    sessions = [date(2024, 4, 1), date(2024, 4, 2), date(2024, 4, 3), date(2024, 4, 4), date(2024, 4, 5)]
    _publish_stale_setup(runtime, sessions=sessions)
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(*sessions), bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 1),
    )
    manifest = load_manifest(path)
    assert manifest.details["kind_stale_sessions"] == 3
    assert manifest.details["kind_covered_through"] == "2024-04-02"


def test_over_budget_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    sessions = [date(2024, 4, day) for day in (1, 2, 3, 4, 5, 8, 9, 11)]
    _publish_stale_setup(runtime, sessions=sessions)
    before = set(runtime.workspace.silver_root.iterdir()) if runtime.workspace.silver_root.is_dir() else set()
    with pytest.raises(PITDataError) as exc_info:
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(*sessions), bridge=_bridge(), kind_keywords=("상장폐지",),
            kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 1),
        )
    message = str(exc_info.value)
    assert "2024-04-02" in message
    assert "6 observed sessions" in message
    assert "budget 5" in message
    after = set(runtime.workspace.silver_root.iterdir()) if runtime.workspace.silver_root.is_dir() else set()
    assert after == before


def test_zero_budget_is_strict(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    sessions = [date(2024, 4, 1), date(2024, 4, 2), date(2024, 4, 3)]
    _publish_stale_setup(runtime, sessions=sessions)
    with pytest.raises(PITDataError):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(*sessions), bridge=_bridge(), kind_keywords=("상장폐지",),
            kind_max_stale_sessions=0, kind_coverage_start=date(2024, 1, 1),
        )


def test_quarter_rollover_no_longer_cliffs(tmp_path: Path) -> None:
    from src.data.kind_notices import require_kind_coverage

    runtime = _runtime(tmp_path)
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-07-01", end="2024-09-30", rows=[])
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-10-01", end="2024-12-26", rows=[])
    sessions = [date(2024, 12, 26), date(2024, 12, 27), date(2024, 12, 30), date(2025, 1, 2)]
    for session in sessions:
        _publish_daily(runtime, session=session, records=[{"ISU_SRT_CD": "005930"}])
    with pytest.raises(PITDataError, match=r"상장폐지:2024-10-01\.\.2024-12-31"):
        require_kind_coverage(
            _catalog(runtime), keywords=("상장폐지",),
            through=date(2025, 1, 2), start=date(2024, 7, 1),
        )
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(*sessions), bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 7, 1),
    )
    manifest = load_manifest(path)
    assert manifest.details["kind_stale_sessions"] == 3
    assert manifest.details["kind_covered_through"] == "2024-12-26"


def test_missing_kind_window_fails_closed_without_coverage(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    sessions = [date(2024, 4, 1), date(2024, 4, 2)]
    for session in sessions:
        _publish_daily(runtime, session=session, records=[{"ISU_SRT_CD": "005930"}])
    with pytest.raises(PITDataError, match="no complete KIND search window"):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(*sessions), bridge=_bridge(), kind_keywords=("상장폐지",),
            kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 1),
        )


def test_past_quarter_gap_still_fails(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    sessions = [date(2024, 4, 1), date(2024, 4, 2)]
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-04-01", end="2024-04-02", rows=[])
    for session in sessions:
        _publish_daily(runtime, session=session, records=[{"ISU_SRT_CD": "005930"}])
    with pytest.raises(PITDataError, match=r"상장폐지:2024-01-01\.\.2024-03-31"):
        materialize_market_actions(
            catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
            calendar=_calendar(*sessions), bridge=_bridge(), kind_keywords=("상장폐지",),
            kind_max_stale_sessions=100, kind_coverage_start=date(2024, 1, 1),
        )


def test_dataset_id_independent_of_freshness_details(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    sessions = [date(2024, 4, 1), date(2024, 4, 2), date(2024, 4, 3), date(2024, 4, 4), date(2024, 4, 5)]
    _publish_stale_setup(runtime, sessions=sessions)
    first = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=tmp_path / "silver-a",
        calendar=_calendar(*sessions), bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=5, kind_coverage_start=date(2024, 1, 1),
    )
    second = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=tmp_path / "silver-b",
        calendar=_calendar(*sessions), bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=3, kind_coverage_start=date(2024, 1, 1),
    )
    assert load_manifest(first).dataset_id == load_manifest(second).dataset_id


def test_current_kind_matches_previous_behaviour(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    sessions = [date(2024, 4, 1), date(2024, 4, 2)]
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-01-01", end="2024-03-31", rows=[])
    _publish_kind_search(runtime, keyword="상장폐지", start="2024-04-01", end="2024-04-02", rows=[])
    for session in sessions:
        _publish_daily(runtime, session=session, records=[{"ISU_SRT_CD": "005930"}])
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(*sessions), bridge=_bridge(), kind_keywords=("상장폐지",),
        kind_max_stale_sessions=0, kind_coverage_start=date(2024, 1, 1),
    )
    manifest = load_manifest(path)
    assert manifest.details["kind_stale_sessions"] == 0
    assert manifest.details["kind_covered_through"] == "2024-04-02"
    assert read_dataset(path).collect().height == 0


def test_flag_cache_roundtrip_is_content_identical(tmp_path: Path) -> None:
    from src.data.market_actions import _read_flag_cache, _write_flag_cache

    flags = {
        date(2024, 1, 3): ("c" * 64, {"000660": True, "005930": False}),
        date(2024, 1, 2): ("a" * 64, {}),
        date(2024, 1, 4): ("b" * 64, {"000001": False}),
    }
    cache = tmp_path / "flags.parquet"
    _write_flag_cache(cache, flags)
    assert _read_flag_cache(cache) == flags
    frame = pl.read_parquet(cache)
    assert frame.schema == {
        "session": pl.Date, "source_hash": pl.String, "ticker": pl.String, "flagged": pl.Boolean,
    }


def test_flag_cache_written_by_new_code_matches_legacy_layout(tmp_path: Path) -> None:
    from src.data.market_actions import _write_flag_cache

    day_a, day_b = date(2024, 1, 2), date(2024, 1, 3)
    _write_flag_cache(
        tmp_path / "flags.parquet",
        {day_b: ("b" * 64, {"005930": True, "000660": False}), day_a: ("a" * 64, {})},
    )
    assert pl.read_parquet(tmp_path / "flags.parquet").to_dicts() == [
        {"session": day_a, "source_hash": "a" * 64, "ticker": None, "flagged": False},
        {"session": day_b, "source_hash": "b" * 64, "ticker": None, "flagged": False},
        {"session": day_b, "source_hash": "b" * 64, "ticker": "000660", "flagged": False},
        {"session": day_b, "source_hash": "b" * 64, "ticker": "005930", "flagged": True},
    ]


def _large_flag_input(n_sessions: int = 150, n_tickers: int = 3000) -> dict:
    from datetime import timedelta

    base = date(2024, 1, 2)
    return {
        base + timedelta(days=index): ("s" * 63 + f"{index % 16:01x}", {f"{ticker:06d}": ticker % 7 == 0 for ticker in range(n_tickers)})
        for index in range(n_sessions)
    }


def test_flag_cache_write_memory_is_per_entry_bounded(tmp_path: Path) -> None:
    import tracemalloc

    from src.data.market_actions import _write_flag_cache

    flags = _large_flag_input()
    entries = sum(len(tickers) + 1 for _, tickers in flags.values())
    tracemalloc.start()
    try:
        _write_flag_cache(tmp_path / "flags.parquet", flags)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= 120 * entries


def test_flag_cache_read_memory_is_per_entry_bounded(tmp_path: Path) -> None:
    import tracemalloc

    from src.data.market_actions import _read_flag_cache, _write_flag_cache

    flags = _large_flag_input()
    entries = sum(len(tickers) + 1 for _, tickers in flags.values())
    cache = tmp_path / "flags.parquet"
    _write_flag_cache(cache, flags)
    tracemalloc.start()
    try:
        assert _read_flag_cache(cache) == flags
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= 80 * entries


def test_flag_cache_rejects_empty_frame_and_bad_hash(tmp_path: Path) -> None:
    from src.data.market_actions import _read_flag_cache, _write_flag_cache

    empty = tmp_path / "empty.parquet"
    _write_flag_cache(empty, {})
    assert _read_flag_cache(empty) == {}
    sentinel_only = tmp_path / "sentinel.parquet"
    _write_flag_cache(sentinel_only, {date(2024, 1, 2): ("a" * 64, {})})
    assert _read_flag_cache(sentinel_only) == {date(2024, 1, 2): ("a" * 64, {})}
    for index, bad_hash in enumerate((None, "")):
        bad = tmp_path / f"bad-{index}.parquet"
        pl.DataFrame(
            [{"session": date(2024, 1, 2), "source_hash": bad_hash, "ticker": None, "flagged": False}],
            schema={"session": pl.Date, "source_hash": pl.String, "ticker": pl.String, "flagged": pl.Boolean},
            strict=False,
        ).write_parquet(bad)
        assert _read_flag_cache(bad) == {}
    mistyped = tmp_path / "bad-dtype.parquet"
    pl.DataFrame(
        {"session": [date(2024, 1, 2)], "source_hash": [1], "ticker": [None], "flagged": [False]},
        schema={"session": pl.Date, "source_hash": pl.Int64, "ticker": pl.String, "flagged": pl.Boolean},
    ).write_parquet(mistyped)
    assert _read_flag_cache(mistyped) == {}


def test_cache_hit_does_not_copy_session_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    import src.data.market_actions as actions_module
    from src.data.market_actions import _load_daily_flag_rows

    runtime = _runtime(tmp_path)
    day = date(2024, 1, 2)
    _publish_daily(runtime, session=day, records=[{"ISU_SRT_CD": "000660", "SECT_TP_NM": ""}])
    catalog = _catalog(runtime)
    calendar = _calendar(day, date(2024, 1, 3))
    cache = tmp_path / "flags.parquet"
    first = _load_daily_flag_rows(catalog, calendar, flag_cache=cache)
    mtime = os.stat(cache).st_mtime_ns
    second = _load_daily_flag_rows(catalog, calendar, flag_cache=cache)
    assert second == first
    assert os.stat(cache).st_mtime_ns == mtime

    entry = catalog.latest(source="krx_daily_market", natural_keys={day.isoformat()})[day.isoformat()]
    inner = {"000660": True}
    monkeypatch.setattr(
        actions_module, "_read_flag_cache", lambda _path: {day: (entry.content_hash, inner)}
    )
    assert _load_daily_flag_rows(catalog, calendar, flag_cache=cache)[day] is inner


@pytest.mark.parametrize("values", [[False, True, False], [-1, 0, None], ["", "yes", None]])
def test_flag_cache_duplicate_tickers_preserve_legacy_truthiness(tmp_path: Path, values: list) -> None:
    from src.data.market_actions import _read_flag_cache

    day = date(2024, 1, 2)
    uncovered = date(2024, 1, 3)
    cache = tmp_path / "duplicates.parquet"
    pl.DataFrame({
        "session": [day, day, uncovered, day, day],
        "source_hash": ["s"] * 5,
        "ticker": [None, "000660", "005930", "000660", "000660"],
        "flagged": [None, values[0], values[1], values[1], values[2]],
    }).write_parquet(cache)
    assert _read_flag_cache(cache) == {day: ("s", {"000660": True})}


def test_flag_cache_parent_creation_fails_closed(tmp_path: Path) -> None:
    from src.data.market_actions import _write_flag_cache

    parent = tmp_path / "not-a-directory"
    parent.write_bytes(b"retained")
    with pytest.raises(PITDataError, match="cannot be written"):
        _write_flag_cache(parent / "flags.parquet", {})
    assert parent.read_bytes() == b"retained"


def test_flag_cache_failed_replace_preserves_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.market_actions as actions_module

    cache = tmp_path / "flags.parquet"
    cache.write_bytes(b"previous-cache")

    def fail_replace(source: str, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(actions_module.os, "rename", fail_replace)
    with pytest.raises(PITDataError, match="cannot be written"):
        actions_module._write_flag_cache(cache, {date(2024, 1, 2): ("s", {"000660": True})})
    assert cache.read_bytes() == b"previous-cache"
    assert list(tmp_path.iterdir()) == [cache]
