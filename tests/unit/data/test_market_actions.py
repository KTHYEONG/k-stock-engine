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
        )


def test_materialize_empty_catalog_publishes_no_rows(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    path = materialize_market_actions(
        catalog=_catalog(runtime), silver_root=runtime.workspace.silver_root,
        calendar=_calendar(date(2024, 1, 2), date(2024, 1, 3)),
        bridge=_bridge(),
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
