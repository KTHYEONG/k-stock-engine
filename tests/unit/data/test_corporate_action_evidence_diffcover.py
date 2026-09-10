"""Supplementary branch coverage for the corporate-action evidence policy.

Covers contract-mandated guards without dedicated scenario skeletons:
unresolved bonus variants, unsupported-event mapping, evidence validation,
and backtest exclusion of unexplained discontinuities.
"""

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from src.core.time import SessionCalendar
from src.data.schemas import PITDataError

_KRX = ZoneInfo("Asia/Seoul")


def _sessions(*days: int) -> tuple[datetime, ...]:
    return tuple(datetime(2024, 1, day, 9, tzinfo=_KRX) for day in days)


def _bonus_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "rcept_no": "20240101000001",
        "corp_code": "00123456",
        "bfic_tisstk_ostk": "10",
        "nstk_ostk_cnt": "10",
        "nstk_ascnt_ps_ostk": "1",
        "nstk_asstd": "2024-01-03",
        "nstk_lstprd": "2024-01-04",
    }
    record.update(overrides)
    return record


def _bonus_page(record: dict[str, object]) -> dict[str, object]:
    return {"endpoint": "fricDecsn.json", "corp_code": "00123456", "status": "000", "records": [record]}


def test_streaming_skips_013_status_page() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {"session": list(sessions), "instrument_id": ["KRX:A"] * 2, "close": [100.0, 100.0]}
    )
    page = {"endpoint": "fricDecsn.json", "corp_code": "1", "status": "013", "records": []}
    assert (
        resolve_opendart_corporate_action_records(
            pages=[page], daily_market=daily, calendar=SessionCalendar(sessions)
        )
        == []
    )


def test_backtest_rejects_legacy_evidence_status() -> None:
    from src.data.backtest_sessions import (
        BacktestMarketInputsPolicy,
        resolve_backtest_corporate_action_evidence,
    )

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {"session": list(sessions), "instrument_id": ["KRX:A"] * 2, "close": [100.0, 100.0]}
    )
    legacy = pl.DataFrame(
        {"instrument_id": ["KRX:A"], "action_type": ["split"], "effective_session": [sessions[1]]}
    )
    with pytest.raises(PITDataError, match=r"legacy corporate-action evidence status missing"):
        resolve_backtest_corporate_action_evidence(
            daily_market=daily,
            corporate_actions=legacy,
            calendar=SessionCalendar(sessions),
            policy=BacktestMarketInputsPolicy(),
        )


def test_streaming_rejects_non_dict_record() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {"session": list(sessions), "instrument_id": ["KRX:A"] * 2, "close": [100.0, 100.0]}
    )
    page = {"endpoint": "fricDecsn.json", "corp_code": "1", "status": "000", "records": [42]}
    with pytest.raises(PITDataError, match=r"invalid OpenDART record"):
        resolve_opendart_corporate_action_records(
            pages=[page], daily_market=daily, calendar=SessionCalendar(sessions)
        )


def test_streaming_rejects_non_datetime_session() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {"session": ["2024-01-02", "2024-01-03"], "instrument_id": ["KRX:A"] * 2, "close": [100.0, 100.0]}
    )
    with pytest.raises(PITDataError, match=r"invalid KRX session"):
        resolve_opendart_corporate_action_records(
            pages=[_bonus_page(_bonus_record())], daily_market=daily, calendar=SessionCalendar(sessions)
        )


def test_streaming_rejects_invalid_close() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {"session": list(sessions), "instrument_id": ["KRX:A"] * 2, "close": [None, 50.0]}
    )
    with pytest.raises(PITDataError, match=r"invalid KRX market value"):
        resolve_opendart_corporate_action_records(
            pages=[_bonus_page(_bonus_record())], daily_market=daily, calendar=SessionCalendar(sessions)
        )


def test_streaming_rejects_non_positive_close() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {"session": list(sessions), "instrument_id": ["KRX:A"] * 2, "close": [0.0, 50.0]}
    )
    with pytest.raises(PITDataError, match=r"invalid KRX market value"):
        resolve_opendart_corporate_action_records(
            pages=[_bonus_page(_bonus_record())], daily_market=daily, calendar=SessionCalendar(sessions)
        )


def test_streaming_reraises_ambiguous_share_basis() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {"session": list(sessions), "instrument_id": ["KRX:A"] * 2, "close": [100.0, 100.0]}
    )
    with pytest.raises(PITDataError, match=r"ambiguous OpenDART share basis"):
        resolve_opendart_corporate_action_records(
            pages=[_bonus_page(_bonus_record(bfic_tisstk_ostk="abc"))],
            daily_market=daily,
            calendar=SessionCalendar(sessions),
        )


def test_streaming_accepts_fractional_bonus_allocation_ratio() -> None:
    from src.data.streaming_normalization import _parse_exact_decimal

    assert _parse_exact_decimal("0.5", field="nstk_ascnt_ps_ostk") == Decimal("0.5")


def test_streaming_zero_allocation_is_unresolved() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3, 4)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 3,
            "close": [100.0, 100.0, 100.0],
            "shares_outstanding": [10.0, 10.0, 10.0],
            "market_cap": [1000.0, 1000.0, 1000.0],
        }
    )
    resolved = resolve_opendart_corporate_action_records(
        pages=[_bonus_page(_bonus_record(nstk_ascnt_ps_ostk="0"))],
        daily_market=daily,
        calendar=SessionCalendar(sessions),
    )
    assert resolved[0]["evidence_status"] == "unresolved"
    assert resolved[0]["evidence_reason"] == "ambiguous_or_missing_dart_share_basis"
    assert resolved[0]["factor"] == 1.0


def test_streaming_basis_mismatch_is_unresolved() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3, 4)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 3,
            "close": [100.0, 100.0, 100.0],
            "shares_outstanding": [10.0, 10.0, 10.0],
            "market_cap": [1000.0, 1000.0, 1000.0],
        }
    )
    resolved = resolve_opendart_corporate_action_records(
        pages=[_bonus_page(_bonus_record(nstk_ostk_cnt="99999"))],
        daily_market=daily,
        calendar=SessionCalendar(sessions),
    )
    assert resolved[0]["evidence_reason"] == "ambiguous_or_missing_dart_share_basis"


def test_streaming_single_bar_raises_missing_bars() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"],
            "close": [100.0],
            "shares_outstanding": [10.0],
            "market_cap": [1000.0],
        }
    )
    with pytest.raises(PITDataError, match=r"missing KRX bars"):
        resolve_opendart_corporate_action_records(
            pages=[_bonus_page(_bonus_record(nstk_asstd="2024-01-02", nstk_lstprd="2024-01-02"))],
            daily_market=daily,
            calendar=SessionCalendar(sessions),
        )


def test_streaming_future_allocation_has_no_candidate() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 100.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 1000.0],
        }
    )
    resolved = resolve_opendart_corporate_action_records(
        pages=[_bonus_page(_bonus_record(nstk_asstd="2024-02-01", nstk_lstprd="2024-01-03"))],
        daily_market=daily,
        calendar=SessionCalendar(sessions),
    )
    assert resolved[0]["evidence_reason"] == "expected_one_price_candidate_got_0"


def test_streaming_flat_prices_have_no_candidate() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3, 4)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 3,
            "close": [100.0, 100.0, 100.0],
            "shares_outstanding": [10.0, 10.0, 20.0],
            "market_cap": [1000.0, 1000.0, 2000.0],
        }
    )
    resolved = resolve_opendart_corporate_action_records(
        pages=[_bonus_page(_bonus_record())], daily_market=daily, calendar=SessionCalendar(sessions)
    )
    assert resolved[0]["evidence_reason"] == "expected_one_price_candidate_got_0"


def test_streaming_late_receipt_is_unresolved() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3, 4, 9)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 4,
            "close": [100.0, 49.0, 49.5, 50.0],
            "shares_outstanding": [10.0, 10.0, 10.0, 20.0],
            "market_cap": [1000.0, 490.0, 495.0, 1000.0],
        }
    )
    resolved = resolve_opendart_corporate_action_records(
        pages=[_bonus_page(_bonus_record(rcept_no="20240103000001", nstk_lstprd="2024-01-09"))],
        daily_market=daily,
        calendar=SessionCalendar(sessions),
    )
    assert resolved[0]["evidence_reason"] == "late_corporate_action_receipt"


def test_streaming_listing_before_effective_is_unresolved() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3, 4, 5)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 4,
            "close": [100.0, 49.0, 50.0, 51.0],
            "shares_outstanding": [10.0, 10.0, 10.0, 20.0],
            "market_cap": [1000.0, 490.0, 500.0, 1020.0],
        }
    )
    resolved = resolve_opendart_corporate_action_records(
        pages=[_bonus_page(_bonus_record(nstk_lstprd="2024-01-02"))],
        daily_market=daily,
        calendar=SessionCalendar(sessions),
    )
    assert resolved[0]["evidence_reason"] == "listing_before_effective_session"


def test_streaming_missing_listing_bar_raises() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3, 4)
    daily = pl.DataFrame(
        {
            "session": list(sessions[:2]),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 49.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 490.0],
        }
    )
    with pytest.raises(PITDataError, match=r"missing KRX bars"):
        resolve_opendart_corporate_action_records(
            pages=[_bonus_page(_bonus_record())],
            daily_market=daily,
            calendar=SessionCalendar(sessions),
        )


def test_streaming_invalid_listing_bars_raise() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3, 4)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 3,
            "close": [100.0, 49.0, 50.0],
            "shares_outstanding": [10.0, 10.0, None],
            "market_cap": [1000.0, 490.0, None],
        }
    )
    with pytest.raises(PITDataError, match=r"invalid KRX market value"):
        resolve_opendart_corporate_action_records(
            pages=[_bonus_page(_bonus_record())],
            daily_market=daily,
            calendar=SessionCalendar(sessions),
        )


def test_streaming_share_delta_mismatch_is_unresolved() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3, 4)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 3,
            "close": [100.0, 49.0, 50.0],
            "shares_outstanding": [10.0, 10.0, 20.0],
            "market_cap": [1000.0, 490.0, 1000.0],
        }
    )
    resolved = resolve_opendart_corporate_action_records(
        pages=[
            _bonus_page(
                _bonus_record(bfic_tisstk_ostk="5", nstk_ostk_cnt="5", nstk_lstprd="2024-01-04")
            )
        ],
        daily_market=daily,
        calendar=SessionCalendar(sessions),
    )
    assert resolved[0]["evidence_reason"] == "krx_listing_share_delta_mismatch"


def test_streaming_cr_page_is_unresolved_reverse_split() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2)
    daily = pl.DataFrame(
        {"session": list(sessions), "instrument_id": ["KRX:A"], "close": [100.0]}
    )
    page = {
        "endpoint": "crDecsn.json",
        "corp_code": "00123456",
        "status": "000",
        "records": [{"rcept_no": "20240101000001"}],
    }
    resolved = resolve_opendart_corporate_action_records(
        pages=[page], daily_market=daily, calendar=SessionCalendar(sessions)
    )
    assert resolved[0]["evidence_reason"] == "unresolved_reverse_split"
    assert resolved[0]["factor"] == 1.0


def test_streaming_event_date_selects_declared_session() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 6)
    daily = pl.DataFrame(
        {"session": list(sessions), "instrument_id": ["KRX:A"] * 2, "close": [100.0, 100.0]}
    )
    page = {
        "endpoint": "cmpMgDecsn.json",
        "corp_code": "00123456",
        "status": "000",
        "records": [{"rcept_no": "20240101000001", "event_date": "2024-01-05"}],
    }
    resolved = resolve_opendart_corporate_action_records(
        pages=[page], daily_market=daily, calendar=SessionCalendar(sessions)
    )
    assert resolved[0]["effective_session"] == sessions[1]


def test_streaming_garbage_event_date_falls_back_to_receipt() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2)
    daily = pl.DataFrame(
        {"session": list(sessions), "instrument_id": ["KRX:A"], "close": [100.0]}
    )
    page = {
        "endpoint": "cmpMgDecsn.json",
        "corp_code": "00123456",
        "status": "000",
        "records": [{"rcept_no": "20240101000001", "event_date": "not-a-date"}],
    }
    resolved = resolve_opendart_corporate_action_records(
        pages=[page], daily_market=daily, calendar=SessionCalendar(sessions)
    )
    assert resolved[0]["effective_session"] == sessions[0]


def test_streaming_ambiguous_mapping_raises() -> None:
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {"session": [sessions[0], sessions[0]], "instrument_id": ["KRX:A", "KRX:B"], "close": [100.0, 100.0]}
    )
    with pytest.raises(PITDataError, match=r"missing OpenDART corp_code mapping"):
        resolve_opendart_corporate_action_records(
            pages=[_bonus_page(_bonus_record())],
            daily_market=daily,
            calendar=SessionCalendar(sessions),
        )


def test_backtest_resolver_rejects_verified_without_effective_column() -> None:
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 101.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 1010.0],
        }
    )
    actions = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"],
            "action_id": ["v1"],
            "action_type": ["split"],
            "factor": [2.0],
            "cash_amount": [0.0],
            "available_at": [sessions[0]],
            "evidence_status": ["verified"],
            "evidence_reason": [None],
        }
    )
    with pytest.raises(PITDataError, match=r"lack effective session"):
        resolve_backtest_corporate_action_evidence(
            daily_market=daily,
            corporate_actions=actions,
            calendar=SessionCalendar(sessions),
            policy=BacktestMarketInputsPolicy(),
        )


def test_backtest_resolver_tolerates_missing_market_cap_column() -> None:
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 101.0],
            "shares_outstanding": [10.0, 10.0],
        }
    )
    actions = pl.DataFrame({"instrument_id": [], "action_id": []})
    resolution = resolve_backtest_corporate_action_evidence(
        daily_market=daily,
        corporate_actions=actions,
        calendar=SessionCalendar(sessions),
        policy=BacktestMarketInputsPolicy(),
    )
    assert resolution.excluded_instruments == frozenset()


def test_backtest_resolver_rejects_non_positive_close() -> None:
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [0.0, 50.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 500.0],
        }
    )
    actions = pl.DataFrame({"instrument_id": [], "action_id": []})
    with pytest.raises(PITDataError, match=r"invalid corporate-action market value"):
        resolve_backtest_corporate_action_evidence(
            daily_market=daily,
            corporate_actions=actions,
            calendar=SessionCalendar(sessions),
            policy=BacktestMarketInputsPolicy(),
        )


def test_backtest_resolver_excludes_unexplained_jump_without_evidence_columns() -> None:
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 40.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 400.0],
        }
    )
    actions = pl.DataFrame({"instrument_id": [], "action_id": []})
    resolution = resolve_backtest_corporate_action_evidence(
        daily_market=daily,
        corporate_actions=actions,
        calendar=SessionCalendar(sessions),
        policy=BacktestMarketInputsPolicy(),
    )
    assert resolution.excluded_instruments == frozenset({"KRX:A"})
    assert resolution.exclusion_reasons["KRX:A"] == ("unexplained_price_discontinuity",)


def test_backtest_resolver_keeps_verified_only_actions() -> None:
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 101.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 1010.0],
        }
    )
    actions = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"],
            "action_id": ["v1"],
            "action_type": ["split"],
            "effective_session": [sessions[1]],
            "factor": [2.0],
            "cash_amount": [0.0],
            "available_at": [sessions[0]],
            "evidence_status": ["verified"],
            "evidence_reason": [None],
        }
    )
    resolution = resolve_backtest_corporate_action_evidence(
        daily_market=daily,
        corporate_actions=actions,
        calendar=SessionCalendar(sessions),
        policy=BacktestMarketInputsPolicy(),
    )
    assert resolution.excluded_instruments == frozenset()
    assert resolution.verified_corporate_actions["action_id"].to_list() == ["v1"]


def test_backtest_validate_rejects_unresolved_evidence_status() -> None:
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 101.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 1010.0],
        }
    )
    actions = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"],
            "action_id": ["u1"],
            "action_type": ["split"],
            "effective_session": [sessions[1]],
            "factor": [2.0],
            "cash_amount": [0.0],
            "available_at": [sessions[0]],
            "evidence_status": ["unresolved"],
            "evidence_reason": ["unsupported_merger"],
        }
    )
    with pytest.raises(PITDataError, match=r"unresolved corporate-action evidence"):
        validate_corporate_action_coverage(
            daily_market=daily,
            corporate_actions=actions,
            calendar=SessionCalendar(sessions),
            decision_time_of=lambda value: value.replace(hour=15, minute=30),
            policy=BacktestMarketInputsPolicy(),
        )


def test_backtest_validate_rejects_unresolved_action_type() -> None:
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 101.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 1010.0],
        }
    )
    actions = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"],
            "action_id": ["u1"],
            "action_type": ["unresolved"],
            "effective_session": [sessions[1]],
            "factor": [1.0],
            "cash_amount": [0.0],
            "available_at": [sessions[0]],
            "evidence_status": ["unresolved"],
            "evidence_reason": ["unsupported_merger"],
        }
    )
    with pytest.raises(PITDataError, match=r"unresolved corporate-action evidence"):
        validate_corporate_action_coverage(
            daily_market=daily,
            corporate_actions=actions,
            calendar=SessionCalendar(sessions),
            decision_time_of=lambda value: value.replace(hour=15, minute=30),
            policy=BacktestMarketInputsPolicy(),
        )


def test_backtest_validate_rejects_unresolved_type_even_if_marked_verified() -> None:
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {"session": list(sessions), "instrument_id": ["KRX:A"] * 2, "close": [100.0, 101.0]}
    )
    actions = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"], "action_id": ["u1"], "action_type": ["unresolved"],
            "effective_session": [sessions[1]], "available_at": [sessions[0]],
            "evidence_status": ["verified"], "evidence_reason": [None],
        }
    )
    with pytest.raises(PITDataError, match=r"unresolved corporate-action evidence"):
        validate_corporate_action_coverage(
            daily_market=daily, corporate_actions=actions, calendar=SessionCalendar(sessions),
            decision_time_of=lambda value: value.replace(hour=15, minute=30),
            policy=BacktestMarketInputsPolicy(),
        )


def test_normalize_rejects_invalid_evidence_status() -> None:
    from src.data.normalization import normalize_corporate_action_records
    from src.data.schemas import PITDataError

    session = datetime(2024, 1, 2, 9, tzinfo=_KRX)
    with pytest.raises(PITDataError, match=r"requires rebuild from raw OpenDART Bronze"):
        normalize_corporate_action_records(
            action_records=[
                {
                    "instrument_id": "KRX:A",
                    "action_id": "x",
                    "type": "split",
                    "effective_session": session,
                    "factor": 2.0,
                    "cash_amount": 0.0,
                    "available_at": session,
                    "evidence_status": "pending",
                    "evidence_reason": None,
                }
            ],
            calendar_sessions=(session,),
            corporate_action_available_at=session,
            corporate_action_source_hash="h",
        )


def test_normalize_rejects_verified_with_reason() -> None:
    from src.data.normalization import normalize_corporate_action_records
    from src.data.schemas import PITDataError

    session = datetime(2024, 1, 2, 9, tzinfo=_KRX)
    with pytest.raises(PITDataError, match=r"reason must be null"):
        normalize_corporate_action_records(
            action_records=[
                {
                    "instrument_id": "KRX:A",
                    "action_id": "x",
                    "type": "split",
                    "effective_session": session,
                    "factor": 2.0,
                    "cash_amount": 0.0,
                    "available_at": session,
                    "evidence_status": "verified",
                    "evidence_reason": "stale",
                }
            ],
            calendar_sessions=(session,),
            corporate_action_available_at=session,
            corporate_action_source_hash="h",
        )


def test_normalize_rejects_unresolved_without_reason() -> None:
    from src.data.normalization import normalize_corporate_action_records
    from src.data.schemas import PITDataError

    session = datetime(2024, 1, 2, 9, tzinfo=_KRX)
    with pytest.raises(PITDataError, match=r"reason missing"):
        normalize_corporate_action_records(
            action_records=[
                {
                    "instrument_id": "KRX:A",
                    "action_id": "x",
                    "type": "unresolved",
                    "effective_session": session,
                    "factor": 1.0,
                    "cash_amount": 0.0,
                    "available_at": session,
                    "evidence_status": "unresolved",
                    "evidence_reason": "",
                }
            ],
            calendar_sessions=(session,),
            corporate_action_available_at=session,
            corporate_action_source_hash="h",
        )


def _corporate_actions_frame(**overrides: object) -> pl.DataFrame:
    from datetime import UTC

    session = datetime(2024, 1, 2, 9, tzinfo=UTC)
    row: dict[str, object] = {
        "instrument_id": "KRX:A",
        "effective_date": session,
        "coverage_end": session,
        "action_id": "a1",
        "type": "split",
        "factor": 2.0,
        "cash_amount": 0.0,
        "source": "KRX",
        "available_at": session,
        "source_hash": "h",
        "share_listing_date": None,
        "share_delta": None,
        "evidence_status": "verified",
        "evidence_reason": None,
    }
    row.update(overrides)
    return pl.DataFrame([row])


def test_silver_rejects_invalid_evidence_status() -> None:
    from datetime import UTC, datetime

    from src.data.schemas import SilverTable
    from src.data.silver import validate_table

    with pytest.raises(PITDataError, match=r"invalid evidence status"):
        validate_table(
            SilverTable.CORPORATE_ACTIONS,
            _corporate_actions_frame(evidence_status="pending"),
            decision_time=datetime(2024, 1, 3, 9, tzinfo=UTC),
        )


def test_silver_retains_unresolved_exclusion_evidence() -> None:
    from datetime import UTC, datetime

    from src.data.schemas import SilverTable
    from src.data.silver import validate_table

    validate_table(
        SilverTable.CORPORATE_ACTIONS,
        _corporate_actions_frame(
            type="unresolved", evidence_status="unresolved", evidence_reason="unsupported_merger"
        ),
        decision_time=datetime(2024, 1, 3, 9, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("overrides", "pattern"),
    [
        ({"type": "no_action"}, r"legacy no_action"),
        ({"type": "split", "evidence_status": "unresolved", "evidence_reason": "bad"}, r"status/type pairing"),
        ({"type": "unresolved", "evidence_status": "unresolved", "evidence_reason": ""}, r"reason missing"),
    ],
)
def test_silver_rejects_invalid_evidence_pairings(overrides, pattern) -> None:
    from datetime import UTC, datetime

    from src.data.schemas import SilverTable
    from src.data.silver import validate_table

    with pytest.raises(PITDataError, match=pattern):
        validate_table(
            SilverTable.CORPORATE_ACTIONS,
            _corporate_actions_frame(**overrides),
            decision_time=datetime(2024, 1, 3, 9, tzinfo=UTC),
        )


def test_silver_rejects_verified_with_reason() -> None:
    from datetime import UTC, datetime

    from src.data.schemas import SilverTable
    from src.data.silver import validate_table

    with pytest.raises(PITDataError, match=r"reason must be null"):
        validate_table(
            SilverTable.CORPORATE_ACTIONS,
            _corporate_actions_frame(evidence_reason="stale"),
            decision_time=datetime(2024, 1, 3, 9, tzinfo=UTC),
        )


def test_find_jumps_matches_effective_date_key() -> None:
    from src.data.backtest_sessions import find_unexplained_price_discontinuities

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 40.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 400.0],
        }
    )
    verified = pl.DataFrame({"instrument_id": ["KRX:A"], "effective_date": [sessions[1]]})
    jumps = find_unexplained_price_discontinuities(daily_market=daily, verified_actions=verified, threshold=0.5)
    assert jumps.height == 0


def test_find_jumps_rejects_keyless_verified_frame() -> None:
    from src.data.backtest_sessions import find_unexplained_price_discontinuities

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 40.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 400.0],
        }
    )
    verified = pl.DataFrame({"instrument_id": ["KRX:A"]})
    with pytest.raises(PITDataError, match=r"lack effective session"):
        find_unexplained_price_discontinuities(daily_market=daily, verified_actions=verified, threshold=0.5)


def test_find_jumps_without_verified_returns_all_jumps() -> None:
    from src.data.backtest_sessions import find_unexplained_price_discontinuities

    sessions = _sessions(2, 3)
    daily = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 40.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 400.0],
        }
    )
    jumps = find_unexplained_price_discontinuities(
        daily_market=daily, verified_actions=pl.DataFrame({"instrument_id": []}), threshold=0.5
    )
    assert jumps.to_dicts() == [{"instrument_id": "KRX:A", "session": sessions[1]}]


def test_mapped_action_instruments_rejects_non_dict_record() -> None:
    from src.data.streaming_normalization import mapped_action_instruments

    with pytest.raises(PITDataError, match=r"invalid OpenDART record"):
        mapped_action_instruments(pages=[{"endpoint": "fricDecsn.json", "records": [42]}])


def test_load_structured_pages_reads_payload() -> None:
    from src.data.streaming_normalization import load_structured_corporate_action_pages

    assert load_structured_corporate_action_pages(action_receipts=()) == []


def test_load_structured_pages_rejects_missing_payload(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.schemas import BronzeReceipt, EvidenceKind
    from src.data.streaming_normalization import load_structured_corporate_action_pages

    decision = datetime(2024, 1, 2, 9, tzinfo=UTC)
    receipt = BronzeReceipt(
        kind=EvidenceKind.CORPORATE_ACTIONS,
        content_hash="h" * 64,
        source_path="fixture",
        retrieved_at=decision,
        ingested_at=decision,
        payload_path=tmp_path / "absent.json",
        metadata_path=tmp_path / "absent.receipt.json",
    )
    with pytest.raises(PITDataError, match=r"missing corporate-action payload"):
        load_structured_corporate_action_pages(action_receipts=(receipt,))


def test_load_structured_pages_rejects_non_dict_payload(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.schemas import BronzeReceipt, EvidenceKind
    from src.data.streaming_normalization import load_structured_corporate_action_pages

    decision = datetime(2024, 1, 2, 9, tzinfo=UTC)
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("[]", encoding="utf-8")
    receipt = BronzeReceipt(
        kind=EvidenceKind.CORPORATE_ACTIONS,
        content_hash="h" * 64,
        source_path="fixture",
        retrieved_at=decision,
        ingested_at=decision,
        payload_path=payload_path,
        metadata_path=tmp_path / "receipt.json",
    )
    with pytest.raises(PITDataError, match=r"invalid corporate-action payload"):
        load_structured_corporate_action_pages(action_receipts=(receipt,))


def _refresh_daily() -> pl.LazyFrame:
    sessions = _sessions(2, 3)
    return pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:A"] * 2,
            "close": [100.0, 101.0],
            "shares_outstanding": [10.0, 10.0],
            "market_cap": [1000.0, 1010.0],
        }
    ).lazy()


def _refresh_calendar() -> object:
    from src.core.time import SessionCalendar

    return SessionCalendar(_sessions(2, 3))


def test_refresh_rejects_naive_decision_time(tmp_path) -> None:
    from datetime import datetime

    import src.data.streaming_normalization as module

    with pytest.raises(PITDataError, match=r"timezone-aware"):
        module.refresh_corporate_action_silver(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2024, 1, 2, 9),
            daily_market=_refresh_daily(),
            calendar=_refresh_calendar(),
        )


def test_refresh_rejects_empty_calendar(tmp_path) -> None:
    from datetime import UTC, datetime

    import src.data.streaming_normalization as module
    from src.core.time import SessionCalendar

    with pytest.raises(PITDataError, match=r"certified KRX calendar"):
        module.refresh_corporate_action_silver(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2024, 1, 2, 9, tzinfo=UTC),
            daily_market=_refresh_daily(),
            calendar=SessionCalendar(()),
        )


def test_refresh_rejects_eager_daily_frame(tmp_path) -> None:
    from datetime import UTC, datetime

    import src.data.streaming_normalization as module

    with pytest.raises(PITDataError, match=r"lazy daily-market scan"):
        module.refresh_corporate_action_silver(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2024, 1, 2, 9, tzinfo=UTC),
            daily_market=_refresh_daily().collect(),
            calendar=_refresh_calendar(),
        )


def test_refresh_rejects_incomplete_scan_columns(tmp_path) -> None:
    from datetime import UTC, datetime

    import src.data.streaming_normalization as module

    sessions = _sessions(2, 3)
    narrow = pl.DataFrame(
        {"session": list(sessions), "instrument_id": ["KRX:A"] * 2, "close": [100.0, 101.0]}
    ).lazy()
    with pytest.raises(PITDataError, match=r"missing columns"):
        module.refresh_corporate_action_silver(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2024, 1, 2, 9, tzinfo=UTC),
            daily_market=narrow,
            calendar=_refresh_calendar(),
        )


def test_refresh_rejects_empty_structured_input(tmp_path) -> None:
    from datetime import UTC, datetime

    import src.data.streaming_normalization as module

    with pytest.raises(PITDataError, match=r"no structured pages"):
        module.refresh_corporate_action_silver(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2024, 1, 2, 9, tzinfo=UTC),
            daily_market=_refresh_daily(),
            calendar=_refresh_calendar(),
        )


def test_refresh_rejects_unmapped_pages(monkeypatch, tmp_path) -> None:
    from datetime import UTC, datetime

    import src.data.streaming_normalization as module

    monkeypatch.setattr(
        module,
        "load_structured_corporate_action_pages",
        lambda **_: [{"endpoint": "fricDecsn.json", "corp_code": "9", "records": []}],
    )
    with pytest.raises(PITDataError, match=r"unmapped corporate-action page"):
        module.refresh_corporate_action_silver(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2024, 1, 2, 9, tzinfo=UTC),
            daily_market=_refresh_daily(),
            calendar=_refresh_calendar(),
        )


def test_refresh_rejects_preview_without_mapped_bars(monkeypatch, tmp_path) -> None:
    from datetime import UTC, datetime

    import src.data.streaming_normalization as module

    monkeypatch.setattr(
        module,
        "load_structured_corporate_action_pages",
        lambda **_: [
            {
                "endpoint": "fricDecsn.json",
                "corp_code": "9",
                "status": "000",
                "requested_instrument_id": "KRX:Z",
                "instrument_mapping_provenance": "opendart_corp_code_direct",
                "records": [],
            }
        ],
    )
    with pytest.raises(PITDataError, match=r"no mapped bars"):
        module.refresh_corporate_action_silver(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2024, 1, 2, 9, tzinfo=UTC),
            daily_market=_refresh_daily(),
            calendar=_refresh_calendar(),
        )


def test_refresh_rejects_rows_without_evidence(monkeypatch, tmp_path) -> None:
    from datetime import UTC, datetime

    import src.data.streaming_normalization as module

    monkeypatch.setattr(
        module,
        "load_structured_corporate_action_pages",
        lambda **_: [
            {
                "endpoint": "fricDecsn.json",
                "corp_code": "9",
                "status": "000",
                "requested_instrument_id": "KRX:A",
                "instrument_mapping_provenance": "opendart_corp_code_direct",
                "records": [],
            }
        ],
    )
    monkeypatch.setattr(
        module, "resolve_opendart_corporate_action_records", lambda **_: [{"instrument_id": "KRX:A"}]
    )
    with pytest.raises(PITDataError, match=r"lacks evidence_status"):
        module.refresh_corporate_action_silver(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2024, 1, 2, 9, tzinfo=UTC),
            daily_market=_refresh_daily(),
            calendar=_refresh_calendar(),
        )


def _materialized_silver_root(tmp_path) -> object:
    from datetime import UTC, datetime

    from src.data.silver import SilverStore, complete_minimal_fixture

    decision = datetime(2026, 9, 6, tzinfo=UTC)
    tables, _receipts, report = complete_minimal_fixture(decision_time=decision)
    silver_root = tmp_path / "silver"
    SilverStore(silver_root).materialize_all(tables, report=report, decision_time=decision)
    return silver_root, decision


def test_market_scan_returns_lazy_projection(tmp_path) -> None:
    import polars as pl

    from src.data.silver import load_latest_silver_market_scan

    silver_root, decision = _materialized_silver_root(tmp_path)
    scan = load_latest_silver_market_scan(
        root=silver_root,
        decision_time=decision,
        columns=("session", "instrument_id", "close", "shares_outstanding", "market_cap"),
    )
    assert isinstance(scan, pl.LazyFrame)
    frame = scan.collect()
    assert frame.columns == ["session", "instrument_id", "close", "shares_outstanding", "market_cap"]
    assert frame.height == 1


def test_market_scan_rejects_naive_decision_time(tmp_path) -> None:
    from datetime import datetime

    from src.data.silver import load_latest_silver_market_scan

    with pytest.raises(PITDataError, match=r"timezone-aware"):
        load_latest_silver_market_scan(
            root=tmp_path / "silver",
            decision_time=datetime(2024, 1, 2, 9),
            columns=("session", "instrument_id", "close", "shares_outstanding", "market_cap"),
        )


def test_market_scan_rejects_wrong_projection(tmp_path) -> None:
    from src.data.silver import load_latest_silver_market_scan

    silver_root, decision = _materialized_silver_root(tmp_path)
    with pytest.raises(PITDataError, match=r"exactly"):
        load_latest_silver_market_scan(
            root=silver_root, decision_time=decision, columns=("session", "instrument_id")
        )


def test_market_scan_rejects_corrupt_dataset(tmp_path) -> None:
    import shutil

    from src.data.silver import load_latest_silver_market_scan

    silver_root, decision = _materialized_silver_root(tmp_path)
    for dataset in (silver_root / "daily_market").iterdir():
        if dataset.is_dir():
            shutil.rmtree(dataset / "partitions", ignore_errors=True)
    with pytest.raises(PITDataError, match=r"invalid certified Silver table"):
        load_latest_silver_market_scan(
            root=silver_root,
            decision_time=decision,
            columns=("session", "instrument_id", "close", "shares_outstanding", "market_cap"),
        )


def test_cli_refresh_reports_failure(monkeypatch, tmp_path, capsys) -> None:
    from datetime import UTC, datetime

    import src.data.cli as cli

    decision = datetime(2026, 9, 9, tzinfo=UTC)
    monkeypatch.setattr(cli, "_parse_dt", lambda _value: decision)
    monkeypatch.setattr(
        cli, "load_latest_silver_table", lambda **kwargs: (_ for _ in ()).throw(ValueError("no calendar"))
    )
    code = cli.main(
        [
            "refresh-corporate-actions",
            "--bronze-root",
            str(tmp_path / "bronze"),
            "--silver-root",
            str(tmp_path / "silver"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--decision-time",
            decision.isoformat(),
        ]
    )
    assert code == 1
