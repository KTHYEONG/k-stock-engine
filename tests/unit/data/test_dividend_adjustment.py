"""Cash-dividend adjustment factor unit tests."""


def test_parse_cash_dividend_records_extracts_common_stock_row_only() -> None:
    from src.data.dividend_adjustment import parse_cash_dividend_records

    raw_records = [
        {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
        {"se": "주당 현금배당금(원)", "stock_knd": "우선주", "thstrm": "1,445", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
        {"se": "주당액면가액(원)", "stock_knd": "-", "thstrm": "100", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
    ]

    result = parse_cash_dividend_records(raw_records, corp_code="00126380")

    assert len(result) == 1
    assert result[0].cash_per_share == 1444.0
    assert result[0].stlm_dt.isoformat() == "2022-12-31"
    assert result[0].rcept_no == "20230307000542"
    assert result[0].corp_code == "00126380"


def test_parse_cash_dividend_records_skips_no_dividend_marker() -> None:
    from src.data.dividend_adjustment import parse_cash_dividend_records

    raw_records = [
        {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "-", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
    ]

    result = parse_cash_dividend_records(raw_records, corp_code="00126380")

    assert result == ()


def test_parse_cash_dividend_records_rejects_malformed_thstrm() -> None:
    import pytest

    from src.data.dividend_adjustment import parse_cash_dividend_records
    from src.data.schemas import PITDataError

    raw_records = [
        {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "abc", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
    ]

    with pytest.raises(PITDataError, match="thstrm"):
        parse_cash_dividend_records(raw_records, corp_code="00126380")


def test_parse_cash_dividend_records_rejects_malformed_stlm_dt() -> None:
    import pytest

    from src.data.dividend_adjustment import parse_cash_dividend_records
    from src.data.schemas import PITDataError

    raw_records = [
        {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444", "stlm_dt": "not-a-date", "rcept_no": "20230307000542"},
    ]

    with pytest.raises(PITDataError, match="stlm_dt"):
        parse_cash_dividend_records(raw_records, corp_code="00126380")


def test_parse_cash_dividend_records_rejects_malformed_rcept_no() -> None:
    import pytest

    from src.data.dividend_adjustment import parse_cash_dividend_records
    from src.data.schemas import PITDataError

    raw_records = [
        {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444", "stlm_dt": "2022-12-31", "rcept_no": "abc"},
    ]

    with pytest.raises(PITDataError, match="rcept_no"):
        parse_cash_dividend_records(raw_records, corp_code="00126380")


def test_filing_date_from_rcept_no_extracts_leading_date() -> None:
    from datetime import date

    from src.data.dividend_adjustment import filing_date_from_rcept_no

    assert filing_date_from_rcept_no("20230307000542") == date(2023, 3, 7)


def test_filing_date_from_rcept_no_rejects_short_value() -> None:
    import pytest

    from src.data.dividend_adjustment import filing_date_from_rcept_no
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="rcept_no"):
        filing_date_from_rcept_no("2023")


def test_resolve_ex_dividend_session_picks_last_session_on_or_before_stlm_dt() -> None:
    from datetime import date, datetime

    from src.core.time import KRX_TZ
    from src.data.dividend_adjustment import resolve_ex_dividend_session

    sessions = (
        datetime(2022, 12, 28, 9, tzinfo=KRX_TZ),
        datetime(2022, 12, 29, 9, tzinfo=KRX_TZ),
        datetime(2022, 12, 30, 9, tzinfo=KRX_TZ),
        datetime(2023, 1, 2, 9, tzinfo=KRX_TZ),
    )

    result = resolve_ex_dividend_session(date(2022, 12, 31), sessions=sessions)

    assert result == datetime(2022, 12, 30, 9, tzinfo=KRX_TZ)


def test_resolve_ex_dividend_session_rejects_missing_prior_coverage() -> None:
    from datetime import date, datetime

    import pytest

    from src.core.time import KRX_TZ
    from src.data.dividend_adjustment import resolve_ex_dividend_session
    from src.data.schemas import PITDataError

    sessions = (datetime(2023, 1, 2, 9, tzinfo=KRX_TZ),)

    with pytest.raises(PITDataError, match="stlm_dt"):
        resolve_ex_dividend_session(date(2022, 12, 31), sessions=sessions)


def test_compute_cash_dividend_factor_matches_expected_ratio() -> None:
    from src.data.dividend_adjustment import compute_cash_dividend_factor

    factor = compute_cash_dividend_factor(close_before_ex=70300.0, cash_per_share=1444.0)

    assert factor == (70300.0 - 1444.0) / 70300.0


def test_compute_cash_dividend_factor_rejects_non_positive_close() -> None:
    import pytest

    from src.data.dividend_adjustment import compute_cash_dividend_factor
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="close_before_ex"):
        compute_cash_dividend_factor(close_before_ex=0.0, cash_per_share=100.0)


def test_compute_cash_dividend_factor_rejects_negative_cash_per_share() -> None:
    import pytest

    from src.data.dividend_adjustment import compute_cash_dividend_factor
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="cash_per_share"):
        compute_cash_dividend_factor(close_before_ex=100.0, cash_per_share=-1.0)


def test_compute_cash_dividend_factor_rejects_dividend_exceeding_close() -> None:
    import pytest

    from src.data.dividend_adjustment import compute_cash_dividend_factor
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="certification blocked"):
        compute_cash_dividend_factor(close_before_ex=100.0, cash_per_share=150.0)


def test_build_cash_dividend_corporate_action_records_end_to_end() -> None:
    from datetime import datetime, timedelta

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.dividend_adjustment import build_cash_dividend_corporate_action_records

    raw_records = [
        {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
        {"se": "주당 현금배당금(원)", "stock_knd": "우선주", "thstrm": "1,445", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
    ]
    start = datetime(2022, 12, 1, 9, tzinfo=KRX_TZ)
    sessions = tuple(start + timedelta(days=i) for i in range(40))
    daily_market = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:005930"] * len(sessions),
            "close": [70000.0 + i * 10 for i in range(len(sessions))],
        }
    )

    records = build_cash_dividend_corporate_action_records(
        raw_records=raw_records,
        corp_code="00126380",
        instrument_id="KRX:005930",
        sessions=sessions,
        daily_market=daily_market,
    )

    assert len(records) == 1
    row = records[0]
    assert row["instrument_id"] == "KRX:005930"
    assert row["type"] == "dividend"
    assert row["cash_amount"] == 1444.0
    assert row["evidence_status"] == "verified"
    assert row["evidence_reason"] is None
    assert 0.0 < row["factor"] < 1.0
    assert row["effective_date"].astimezone(KRX_TZ).date().isoformat() == "2022-12-31"
    assert row["available_at"].astimezone(KRX_TZ).date().isoformat() == "2023-03-07"


def test_build_cash_dividend_corporate_action_records_returns_empty_without_dividend() -> None:
    from datetime import datetime, timedelta

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.dividend_adjustment import build_cash_dividend_corporate_action_records

    raw_records = [
        {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "-", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
    ]
    start = datetime(2022, 12, 1, 9, tzinfo=KRX_TZ)
    sessions = tuple(start + timedelta(days=i) for i in range(10))
    daily_market = pl.DataFrame({"session": list(sessions), "instrument_id": ["KRX:005930"] * len(sessions), "close": [100.0] * len(sessions)})

    records = build_cash_dividend_corporate_action_records(
        raw_records=raw_records, corp_code="00126380", instrument_id="KRX:005930", sessions=sessions, daily_market=daily_market,
    )

    assert records == []


def test_build_cash_dividend_corporate_action_records_rejects_missing_close() -> None:
    from datetime import datetime, timedelta

    import polars as pl
    import pytest

    from src.core.time import KRX_TZ
    from src.data.dividend_adjustment import build_cash_dividend_corporate_action_records
    from src.data.schemas import PITDataError

    raw_records = [
        {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
    ]
    start = datetime(2022, 12, 1, 9, tzinfo=KRX_TZ)
    sessions = tuple(start + timedelta(days=i) for i in range(40))
    daily_market = pl.DataFrame({"session": [], "instrument_id": [], "close": []}, schema={"session": pl.Datetime(time_zone="Asia/Seoul"), "instrument_id": pl.String, "close": pl.Float64})

    with pytest.raises(PITDataError, match="missing daily close"):
        build_cash_dividend_corporate_action_records(
            raw_records=raw_records, corp_code="00126380", instrument_id="KRX:005930", sessions=sessions, daily_market=daily_market,
        )


def test_build_cash_dividend_corporate_action_records_output_feeds_normalize_corporate_action_records() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.dividend_adjustment import build_cash_dividend_corporate_action_records
    from src.data.normalization import normalize_corporate_action_records

    raw_records = [
        {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
    ]
    start = datetime(2022, 12, 1, 9, tzinfo=KRX_TZ)
    sessions = tuple(start + timedelta(days=i) for i in range(40))
    daily_market = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:005930"] * len(sessions),
            "close": [70000.0 + i * 10 for i in range(len(sessions))],
        }
    )

    records = build_cash_dividend_corporate_action_records(
        raw_records=raw_records, corp_code="00126380", instrument_id="KRX:005930", sessions=sessions, daily_market=daily_market,
    )

    frame = normalize_corporate_action_records(
        action_records=records,
        calendar_sessions=sessions,
        corporate_action_available_at=datetime(2023, 3, 7, tzinfo=UTC),
        corporate_action_source_hash="h" * 64,
    )

    assert frame.height == 1
    assert frame["type"].to_list() == ["dividend"]
    assert frame["evidence_status"].to_list() == ["verified"]


def test_load_dividend_corporate_action_pages_reads_only_dividend_prefixed_receipts(tmp_path) -> None:
    from datetime import UTC, datetime
    import json

    from src.data.dividend_adjustment import load_dividend_corporate_action_pages
    from src.data.schemas import BronzeReceipt, EvidenceKind

    decision = datetime(2024, 1, 2, 9, tzinfo=UTC)
    dividend_payload = tmp_path / "dividend.json"
    dividend_payload.write_text(json.dumps({"corp_code": "00126380", "records": []}), encoding="utf-8")
    other_payload = tmp_path / "other.json"
    other_payload.write_text(json.dumps({"endpoint": "fricDecsn.json"}), encoding="utf-8")

    dividend_receipt = BronzeReceipt(
        kind=EvidenceKind.CORPORATE_ACTIONS, content_hash="a" * 64,
        source_path="opendart_dividend:alotMatter.json:00126380:2022:11011",
        retrieved_at=decision, ingested_at=decision,
        payload_path=dividend_payload, metadata_path=tmp_path / "a.receipt.json",
    )
    other_receipt = BronzeReceipt(
        kind=EvidenceKind.CORPORATE_ACTIONS, content_hash="b" * 64,
        source_path="opendart_structured_decisions:fricDecsn.json:00126380",
        retrieved_at=decision, ingested_at=decision,
        payload_path=other_payload, metadata_path=tmp_path / "b.receipt.json",
    )

    pages = load_dividend_corporate_action_pages(action_receipts=(dividend_receipt, other_receipt))

    assert len(pages) == 1
    assert pages[0]["corp_code"] == "00126380"


def test_load_dividend_corporate_action_pages_rejects_missing_payload(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.dividend_adjustment import load_dividend_corporate_action_pages
    from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError

    decision = datetime(2024, 1, 2, 9, tzinfo=UTC)
    receipt = BronzeReceipt(
        kind=EvidenceKind.CORPORATE_ACTIONS, content_hash="a" * 64,
        source_path="opendart_dividend:alotMatter.json:00126380:2022:11011",
        retrieved_at=decision, ingested_at=decision,
        payload_path=tmp_path / "absent.json", metadata_path=tmp_path / "absent.receipt.json",
    )

    with pytest.raises(PITDataError, match="missing dividend-disclosure payload"):
        load_dividend_corporate_action_pages(action_receipts=(receipt,))


def test_resolve_dividend_corporate_action_records_uses_direct_mapping() -> None:
    from datetime import datetime, timedelta

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.dividend_adjustment import resolve_dividend_corporate_action_records

    start = datetime(2022, 12, 1, 9, tzinfo=KRX_TZ)
    sessions = tuple(start + timedelta(days=i) for i in range(40))
    daily_market = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:005930"] * len(sessions),
            "close": [70000.0 + i * 10 for i in range(len(sessions))],
        }
    )
    page = {
        "corp_code": "00126380",
        "requested_instrument_id": "KRX:005930",
        "instrument_mapping_provenance": "opendart_corp_code_direct",
        "records": [
            {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
        ],
    }

    records = resolve_dividend_corporate_action_records(pages=[page], daily_market=daily_market, sessions=sessions)

    assert len(records) == 1
    assert records[0]["instrument_id"] == "KRX:005930"
    assert records[0]["type"] == "dividend"


def test_resolve_dividend_corporate_action_records_rejects_ambiguous_mapping() -> None:
    from datetime import datetime, timedelta

    import polars as pl
    import pytest

    from src.core.time import KRX_TZ
    from src.data.dividend_adjustment import resolve_dividend_corporate_action_records
    from src.data.schemas import PITDataError

    start = datetime(2022, 12, 1, 9, tzinfo=KRX_TZ)
    sessions = tuple(start + timedelta(days=i) for i in range(5))
    daily_market = pl.DataFrame(
        {
            "session": [sessions[0], sessions[0]],
            "instrument_id": ["KRX:005930", "KRX:000660"],
            "close": [70000.0, 80000.0],
        }
    )
    page = {"corp_code": "00126380", "records": []}

    with pytest.raises(PITDataError, match="missing OpenDART corp_code mapping"):
        resolve_dividend_corporate_action_records(pages=[page], daily_market=daily_market, sessions=sessions)


def test_resolve_dividend_corporate_action_records_falls_back_to_single_instrument() -> None:
    from datetime import datetime, timedelta

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.dividend_adjustment import resolve_dividend_corporate_action_records

    start = datetime(2022, 12, 1, 9, tzinfo=KRX_TZ)
    sessions = tuple(start + timedelta(days=i) for i in range(40))
    daily_market = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:005930"] * len(sessions),
            "close": [70000.0 + i * 10 for i in range(len(sessions))],
        }
    )
    page = {
        "corp_code": "00126380",
        "records": [
            {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
        ],
    }

    records = resolve_dividend_corporate_action_records(pages=[page], daily_market=daily_market, sessions=sessions)

    assert len(records) == 1
    assert records[0]["instrument_id"] == "KRX:005930"


def test_resolve_dividend_corporate_action_records_skips_page_without_corp_code() -> None:
    from datetime import datetime, timedelta

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.dividend_adjustment import resolve_dividend_corporate_action_records

    start = datetime(2022, 12, 1, 9, tzinfo=KRX_TZ)
    sessions = tuple(start + timedelta(days=i) for i in range(5))
    daily_market = pl.DataFrame({"session": [sessions[0]], "instrument_id": ["KRX:005930"], "close": [70000.0]})
    page = {"corp_code": "", "records": []}

    records = resolve_dividend_corporate_action_records(pages=[page], daily_market=daily_market, sessions=sessions)

    assert records == []


def test_load_dividend_corporate_action_pages_rejects_non_dict_payload(tmp_path) -> None:
    from datetime import UTC, datetime
    import json

    import pytest

    from src.data.dividend_adjustment import load_dividend_corporate_action_pages
    from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError

    decision = datetime(2024, 1, 2, 9, tzinfo=UTC)
    payload_path = tmp_path / "list_payload.json"
    payload_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    receipt = BronzeReceipt(
        kind=EvidenceKind.CORPORATE_ACTIONS, content_hash="a" * 64,
        source_path="opendart_dividend:alotMatter.json:00126380:2022:11011",
        retrieved_at=decision, ingested_at=decision,
        payload_path=payload_path, metadata_path=tmp_path / "a.receipt.json",
    )

    with pytest.raises(PITDataError, match="invalid dividend-disclosure payload"):
        load_dividend_corporate_action_pages(action_receipts=(receipt,))


def test_parse_cash_dividend_records_skips_non_positive_cash_per_share() -> None:
    from src.data.dividend_adjustment import parse_cash_dividend_records

    raw_records = [
        {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "-100", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
    ]

    result = parse_cash_dividend_records(raw_records, corp_code="00126380")

    assert result == ()


def test_filing_date_from_rcept_no_rejects_invalid_calendar_date() -> None:
    import pytest

    from src.data.dividend_adjustment import filing_date_from_rcept_no
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="rcept_no"):
        filing_date_from_rcept_no("20231307000542")


def test_build_cash_dividend_corporate_action_records_rejects_missing_prior_session() -> None:
    from datetime import datetime, timedelta

    import polars as pl
    import pytest

    from src.core.time import KRX_TZ
    from src.data.dividend_adjustment import build_cash_dividend_corporate_action_records
    from src.data.schemas import PITDataError

    start = datetime(2022, 12, 31, 9, tzinfo=KRX_TZ)
    sessions = tuple(start + timedelta(days=i) for i in range(5))
    daily_market = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:005930"] * len(sessions),
            "close": [70000.0] * len(sessions),
        }
    )
    raw_records = [
        {"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444", "stlm_dt": "2022-12-31", "rcept_no": "20230307000542"},
    ]

    with pytest.raises(PITDataError, match="no prior certified session"):
        build_cash_dividend_corporate_action_records(
            raw_records=raw_records, corp_code="00126380", instrument_id="KRX:005930", sessions=sessions, daily_market=daily_market,
        )
