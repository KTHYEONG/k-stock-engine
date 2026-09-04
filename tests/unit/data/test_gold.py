"""Unit tests for src/data/gold.py — Gold-layer pre-flight checks."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from src.core.time import KRX_TZ, SessionCalendar
from src.data.gold import (
    WARMUP_SESSIONS,
    BarExclusionReason,
    DartExclusionReason,
    GoldAuditManifest,
    audit_bar_continuity,
    audit_dart_fact_eligibility,
    build_gold_audit_manifest,
    check_warmup_sessions,
    exclude_sentinel_corporate_actions,
    materialize_gold_window,
    write_gold_audit_artifact,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

def _session(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 9, 0, tzinfo=KRX_TZ)


def _make_calendar(start: date, n: int) -> SessionCalendar:
    """Build a simple calendar: n sessions starting at start (weekday-agnostic)."""
    from datetime import timedelta
    sessions = []
    cur = start
    while len(sessions) < n:
        sessions.append(_session(cur))
        cur += timedelta(days=1)
    return SessionCalendar(tuple(sessions))


def _make_daily_market(instrument_ids: list[str], sessions: list[datetime]) -> pl.DataFrame:
    rows = [
        {
            "session": s,
            "instrument_id": iid,
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
            "trading_value": 100_000.0,
            "market_cap": 1_000_000.0,
            "shares_outstanding": 10_000.0,
            "available_at": s,
            "source_hash": "abc",
        }
        for iid in instrument_ids
        for s in sessions
    ]
    return pl.DataFrame(rows)


def _make_financial_facts(
    company_ids: list[str],
    periods: list[str],
    facts: list[str],
    available_at: datetime,
) -> pl.DataFrame:
    rows = [
        {
            "company_id": cid,
            "fiscal_period": period,
            "filing_id": f"{cid}_{period}",
            "fact": fact,
            "published_at": available_at,
            "available_at": available_at,
            "value": 1_000_000.0,
            "unit": "KRW",
            "consolidated": True,
            "restatement_id": "",
            "source_hash": "abc",
            "source_kind": "opendart_standard",
            "mapping_version": "dart-fact-map-v1",
            "raw_document_hash": None,
        }
        for cid in company_ids
        for period in periods
        for fact in facts
    ]
    return pl.DataFrame(rows)


def _make_corporate_actions(instrument_ids: list[str], action_type: str = "no_action") -> pl.DataFrame:
    rows = [
        {
            "instrument_id": iid,
            "effective_date": datetime(2016, 1, 4, 9, 0, tzinfo=KRX_TZ),
            "action_id": f"ca_{iid}",
            "type": action_type,
            "factor": 1.0,
            "cash_amount": 0.0,
            "source": "test",
            "available_at": datetime(2016, 1, 4, 9, 0, tzinfo=KRX_TZ),
            "source_hash": "abc",
        }
        for iid in instrument_ids
    ]
    return pl.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# 1. Warmup check
# ──────────────────────────────────────────────────────────────────

def test_check_warmup_sessions_sufficient() -> None:
    """60 sessions before validation_start → warmup_ok=True."""
    val_start = date(2016, 4, 1)
    # Build 61 sessions before val_start + 10 after
    cal = _make_calendar(date(2016, 1, 1), 80)
    result = check_warmup_sessions(cal, validation_start=val_start)
    # Count sessions before val_start
    expected_found = sum(1 for s in cal.sessions if s.astimezone(KRX_TZ).date() < val_start)
    assert result.warmup_sessions_found == expected_found
    assert result.warmup_sessions_required == WARMUP_SESSIONS
    if expected_found >= WARMUP_SESSIONS:
        assert result.warmup_ok is True
    else:
        assert result.warmup_ok is False


def test_check_warmup_sessions_insufficient() -> None:
    """Only 10 sessions before validation_start → warmup_ok=False."""
    val_start = date(2016, 1, 15)
    cal = _make_calendar(date(2016, 1, 1), 80)
    result = check_warmup_sessions(cal, validation_start=val_start, warmup_required=60)
    assert result.warmup_ok is False
    assert result.warmup_sessions_found < 60


def test_check_warmup_sessions_exactly_met() -> None:
    """Exactly 60 sessions before validation_start → warmup_ok=True."""
    # 60 days from 2016-01-01 lands on 2016-03-01
    cal = _make_calendar(date(2016, 1, 1), 120)
    # Use 61st session date as validation_start
    val_start = cal.sessions[60].astimezone(KRX_TZ).date()
    result = check_warmup_sessions(cal, validation_start=val_start, warmup_required=60)
    assert result.warmup_sessions_found == 60
    assert result.warmup_ok is True


def test_check_warmup_sessions_first_validation_session() -> None:
    """first_validation_session is the first session on or after validation_start."""
    cal = _make_calendar(date(2016, 1, 1), 100)
    val_start = date(2016, 2, 1)
    result = check_warmup_sessions(cal, validation_start=val_start)
    if result.first_validation_session is not None:
        assert result.first_validation_session.astimezone(KRX_TZ).date() >= val_start


# ──────────────────────────────────────────────────────────────────
# 2. Bar continuity audit
# ──────────────────────────────────────────────────────────────────

def test_audit_bar_continuity_clean() -> None:
    """All sessions present, valid OHLC → eligible=True."""
    window_start = date(2016, 1, 4)
    window_end = date(2016, 1, 15)
    cal = _make_calendar(window_start, 10)
    sessions = [s for s in cal.sessions if window_start <= s.astimezone(KRX_TZ).date() <= window_end]
    dm = _make_daily_market(["KRX:000001"], sessions)
    results = audit_bar_continuity(dm, cal, window_start=window_start, window_end=window_end)
    assert len(results) == 1
    r = results[0]
    assert r.instrument_id == "KRX:000001"
    assert r.eligible is True
    assert r.exclusion_reasons == ()


def test_audit_bar_continuity_missing_sessions() -> None:
    """Missing a session → MISSING_SESSIONS reason, ineligible."""
    window_start = date(2016, 1, 4)
    window_end = date(2016, 1, 15)
    cal = _make_calendar(window_start, 10)
    sessions = [s for s in cal.sessions if window_start <= s.astimezone(KRX_TZ).date() <= window_end]
    # Drop one session
    incomplete_sessions = sessions[:-1]
    dm = _make_daily_market(["KRX:000001"], incomplete_sessions)
    results = audit_bar_continuity(dm, cal, window_start=window_start, window_end=window_end)
    assert len(results) == 1
    r = results[0]
    assert r.eligible is False
    assert BarExclusionReason.MISSING_SESSIONS in r.exclusion_reasons


def test_audit_bar_continuity_ohlc_violation() -> None:
    """low > high violates OHLC → OHLC_VIOLATION reason."""
    window_start = date(2016, 1, 4)
    window_end = date(2016, 1, 6)
    cal = _make_calendar(window_start, 3)
    sessions = list(cal.sessions)
    rows = [
        {
            "session": sessions[0],
            "instrument_id": "KRX:000001",
            "open": 100.0,
            "high": 80.0,   # high < open: violation
            "low": 90.0,
            "close": 95.0,
            "volume": 1000.0,
            "trading_value": 100_000.0,
            "market_cap": 1_000_000.0,
            "shares_outstanding": 10_000.0,
            "available_at": sessions[0],
            "source_hash": "abc",
        },
        {
            "session": sessions[1],
            "instrument_id": "KRX:000001",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
            "trading_value": 100_000.0,
            "market_cap": 1_000_000.0,
            "shares_outstanding": 10_000.0,
            "available_at": sessions[1],
            "source_hash": "abc",
        },
        {
            "session": sessions[2],
            "instrument_id": "KRX:000001",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
            "trading_value": 100_000.0,
            "market_cap": 1_000_000.0,
            "shares_outstanding": 10_000.0,
            "available_at": sessions[2],
            "source_hash": "abc",
        },
    ]
    dm = pl.DataFrame(rows)
    results = audit_bar_continuity(dm, cal, window_start=window_start, window_end=window_end)
    assert len(results) == 1
    r = results[0]
    assert r.eligible is False
    assert BarExclusionReason.OHLC_VIOLATION in r.exclusion_reasons


def test_audit_bar_continuity_negative_trading_value() -> None:
    """Negative trading_value → NEGATIVE_TRADING_VALUE reason."""
    window_start = date(2016, 1, 4)
    window_end = date(2016, 1, 5)
    cal = _make_calendar(window_start, 2)
    sessions = list(cal.sessions)
    rows = [
        {
            "session": s,
            "instrument_id": "KRX:000002",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
            "trading_value": -1.0,  # negative!
            "market_cap": 1_000_000.0,
            "shares_outstanding": 10_000.0,
            "available_at": s,
            "source_hash": "abc",
        }
        for s in sessions
    ]
    dm = pl.DataFrame(rows)
    results = audit_bar_continuity(dm, cal, window_start=window_start, window_end=window_end)
    assert len(results) == 1
    assert BarExclusionReason.NEGATIVE_TRADING_VALUE in results[0].exclusion_reasons


def test_audit_bar_continuity_empty_market() -> None:
    """Empty daily_market → empty result tuple."""
    cal = _make_calendar(date(2016, 1, 4), 5)
    dm = pl.DataFrame({"session": [], "instrument_id": [], "open": [], "high": [], "low": [], "close": [],
                       "volume": [], "trading_value": [], "market_cap": [], "shares_outstanding": [],
                       "available_at": [], "source_hash": []})
    results = audit_bar_continuity(dm, cal, window_start=date(2016, 1, 4), window_end=date(2016, 1, 8))
    assert results == ()


# ──────────────────────────────────────────────────────────────────
# 3. DART fact eligibility
# ──────────────────────────────────────────────────────────────────

def test_audit_dart_fact_eligibility_eligible() -> None:
    """Company with 4 consecutive quarters + all required facts → eligible."""
    decision_time = datetime(2016, 5, 1, 9, 0, tzinfo=KRX_TZ)
    available_at = datetime(2016, 4, 1, 9, 0, tzinfo=KRX_TZ)
    from src.data.gold import _REQUIRED_FACTS
    periods = ["2015Q4", "2015Q3", "2015Q2", "2015Q1"]
    ff = _make_financial_facts(["CID:001"], periods, list(_REQUIRED_FACTS), available_at)
    results = audit_dart_fact_eligibility(ff, decision_time=decision_time)
    assert len(results) == 1
    r = results[0]
    assert r.company_id == "CID:001"
    assert r.eligible is True
    assert r.consecutive_quarters_found >= 4


def test_audit_dart_fact_eligibility_insufficient_quarters() -> None:
    """Only 2 consecutive quarters → INSUFFICIENT_QUARTERS."""
    decision_time = datetime(2016, 5, 1, 9, 0, tzinfo=KRX_TZ)
    available_at = datetime(2016, 4, 1, 9, 0, tzinfo=KRX_TZ)
    from src.data.gold import _REQUIRED_FACTS
    periods = ["2015Q4", "2015Q3"]  # only 2
    ff = _make_financial_facts(["CID:002"], periods, list(_REQUIRED_FACTS), available_at)
    results = audit_dart_fact_eligibility(ff, decision_time=decision_time)
    assert len(results) == 1
    r = results[0]
    assert r.eligible is False
    assert DartExclusionReason.INSUFFICIENT_QUARTERS in r.exclusion_reasons


def test_audit_dart_fact_eligibility_missing_required_facts() -> None:
    """4 quarters but missing required facts → MISSING_REQUIRED_FACTS."""
    decision_time = datetime(2016, 5, 1, 9, 0, tzinfo=KRX_TZ)
    available_at = datetime(2016, 4, 1, 9, 0, tzinfo=KRX_TZ)
    periods = ["2015Q4", "2015Q3", "2015Q2", "2015Q1"]
    # Only provide some facts, not all required
    ff = _make_financial_facts(["CID:003"], periods, ["sales", "net_income"], available_at)
    results = audit_dart_fact_eligibility(ff, decision_time=decision_time)
    assert len(results) == 1
    r = results[0]
    assert r.eligible is False
    assert DartExclusionReason.MISSING_REQUIRED_FACTS in r.exclusion_reasons


def test_audit_dart_fact_eligibility_pit_filter() -> None:
    """Facts available_at after decision_time are excluded (PIT safety)."""
    decision_time = datetime(2016, 1, 1, 9, 0, tzinfo=KRX_TZ)
    future_at = datetime(2016, 6, 1, 9, 0, tzinfo=KRX_TZ)  # after decision_time
    from src.data.gold import _REQUIRED_FACTS
    periods = ["2015Q4", "2015Q3", "2015Q2", "2015Q1"]
    ff = _make_financial_facts(["CID:004"], periods, list(_REQUIRED_FACTS), future_at)
    results = audit_dart_fact_eligibility(ff, decision_time=decision_time)
    # All facts are in the future → no PIT facts available
    assert len(results) == 0


def test_audit_dart_fact_eligibility_empty() -> None:
    """Empty financial_facts → empty result."""
    decision_time = datetime(2016, 5, 1, 9, 0, tzinfo=KRX_TZ)
    ff = pl.DataFrame({"company_id": [], "fiscal_period": [], "filing_id": [], "fact": [],
                       "published_at": [], "available_at": [], "value": [], "unit": [],
                       "consolidated": [], "restatement_id": [], "source_hash": [],
                       "source_kind": [], "mapping_version": [], "raw_document_hash": []})
    results = audit_dart_fact_eligibility(ff, decision_time=decision_time)
    assert results == ()


# ──────────────────────────────────────────────────────────────────
# 4. Corporate action sentinel exclusion
# ──────────────────────────────────────────────────────────────────

def test_exclude_sentinel_ca_sentinel_only() -> None:
    """Instruments with only no_action → excluded."""
    ca = _make_corporate_actions(["KRX:000001", "KRX:000002"], "no_action")
    excluded = exclude_sentinel_corporate_actions(ca, frozenset(["KRX:000001", "KRX:000002"]))
    assert excluded == frozenset(["KRX:000001", "KRX:000002"])


def test_exclude_sentinel_ca_real_action() -> None:
    """Instrument with a real action type → not excluded."""
    ca = _make_corporate_actions(["KRX:000001"], "split")
    excluded = exclude_sentinel_corporate_actions(ca, frozenset(["KRX:000001"]))
    assert "KRX:000001" not in excluded


def test_exclude_sentinel_ca_absent_from_table() -> None:
    """Instrument absent from CA table → excluded (no verified data)."""
    ca = _make_corporate_actions(["KRX:OTHER"], "split")
    excluded = exclude_sentinel_corporate_actions(ca, frozenset(["KRX:000001"]))
    assert "KRX:000001" in excluded


def test_exclude_sentinel_ca_empty_table() -> None:
    """Empty CA table → all candidates excluded."""
    ca = pl.DataFrame({"instrument_id": [], "effective_date": [], "action_id": [], "type": [],
                       "factor": [], "cash_amount": [], "source": [], "available_at": [], "source_hash": []})
    candidates = frozenset(["KRX:A", "KRX:B"])
    excluded = exclude_sentinel_corporate_actions(ca, candidates)
    assert excluded == candidates


def test_exclude_sentinel_ca_mixed() -> None:
    """Mix of real and sentinel actions → only sentinel-only excluded."""
    import polars as pl
    rows = [
        {"instrument_id": "KRX:REAL", "effective_date": datetime(2016,1,4,9,0,tzinfo=KRX_TZ),
         "action_id": "ca1", "type": "split", "factor": 2.0, "cash_amount": 0.0,
         "source": "test", "available_at": datetime(2016,1,4,9,0,tzinfo=KRX_TZ), "source_hash": "x"},
        {"instrument_id": "KRX:SENTINEL", "effective_date": datetime(2016,1,4,9,0,tzinfo=KRX_TZ),
         "action_id": "ca2", "type": "no_action", "factor": 1.0, "cash_amount": 0.0,
         "source": "test", "available_at": datetime(2016,1,4,9,0,tzinfo=KRX_TZ), "source_hash": "x"},
    ]
    ca = pl.DataFrame(rows)
    candidates = frozenset(["KRX:REAL", "KRX:SENTINEL"])
    excluded = exclude_sentinel_corporate_actions(ca, candidates)
    assert "KRX:SENTINEL" in excluded
    assert "KRX:REAL" not in excluded


# ──────────────────────────────────────────────────────────────────
# 5. Composite manifest
# ──────────────────────────────────────────────────────────────────

def test_build_gold_audit_manifest_basic() -> None:
    """Composite manifest runs without error and returns GoldAuditManifest."""
    val_start = date(2016, 4, 1)
    val_end = date(2016, 4, 5)
    # Calendar with enough warmup + window
    cal = _make_calendar(date(2016, 1, 1), 100)
    sessions = [s for s in cal.sessions if val_start <= s.astimezone(KRX_TZ).date() <= val_end]
    decision_time = datetime(2016, 4, 30, 9, 0, tzinfo=KRX_TZ)

    dm = _make_daily_market(["KRX:000001"], sessions)
    from src.data.gold import _REQUIRED_FACTS
    ff = _make_financial_facts(
        ["CID:A"],
        ["2015Q4", "2015Q3", "2015Q2", "2015Q1"],
        list(_REQUIRED_FACTS),
        datetime(2016, 3, 30, 9, 0, tzinfo=KRX_TZ),
    )
    sm = pl.DataFrame({
        "instrument_id": ["KRX:000001"],
        "company_id": ["CID:A"],
        "ticker": ["000001"],
        "market": ["KOSPI"],
        "sector": ["Technology"],
        "listing_date": [datetime(2010, 1, 1, tzinfo=KRX_TZ)],
        "delisting_date": [None],
        "share_class": ["common"],
        "status": ["listed"],
        "valid_from": [datetime(2016, 1, 1, 9, 0, tzinfo=KRX_TZ)],
        "valid_to": [None],
        "available_at": [datetime(2016, 1, 1, 9, 0, tzinfo=KRX_TZ)],
        "source_hash": ["abc"],
    })
    ca = _make_corporate_actions(["KRX:000001"], "split")  # real action → not excluded

    manifest = build_gold_audit_manifest(
        calendar=cal,
        security_master=sm,
        daily_market=dm,
        financial_facts=ff,
        corporate_actions=ca,
        decision_time=decision_time,
        validation_start=val_start,
        validation_end=val_end,
    )
    assert isinstance(manifest, GoldAuditManifest)
    assert len(manifest.manifest_hash) == 64  # sha256 hex
    # Warmup should be OK (>60 sessions before 2016-04-01)
    assert manifest.warmup.warmup_ok is True
    # KRX:000001 should be eligible
    assert "KRX:000001" in manifest.eligible_instrument_ids


def test_build_gold_audit_manifest_sentinel_ca_excluded() -> None:
    """Instrument with sentinel CA is excluded from eligible set."""
    val_start = date(2016, 4, 1)
    val_end = date(2016, 4, 3)
    cal = _make_calendar(date(2016, 1, 1), 100)
    sessions = [s for s in cal.sessions if val_start <= s.astimezone(KRX_TZ).date() <= val_end]
    decision_time = datetime(2016, 4, 30, 9, 0, tzinfo=KRX_TZ)

    dm = _make_daily_market(["KRX:SENT"], sessions)
    from src.data.gold import _REQUIRED_FACTS
    ff = _make_financial_facts(
        ["CID:SENT"],
        ["2015Q4", "2015Q3", "2015Q2", "2015Q1"],
        list(_REQUIRED_FACTS),
        datetime(2016, 3, 30, 9, 0, tzinfo=KRX_TZ),
    )
    sm = pl.DataFrame({
        "instrument_id": ["KRX:SENT"],
        "company_id": ["CID:SENT"],
        "ticker": ["SENT"],
        "market": ["KOSPI"],
        "sector": ["Technology"],
        "listing_date": [datetime(2010, 1, 1, tzinfo=KRX_TZ)],
        "delisting_date": [None],
        "share_class": ["common"],
        "status": ["listed"],
        "valid_from": [datetime(2016, 1, 1, 9, 0, tzinfo=KRX_TZ)],
        "valid_to": [None],
        "available_at": [datetime(2016, 1, 1, 9, 0, tzinfo=KRX_TZ)],
        "source_hash": ["abc"],
    })
    ca = _make_corporate_actions(["KRX:SENT"], "no_action")  # sentinel

    manifest = build_gold_audit_manifest(
        calendar=cal,
        security_master=sm,
        daily_market=dm,
        financial_facts=ff,
        corporate_actions=ca,
        decision_time=decision_time,
        validation_start=val_start,
        validation_end=val_end,
    )
    assert "KRX:SENT" in manifest.ca_excluded_instrument_ids
    assert "KRX:SENT" not in manifest.eligible_instrument_ids


def test_write_gold_audit_artifact(tmp_path: Path) -> None:
    """write_gold_audit_artifact produces valid JSON at the given path."""
    import json
    val_start = date(2016, 4, 1)
    val_end = date(2016, 4, 3)
    cal = _make_calendar(date(2016, 1, 1), 100)
    sessions = [s for s in cal.sessions if val_start <= s.astimezone(KRX_TZ).date() <= val_end]
    decision_time = datetime(2016, 4, 30, 9, 0, tzinfo=KRX_TZ)
    dm = _make_daily_market(["KRX:000001"], sessions)
    ff = pl.DataFrame({"company_id": [], "fiscal_period": [], "filing_id": [], "fact": [],
                       "published_at": [], "available_at": [], "value": [], "unit": [],
                       "consolidated": [], "restatement_id": [], "source_hash": [],
                       "source_kind": [], "mapping_version": [], "raw_document_hash": []})
    sm = pl.DataFrame({"instrument_id": [], "company_id": [], "ticker": [], "market": [],
                       "sector": [], "listing_date": [], "delisting_date": [], "share_class": [],
                       "status": [], "valid_from": [], "valid_to": [], "available_at": [], "source_hash": []})
    ca = pl.DataFrame({"instrument_id": [], "effective_date": [], "action_id": [], "type": [],
                       "factor": [], "cash_amount": [], "source": [], "available_at": [], "source_hash": []})
    manifest = build_gold_audit_manifest(
        calendar=cal, security_master=sm, daily_market=dm,
        financial_facts=ff, corporate_actions=ca,
        decision_time=decision_time, validation_start=val_start, validation_end=val_end,
    )
    out_path = tmp_path / "gold_audit.json"
    result_path = write_gold_audit_artifact(manifest, out_path)
    assert result_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["manifest_hash"] == manifest.manifest_hash
    assert "warmup" in payload
    assert "bar_audit" in payload
    assert "dart_eligibility" in payload
    assert "eligible_instruments" in payload


def test_materialize_gold_window(tmp_path: Path) -> None:
    """materialize_gold_window executes full audit, universe decisions, and summary artifact."""
    import json

    from src.strategy.universe import UniversePolicy

    val_start = date(2016, 4, 1)
    val_end = date(2016, 4, 3)
    cal = _make_calendar(date(2016, 1, 1), 100)
    decision_time = datetime(2016, 4, 30, 15, 30, tzinfo=KRX_TZ)

    dm = _make_daily_market(["KRX:000001"], list(cal.sessions))
    ff = pl.DataFrame({"company_id": [], "fiscal_period": [], "filing_id": [], "fact": [],
                       "published_at": [], "available_at": [], "value": [], "unit": [],
                       "consolidated": [], "restatement_id": [], "source_hash": [],
                       "source_kind": [], "mapping_version": [], "raw_document_hash": []})
    sm = pl.DataFrame({
        "instrument_id": ["KRX:000001"], "company_id": ["C1"], "ticker": ["000001"],
        "market": ["KOSPI"], "sector": ["Industrials"], "listing_date": [cal.sessions[0]],
        "delisting_date": [None], "share_class": ["common"], "status": ["listed"],
        "valid_from": [cal.sessions[0]], "valid_to": [None],
        "available_at": [cal.sessions[0]], "source_hash": ["abc"],
    })
    ca = _make_corporate_actions(["KRX:000001"], "no_action")

    artifact_root = tmp_path / "artifacts"
    gold_root = tmp_path / "gold"

    report = materialize_gold_window(
        calendar=cal,
        security_master=sm,
        daily_market=dm,
        financial_facts=ff,
        corporate_actions=ca,
        validation_start=val_start,
        validation_end=val_end,
        decision_time=decision_time,
        artifact_root=artifact_root,
        gold_root=gold_root,
        universe_policy=UniversePolicy(minimum_listing_sessions=10, liquidity_window_sessions=10, minimum_median_trading_value_krw=100.0),
    )

    assert report.universe_decisions_count > 0
    assert report.summary_artifact_path != ""
    assert Path(report.summary_artifact_path).exists()
    summary = json.loads(Path(report.summary_artifact_path).read_text(encoding="utf-8"))
    assert summary["manifest_hash"] == report.manifest.manifest_hash
    assert summary["sessions_evaluated"] > 0
