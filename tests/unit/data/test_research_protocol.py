"""Lockbox segment enforcement and protocol binding invariants."""
from __future__ import annotations

from datetime import date, datetime, UTC
from pathlib import Path

import pytest

from src.data.research_protocol import (
    FinalistRecord,
    LockboxError,
    LockboxLedger,
    Segment,
    load_research_protocol,
)


def _protocol(tmp_path: Path | None = None):  # type: ignore[no-untyped-def]
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    return load_research_protocol(Path("config/research/protocol.toml"), scope), scope


def _ledger(tmp_path: Path, now: datetime | None = None) -> LockboxLedger:
    protocol, _ = _protocol()
    moment = now or datetime(2026, 9, 29, tzinfo=UTC)
    return LockboxLedger(state_root=tmp_path / "state", protocol=protocol, now=lambda: moment)


def _finalist(spec_hash: str = "abc123") -> FinalistRecord:
    return FinalistRecord(
        spec_hash=spec_hash,
        family="fam",
        discovery_trial_id="trial0123456789abcdef",
        gate_report_digest="digest",
        registered_at=datetime(2026, 9, 29, tzinfo=UTC),
    )


def test_protocol_binds_scope_boundaries() -> None:
    protocol, _ = _protocol()
    assert protocol.discovery_start == date(2017, 4, 1)
    assert protocol.discovery_window() == (date(2017, 4, 1), date(2023, 12, 31))
    assert protocol.holdout_window() == (date(2024, 1, 1), date(2025, 12, 31))
    assert protocol.forward_start == date(2026, 1, 1)
    assert protocol.segment_of(date(2016, 6, 1)) is Segment.DISCOVERY
    assert protocol.segment_of(date(2023, 12, 31)) is Segment.DISCOVERY
    assert protocol.segment_of(date(2024, 1, 1)) is Segment.HOLDOUT
    assert protocol.segment_of(date(2026, 1, 1)) is Segment.FORWARD


def test_discovery_always_authorized(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    auth = ledger.authorize(start=date(2017, 4, 3), end=date(2023, 12, 28), spec_hash=None)
    assert auth.segment is Segment.DISCOVERY
    assert auth.evidence is True
    assert not ledger.is_open(Segment.HOLDOUT)


def test_holdout_refused_for_non_finalist(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(LockboxError):
        ledger.authorize(start=date(2024, 1, 2), end=date(2024, 6, 28), spec_hash=None)
    assert not ledger.is_open(Segment.HOLDOUT)


def test_first_finalist_holdout_run_opens_holdout_irreversibly(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_finalists([_finalist()])
    auth = ledger.authorize(start=date(2024, 1, 2), end=date(2024, 6, 28), spec_hash="abc123")
    assert auth.evidence is True
    assert ledger.is_open(Segment.HOLDOUT)
    protocol, _ = _protocol()
    reopened = LockboxLedger(
        state_root=tmp_path / "state", protocol=protocol, now=lambda: datetime(2026, 9, 30, tzinfo=UTC)
    )
    assert reopened.is_open(Segment.HOLDOUT)


def test_no_finalists_after_opening(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_finalists([_finalist()])
    ledger.authorize(start=date(2024, 1, 2), end=date(2024, 6, 28), spec_hash="abc123")
    with pytest.raises(LockboxError):
        ledger.register_finalists([_finalist("other")])
    assert len(ledger.finalists()) == 1


def test_finalist_cap_enforced(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(LockboxError):
        ledger.register_finalists([_finalist(f"hash-{i}") for i in range(4)])
    assert ledger.finalists() == ()


def test_forward_requires_holdout_pass(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_finalists([_finalist()])
    ledger.authorize(start=date(2024, 1, 2), end=date(2024, 6, 28), spec_hash="abc123")
    ledger.record_holdout_verdict(spec_hash="abc123", passed=False, report_digest="d0")
    with pytest.raises(LockboxError):
        ledger.authorize(start=date(2026, 2, 1), end=date(2026, 3, 1), spec_hash="abc123")
    assert not ledger.is_open(Segment.FORWARD)


def test_forward_opens_on_holdout_pass(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_finalists([_finalist()])
    ledger.authorize(start=date(2024, 1, 2), end=date(2024, 6, 28), spec_hash="abc123")
    ledger.record_holdout_verdict(spec_hash="abc123", passed=True, report_digest="d1")
    auth = ledger.authorize(start=date(2026, 2, 1), end=date(2026, 3, 1), spec_hash="abc123")
    assert auth.evidence is True
    assert ledger.is_open(Segment.FORWARD)


def test_non_finalist_after_opening_is_non_evidence(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_finalists([_finalist()])
    ledger.authorize(start=date(2024, 1, 2), end=date(2024, 6, 28), spec_hash="abc123")
    auth = ledger.authorize(start=date(2024, 1, 2), end=date(2024, 6, 28), spec_hash=None)
    assert auth.evidence is False


def test_corrupt_ledger_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_finalists([_finalist()])
    lockbox = tmp_path / "state" / "research" / "lockbox.json"
    lockbox.write_text(lockbox.read_text(encoding="utf-8")[:10], encoding="utf-8")
    protocol, _ = _protocol()
    with pytest.raises(LockboxError):
        LockboxLedger(state_root=tmp_path / "state", protocol=protocol, now=lambda: datetime.now(UTC))


def test_window_classified_by_its_end(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(LockboxError):
        ledger.authorize(start=date(2023, 6, 1), end=date(2024, 2, 1), spec_hash=None)


def test_content_hash_stable_and_bound_to_calendar(tmp_path: Path) -> None:
    protocol, _ = _protocol()
    assert protocol.content_hash == protocol.content_hash
    assert len(protocol.content_hash) == 64
    assert protocol.holdout_window() == (date(2024, 1, 1), date(2025, 12, 31))


def test_is_open_discovery_always_true(tmp_path: Path) -> None:
    assert _ledger(tmp_path).is_open(Segment.DISCOVERY) is True


def test_register_empty_is_noop(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_finalists([])
    assert ledger.finalists() == ()


def test_register_duplicate_spec_hash_refused(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_finalists([_finalist("dup")])
    with pytest.raises(LockboxError):
        ledger.register_finalists([_finalist("dup")])
    with pytest.raises(LockboxError):
        ledger.register_finalists([_finalist("x"), _finalist("x")])


def test_authorize_rejects_inverted_window(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        ledger.authorize(start=date(2024, 6, 1), end=date(2024, 1, 1), spec_hash=None)


def test_forward_non_finalist_after_open_is_non_evidence(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_finalists([_finalist()])
    ledger.authorize(start=date(2024, 1, 2), end=date(2024, 6, 28), spec_hash="abc123")
    ledger.record_holdout_verdict(spec_hash="abc123", passed=True, report_digest="d1")
    ledger.authorize(start=date(2026, 2, 1), end=date(2026, 3, 1), spec_hash="abc123")
    auth = ledger.authorize(start=date(2026, 2, 1), end=date(2026, 3, 1), spec_hash=None)
    assert auth.evidence is False


def test_forward_sealed_for_unknown_spec(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(LockboxError):
        ledger.authorize(start=date(2026, 2, 1), end=date(2026, 3, 1), spec_hash="nope")


def test_verdict_unknown_finalist_refused(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(LockboxError):
        ledger.record_holdout_verdict(spec_hash="nope", passed=True, report_digest="d")


def test_verdict_before_open_refused(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_finalists([_finalist()])
    with pytest.raises(LockboxError):
        ledger.record_holdout_verdict(spec_hash="abc123", passed=True, report_digest="d")
    assert ledger.holdout_passed("abc123") is False


def test_verdict_duplicate_refused(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_finalists([_finalist()])
    ledger.authorize(start=date(2024, 1, 2), end=date(2024, 6, 28), spec_hash="abc123")
    ledger.record_holdout_verdict(spec_hash="abc123", passed=True, report_digest="d")
    with pytest.raises(LockboxError):
        ledger.record_holdout_verdict(spec_hash="abc123", passed=False, report_digest="d2")


def test_lockbox_version_mismatch_fails_closed(tmp_path: Path) -> None:
    import json as _json

    ledger = _ledger(tmp_path)
    ledger.register_finalists([_finalist()])
    lockbox = tmp_path / "state" / "research" / "lockbox.json"
    raw = _json.loads(lockbox.read_text(encoding="utf-8"))
    raw["protocol_version"] = "other-version"
    lockbox.write_text(_json.dumps(raw), encoding="utf-8")
    protocol, _ = _protocol()
    with pytest.raises(LockboxError):
        LockboxLedger(state_root=tmp_path / "state", protocol=protocol, now=lambda: datetime.now(UTC))


def test_lockbox_invalid_shape_fails_closed(tmp_path: Path) -> None:
    lockbox_dir = tmp_path / "state" / "research"
    lockbox_dir.mkdir(parents=True)
    (lockbox_dir / "lockbox.json").write_text("[1, 2]\n", encoding="utf-8")
    protocol, _ = _protocol()
    with pytest.raises(LockboxError):
        LockboxLedger(state_root=tmp_path / "state", protocol=protocol, now=lambda: datetime.now(UTC))
    (lockbox_dir / "lockbox.json").write_text(
        '{"protocol_version": "research-protocol-v1", "finalists": [{}], "verdicts": {}, "audit": []}',
        encoding="utf-8",
    )
    with pytest.raises(LockboxError):
        LockboxLedger(state_root=tmp_path / "state", protocol=protocol, now=lambda: datetime.now(UTC))
    (lockbox_dir / "lockbox.json").write_text(
        '{"protocol_version": "research-protocol-v1", "finalists": [], "verdicts": [], "audit": []}',
        encoding="utf-8",
    )
    with pytest.raises(LockboxError):
        LockboxLedger(state_root=tmp_path / "state", protocol=protocol, now=lambda: datetime.now(UTC))


def test_lockbox_write_failure_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    research_dir = tmp_path / "state" / "research"
    research_dir.mkdir(parents=True)
    (research_dir / "lockbox.json").write_text("{}", encoding="utf-8")
    (tmp_path / "blocker").write_text("x", encoding="utf-8")
    (research_dir / "lockbox.json").unlink()
    research_dir.rmdir()
    (tmp_path / "state" / "research").write_text("not-a-dir", encoding="utf-8")
    with pytest.raises(LockboxError):
        ledger.register_finalists([_finalist("w")])
    (tmp_path / "state" / "research").unlink()


def test_check_ratio_rejects_non_number() -> None:
    from src.config import ConfigError
    from src.data.research_protocol import _check_ratio

    with pytest.raises(ConfigError):
        _check_ratio("min_dsr", True)  # type: ignore[arg-type]


def _protocol_text_with(patch: str) -> str:
    from pathlib import Path as _Path

    return patch + _Path("config/research/protocol.toml").read_text(encoding="utf-8")


def test_load_missing_and_invalid_protocol(tmp_path: Path) -> None:
    from src.config import ConfigError
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    with pytest.raises(ConfigError):
        load_research_protocol(tmp_path / "absent.toml", scope)
    broken = tmp_path / "broken.toml"
    broken.write_text("version = [\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_research_protocol(broken, scope)
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(_protocol_text_with('\nunknown_key = "x"\n'), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_research_protocol(unknown, scope)
    bad_start = tmp_path / "bad-start.toml"
    text = _protocol_text_with("")
    text = text.replace('discovery_start = "2017-04-01"', 'discovery_start = "not-a-date"')
    bad_start.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_research_protocol(bad_start, scope)


def test_load_discovery_bounds_and_sections(tmp_path: Path) -> None:
    from src.config import ConfigError
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    early = tmp_path / "early.toml"
    text = Path("config/research/protocol.toml").read_text(encoding="utf-8")
    early.write_text(text.replace('discovery_start = "2017-04-01"', 'discovery_start = "2016-01-01"'), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_research_protocol(early, scope)
    late = tmp_path / "late.toml"
    late.write_text(text.replace('discovery_start = "2017-04-01"', 'discovery_start = "2024-06-01"'), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_research_protocol(late, scope)
    nogates = tmp_path / "nogates.toml"
    nogates.write_text(
        'version = "research-protocol-v1"\ndiscovery_start = "2017-04-01"\nmax_finalists = 3\n'
        'prior_trials = 415\nsessions_per_year = 252\nprimary_capital_krw = 10000000\n'
        '[benchmarks]\nuniverse_min_adtv20_krw = 1\nuniverse_min_price_krw = 1\nrebalance = "M"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_research_protocol(nogates, scope)
    bad_model = tmp_path / "bad-model.toml"
    bad_model.write_text(text.replace('rebalance = "M"', 'rebalance = "Q"'), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_research_protocol(bad_model, scope)


def test_load_gate_domains(tmp_path: Path) -> None:
    from src.config import ConfigError
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = Path("config/research/protocol.toml").read_text(encoding="utf-8")

    def _bad(old: str, new: str) -> Path:
        path = tmp_path / f"bad-{len(old)}-{abs(hash(new)) % 100000}.toml"
        path.write_text(base.replace(old, new), encoding="utf-8")
        return path

    with pytest.raises(ConfigError):
        load_research_protocol(_bad("max_finalists = 3", "max_finalists = 0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("prior_trials = 415", "prior_trials = -1"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("sessions_per_year = 252", "sessions_per_year = 0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("primary_capital_krw = 10_000_000", "primary_capital_krw = 0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("universe_min_adtv20_krw = 500_000_000", "universe_min_adtv20_krw = 0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("bootstrap_block_sessions = 63", "bootstrap_block_sessions = 0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("cscv_blocks = 8", "cscv_blocks = 3"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("min_dsr = 0.95", "min_dsr = 1.5"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("min_dsr = 0.95", 'min_dsr = "high"'), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("stress_extra_slippage = 0.002", "stress_extra_slippage = -0.5"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("perturbation_cuts = 3", "perturbation_cuts = 0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("ledger_capitals = [1_000_000, 10_000_000, 100_000_000]", "ledger_capitals = []"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("parity_max_capital = 10_000_000", "parity_max_capital = 0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("min_active_t = 2.0", "min_active_t = -1.0"), scope)
