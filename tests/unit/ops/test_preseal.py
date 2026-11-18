"""Pre-seal read-only check invariants: verdicts, exit code, non-trading skip, no writes, unit wiring, paging."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from src.core.pit import PITDataError
from src.core.time import KRX_TZ
from src.ops import preseal
from src.ops.alert_policy import PAGE_TEXT, IncidentKind, build_resolved, unit_incidents

REPO = Path(__file__).resolve().parents[3]
SYSTEMD = REPO / "deploy" / "systemd"
TODAY = date(2026, 11, 18)
SEAL_AT = datetime(2026, 11, 18, 8, 32, tzinfo=KRX_TZ)
MORNING_LOOK = datetime(2026, 11, 18, 7, 30, tzinfo=KRX_TZ)
SESSIONS = [date(2026, 11, d) for d in range(10, 18)]  # 11-10 .. 11-17, consecutive for the pure helpers


def _policy(max_age: int = 36) -> SimpleNamespace:
    return SimpleNamespace(relay_max_age_hours=max_age, relay_clock_skew_seconds=300)


def _runtime(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        workspace=SimpleNamespace(
            bronze_root=root / "bronze", silver_root=root / "silver", state_root=root / "state",
        )
    )


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _seeded_tree(root: Path) -> None:
    for rel, text in {
        "bronze/catalog/receipts.jsonl": "{}\n",
        "silver/investor_flow/part.parquet": "silver",
        "state/operate/state.json": "{\"last_session\": \"2026-11-12\"}",
        "state/decisions/chain.jsonl": "chain\n",
    }.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def _stub_entrypoints(monkeypatch: pytest.MonkeyPatch, root: Path, verdicts: dict[str, str]) -> None:
    import src.config as config_module
    import src.data.runtime as runtime_module
    import src.ops.guards as guards
    import src.research.operate as operate

    def _result(check: str) -> preseal.CheckResult:
        return preseal.CheckResult(check, verdicts.get(check, "PASS"))

    monkeypatch.setattr(guards, "trading_day", lambda day: True)
    monkeypatch.setattr(runtime_module, "resolve_data_runtime", lambda data_root: _runtime(root))
    monkeypatch.setattr(operate, "load_state", lambda path: SimpleNamespace(last_session=date(2026, 11, 12)))
    monkeypatch.setattr(config_module, "load_provider_policy", lambda runtime: SimpleNamespace(kind=_policy()))
    monkeypatch.setattr(config_module, "load_runtime_config", lambda: SimpleNamespace())
    monkeypatch.setattr(preseal, "check_kind_consistency", lambda bronze: _result("KIND_CONSISTENCY"))
    monkeypatch.setattr(
        preseal, "check_flow_freshness",
        lambda runtime, *, sessions, last_session, today: _result("FLOW_FRESHNESS"),
    )
    monkeypatch.setattr(
        preseal, "check_kind_relay_age",
        lambda state_root, *, policy, now, seal_at: _result("KIND_RELAY_AGE"),
    )


def test_kind_consistency_fails_on_contradiction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.kind_notices as kind_notices

    def _contradiction(catalog: object):  # noqa: ANN202 - generator stub
        raise PITDataError("contradictory KIND notice fields for 20261117000123")
        yield  # pragma: no cover

    monkeypatch.setattr(kind_notices, "iter_kind_notices", _contradiction)
    result = preseal.check_kind_consistency(tmp_path / "bronze")
    assert result.verdict == preseal.FAIL
    assert "20261117000123" in dict(result.fields)["detail"]


def test_kind_consistency_passes_name_only_difference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.kind_notices as kind_notices

    # The seal's iterator already folds name-only differences; the check passes when it completes.
    monkeypatch.setattr(kind_notices, "iter_kind_notices", lambda catalog: iter((object(), object())))
    result = preseal.check_kind_consistency(tmp_path / "bronze")
    assert (result.verdict, dict(result.fields)["rows"]) == (preseal.PASS, "2")


def _flow_patches(monkeypatch: pytest.MonkeyPatch, *, newest: date | None, freshness_error: list | None) -> list:
    import src.config.live_gate as live_gate
    import src.config.runtime as runtime_config
    import src.data.dataset_registry as registry_module
    import src.data.datasets as datasets
    import src.data.investor_flow_freshness as freshness

    calls: list[list[date]] = []

    class _Registry:
        def __init__(self, state_root: Path) -> None:
            self.state_root = state_root

        def require(self, kind: str) -> str:
            return f"{kind}_gen"

    def _read(path: Path, *, columns: object = None) -> pl.LazyFrame:
        return pl.DataFrame({"session": [newest]} if newest else {"session": []},
                            schema={"session": pl.Date}).lazy()

    def _check(**kwargs: object) -> tuple:
        calls.append(list(kwargs["source_sessions"]))  # type: ignore[arg-type]
        if freshness_error is not None:
            raise freshness.StaleInvestorFlowError(freshness_error)
        return ()

    monkeypatch.setattr(registry_module, "DatasetRegistry", _Registry)
    monkeypatch.setattr(datasets, "read_dataset", _read)
    monkeypatch.setattr(freshness, "check_investor_flow_freshness", _check)
    monkeypatch.setattr(runtime_config, "load_runtime_config", lambda: SimpleNamespace(live_gate=Path("gate")))
    monkeypatch.setattr(live_gate, "load_live_gate_policy", lambda path: None)
    return calls


def test_required_source_sessions_is_previous_session_per_new_position() -> None:
    from src.research.operate import required_source_sessions

    assert required_source_sessions(SESSIONS, 3) == tuple(SESSIONS[2:7])
    assert required_source_sessions(SESSIONS, len(SESSIONS)) == ()


def test_flow_freshness_fails_when_required_session_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.data.investor_flow_freshness import FlowFreshness, StaleInvestorFlowError

    empty = FlowFreshness(
        source_session=date(2026, 11, 13), required_cells=10, present_cells=0, late_cells=0,
        missing_cells=10, missing_fraction=1.0,
    )
    _flow_patches(monkeypatch, newest=date(2026, 11, 17), freshness_error=[empty])
    assert StaleInvestorFlowError.reason == "stale_investor_flow"
    result = preseal.check_flow_freshness(
        _runtime(tmp_path), sessions=SESSIONS, last_session=date(2026, 11, 12), today=TODAY,
    )
    assert result.verdict == preseal.FAIL
    assert dict(result.fields)["missing"] == "2026-11-13:1.000000"


def test_flow_freshness_warns_when_silver_not_built(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _flow_patches(monkeypatch, newest=date(2026, 11, 14), freshness_error=None)
    result = preseal.check_flow_freshness(
        _runtime(tmp_path), sessions=SESSIONS, last_session=date(2026, 11, 12), today=TODAY,
    )
    assert result.verdict == preseal.WARN
    assert dict(result.fields)["reason"] == "silver_not_built"
    assert calls == [[date(2026, 11, 12), date(2026, 11, 13), date(2026, 11, 14)]]


def test_flow_freshness_edge_verdicts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.datasets as datasets

    calls = _flow_patches(monkeypatch, newest=date(2026, 11, 17), freshness_error=None)
    outside = preseal.check_flow_freshness(
        _runtime(tmp_path), sessions=SESSIONS, last_session=date(2026, 1, 1), today=TODAY,
    )
    assert dict(outside.fields)["reason"] == "state_outside_calendar"
    nothing_new = preseal.check_flow_freshness(
        _runtime(tmp_path), sessions=SESSIONS, last_session=SESSIONS[-1], today=TODAY,
    )
    assert (nothing_new.verdict, dict(nothing_new.fields)["required"]) == (preseal.PASS, "0")
    built = preseal.check_flow_freshness(
        _runtime(tmp_path), sessions=SESSIONS, last_session=date(2026, 11, 12), today=TODAY,
    )
    assert (built.verdict, len(calls[0])) == (preseal.PASS, 5)

    def _boom(path: Path, *, columns: object = None) -> None:
        raise OSError("gone")

    monkeypatch.setattr(datasets, "read_dataset", _boom)
    unreadable = preseal.check_flow_freshness(
        _runtime(tmp_path), sessions=SESSIONS, last_session=date(2026, 11, 12), today=TODAY,
    )
    assert dict(unreadable.fields)["reason"] == "flow_dataset_unreadable"


def test_relay_age_missing_and_undecodable_bundle_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.kind_relay as kind_relay
    from src.data import kind_relay_import

    bundle = tmp_path / "bundle.json"
    monkeypatch.setattr(kind_relay_import, "relay_inbox_path", lambda state_root: bundle)
    missing = preseal.check_kind_relay_age(tmp_path, policy=_policy(), now=MORNING_LOOK, seal_at=SEAL_AT)
    assert dict(missing.fields)["reason"] == "bundle_missing"
    bundle.write_bytes(b"{}")

    def _bad(raw: bytes) -> None:
        raise PITDataError("relay bundle schema mismatch")

    monkeypatch.setattr(kind_relay, "decode_relay_bundle", _bad)
    bad = preseal.check_kind_relay_age(tmp_path, policy=_policy(), now=MORNING_LOOK, seal_at=SEAL_AT)
    assert (bad.verdict, dict(bad.fields)["reason"]) == (preseal.FAIL, "relay_bundle_schema_mismatch")


def test_relay_age_fail_and_warn_thresholds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.kind_relay as kind_relay
    from src.data import kind_relay_import

    bundle = tmp_path / "bundle.json"
    bundle.write_bytes(b"{}")
    monkeypatch.setattr(kind_relay_import, "relay_inbox_path", lambda state_root: bundle)
    verdicts: dict[str, str] = {}
    for label, age_hours, expected in (("40h", 40, preseal.FAIL), ("35h40", 35.0 + 40 / 60, preseal.WARN),
                                       ("5h", 5, preseal.PASS)):
        fetched = MORNING_LOOK - timedelta(hours=age_hours)
        monkeypatch.setattr(
            kind_relay, "decode_relay_bundle", lambda raw, fetched=fetched: SimpleNamespace(fetched_at=fetched)
        )
        verdicts[label] = preseal.check_kind_relay_age(
            tmp_path, policy=_policy(), now=MORNING_LOOK, seal_at=SEAL_AT,
        ).verdict
        assert verdicts[label] == expected, label


def test_non_trading_day_skipped_without_checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.ops.guards as guards

    monkeypatch.setattr(guards, "trading_day", lambda day: False)

    def _must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("checks must not run on a non-trading day")

    monkeypatch.setattr(preseal, "check_kind_consistency", _must_not_run)
    monkeypatch.setattr(preseal, "check_flow_freshness", _must_not_run)
    monkeypatch.setattr(preseal, "check_kind_relay_age", _must_not_run)
    assert preseal.preseal_check(tmp_path, today=date(2026, 11, 21), now=MORNING_LOOK) == 0
    events = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "ops" / "events").rglob("*") if path.is_file())
    assert "preseal" in events
    assert "SKIPPED" in events


def test_fail_exit_one_and_records_degraded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _seeded_tree(tmp_path)
    _stub_entrypoints(monkeypatch, tmp_path, {"KIND_CONSISTENCY": preseal.FAIL})
    assert preseal.preseal_check(tmp_path, today=TODAY, now=MORNING_LOOK) == 1
    events = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "ops" / "events").rglob("*") if path.is_file())
    assert "DEGRADED" in events
    assert "KIND_CONSISTENCY" in events


def test_unreadable_state_and_raising_check_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.research.operate as operate

    _seeded_tree(tmp_path)
    _stub_entrypoints(monkeypatch, tmp_path, {})

    def _no_state(path: Path) -> None:
        raise OSError("missing")

    def _explode(bronze: Path) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(operate, "load_state", _no_state)
    monkeypatch.setattr(preseal, "check_kind_consistency", _explode)
    assert preseal.preseal_check(tmp_path, today=TODAY, now=MORNING_LOOK) == 1
    events = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "ops" / "events").rglob("*") if path.is_file())
    assert "KIND_CONSISTENCY" in events
    assert "FLOW_FRESHNESS" in events


def test_no_writes_outside_events_and_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _seeded_tree(tmp_path)
    before = _tree_digest(tmp_path)
    _stub_entrypoints(monkeypatch, tmp_path, {"FLOW_FRESHNESS": preseal.WARN})
    assert preseal.preseal_check(tmp_path, today=TODAY, now=MORNING_LOOK) == 0
    after = _tree_digest(tmp_path)
    outside = {key: value for key, value in after.items() if not key.startswith("ops/")}
    assert outside == before
    assert any(key.startswith("ops/events/") for key in after)


def test_preseal_unit_files_installed() -> None:
    timer = (SYSTEMD / "kse-preseal.timer").read_text(encoding="utf-8")
    service = (SYSTEMD / "kse-preseal.service").read_text(encoding="utf-8")
    assert "OnCalendar=Mon..Fri 07:30:00 Asia/Seoul" in timer
    assert "Unit=kse-preseal.service" in timer
    assert "OnFailure=kse-alert@%n.service" in service
    assert "ExecStart=%h/.local/lib/kse/kse-run preseal" in service
    assert "kse-run" in (REPO / "deploy" / "kse-run").read_text(encoding="utf-8")
    assert "preseal)" in (REPO / "deploy" / "kse-run").read_text(encoding="utf-8")


def test_alert_policy_pages_preseal_once_per_day(tmp_path: Path) -> None:
    assert unit_incidents("kse-preseal.service", today=TODAY, data_root=tmp_path) == (IncidentKind.PRESEAL_BLOCKED,)
    assert unit_incidents("kse-preseal.service", today=TODAY, data_root=tmp_path) == (
        unit_incidents("kse-preseal.service", today=TODAY, data_root=tmp_path)
    )
    text = PAGE_TEXT[IncidentKind.PRESEAL_BLOCKED]
    assert "preseal" in text.what
    assert "kse-morning-late" in text.next_step
    resolved = build_resolved(IncidentKind.PRESEAL_BLOCKED, TODAY)
    assert resolved.key == f"resolved|preseal_blocked|{TODAY.isoformat()}"
