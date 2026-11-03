"""Operator page policy: which incidents mail a human, page text, failure ledger and incident state."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final

from src.core.time import KRX_TZ
from src.ops.alerts import AlertMessage

__all__ = [
    "DEFAULT_ALERT_POLICY",
    "FAILURE_LEDGER_RELPATH",
    "INCIDENT_STATE_RELPATH",
    "PAGE_TEXT",
    "AlertPolicy",
    "IncidentKind",
    "PageText",
    "audit_incidents",
    "backup_is_stale",
    "build_page",
    "build_resolved",
    "consecutive_failure_days",
    "evaluated_kinds",
    "incident_key",
    "load_open_incidents",
    "record_unit_failure",
    "save_open_incidents",
    "streak_issue_key",
    "unit_incidents",
]

FAILURE_LEDGER_RELPATH: Final = "ops/alerts/failures.jsonl"
INCIDENT_STATE_RELPATH: Final = "ops/alerts/incidents.json"
_EVENTS_RELPATH: Final = "ops/events"
_NEUTRAL_OUTCOMES: Final = frozenset({"SKIPPED"})


class IncidentKind(StrEnum):
    """Incidents that need a human now; one mail per kind per KST day."""

    NO_SEAL = "no_seal"
    EVENTS_UNREADABLE = "events_unreadable"
    BACKUP_STALE = "backup_stale"
    COLLECT_EVE_STREAK = "collect_eve_streak"
    COLLECT_DBSEC_STREAK = "collect_dbsec_streak"
    AUCTION_OPEN_STREAK = "auction_open_streak"


_KIND_VALUES: Final = frozenset(kind.value for kind in IncidentKind)


@dataclass(frozen=True, slots=True)
class PageText:
    """Korean page copy; ``{days}`` in any field is the policy streak threshold."""

    summary: str
    what: str
    auto_action: str
    next_step: str


PAGE_TEXT: Final[Mapping[IncidentKind, PageText]] = MappingProxyType({
    IncidentKind.NO_SEAL: PageText(
        summary="장전 매매 결정 미봉인",
        what="오늘 장 시작 전 매매 결정이 봉인되지 않았습니다.",
        auto_action="아침 봉인 실패 직후 늦은 봉인(morning-late)을 자동 실행했지만 결정이 만들어지지 않았습니다.",
        next_step="kse-morning-late 로그에서 원인(KRX 지연, 결정 체인 손상, 메모리 부족 등)을 고친 뒤 kse-morning-late를 다시 실행하세요.",
    ),
    IncidentKind.EVENTS_UNREADABLE: PageText(
        summary="작업 결과 기록 판독 불가",
        what="오늘 작업 결과 기록(events)을 읽을 수 없어 일일 감사가 작업 상태를 판단하지 못했습니다.",
        auto_action="없음. 손상된 기록은 자동으로 고치지 않습니다.",
        next_step="서버에서 오늘 events 파일의 손상된 줄을 확인해 복구하세요.",
    ),
    IncidentKind.BACKUP_STALE: PageText(
        summary="오프사이트 백업 2일 이상 중단",
        what="마지막 정상 오프사이트 백업이 2일 이상 지났습니다.",
        auto_action="매일 02:30 백업이 자동으로 재시도됐지만 계속 실패했습니다.",
        next_step="kse-backup 로그에서 Drive 잠금이나 rclone 오류를 확인하고 장외 시간에 백업을 다시 실행하세요.",
    ),
    IncidentKind.COLLECT_EVE_STREAK: PageText(
        summary="저녁 수집 {days}거래일 연속 실패",
        what="저녁 수집(collect-eve)이 {days}거래일 연속 실패했습니다.",
        auto_action="거래일마다 일정대로 자동 재실행됐지만 계속 실패했습니다.",
        next_step="kse-collect-eve 로그에서 공통 원인을 찾아 고치세요. 다음 아침 봉인 입력이 부족해질 수 있습니다.",
    ),
    IncidentKind.COLLECT_DBSEC_STREAK: PageText(
        summary="DB증권 수집 {days}거래일 연속 실패",
        what="DB증권 보조 수집(collect-dbsec)이 {days}거래일 연속 실패했습니다.",
        auto_action="거래일마다 일정대로 자동 재실행됐지만 계속 실패했습니다.",
        next_step="kse-collect-dbsec 로그에서 토큰이나 설정 오류를 확인해 고치세요.",
    ),
    IncidentKind.AUCTION_OPEN_STREAK: PageText(
        summary="장전 동시호가 수집 {days}거래일 연속 실패",
        what="장전 동시호가 수집(auction-open)이 {days}거래일 연속 실패했습니다.",
        auto_action="거래일마다 일정대로 자동 재실행됐지만 계속 실패했습니다.",
        next_step="kse-auction-open 로그와 그날 아침 봉인 여부를 확인해 원인을 고치세요.",
    ),
})


@dataclass(frozen=True, slots=True)
class AlertPolicy:
    """Page/silence contract for unit failures and audit issues; everything unlisted is silent."""

    streak_threshold: int = 3
    streak_lookback_days: int = 14
    backup_stale_after: timedelta = timedelta(days=2)
    recovery_units: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({"kse-morning.service": "kse-morning-late.service"})
    )
    critical_units: Mapping[str, IncidentKind] = field(
        default_factory=lambda: MappingProxyType({"kse-morning-late.service": IncidentKind.NO_SEAL})
    )
    streak_units: Mapping[str, tuple[str, IncidentKind]] = field(
        default_factory=lambda: MappingProxyType({
            "kse-collect-eve.service": ("collect_eve", IncidentKind.COLLECT_EVE_STREAK),
            "kse-collect-dbsec.service": ("collect_dbsec", IncidentKind.COLLECT_DBSEC_STREAK),
            "kse-auction-open.service": ("auction_open", IncidentKind.AUCTION_OPEN_STREAK),
        })
    )
    audit_unit: str = "kse-daily-audit.service"
    audit_issue_kinds: Mapping[str, IncidentKind] = field(
        default_factory=lambda: MappingProxyType({
            "decision": IncidentKind.NO_SEAL,
            "events": IncidentKind.EVENTS_UNREADABLE,
            IncidentKind.BACKUP_STALE.value: IncidentKind.BACKUP_STALE,
            IncidentKind.COLLECT_EVE_STREAK.value: IncidentKind.COLLECT_EVE_STREAK,
            IncidentKind.COLLECT_DBSEC_STREAK.value: IncidentKind.COLLECT_DBSEC_STREAK,
            IncidentKind.AUCTION_OPEN_STREAK.value: IncidentKind.AUCTION_OPEN_STREAK,
        })
    )
    closed_day_kinds: frozenset[IncidentKind] = frozenset({IncidentKind.BACKUP_STALE})

    def __post_init__(self) -> None:
        if self.streak_threshold < 1:
            raise ValueError("streak_threshold must be >= 1")
        if self.streak_lookback_days < self.streak_threshold:
            raise ValueError("streak_lookback_days must cover streak_threshold")
        if self.backup_stale_after <= timedelta(0):
            raise ValueError("backup_stale_after must be positive")


DEFAULT_ALERT_POLICY: Final = AlertPolicy()


def streak_issue_key(job: str) -> str:
    """Audit issue key raised when ``job`` hits the consecutive-failure threshold."""
    return f"{job}_streak"


def incident_key(kind: IncidentKind, day: date) -> str:
    """Shared dedup key so the unit and audit paths mail one incident once per KST day."""
    return f"incident|{kind.value}|{day.isoformat()}"


def _format(text: str, policy: AlertPolicy) -> str:
    return text.replace("{days}", str(policy.streak_threshold))


def build_page(kind: IncidentKind, day: date, policy: AlertPolicy = DEFAULT_ALERT_POLICY) -> AlertMessage:
    """``[긴급][kse]`` page with exactly three body lines and the shared incident key."""
    text = PAGE_TEXT[kind]
    body = "\n".join((
        f"무슨 일: {_format(text.what, policy)}",
        f"자동 조치: {_format(text.auto_action, policy)}",
        f"지금 할 일: {_format(text.next_step, policy)}",
    ))
    return AlertMessage(
        subject=f"[긴급][kse] {_format(text.summary, policy)}", body=body, key=incident_key(kind, day)
    )


def build_resolved(kind: IncidentKind, day: date, policy: AlertPolicy = DEFAULT_ALERT_POLICY) -> AlertMessage:
    """One-line ``해결됨`` notice for a previously paged incident."""
    summary = _format(PAGE_TEXT[kind].summary, policy)
    return AlertMessage(
        subject=f"[해결][kse] {summary}", body=f"해결됨: {summary}",
        key=f"resolved|{kind.value}|{day.isoformat()}",
    )


def audit_incidents(
    open_issues: Iterable[Mapping[str, object]], policy: AlertPolicy = DEFAULT_ALERT_POLICY
) -> tuple[IncidentKind, ...]:
    """Distinct critical incidents among audit issue keys, in first-seen order."""
    found: list[IncidentKind] = []
    for item in open_issues:
        kind = policy.audit_issue_kinds.get(str(item.get("key", "")))
        if kind is not None and kind not in found:
            found.append(kind)
    return tuple(found)


def evaluated_kinds(day_kind: str, policy: AlertPolicy = DEFAULT_ALERT_POLICY) -> frozenset[IncidentKind]:
    """Incident kinds an audit of ``day_kind`` actually checks; only these may resolve."""
    return frozenset(IncidentKind) if day_kind == "trading" else policy.closed_day_kinds


def backup_is_stale(status: Mapping[str, object], now: datetime, policy: AlertPolicy = DEFAULT_ALERT_POLICY) -> bool:
    """True when the last successful backup is at least ``backup_stale_after`` old; False when unknown."""
    raw = status.get("finished_at") if status.get("rc") == 0 else status.get("last_ok_at")
    if not isinstance(raw, str):
        return False
    try:
        last_ok = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if last_ok.tzinfo is None:
        return False
    moment = now if now.tzinfo is not None else now.replace(tzinfo=KRX_TZ)
    return moment - last_ok >= policy.backup_stale_after


def _jsonl_records(path: Path) -> list[dict[str, object]]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return []
    records: list[dict[str, object]] = []
    for line in content.splitlines():
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _latest_outcome(events_root: Path, day: date, job: str) -> str | None:
    target = Path(events_root) / day.strftime("%Y-%m") / f"{day.isoformat()}.jsonl"
    latest: str | None = None
    for record in _jsonl_records(target):
        if record.get("job") == job:
            latest = str(record.get("outcome"))
    return latest


def _failure_days(failures_path: Path, unit: str) -> set[date]:
    days: set[date] = set()
    for record in _jsonl_records(failures_path):
        if record.get("unit") != unit:
            continue
        try:
            days.add(date.fromisoformat(str(record.get("date"))))
        except ValueError:
            continue
    return days


def consecutive_failure_days(
    job: str,
    unit: str,
    *,
    today: date,
    data_root: Path,
    policy: AlertPolicy = DEFAULT_ALERT_POLICY,
) -> int:
    """Failed run days ending today; a day fails when its unit failed and no later OK outcome exists.

    Days without evidence or only ``SKIPPED`` (weekends, holidays) are neutral; an exit-0 ``DEGRADED``
    breaks the streak. Returns 0 when today itself has not failed.
    """
    root = Path(data_root)
    failed = _failure_days(root / FAILURE_LEDGER_RELPATH, unit)
    events_root = root / _EVENTS_RELPATH
    count = 0
    for offset in range(policy.streak_lookback_days):
        day = today - timedelta(days=offset)
        outcome = _latest_outcome(events_root, day, job)
        if day in failed and outcome != "OK":
            count += 1
            continue
        if offset == 0:
            return 0
        if outcome is None or outcome in _NEUTRAL_OUTCOMES:
            continue
        break
    return count


def unit_incidents(
    unit: str,
    *,
    today: date,
    data_root: Path,
    audit_critical: Collection[str] = (),
    policy: AlertPolicy = DEFAULT_ALERT_POLICY,
) -> tuple[IncidentKind, ...]:
    """Critical incidents implied by one unit failure already recorded in the failure ledger."""
    if unit in policy.recovery_units:
        return ()
    critical = policy.critical_units.get(unit)
    if critical is not None:
        return (critical,)
    streak = policy.streak_units.get(unit)
    if streak is not None:
        job, kind = streak
        days = consecutive_failure_days(job, unit, today=today, data_root=data_root, policy=policy)
        return (kind,) if days >= policy.streak_threshold else ()
    if unit == policy.audit_unit:
        return tuple(IncidentKind(value) for value in audit_critical if value in _KIND_VALUES)
    return ()


def record_unit_failure(
    data_root: Path,
    *,
    unit: str,
    result: str,
    exit_status: str,
    cause: str,
    now: datetime,
) -> None:
    """Append one unit failure to the ledger the streak policy and the ops check read."""
    moment = now if now.tzinfo is not None else now.replace(tzinfo=KRX_TZ)
    kst = moment.astimezone(KRX_TZ)
    record = {
        "ts": kst.isoformat(timespec="seconds"), "date": kst.date().isoformat(), "unit": unit,
        "result": result, "exit_status": exit_status, "cause": cause,
    }
    target = Path(data_root) / FAILURE_LEDGER_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_open_incidents(data_root: Path) -> dict[IncidentKind, str]:
    """Paged incidents not yet announced as resolved, mapped to the day first paged."""
    try:
        payload = json.loads((Path(data_root) / INCIDENT_STATE_RELPATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    raw = payload.get("open") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {IncidentKind(key): str(value) for key, value in raw.items() if key in _KIND_VALUES}


def save_open_incidents(data_root: Path, incidents: Mapping[IncidentKind, str]) -> None:
    """Atomically replace the open incident state."""
    target = Path(data_root) / INCIDENT_STATE_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"open": {kind.value: day for kind, day in sorted(incidents.items())}}
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(target.parent), prefix=".incidents.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)
