"""Operator alert layer: message building, redaction, dedup, outbox and Gmail delivery."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import smtplib
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from enum import Enum
from pathlib import Path
from typing import Final, Protocol
from unicodedata import category

from src.config.secrets import read_secret
from src.core.time import KRX_TZ

__all__ = [
    "AlertMessage",
    "AlertSender",
    "DispatchOutcome",
    "GmailSender",
    "build_audit_summary",
    "build_unit_failure",
    "collect_secret_values",
    "count_undelivered",
    "dispatch",
    "drain_outbox",
    "redact",
    "sender_from_env",
]

_LOG = logging.getLogger(__name__)

_MASK: Final = "***"
_MAX_SUBJECT: Final = 120
_MAX_BODY: Final = 4000
_SUPPRESS_WINDOW: Final = timedelta(hours=24)
_GMAIL_HOST: Final = "smtp.gmail.com"
_GMAIL_PORT: Final = 465

_BEARER_RE: Final = re.compile(r"(Bearer\s+)[^\s\"'`,;}\]]+", re.IGNORECASE)
_CRED_RE: Final = re.compile(
    r"""(?P<prefix>(?<![\w.\-])["']?[\w.\-]*(?:key|secret|password|token)["']?[ \t]*(?:=>|:|=)[ \t]*)"""
    r"""(?P<value>"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'|[^\s"'`,;}\]]+)""",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AlertMessage:
    """One operator notification with its de-duplication identity."""

    subject: str
    body: str
    key: str


class DispatchOutcome(Enum):
    """Delivery result of a dispatch call."""

    SENT = "SENT"
    SUPPRESSED = "SUPPRESSED"
    QUEUED = "QUEUED"


class AlertSender(Protocol):
    """Delivery backend; raises on delivery failure."""

    def send(self, subject: str, body: str) -> None: ...


@dataclass(frozen=True, slots=True)
class GmailSender:
    """Gmail delivery backend over SMTP SSL."""

    user: str = field(repr=False)
    app_password: str = field(repr=False)
    to_addr: str = field(repr=False)

    def send(self, subject: str, body: str) -> None:
        message = EmailMessage()
        message["From"] = self.user
        message["To"] = self.to_addr
        message["Subject"] = subject
        message.set_content(body)
        with smtplib.SMTP_SSL(_GMAIL_HOST, _GMAIL_PORT, timeout=10) as smtp:
            smtp.login(self.user, self.app_password)
            refused = smtp.send_message(message)
            if refused:
                raise smtplib.SMTPRecipientsRefused(refused)


def sender_from_env(environ: Mapping[str, str] | None = None) -> AlertSender | None:
    """Gmail sender from alert env vars; None when any is blank."""
    if environ is None:
        try:
            user = read_secret("ALERT_GMAIL_USER", default="")
            password = read_secret("ALERT_GMAIL_APP_PASSWORD", default="")
            to_addr = read_secret("ALERT_GMAIL_TO", default="")
        except Exception:
            _LOG.debug("[SYS] alert sender unavailable")
            return None
    else:
        user = environ.get("ALERT_GMAIL_USER", "")
        password = environ.get("ALERT_GMAIL_APP_PASSWORD", "")
        to_addr = environ.get("ALERT_GMAIL_TO", "")
    if not user.strip() or not password.strip() or not to_addr.strip():
        return None
    return GmailSender(user=user, app_password=password, to_addr=to_addr)


def collect_secret_values(environ: Mapping[str, str]) -> frozenset[str]:
    """Env values whose key name looks credential-shaped and length >= 6."""
    found: set[str] = set()
    for name, value in environ.items():
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        if len(value) < 6:
            continue
        upper = name.upper()
        if upper == "ECOS_API" or upper.endswith(("KEY", "SECRET", "PASSWORD", "TOKEN")):
            found.add(value)
    return frozenset(found)


def redact(text: str, secret_values: Collection[str]) -> str:
    """Mask configured secret values and credential-shaped key/value pairs."""
    candidates = sorted(
        {v for v in secret_values if isinstance(v, str) and len(v) >= 6},
        key=len,
        reverse=True,
    )
    for value in candidates:
        text = text.replace(value, _MASK)
    text = _BEARER_RE.sub(r"\1" + _MASK, text)
    def mask_pair(match: re.Match[str]) -> str:
        value = match.group("value")
        quote = value[0] if value[0] in "\"'" else ""
        return match.group("prefix") + quote + _MASK + quote

    return _CRED_RE.sub(mask_pair, text)


def _as_kst(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=KRX_TZ)
    return now.astimezone(KRX_TZ)


def _normalize_subject(subject: str) -> str:
    single = _sanitize_line(" ".join(subject.split()))
    if not single.startswith("[kse] "):
        single = "[kse] " + single
    return single[:_MAX_SUBJECT]


def _sanitize_line(line: str) -> str:
    cleaned = line.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return "".join(ch if ch == "\t" or category(ch) not in ("Cc", "Cs") else "?" for ch in cleaned)


def build_unit_failure(
    unit: str,
    *,
    result: str,
    exit_status: str,
    journal_tail: str,
    now: datetime,
    open_issues: Sequence[Mapping[str, str]] = (),
) -> AlertMessage:
    """Failure message for one systemd unit."""
    moment = _as_kst(now)
    key = f"unit|{unit}|{moment.date().isoformat()}"
    subject = _normalize_subject(redact(f"[kse] 유닛 실행 실패: {unit}", ()))
    lines = redact(journal_tail, ()).splitlines()
    kept = [_sanitize_line(line)[:300] for line in lines[-40:]]
    if not journal_tail.strip() or not kept:
        kept = ["(no journal output)"]
    parts = [
        f"unit: {unit}",
        f"Result: {result}",
        f"ExecMainStatus: {exit_status}",
        f"time(KST): {moment.isoformat()}",
    ]
    if open_issues:
        parts.append(f"open issues ({len(open_issues)}):")
        parts.extend(f"- {item.get('key', '')}: {item.get('text', '')}" for item in open_issues)
    parts.append("journal (last 40 lines):")
    journal_start = len(parts)
    parts.extend(kept)
    parts.append(f"hint: journalctl --user -u {unit} -n 50 --no-pager")
    body = redact("\n".join(parts), ())
    while len(body) > _MAX_BODY and len(parts) > journal_start + 1:
        del parts[journal_start]
        body = redact("\n".join(parts), ())
    return AlertMessage(subject=subject, body=body[:_MAX_BODY], key=key)


def build_audit_summary(heartbeat: Mapping[str, object]) -> AlertMessage | None:
    """Audit summary with the open issue list; None when healthy."""
    severity = str(heartbeat.get("severity", ""))
    raw_issues = heartbeat.get("open_issues", [])
    issues: list[Mapping[str, object]] = (
        [item for item in raw_issues if isinstance(item, Mapping)] if isinstance(raw_issues, (list, tuple)) else []
    )
    if severity in ("OK", "HOLIDAY_SKIP") and not issues:
        return None
    raw_subject = str(heartbeat.get("subject", "")).strip()
    if raw_subject[:4].lower() == "kse ":
        raw_subject = raw_subject[4:].lstrip()
    if not raw_subject:
        raw_subject = f"일일 감사 {severity}"
    subject = _normalize_subject(redact(f"[kse] {raw_subject}", ()))
    snapshot_date = str(heartbeat.get("snapshot_date", ""))
    day_kind = str(heartbeat.get("day_kind", ""))
    keys = sorted({str(item.get("key", "")) for item in issues})
    key = f"audit|{snapshot_date}|{','.join(keys)}"
    lines = [
        f"severity: {severity}",
        f"snapshot_date: {snapshot_date}",
        f"day_kind: {day_kind}",
        f"open issues ({len(issues)}):",
    ]
    lines.extend(f"- {item.get('key', '')}: {item.get('text', '')}" for item in issues)
    body = redact("\n".join(lines), ())[:_MAX_BODY]
    return AlertMessage(subject=subject, body=body, key=key)


def _ledger_path(alerts_dir: Path) -> Path:
    return Path(alerts_dir) / "sent.jsonl"


def _outbox_dir(alerts_dir: Path) -> Path:
    return Path(alerts_dir) / "outbox"


def _is_suppressed(key: str, alerts_dir: Path, now: datetime) -> bool:
    moment = _as_kst(now)
    try:
        content = _ledger_path(alerts_dir).read_text(encoding="utf-8")
    except OSError:
        return False
    for line in content.splitlines():
        try:
            entry = json.loads(line)
            stamp = datetime.fromisoformat(str(entry["ts"]))
            entry_key = str(entry["key"])
        except (ValueError, KeyError, TypeError):
            continue
        if entry_key != key:
            continue
        aware = stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=KRX_TZ)
        delta = (moment - aware).total_seconds()
        if 0 <= delta < _SUPPRESS_WINDOW.total_seconds():
            return True
    return False


def _append_ledger(alerts_dir: Path, now: datetime, key: str, subject: str) -> None:
    target = _ledger_path(alerts_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    moment = _as_kst(now)
    record = {"ts": moment.isoformat(), "key": key, "subject": subject}
    serialized = json.dumps(record, ensure_ascii=True) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _enqueue(alerts_dir: Path, now: datetime, subject: str, body: str, key: str) -> None:
    outbox = _outbox_dir(alerts_dir)
    outbox.mkdir(parents=True, exist_ok=True)
    moment = _as_kst(now)
    stamp = moment.astimezone(UTC).strftime("%Y%m%dT%H%M%S-%fZ")
    digest = hashlib.sha256(key.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    payload = {"subject": subject, "body": body, "key": key, "enqueued_at": moment.isoformat()}
    serialized = json.dumps(payload, ensure_ascii=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(outbox), prefix=".outbox.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o600)
        os.replace(temporary, outbox / f"{stamp}-{digest}.json")
        temporary = None
        directory_fd = os.open(outbox, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                _LOG.debug("[SYS] alert outbox tmp cleanup failed")


def dispatch(
    message: AlertMessage,
    *,
    sender: AlertSender | None,
    alerts_dir: Path,
    now: datetime,
    secret_values: Collection[str] = (),
) -> DispatchOutcome:
    """Redact, de-duplicate, send; undeliverable messages are queued."""
    subject = _normalize_subject(redact(message.subject, secret_values))
    body = redact(message.body, secret_values).encode("utf-8", errors="replace").decode("utf-8")[:_MAX_BODY]
    key = message.key
    try:
        suppressed = _is_suppressed(key, Path(alerts_dir), now)
    except (OSError, ValueError, TypeError):
        _LOG.debug("[SYS] alert ledger unreadable")
        suppressed = False
    if suppressed:
        _LOG.info("[SYS] alert outcome=SUPPRESSED")
        return DispatchOutcome.SUPPRESSED
    if sender is None:
        try:
            _enqueue(Path(alerts_dir), now, subject, body, key)
        except (OSError, ValueError, TypeError):
            _LOG.debug("[SYS] alert enqueue failed")
        _LOG.info("[SYS] alert outcome=QUEUED")
        return DispatchOutcome.QUEUED
    delivered = False
    for _ in range(2):
        try:
            sender.send(subject, body)
            delivered = True
            break
        except (smtplib.SMTPException, OSError, TimeoutError):
            _LOG.debug("[SYS] alert send attempt failed")
            continue
    if delivered:
        try:
            _append_ledger(Path(alerts_dir), now, key, subject)
        except (OSError, ValueError, TypeError):
            _LOG.debug("[SYS] alert ledger append failed")
        _LOG.info("[SYS] alert outcome=SENT")
        return DispatchOutcome.SENT
    try:
        _enqueue(Path(alerts_dir), now, subject, body, key)
    except (OSError, ValueError, TypeError):
        _LOG.debug("[SYS] alert enqueue failed")
    _LOG.info("[SYS] alert outcome=QUEUED")
    return DispatchOutcome.QUEUED


def drain_outbox(
    *,
    sender: AlertSender | None,
    alerts_dir: Path,
    secret_values: Collection[str] = (),
    max_items: int = 20,
) -> int:
    """Re-deliver queued messages oldest-first; returns the number delivered."""
    if sender is None or max_items <= 0:
        return 0
    try:
        names = sorted(path.name for path in _outbox_dir(alerts_dir).iterdir() if path.is_file())
    except OSError:
        _LOG.debug("[SYS] alert outbox unreadable")
        return 0
    delivered = 0
    for name in names:
        if delivered >= max_items:
            break
        if not name.endswith(".json"):
            continue
        path = _outbox_dir(alerts_dir) / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not all(
                isinstance(payload.get(field), str) for field in ("subject", "body", "key", "enqueued_at")
            ):
                continue
            subject = _normalize_subject(redact(payload["subject"], secret_values))
            body = redact(payload["body"], secret_values).encode("utf-8", errors="replace").decode("utf-8")[:_MAX_BODY]
            key = payload["key"]
        except (OSError, ValueError, KeyError, TypeError):
            _LOG.debug("[SYS] alert outbox corrupt file skipped")
            continue
        try:
            sender.send(subject, body)
        except (smtplib.SMTPException, OSError, TimeoutError):
            _LOG.info("[SYS] alert drain stopped delivered=%d", delivered)
            break
        delivered += 1
        try:
            _append_ledger(alerts_dir, datetime.now(KRX_TZ), key, subject)
        except (OSError, ValueError, TypeError):
            _LOG.debug("[SYS] alert ledger append failed")
        try:
            path.unlink()
        except OSError:
            _LOG.debug("[SYS] alert outbox delete failed")
            break
        _LOG.info("[SYS] alert outcome=SENT")
    return delivered


def count_undelivered(alerts_dir: Path) -> int:
    """Number of queued messages; 0 when the directory does not exist."""
    try:
        outbox = _outbox_dir(alerts_dir)
        if not outbox.is_dir():
            return 0
        return sum(1 for path in outbox.iterdir() if path.is_file() and path.name.endswith(".json"))
    except OSError:
        _LOG.debug("[SYS] alert count failed")
        return 0
