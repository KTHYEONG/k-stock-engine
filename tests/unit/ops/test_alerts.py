"""Core alert layer invariant guards."""

from __future__ import annotations

import json
import smtplib
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import src.ops.alerts as alerts_mod
from src.ops.alerts import (
    AlertMessage,
    DispatchOutcome,
    GmailSender,
    collect_secret_values,
    count_undelivered,
    dispatch,
    drain_outbox,
    failure_cause,
    redact,
    sender_from_env,
)

KRX_TZ = ZoneInfo("Asia/Seoul")


def _now() -> datetime:
    return datetime(2026, 10, 8, 10, 0, tzinfo=KRX_TZ)


def build_unit_failure(
    unit: str, *, result: str, exit_status: str, journal_tail: str, now: datetime,
) -> AlertMessage:
    moment = now if now.tzinfo is not None else now.replace(tzinfo=KRX_TZ)
    return AlertMessage(
        subject=f"[kse][실패] {unit}",
        body=f"{result} {exit_status} {failure_cause(journal_tail)}",
        key=f"unit|{unit}|{moment.astimezone(KRX_TZ).date().isoformat()}",
    )


class _Recording:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def send(self, subject: str, body: str) -> None:
        self.calls.append((subject, body))


class _Failing:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def send(self, subject: str, body: str) -> None:
        self.calls += 1
        raise self.error


class _FailOnSecond:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0
        self.subjects: list[str] = []

    def send(self, subject: str, body: str) -> None:
        self.calls += 1
        self.subjects.append(subject)
        if self.calls == 2:
            raise self.error


class _Flaky:
    def __init__(self) -> None:
        self.calls = 0

    def send(self, subject: str, body: str) -> None:
        self.calls += 1
        if self.calls == 1:
            raise OSError("boom")


def test_failure_cause_is_bounded_single_line() -> None:
    tail = "\n".join(f"L{i:05d}-" + "x" * 1000 for i in range(500))
    cause = failure_cause(tail)
    assert cause.startswith("L00499-")
    assert len(cause) == 300
    assert "\n" not in cause


def test_cause_prefers_last_error_line_over_trailing_noise() -> None:
    tail = "\n".join([
        "starting job",
        "Traceback (most recent call last):",
        "ValueError: snapshot missing for 2026-10-08",
        "kse-backup.service: Main process exited, code=exited, status=1/FAILURE",
        "   ",
    ])
    assert failure_cause(tail) == "kse-backup.service: Main process exited, code=exited, status=1/FAILURE"
    assert failure_cause("a\nValueError: boom\nbye") == "ValueError: boom"
    assert failure_cause("a\nlast words") == "last words"
    assert failure_cause("") == "(no journal output)"
    assert "\x00" not in failure_cause("a\x00b\x1f\x7f")


def test_secrets_never_survive(tmp_path: Path) -> None:
    secret = "abcd1234"
    tail = f"failed with password={secret} and token Bearer {secret}"
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail=tail, now=_now(),
    )
    sender = _Recording()
    outcome = dispatch(
        message, sender=sender, alerts_dir=tmp_path,
        now=_now(), secret_values={secret},
    )
    assert outcome == DispatchOutcome.SENT
    for subject, body in sender.calls:
        assert secret not in subject
        assert secret not in body


def test_secrets_never_survive_queued_file(tmp_path: Path) -> None:
    secret = "abcd1234"
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1",
        journal_tail=f"leak {secret}", now=_now(),
    )
    outcome = dispatch(message, sender=None, alerts_dir=tmp_path, now=_now(), secret_values={secret})
    assert outcome == DispatchOutcome.QUEUED
    files = list((tmp_path / "outbox").glob("*.json"))
    assert len(files) == 1
    assert secret not in files[0].read_text(encoding="utf-8")
    assert files[0].stat().st_mode & 0o777 == 0o600


def test_redaction_shapes() -> None:
    assert redact("api_key=supersecret1", set()) == "api_key=***"
    assert redact('"app_password": "hunter22"', set()) == '"app_password": "***"'
    assert redact("token: abcdefgh", set()) == "token: ***"
    assert redact("Bearer abcdefgh", set()) == "Bearer ***"
    assert redact("short", {"abc"}) == "short"
    assert redact("has abcd1234 inside", {"abcd1234", "abcd123456"}) == "has *** inside"


def test_collect_secret_values() -> None:
    env = {
        "MY_API_KEY": "s3cret-value",
        "OTHER_SECRET": "shrt",
        "APP_PASSWORD": "pw123456",
        "AUTH_TOKEN": "tok123456",
        "ECOS_API": "ecos123456",
        "PLAIN": "notcollected123",
    }
    found = collect_secret_values(env)
    assert found == frozenset({"s3cret-value", "pw123456", "tok123456", "ecos123456"})
    mixed = collect_secret_values({123: "longvalue123", "MY_KEY": 456})  # type: ignore[dict-item]
    assert mixed == frozenset()


@pytest.mark.parametrize("subject", ["[kse] x", "[긴급][kse] 요약", "[해결][kse] 요약"])
def test_tagged_subjects_keep_their_tag(tmp_path: Path, subject: str) -> None:
    sender = _Recording()
    dispatch(AlertMessage(subject=subject, body="b", key=subject), sender=sender, alerts_dir=tmp_path, now=_now())
    assert sender.calls[0][0] == subject


def test_untagged_subject_gets_kse_tag(tmp_path: Path) -> None:
    sender = _Recording()
    dispatch(AlertMessage(subject="plain", body="b", key="k"), sender=sender, alerts_dir=tmp_path, now=_now())
    assert sender.calls[0][0] == "[kse] plain"


def test_notice_subject_tag_is_preserved(tmp_path: Path) -> None:
    sender = _Recording()
    subject = "[알림][kse] x"
    dispatch(AlertMessage(subject=subject, body="b", key="n"), sender=sender, alerts_dir=tmp_path, now=_now())
    assert sender.calls[0][0] == subject


def test_same_cause_sent_once_per_day(tmp_path: Path) -> None:
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    sender = _Recording()
    assert dispatch(message, sender=sender, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SENT
    assert dispatch(message, sender=sender, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SUPPRESSED
    assert len(sender.calls) == 1
    other = build_unit_failure(
        "other.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    assert dispatch(other, sender=sender, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SENT


def test_failed_send_is_queued_not_lost(tmp_path: Path) -> None:
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    sender = _Failing(OSError("down"))
    outcome = dispatch(message, sender=sender, alerts_dir=tmp_path, now=_now())
    assert outcome == DispatchOutcome.QUEUED
    assert sender.calls == 2
    assert count_undelivered(tmp_path) == 1
    assert not (tmp_path / "sent.jsonl").exists()


def test_no_credentials_queues(tmp_path: Path) -> None:
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    assert dispatch(message, sender=None, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.QUEUED
    assert sender_from_env({}) is None
    assert sender_from_env({"ALERT_GMAIL_USER": " ", "ALERT_GMAIL_APP_PASSWORD": "x",
                            "ALERT_GMAIL_TO": "y"}) is None
    sender = sender_from_env({"ALERT_GMAIL_USER": "u", "ALERT_GMAIL_APP_PASSWORD": "p",
                              "ALERT_GMAIL_TO": "t"})
    assert isinstance(sender, GmailSender)


def test_sender_from_env_default_path(monkeypatch: object) -> None:
    monkeypatch.setattr(alerts_mod, "read_secret", lambda name, default="": {"ALERT_GMAIL_USER": "u",
        "ALERT_GMAIL_APP_PASSWORD": "p", "ALERT_GMAIL_TO": "t"}[name])  # type: ignore[attr-defined]
    assert isinstance(sender_from_env(None), GmailSender)
    def _boom(name: str, default: str | None = None) -> str:
        raise ValueError("missing")
    monkeypatch.setattr(alerts_mod, "read_secret", _boom)  # type: ignore[attr-defined]
    assert sender_from_env(None) is None


def test_drain_oldest_first_stops_on_failure(tmp_path: Path) -> None:
    base = _now()
    for i, unit in enumerate(["a.service", "b.service", "c.service"]):
        message = build_unit_failure(
            unit, result="failed", exit_status="1", journal_tail="x",
            now=base + timedelta(seconds=i))
        assert dispatch(message, sender=None, alerts_dir=tmp_path,
                        now=base + timedelta(seconds=i)) == DispatchOutcome.QUEUED
    assert count_undelivered(tmp_path) == 3
    sender = _FailOnSecond(OSError("down"))
    delivered = drain_outbox(sender=sender, alerts_dir=tmp_path)
    assert delivered == 1
    assert count_undelivered(tmp_path) == 2
    assert sender.subjects == ["[kse][실패] a.service", "[kse][실패] b.service"]
    remaining = [json.loads(path.read_text(encoding="utf-8"))["key"]
                 for path in (tmp_path / "outbox").glob("*.json")]
    assert sorted(remaining) == ["unit|b.service|2026-10-08", "unit|c.service|2026-10-08"]


def test_queued_does_not_suppress_later_send(tmp_path: Path) -> None:
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    assert dispatch(message, sender=None, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.QUEUED
    sender = _Recording()
    assert dispatch(message, sender=sender, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SENT
    assert len(sender.calls) == 1


def test_alert_path_never_raises(tmp_path: Path) -> None:
    target = tmp_path / "as-file"
    target.write_text("x", encoding="utf-8")
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    assert dispatch(message, sender=None, alerts_dir=target, now=_now()) == DispatchOutcome.QUEUED
    assert dispatch(message, sender=_Recording(), alerts_dir=target, now=_now()) == DispatchOutcome.SENT


def test_corrupt_outbox_preserved(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox"
    outbox.mkdir(parents=True)
    (outbox / "000-bad.json").write_text("{not json", encoding="utf-8")
    (outbox / "zzz-note.txt").write_text("ignore", encoding="utf-8")
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    assert dispatch(message, sender=None, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.QUEUED
    sender = _Recording()
    delivered = drain_outbox(sender=sender, alerts_dir=tmp_path)
    assert delivered == 1
    assert (outbox / "000-bad.json").exists()


def test_retry_then_success_and_ledger_failure(tmp_path: Path) -> None:
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    flaky = _Flaky()
    assert dispatch(message, sender=flaky, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SENT
    assert flaky.calls == 2
    ledger = tmp_path / "sent.jsonl"
    ledger.write_text("corrupt line\n", encoding="utf-8")
    other = build_unit_failure(
        "z.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    assert dispatch(other, sender=_Recording(), alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SENT
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "sent.jsonl").mkdir()
    blocked = build_unit_failure(
        "b.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    assert dispatch(blocked, sender=_Recording(), alerts_dir=busy, now=_now()) == DispatchOutcome.SENT


def test_edge_journal_and_naive_now(tmp_path: Path) -> None:
    naive = datetime(2026, 10, 8, 10, 0)
    long_subject = AlertMessage(subject="s\nline2 " + "y" * 200, body="b", key="k")
    outcome = dispatch(long_subject, sender=_Recording(), alerts_dir=tmp_path, now=naive)
    assert outcome == DispatchOutcome.SENT


def test_drain_and_count_edge(tmp_path: Path) -> None:
    assert drain_outbox(sender=None, alerts_dir=tmp_path) == 0
    assert drain_outbox(sender=_Recording(), alerts_dir=tmp_path / "missing") == 0
    assert count_undelivered(tmp_path / "missing") == 0
    assert count_undelivered(tmp_path) == 0
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    dispatch(message, sender=None, alerts_dir=tmp_path, now=_now())
    assert count_undelivered(tmp_path) == 1
    assert drain_outbox(sender=_Recording(), alerts_dir=tmp_path, max_items=0) == 0


def test_gmail_sender_uses_ssl(monkeypatch: object) -> None:
    seen: dict[str, object] = {}

    class _FakeSMTP:
        def __init__(self, *args: object, **kwargs: object) -> None:
            seen["args"] = args
            seen["kwargs"] = kwargs

        def __enter__(self) -> _FakeSMTP:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def login(self, user: str, password: str) -> None:
            seen["login"] = (user, password)

        def send_message(self, message: object) -> None:
            seen["message"] = message

    monkeypatch.setattr(smtplib, "SMTP_SSL", _FakeSMTP)  # type: ignore[attr-defined]
    GmailSender(user="u", app_password="p", to_addr="t").send("s", "b")
    assert seen["args"] == ("smtp.gmail.com", 465)
    assert seen["kwargs"] == {"timeout": 10}


def test_smtp_exception_retried_as_queued(tmp_path: Path) -> None:
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    outcome = dispatch(
        message, sender=_Failing(smtplib.SMTPException("smtp down")),
        alerts_dir=tmp_path, now=_now())
    assert outcome == DispatchOutcome.QUEUED
    outcome = dispatch(
        message, sender=_Failing(TimeoutError("t/o")), alerts_dir=tmp_path, now=_now())
    assert outcome == DispatchOutcome.QUEUED


def test_suppression_window_expiry(tmp_path: Path) -> None:
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    sender = _Recording()
    assert dispatch(message, sender=sender, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SENT
    later = _now() + timedelta(hours=25)
    fresh = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=later)
    assert dispatch(fresh, sender=sender, alerts_dir=tmp_path, now=later) == DispatchOutcome.SENT


def test_outbox_payload_roundtrip(tmp_path: Path) -> None:
    message = AlertMessage(subject="[kse] hello", body="world", key="k1")
    assert dispatch(message, sender=None, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.QUEUED
    files = list((tmp_path / "outbox").glob("*.json"))
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["subject"] == "[kse] hello"
    assert payload["key"] == "k1"
    assert "enqueued_at" in payload
    sender = _Recording()
    assert drain_outbox(sender=sender, alerts_dir=tmp_path) == 1
    assert count_undelivered(tmp_path) == 0


def test_enqueue_failure_cleanup(tmp_path: Path, monkeypatch: object) -> None:
    import os as _os

    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    monkeypatch.setattr(_os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))  # type: ignore[attr-defined]
    assert dispatch(message, sender=None, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.QUEUED

    def _boom_unlink(self: Path, *a: object, **k: object) -> None:
        raise OSError("gone")

    monkeypatch.setattr(Path, "unlink", _boom_unlink)  # type: ignore[attr-defined]
    assert dispatch(message, sender=None, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.QUEUED


def test_dispatch_ledger_unreadable_branch(tmp_path: Path) -> None:
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    (tmp_path / "sent.jsonl").write_bytes(b"\xff")
    assert dispatch(message, sender=_Recording(), alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SENT


def test_failed_send_with_unwritable_dir(tmp_path: Path) -> None:
    target = tmp_path / "as-file"
    target.write_text("x", encoding="utf-8")
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    assert dispatch(message, sender=_Failing(OSError("down")), alerts_dir=target,
                    now=_now()) == DispatchOutcome.QUEUED


def test_drain_unlink_failure(tmp_path: Path, monkeypatch: object) -> None:
    message = build_unit_failure(
        "u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    assert dispatch(message, sender=None, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.QUEUED
    real_unlink = Path.unlink

    def _boom(self: Path, *a: object, **k: object) -> None:
        raise OSError("gone")

    monkeypatch.setattr(Path, "unlink", _boom)  # type: ignore[attr-defined]
    assert drain_outbox(sender=_Recording(), alerts_dir=tmp_path) == 1
    assert count_undelivered(tmp_path) == 1
    monkeypatch.setattr(Path, "unlink", real_unlink)  # type: ignore[attr-defined]


def test_count_undelivered_oserror(tmp_path: Path, monkeypatch: object) -> None:
    (tmp_path / "outbox").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "is_dir", lambda self: (_ for _ in ()).throw(OSError("io")))  # type: ignore[attr-defined]
    assert count_undelivered(tmp_path) == 0


@pytest.mark.parametrize(("text", "expected"), [
    ('"APP_PASSWORD": "abcd efgh ijkl mnop"', '"APP_PASSWORD": "***"'),
    ("secret='value with spaces'", "secret='***'"),
    ('TOKEN = "abc,def;ghi"', 'TOKEN = "***"'),
    ('"password": "abc\\\"def"', '"password": "***"'),
    ("secret='abc\\'def'", "secret='***'"),
    ("ordinary=visible", "ordinary=visible"),
    ("Bearer AbCdEf", "Bearer ***"),
])
def test_redaction_masks_entire_quoted_values(text: str, expected: str) -> None:
    assert redact(text, ()) == expected


def test_logging_does_not_expose_credentials(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    secret = "abcd efgh ijkl mnop"
    recipient = "operator@example.invalid"
    message = AlertMessage(f"[kse] {recipient}", f"password={secret}", f"key|{secret}|{recipient}")
    caplog.set_level(logging.DEBUG, logger="src.ops.alerts")
    assert dispatch(message, sender=_Recording(), alerts_dir=tmp_path, now=_now(),
                    secret_values={secret}) == DispatchOutcome.SENT
    assert secret not in caplog.text
    assert recipient not in caplog.text
    assert secret not in repr(GmailSender("user", secret, recipient))
    assert recipient not in repr(GmailSender("user", secret, recipient))


def test_same_key_exact_24_hour_boundary(tmp_path: Path) -> None:
    sender = _Recording()
    message = AlertMessage("[kse] boundary", "body", "same")
    assert dispatch(message, sender=sender, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SENT
    assert dispatch(message, sender=sender, alerts_dir=tmp_path,
                    now=_now() + timedelta(hours=24) - timedelta(microseconds=1)) == DispatchOutcome.SUPPRESSED
    assert dispatch(message, sender=sender, alerts_dir=tmp_path,
                    now=_now() + timedelta(hours=24)) == DispatchOutcome.SENT
    assert len(sender.calls) == 2


def test_drain_records_delivery_and_skips_corrupt_files_before_limit(tmp_path: Path) -> None:
    message = AlertMessage("[kse] queued", "body", "same")
    dispatch(message, sender=None, alerts_dir=tmp_path, now=_now())
    outbox = tmp_path / "outbox"
    (outbox / "000-invalid.json").write_bytes(b"\xff")
    (outbox / "001-invalid.json").write_text('{"subject": null}', encoding="utf-8")
    (outbox / "002-tmp.tmp").write_text("temporary", encoding="utf-8")
    sender = _Recording()
    assert drain_outbox(sender=sender, alerts_dir=tmp_path, max_items=1) == 1
    assert sender.calls == [("[kse] queued", "body")]
    assert count_undelivered(tmp_path) == 2
    assert dispatch(message, sender=sender, alerts_dir=tmp_path,
                    now=datetime.now(KRX_TZ)) == DispatchOutcome.SUPPRESSED
    assert len(sender.calls) == 1


def test_drain_negative_limit_does_not_send(tmp_path: Path) -> None:
    dispatch(AlertMessage("[kse] queued", "body", "k"), sender=None, alerts_dir=tmp_path, now=_now())
    sender = _Recording()
    assert drain_outbox(sender=sender, alerts_dir=tmp_path, max_items=-1) == 0
    assert sender.calls == []


def test_journal_masks_credentials_before_line_cut() -> None:
    journal = "x" * 280 + ' APP_PASSWORD="abcd efgh ijkl mnop"\x7f\x85'
    cause = failure_cause(redact(journal, ()))
    assert "abcd" not in cause
    assert "efgh" not in cause
    assert "\x7f" not in cause
    assert "\x85" not in cause


def test_gmail_recipient_refusal_is_delivery_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class RefusingSMTP:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.calls = 0

        def __enter__(self) -> RefusingSMTP:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def login(self, user: str, password: str) -> None:
            return None

        def send_message(self, message: object) -> dict[str, tuple[int, bytes]]:
            return {"operator@example.invalid": (550, b"rejected")}

    monkeypatch.setattr(smtplib, "SMTP_SSL", RefusingSMTP)
    sender = GmailSender("user", "password", "operator@example.invalid")
    assert dispatch(AlertMessage("[kse] test", "body", "k"), sender=sender,
                    alerts_dir=tmp_path, now=_now()) == DispatchOutcome.QUEUED
    assert count_undelivered(tmp_path) == 1
    assert not (tmp_path / "sent.jsonl").exists()


def test_persistence_failure_during_flush_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    def fail_fsync(fd: int) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    assert dispatch(AlertMessage("[kse] test", "body", "k"), sender=None,
                    alerts_dir=tmp_path, now=_now()) == DispatchOutcome.QUEUED
    assert list((tmp_path / "outbox").iterdir()) == []


def test_drain_ledger_permission_failure_is_nonfatal(tmp_path: Path) -> None:
    dispatch(AlertMessage("[kse] test", "body", "k"), sender=None, alerts_dir=tmp_path, now=_now())
    (tmp_path / "sent.jsonl").mkdir()
    sender = _Recording()
    assert drain_outbox(sender=sender, alerts_dir=tmp_path) == 1
    assert count_undelivered(tmp_path) == 0
    assert len(sender.calls) == 1


def test_drain_limits_deliveries(tmp_path: Path) -> None:
    for index in range(2):
        dispatch(AlertMessage("[kse] test", str(index), str(index)), sender=None,
                 alerts_dir=tmp_path, now=_now() + timedelta(seconds=index))
    sender = _Recording()
    assert drain_outbox(sender=sender, alerts_dir=tmp_path, max_items=1) == 1
    assert sender.calls == [("[kse] test", "0")]
    assert count_undelivered(tmp_path) == 1


def test_invalid_unicode_is_safe_for_delivery_and_persistence(tmp_path: Path) -> None:
    message = AlertMessage("[kse] bad\udcff", "bad\udcff", "key\udcff")
    sender = _Recording()
    assert dispatch(message, sender=sender, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SENT
    assert sender.calls == [("[kse] bad?", "bad?")]
    assert dispatch(message, sender=None, alerts_dir=tmp_path,
                    now=_now() + timedelta(hours=24)) == DispatchOutcome.QUEUED
    assert drain_outbox(sender=sender, alerts_dir=tmp_path) == 1
    assert sender.calls == [("[kse] bad?", "bad?"), ("[kse] bad?", "bad?")]


def test_programmer_errors_are_not_swallowed(tmp_path: Path) -> None:
    sender = _Failing(ValueError("invalid sender implementation"))
    with pytest.raises(ValueError, match="invalid sender implementation"):
        dispatch(AlertMessage("[kse] test", "body", "k"), sender=sender,
                 alerts_dir=tmp_path, now=_now())
    assert sender.calls == 1
    assert count_undelivered(tmp_path) == 0


def test_drain_drops_queued_copy_of_cause_already_sent(tmp_path: Path) -> None:
    message = build_unit_failure("u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    assert dispatch(message, sender=None, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.QUEUED
    sender = _Recording()
    later = _now() + timedelta(minutes=5)
    assert dispatch(message, sender=sender, alerts_dir=tmp_path, now=later) == DispatchOutcome.SENT
    assert drain_outbox(sender=sender, alerts_dir=tmp_path, now=later + timedelta(minutes=1)) == 0
    assert len(sender.calls) == 1
    assert count_undelivered(tmp_path) == 0
    assert drain_outbox(sender=sender, alerts_dir=tmp_path, now=later) == 0


def test_drain_resends_queued_cause_after_window(tmp_path: Path) -> None:
    message = build_unit_failure("u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    sender = _Recording()
    assert dispatch(message, sender=sender, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SENT
    assert dispatch(message, sender=None, alerts_dir=tmp_path, now=_now() + timedelta(days=2)) == DispatchOutcome.QUEUED
    assert drain_outbox(sender=sender, alerts_dir=tmp_path, now=_now() + timedelta(days=2)) == 1
    assert len(sender.calls) == 2


def test_drain_stops_when_suppressed_copy_cannot_be_deleted(tmp_path: Path, monkeypatch: object) -> None:
    message = build_unit_failure("u.service", result="failed", exit_status="1", journal_tail="x", now=_now())
    assert dispatch(message, sender=None, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.QUEUED
    sender = _Recording()
    assert dispatch(message, sender=sender, alerts_dir=tmp_path, now=_now()) == DispatchOutcome.SENT

    def _boom(self: Path, *a: object, **k: object) -> None:
        raise OSError("ro")

    monkeypatch.setattr(Path, "unlink", _boom)  # type: ignore[attr-defined]
    assert drain_outbox(sender=sender, alerts_dir=tmp_path, now=_now()) == 0
    assert len(sender.calls) == 1
    assert count_undelivered(tmp_path) == 1
