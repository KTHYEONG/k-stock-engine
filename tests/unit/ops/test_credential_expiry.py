"""Credential registry reader: kse-owned active dated entries only, no values in errors."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.ops.credential_expiry import ExpiringCredential, load_kse_credentials

TODAY = date(2026, 11, 9)
MARKER = "SECRET-MARKER-123"


def _entry(name: str, **fields: object) -> str:
    base: dict[str, object] = {
        "provider": "p", "store": "quant-secrets/kse-alert.env", "env_vars": ["X"],
        "consumers": ["kse"], "status": "active", "confidence": "confirmed",
        "expires_on": date(2026, 12, 9), "basis": "b", "renew_hint": "h",
    }
    base.update(fields)
    lines = [f'name = "{name}"']
    for key, value in base.items():
        if value is None:
            continue
        if isinstance(value, date):
            lines.append(f"{key} = {value.isoformat()}")
        elif isinstance(value, list):
            lines.append(f"{key} = [" + ", ".join(f'"{item}"' for item in value) + "]")
        else:
            lines.append(f'{key} = "{value}"')
    return "[[credential]]\n" + "\n".join(lines) + "\n"


def _registry(*entries: str, schema: str = "credential-registry.v1") -> str:
    return f'schema = "{schema}"\n\n' + "\n".join(entries)


def test_kse_entries_only_active_with_dates(tmp_path: Path) -> None:
    path = tmp_path / "credential-registry.toml"
    path.write_text(_registry(
        _entry("KCA_ONLY", consumers=["kca"], expires_on=date(2026, 11, 12)),
        _entry("RETIRING", status="retiring", expires_on=None),
        _entry("NO_EXPIRY", status="no_expiry", expires_on=None),
        _entry("KSE_LATE", expires_on=date(2026, 12, 19)),
        _entry("KSE_SOON", expires_on=date(2026, 11, 16), confidence="estimated"),
        _entry("KSE_NO_DATE", confidence="unknown", expires_on=None),
    ), encoding="utf-8")
    assert load_kse_credentials(path, TODAY) == (
        ExpiringCredential(name="KSE_SOON", days_left=7, expires_on=date(2026, 11, 16), estimated=True),
        ExpiringCredential(name="KSE_LATE", days_left=40, expires_on=date(2026, 12, 19), estimated=False),
    )


def test_bad_schema_or_entry_raises_without_contents(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.toml"
    wrong.write_text(_registry(_entry("A"), schema="credential-registry.v2"), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_kse_credentials(wrong, TODAY)

    malformed = tmp_path / "malformed.toml"
    malformed.write_text(_registry(_entry("A", expires_on=MARKER)), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed expires_on") as bad_info:
        load_kse_credentials(malformed, TODAY)
    assert MARKER not in str(bad_info.value)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('credential = ["bare"]', "not a table"),
        ('credential = "text"', "credential section is malformed"),
        ('[[credential]]\nname = 1\nstatus = "active"\nconsumers = ["kse"]\nconfidence = "confirmed"\n'
         "expires_on = 2026-12-09", "malformed scalar field"),
        ('[[credential]]\nname = "A"\nstatus = "active"\nconsumers = "kse"\nconfidence = "confirmed"\n'
         "expires_on = 2026-12-09", "malformed consumers"),
    ],
)
def test_malformed_registry_shapes_raise(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "registry.toml"
    path.write_text(f'schema = "credential-registry.v1"\n{body}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_kse_credentials(path, TODAY)
