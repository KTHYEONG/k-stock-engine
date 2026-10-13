"""KIND relay bundle: canonical encoding for runner-to-VPS evidence transport."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Final, Literal

from src.config.providers import KindPolicy
from src.core.pit import EvidenceKind, PITDataError
from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE, KIND_NOTICE_SEARCH_SOURCE
from src.data.receipt_catalog import EvidenceStatus
from src.data.scoped_ingestion import ScopedRawPayload

__all__ = [
    "RELAY_MAX_BYTES",
    "RELAY_MAX_ITEMS",
    "RELAY_SCHEMA",
    "RelayBundle",
    "check_relay_freshness",
    "decode_relay_bundle",
    "encode_relay_bundle",
    "kind_transport",
]

RELAY_SCHEMA: Final = "kind-relay-v1"
RELAY_MAX_BYTES: Final = 64 * 1024 * 1024
RELAY_MAX_ITEMS: Final = 5000
_KST = timedelta(hours=9)
_KIND_SOURCES = frozenset({KIND_NOTICE_SEARCH_SOURCE, KIND_NOTICE_DOCUMENT_SOURCE})
_TOP_KEYS = frozenset({"schema", "fetched_at", "fetch_date_kst", "producer", "items", "items_sha256"})
_ITEM_KEYS = frozenset(
    {
        "kind",
        "source",
        "natural_key",
        "as_of",
        "fiscal_period",
        "status",
        "source_label",
        "retrieved_at",
        "payload_b64",
        "payload_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class RelayBundle:
    """One producer run's raw KIND evidence plus its observation identity."""

    fetched_at: datetime
    fetch_date_kst: date
    producer: Mapping[str, str]
    items: tuple[ScopedRawPayload, ...]


def _canonical(document: object) -> bytes:
    return json.dumps(
        document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _item_document(payload: ScopedRawPayload) -> dict[str, object]:
    return {
        "kind": payload.kind.value,
        "source": payload.source,
        "natural_key": payload.natural_key,
        "as_of": payload.as_of.isoformat() if payload.as_of is not None else None,
        "fiscal_period": payload.fiscal_period,
        "status": payload.status.value,
        "source_label": payload.source_label,
        "retrieved_at": payload.retrieved_at.isoformat(),
        "payload_b64": base64.b64encode(payload.payload).decode("ascii"),
        "payload_sha256": hashlib.sha256(payload.payload).hexdigest(),
    }


def encode_relay_bundle(bundle: RelayBundle) -> bytes:
    """Canonical UTF-8 JSON encoding; deterministic for equal bundles."""
    if bundle.fetched_at.utcoffset() is None:
        raise PITDataError("relay bundle fetched_at must be timezone-aware")
    items = sorted(
        (_item_document(item) for item in bundle.items),
        key=lambda entry: (str(entry["source"]), str(entry["natural_key"])),
    )
    items_sha256 = hashlib.sha256(_canonical(items)).hexdigest()
    document = {
        "schema": RELAY_SCHEMA,
        "fetched_at": bundle.fetched_at.astimezone(UTC).isoformat(),
        "fetch_date_kst": bundle.fetch_date_kst.isoformat(),
        "producer": dict(bundle.producer),
        "items": items,
        "items_sha256": items_sha256,
    }
    return _canonical(document)


def _require_top(document: object) -> dict[str, object]:
    if not isinstance(document, dict):
        raise PITDataError("relay bundle is not a JSON object")
    if set(document) != set(_TOP_KEYS):
        raise PITDataError("relay bundle has unknown or missing keys")
    return document


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise PITDataError(f"relay bundle has duplicate JSON key {key!r}")
        document[key] = value
    return document


def decode_relay_bundle(raw: bytes) -> RelayBundle:
    """Strictly parse and integrity-check a bundle; raises PITDataError on any deviation."""
    try:
        return _decode_relay_bundle(raw)
    except PITDataError:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise PITDataError("relay bundle has invalid JSON or field values") from exc


def _decode_relay_bundle(raw: bytes) -> RelayBundle:
    if len(raw) > RELAY_MAX_BYTES:
        raise PITDataError("relay bundle exceeds the byte cap")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PITDataError("relay bundle is not valid JSON") from exc
    try:
        document = json.loads(text, object_pairs_hook=_unique_object)
    except PITDataError:
        raise
    except ValueError as exc:
        raise PITDataError("relay bundle is not valid JSON") from exc
    top = _require_top(document)
    if top["schema"] != RELAY_SCHEMA:
        raise PITDataError("relay bundle has unknown schema")
    fetched_raw = top["fetched_at"]
    fetch_date_raw = top["fetch_date_kst"]
    producer_raw = top["producer"]
    items_raw = top["items"]
    items_sha256 = top["items_sha256"]
    if not isinstance(fetched_raw, str) or not isinstance(fetch_date_raw, str):
        raise PITDataError("relay bundle has invalid fetched_at")
    if not isinstance(items_sha256, str):
        raise PITDataError("relay bundle has invalid items_sha256")
    try:
        fetched_at = datetime.fromisoformat(fetched_raw)
    except ValueError as exc:
        raise PITDataError("relay bundle has invalid fetched_at") from exc
    if fetched_at.tzinfo is None or fetched_at.utcoffset() != timedelta(0):
        raise PITDataError("relay bundle fetched_at is not UTC")
    try:
        fetch_date_kst = date.fromisoformat(fetch_date_raw)
    except ValueError as exc:
        raise PITDataError("relay bundle has invalid fetch_date_kst") from exc
    if fetch_date_kst != (fetched_at + _KST).date():
        raise PITDataError("relay bundle fetch_date_kst is inconsistent with fetched_at")
    if not isinstance(producer_raw, dict):
        raise PITDataError("relay bundle has invalid producer")
    for key, value in producer_raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise PITDataError("relay bundle has invalid producer")
    if not isinstance(items_raw, list):
        raise PITDataError("relay bundle items is not a list")
    if len(items_raw) > RELAY_MAX_ITEMS:
        raise PITDataError("relay bundle exceeds the item cap")
    if hashlib.sha256(_canonical(items_raw)).hexdigest() != items_sha256:
        raise PITDataError("relay bundle items_sha256 mismatch")
    seen: set[tuple[str, str]] = set()
    items: list[ScopedRawPayload] = []
    for entry in items_raw:
        if not isinstance(entry, dict) or set(entry) != set(_ITEM_KEYS):
            raise PITDataError("relay bundle item has unknown or missing keys")
        source = entry["source"]
        natural_key = entry["natural_key"]
        if not isinstance(source, str) or source not in _KIND_SOURCES or not isinstance(natural_key, str) or not natural_key:
            raise PITDataError("relay bundle item has invalid source")
        identity = (str(source), str(natural_key))
        if identity in seen:
            raise PITDataError(f"relay bundle carries duplicate {identity[1]}")
        seen.add(identity)
        try:
            kind = EvidenceKind(str(entry["kind"]))
        except ValueError as exc:
            raise PITDataError("relay bundle item has invalid kind") from exc
        try:
            status = EvidenceStatus(str(entry["status"]))
        except ValueError as exc:
            raise PITDataError("relay bundle item has invalid status") from exc
        as_of_raw = entry["as_of"]
        if as_of_raw is None:
            as_of = None
        elif isinstance(as_of_raw, str):
            try:
                as_of = date.fromisoformat(as_of_raw)
            except ValueError as exc:
                raise PITDataError("relay bundle item has invalid as_of") from exc
        else:
            raise PITDataError("relay bundle item has invalid as_of")
        fiscal_period = entry["fiscal_period"]
        if fiscal_period is not None and not isinstance(fiscal_period, str):
            raise PITDataError("relay bundle item has invalid fiscal_period")
        source_label = entry["source_label"]
        retrieved_raw = entry["retrieved_at"]
        if not isinstance(source_label, str) or not source_label or not isinstance(retrieved_raw, str):
            raise PITDataError("relay bundle item has invalid source_label")
        try:
            retrieved_at = datetime.fromisoformat(retrieved_raw)
        except ValueError as exc:
            raise PITDataError("relay bundle item has invalid retrieved_at") from exc
        if retrieved_at.tzinfo is None:
            raise PITDataError("relay bundle item retrieved_at is naive")
        payload_b64 = entry["payload_b64"]
        payload_sha256 = entry["payload_sha256"]
        if not isinstance(payload_b64, str) or not isinstance(payload_sha256, str):
            raise PITDataError("relay bundle item has invalid payload")
        try:
            payload = base64.b64decode(payload_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PITDataError("relay bundle item has invalid payload") from exc
        if hashlib.sha256(payload).hexdigest() != payload_sha256:
            raise PITDataError(f"relay bundle payload_sha256 mismatch for {natural_key}")
        items.append(
            ScopedRawPayload(
                kind=kind,
                source=str(source),
                natural_key=str(natural_key),
                as_of=as_of,
                fiscal_period=fiscal_period,
                status=status,
                payload=payload,
                retrieved_at=retrieved_at,
                source_label=source_label,
            )
        )
    return RelayBundle(
        fetched_at=fetched_at,
        fetch_date_kst=fetch_date_kst,
        producer=dict(producer_raw),
        items=tuple(items),
    )


def check_relay_freshness(
    bundle: RelayBundle, *, now: datetime, policy: KindPolicy, last_accepted_at: datetime | None
) -> None:
    """Refuse future-dated, over-age or regressing bundles; raises PITDataError."""
    moments = (bundle.fetched_at, now, last_accepted_at)
    if any(moment is not None and moment.utcoffset() is None for moment in moments):
        raise PITDataError("relay freshness requires timezone-aware timestamps")
    skew = timedelta(seconds=int(policy.relay_clock_skew_seconds))
    max_age = timedelta(hours=int(policy.relay_max_age_hours))
    if bundle.fetched_at - now > skew:
        raise PITDataError("relay bundle is future-dated")
    if now - bundle.fetched_at > max_age:
        raise PITDataError("relay bundle is over-age")
    if last_accepted_at is not None and bundle.fetched_at < last_accepted_at:
        raise PITDataError("relay bundle regresses the accepted watermark")


def kind_transport(environ: Mapping[str, str]) -> Literal["direct", "relay"]:
    """Select the KIND transport from ``KSE_KIND_TRANSPORT`` (default ``direct``); raises PITDataError otherwise."""
    value = environ.get("KSE_KIND_TRANSPORT", "direct")
    if value is None or value == "":
        return "direct"
    if value in ("direct", "relay"):
        return value  # type: ignore[return-value]
    raise PITDataError(f"unknown KIND transport {value!r}")
