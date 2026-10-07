"""Provider-owned storage contract of every evidence source.

A source is the unit that owns storage: which provider answered, which endpoint,
which envelope shape the bytes have, and whether coverage is recorded per natural
key or per answered session range. Nothing is stored until a payload satisfies
the contract of its own source, so one provider's page can never be read as
another's and a range-shaped source never degenerates into per-cell receipts.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from src.core.pit import EvidenceKind, PITDataError
from src.data.receipt_catalog import EvidenceStatus

__all__ = [
    "DART_CORP_CODES_SOURCE",
    "DART_CORP_DISCLOSURES_SOURCE",
    "DART_DISCLOSURES_SOURCE",
    "DART_DISCLOSURE_WINDOWS_SOURCE",
    "DART_DOCUMENT_SOURCE",
    "DIVIDEND_DECISION_SOURCE",
    "EARNINGS_RELEASE_SOURCE",
    "FINANCIAL_FACTS_SOURCE",
    "KIND_NOTICE_DOCUMENT_SOURCE",
    "KIND_NOTICE_SEARCH_SOURCE",
    "KIS_FLOW_SOURCE",
    "KIS_INDUSTRY_SOURCE",
    "KRX_CASH_SERIES_SOURCE",
    "KRX_DAILY_MARKET_SOURCE",
    "KRX_HEDGE_SERIES_SOURCE",
    "KRX_SECURITY_MASTER_SOURCE",
    "KRX_TREND_SERIES_SOURCE",
    "LS_FLOW_SOURCE",
    "SOURCE_CONTRACTS",
    "CoverageShape",
    "EnvelopeFormat",
    "SourceContract",
    "canonical_raw_rows_bytes",
    "envelope_carries_rows",
    "raw_rows_envelope",
    "source_contract",
    "validate_envelope",
]

LS_FLOW_SOURCE = "ls_investor_flow"
KIS_FLOW_SOURCE = "kis_investor_flow"
KIS_INDUSTRY_SOURCE = "kis_industry"
KRX_DAILY_MARKET_SOURCE = "krx_daily_market"
KRX_HEDGE_SERIES_SOURCE = "krx_hedge_series"
KRX_TREND_SERIES_SOURCE = "krx_trend_series"
KRX_CASH_SERIES_SOURCE = "krx_cash_series"
KRX_SECURITY_MASTER_SOURCE = "krx_security_master"
FINANCIAL_FACTS_SOURCE = "financial_facts"
DART_DISCLOSURES_SOURCE = "dart_disclosures"
DART_CORP_DISCLOSURES_SOURCE = "dart_corp_disclosures"
DART_DISCLOSURE_WINDOWS_SOURCE = "dart_disclosure_windows"
DIVIDEND_DECISION_SOURCE = "opendart:dividend_decision"
EARNINGS_RELEASE_SOURCE = "opendart:earnings_release"
DART_CORP_CODES_SOURCE = "dart_corp_codes"
DART_DOCUMENT_SOURCE = "dart_documents"
KIND_NOTICE_SEARCH_SOURCE = "kind_notice_search"
KIND_NOTICE_DOCUMENT_SOURCE = "kind_notice_documents"

_CANONICAL_SEPARATORS = (",", ":")
_IDENTITY_FIELDS = ("envelope", "provider", "endpoint")


class EnvelopeFormat(StrEnum):
    """Serialized shape a provider response is stored in."""

    RAW_ROWS_V1 = "raw-rows-v1"  # {"envelope","provider","endpoint","query","rows"}; validated here
    NATIVE = "native"  # provider page shape owned by its Silver parser (KRX, DART)


class CoverageShape(StrEnum):
    """How a source records the coverage it already answered."""

    KEYED = "keyed"  # one receipt per natural key (KRX session, DART identity, industry snapshot)
    RANGED = "ranged"  # answered [start, end] session ranges per subject (investor flow)


@dataclass(frozen=True, slots=True)
class SourceContract:
    """Storage contract of one evidence source.

    A source is provider-specific: two providers never share a source, so one
    provider's page can never be read as another's. ``row_fields`` and
    ``query_fields`` apply to ``RAW_ROWS_V1`` only.

    ``endpoints`` pins the endpoint labels a payload may declare. It is a set
    because one source may answer through more than one endpoint label
    (``kis_industry`` uses both ``inquire-price`` and ``search-stock-info``);
    an empty set means the source pins no label and the check is skipped
    (KRX and DART pages carry no endpoint label at all).
    """

    source: str
    kind: EvidenceKind
    provider: str
    endpoints: frozenset[str]
    envelope: EnvelopeFormat
    coverage: CoverageShape
    row_fields: frozenset[str]
    query_fields: frozenset[str]

    def endpoint_label(self) -> str:
        """Return the single endpoint a serialized envelope must declare.

        Raises:
            PITDataError: the source does not pin exactly one endpoint, so no
                envelope can be attributed to it.
        """
        if len(self.endpoints) != 1:
            raise PITDataError(f"source {self.source!r} does not pin exactly one endpoint")
        return next(iter(self.endpoints))


_LS_FLOW_ROWS: Final[frozenset[str]] = frozenset(
    {"date", "close", "volume", "value"}
    | {f"tjj{index:04d}" for index in range(12)}
    | {"tjj0016", "tjj0017", "tjj0018"}
)
_KIS_FLOW_ROWS: Final[frozenset[str]] = frozenset(
    {
        "stck_bsop_date",
        "prsn_ntby_qty",
        "frgn_ntby_qty",
        "orgn_ntby_qty",
        "etc_ntby_qty",
    }
)

SOURCE_CONTRACTS: Final[Mapping[str, SourceContract]] = {
    contract.source: contract
    for contract in (
        SourceContract(
            source=LS_FLOW_SOURCE,
            kind=EvidenceKind.INVESTOR_FLOW,
            provider="LS",
            endpoints=frozenset({"t1702"}),
            envelope=EnvelopeFormat.RAW_ROWS_V1,
            coverage=CoverageShape.RANGED,
            row_fields=_LS_FLOW_ROWS,
            query_fields=frozenset({"symbol", "start", "end"}),
        ),
        SourceContract(
            source=KIS_FLOW_SOURCE,
            kind=EvidenceKind.INVESTOR_FLOW,
            provider="KIS",
            endpoints=frozenset({"investor-trade-by-stock-daily"}),
            envelope=EnvelopeFormat.RAW_ROWS_V1,
            coverage=CoverageShape.RANGED,
            row_fields=_KIS_FLOW_ROWS,
            query_fields=frozenset({"symbol", "anchor"}),
        ),
        SourceContract(
            source=KIS_INDUSTRY_SOURCE,
            kind=EvidenceKind.INDUSTRY,
            provider="KIS",
            endpoints=frozenset({"inquire-price", "search-stock-info"}),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=KRX_DAILY_MARKET_SOURCE,
            kind=EvidenceKind.DAILY_MARKET,
            provider="KRX",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=KRX_HEDGE_SERIES_SOURCE,
            kind=EvidenceKind.DAILY_MARKET,
            provider="KRX",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=KRX_TREND_SERIES_SOURCE,
            kind=EvidenceKind.DAILY_MARKET,
            provider="KRX",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=KRX_CASH_SERIES_SOURCE,
            kind=EvidenceKind.DAILY_MARKET,
            provider="KRX",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=KRX_SECURITY_MASTER_SOURCE,
            kind=EvidenceKind.SECURITY_MASTER,
            provider="KRX",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=FINANCIAL_FACTS_SOURCE,
            kind=EvidenceKind.FINANCIAL_FACTS,
            provider="DART",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=DART_DISCLOSURES_SOURCE,
            kind=EvidenceKind.DISCLOSURES,
            provider="DART",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=DART_CORP_DISCLOSURES_SOURCE,
            kind=EvidenceKind.DISCLOSURES,
            provider="DART",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=DART_DISCLOSURE_WINDOWS_SOURCE,
            kind=EvidenceKind.DISCLOSURES,
            provider="DART",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=DIVIDEND_DECISION_SOURCE,
            kind=EvidenceKind.CORPORATE_ACTIONS,
            provider="DART",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=EARNINGS_RELEASE_SOURCE,
            kind=EvidenceKind.DISCLOSURES,
            provider="DART",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=DART_CORP_CODES_SOURCE,
            kind=EvidenceKind.SECURITY_MASTER,
            provider="DART",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=DART_DOCUMENT_SOURCE,
            kind=EvidenceKind.DISCLOSURES,
            provider="DART",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=KIND_NOTICE_SEARCH_SOURCE,
            kind=EvidenceKind.DISCLOSURES,
            provider="KIND",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
        SourceContract(
            source=KIND_NOTICE_DOCUMENT_SOURCE,
            kind=EvidenceKind.DISCLOSURES,
            provider="KIND",
            endpoints=frozenset(),
            envelope=EnvelopeFormat.NATIVE,
            coverage=CoverageShape.KEYED,
            row_fields=frozenset(),
            query_fields=frozenset(),
        ),
    )
}


def source_contract(source: str) -> SourceContract:
    """Return the registered contract.

    Raises:
        PITDataError: the source is not registered (fail closed; nothing unregistered is ever stored).
    """
    try:
        return SOURCE_CONTRACTS[source]
    except KeyError:
        raise PITDataError(f"unregistered evidence source {source!r}: expected one of {sorted(SOURCE_CONTRACTS)}") from None


def _canonical_bytes(document: object) -> bytes:
    try:
        return json.dumps(
            document, sort_keys=True, ensure_ascii=False, separators=_CANONICAL_SEPARATORS
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PITDataError("evidence payload is not JSON-serializable") from exc


def _load_document(payload: bytes) -> object:
    """Parse provider bytes into a JSON document.

    Raises:
        PITDataError: the payload is not valid JSON.
    """
    try:
        return json.loads(payload)
    except ValueError as exc:
        raise PITDataError("evidence payload is not valid JSON") from exc


def _require_query_fields(contract: SourceContract, query: Mapping[str, object]) -> None:
    missing = sorted(field for field in contract.query_fields if not str(query.get(field, "")).strip())
    if missing:
        raise PITDataError(f"envelope query is missing contract fields {missing} for {contract.source!r}")


def _require_row_fields(contract: SourceContract, row: Mapping[str, object], *, position: int) -> None:
    missing = sorted(field for field in contract.row_fields if field not in row)
    if missing:
        raise PITDataError(f"row {position} of {contract.source!r} is missing contract fields {missing}")


def raw_rows_envelope(
    contract: SourceContract, *, query: Mapping[str, str], rows: Sequence[Mapping[str, object]]
) -> bytes:
    """Serialize one provider response as a canonical ``RAW_ROWS_V1`` envelope.

    Rows are stored exactly as the provider returned them. No mapped or derived
    field is added, because Silver parsers are the only interpreters of
    provider rows. Row fields are a floor, not a whitelist: extra provider
    fields are kept.

    Raises:
        PITDataError: the contract is not ``RAW_ROWS_V1``, a query field is
            missing, or a row lacks a contract row field.
    """
    if contract.envelope is not EnvelopeFormat.RAW_ROWS_V1:
        raise PITDataError(f"source {contract.source!r} does not store a RAW_ROWS_V1 envelope")
    _require_query_fields(contract, query)
    stored_rows: list[Mapping[str, object]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PITDataError(f"row {position} of {contract.source!r} is not an object")
        _require_row_fields(contract, row, position=position)
        stored_rows.append(dict(row))
    return _canonical_bytes(
        {
            "envelope": EnvelopeFormat.RAW_ROWS_V1.value,
            "provider": contract.provider,
            "endpoint": contract.endpoint_label(),
            "query": {str(key): str(value) for key, value in query.items()},
            "rows": stored_rows,
        }
    )


def _validate_identity(contract: SourceContract, document: Mapping[str, object]) -> None:
    expected: dict[str, object] = {
        "envelope": contract.envelope.value,
        "provider": contract.provider,
    }
    for field, value in expected.items():
        declared = document.get(field)
        if declared is not None and str(declared) != str(value):
            raise PITDataError(f"payload declares {field} {declared!r} but {contract.source!r} is {value!r}")
    if not contract.endpoints:
        return
    declared_endpoint = document.get("endpoint")
    if declared_endpoint is not None and str(declared_endpoint) not in contract.endpoints:
        raise PITDataError(
            f"payload declares endpoint {declared_endpoint!r} but {contract.source!r} is "
            f"{sorted(contract.endpoints)}"
        )


def _validate_raw_rows(contract: SourceContract, document: Mapping[str, object], *, status: EvidenceStatus) -> None:
    _validate_identity(contract, document)
    raw_query = document.get("query")
    _require_query_fields(contract, raw_query if isinstance(raw_query, Mapping) else {})
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise PITDataError(f"envelope of {contract.source!r} does not carry a rows list")
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PITDataError(f"row {position} of {contract.source!r} is not an object")
        _require_row_fields(contract, row, position=position)
    if status is EvidenceStatus.SUCCESS and not rows:
        raise PITDataError(f"successful payload of {contract.source!r} carries no rows")
    if status is not EvidenceStatus.SUCCESS and rows:
        raise PITDataError(f"non-successful payload of {contract.source!r} carries rows")


def validate_envelope(contract: SourceContract, payload: bytes, *, status: EvidenceStatus) -> None:
    """Fail closed when a payload does not satisfy its source contract.

    A ``RAW_ROWS_V1`` payload is held to its declared provider, endpoint, query
    fields and row fields, and a payload may only claim success when it carries
    rows. A ``NATIVE`` payload keeps its provider page shape for its own Silver
    parser, so only the identity labels it chooses to declare are checked.

    Raises:
        PITDataError: invalid JSON; wrong envelope, provider or endpoint; missing
            query fields; a ``success`` payload with no rows; a row missing a
            contract field; or a payload that is not ``success`` carrying rows.
    """
    document = _load_document(payload)
    if contract.envelope is EnvelopeFormat.RAW_ROWS_V1:
        if not isinstance(document, Mapping):
            raise PITDataError(f"envelope of {contract.source!r} is not a JSON object")
        _validate_raw_rows(contract, document, status=status)
        return
    if isinstance(document, Mapping):
        _validate_identity(contract, document)


def canonical_raw_rows_bytes(contract: SourceContract, payload: bytes) -> bytes:
    """Return the canonical serialization a ``RAW_ROWS_V1`` payload must already be in.

    Raises:
        PITDataError: the payload is not valid JSON.
    """
    if contract.envelope is not EnvelopeFormat.RAW_ROWS_V1:
        return payload
    return _canonical_bytes(_load_document(payload))


def envelope_carries_rows(contract: SourceContract, payload: bytes) -> bool:
    """Whether a payload holds at least one provider row.

    The status of a stored payload is a property of its bytes, so a caller can
    never claim ``success`` for a page that answered with nothing, and a page
    that carries rows is never filed as an unanswered ``empty`` request.

    Raises:
        PITDataError: the payload is not valid JSON.
    """
    if contract.envelope is not EnvelopeFormat.RAW_ROWS_V1:
        return True
    document = _load_document(payload)
    rows = document.get("rows") if isinstance(document, Mapping) else None
    return bool(rows)
