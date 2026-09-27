"""Source contracts are the fail-closed gate in front of Bronze."""
from __future__ import annotations

import json
from typing import Any

import pytest

from src.core.pit import EvidenceKind, PITDataError
from src.data.evidence_sources import (
    KIS_FLOW_SOURCE,
    KIS_INDUSTRY_SOURCE,
    LS_FLOW_SOURCE,
    SOURCE_CONTRACTS,
    CoverageShape,
    EnvelopeFormat,
    raw_rows_envelope,
    source_contract,
    validate_envelope,
)
from src.data.receipt_catalog import EvidenceStatus


def _ls_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "date": "20260904",
        "close": 70100,
        "volume": 12_000_000,
        "value": 840_000_000_000,
    }
    row.update({f"tjj{index:04d}": index for index in range(12)})
    row.update({"tjj0016": 30, "tjj0017": 50, "tjj0018": -20})
    row.update(overrides)
    return row


def _kis_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "stck_bsop_date": "20260904",
        "prsn_ntby_qty": "-120000",
        "frgn_ntby_qty": "30000",
        "orgn_ntby_qty": "-20000",
        "etc_ntby_qty": "110000",
    }
    row.update(overrides)
    return row


def _ls_envelope(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(
        raw_rows_envelope(
            source_contract(LS_FLOW_SOURCE),
            query={"symbol": "005930", "start": "2026-09-01", "end": "2026-09-04"},
            rows=[_ls_row()],
        )
    )
    document.update(overrides)
    return document


def _rebuild(document: dict[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def test_unregistered_source_is_rejected() -> None:
    with pytest.raises(PITDataError, match="unregistered evidence source"):
        source_contract("investor_flow")


def test_every_registered_source_declares_one_shape() -> None:
    assert set(SOURCE_CONTRACTS) >= {LS_FLOW_SOURCE, KIS_FLOW_SOURCE, KIS_INDUSTRY_SOURCE}
    for contract in SOURCE_CONTRACTS.values():
        assert contract.coverage in set(CoverageShape)
        assert contract.envelope in set(EnvelopeFormat)
        assert contract.kind in set(EvidenceKind)
    assert source_contract(LS_FLOW_SOURCE).coverage is CoverageShape.RANGED
    assert source_contract(KIS_FLOW_SOURCE).coverage is CoverageShape.RANGED
    assert source_contract(KIS_INDUSTRY_SOURCE).coverage is CoverageShape.KEYED
    assert source_contract(KIS_INDUSTRY_SOURCE).envelope is EnvelopeFormat.NATIVE
    assert source_contract(LS_FLOW_SOURCE).endpoint_label() == "t1702"
    assert source_contract(KIS_FLOW_SOURCE).endpoint_label() == "investor-trade-by-stock-daily"


def test_kis_value_only_row_is_not_evidence() -> None:
    """KIS answers amounts as well as quantities; an amount is not a tradable quantity."""
    contract = source_contract(KIS_FLOW_SOURCE)
    priced = {
        "stck_bsop_date": "20260904",
        "prsn_ntby_tr_pbmn": "840000000000",
        "frgn_ntby_tr_pbmn": "0",
        "orgn_ntby_tr_pbmn": "0",
        "etc_ntby_tr_pbmn": "0",
    }

    with pytest.raises(PITDataError, match="prsn_ntby_qty"):
        raw_rows_envelope(
            contract, query={"symbol": "005930", "anchor": "20260904"}, rows=[priced]
        )


def test_ls_row_without_a_contract_field_is_rejected() -> None:
    contract = source_contract(LS_FLOW_SOURCE)
    row = _ls_row()
    del row["tjj0011"]

    with pytest.raises(PITDataError, match="tjj0011"):
        raw_rows_envelope(
            contract, query={"symbol": "005930", "start": "2026-09-04", "end": "2026-09-04"}, rows=[row]
        )


def test_extra_provider_fields_are_kept() -> None:
    contract = source_contract(LS_FLOW_SOURCE)
    payload = raw_rows_envelope(
        contract,
        query={"symbol": "005930", "start": "2026-09-04", "end": "2026-09-04"},
        rows=[_ls_row(provider_only_field="kept")],
    )
    row = json.loads(payload)["rows"][0]

    assert row["provider_only_field"] == "kept"
    validate_envelope(contract, payload, status=EvidenceStatus.SUCCESS)


def test_missing_query_field_is_rejected() -> None:
    contract = source_contract(KIS_FLOW_SOURCE)

    with pytest.raises(PITDataError, match="anchor"):
        raw_rows_envelope(contract, query={"symbol": "005930"}, rows=[_kis_row()])


def test_envelope_requires_a_row_object_not_a_scalar() -> None:
    contract = source_contract(KIS_FLOW_SOURCE)

    with pytest.raises(PITDataError, match="not an object"):
        raw_rows_envelope(  # type: ignore[arg-type]
            contract, query={"symbol": "005930", "anchor": "20260904"}, rows=["not-a-row"]  # type: ignore[list-item]
        )


def test_native_source_refuses_a_raw_rows_envelope() -> None:
    with pytest.raises(PITDataError, match="does not store a RAW_ROWS_V1 envelope"):
        raw_rows_envelope(source_contract(KIS_INDUSTRY_SOURCE), query={}, rows=[])


def test_success_needs_rows() -> None:
    contract = source_contract(KIS_FLOW_SOURCE)
    payload = raw_rows_envelope(
        contract, query={"symbol": "005930", "anchor": "20260904"}, rows=[]
    )

    validate_envelope(contract, payload, status=EvidenceStatus.EMPTY)
    with pytest.raises(PITDataError, match="carries no rows"):
        validate_envelope(contract, payload, status=EvidenceStatus.SUCCESS)


def test_empty_carries_no_rows() -> None:
    contract = source_contract(LS_FLOW_SOURCE)
    payload = raw_rows_envelope(
        contract, query={"symbol": "005930", "start": "2026-09-04", "end": "2026-09-04"}, rows=[_ls_row()]
    )

    with pytest.raises(PITDataError, match="carries rows"):
        validate_envelope(contract, payload, status=EvidenceStatus.EMPTY)
    with pytest.raises(PITDataError, match="carries rows"):
        validate_envelope(contract, payload, status=EvidenceStatus.PROVIDER_ERROR)


def test_foreign_provider_is_rejected() -> None:
    kis = source_contract(KIS_FLOW_SOURCE)
    ls = source_contract(LS_FLOW_SOURCE)
    ls_payload = raw_rows_envelope(
        ls, query={"symbol": "005930", "start": "2026-09-04", "end": "2026-09-04"}, rows=[_ls_row()]
    )
    kis_payload = raw_rows_envelope(
        kis, query={"symbol": "005930", "anchor": "20260904"}, rows=[_kis_row()]
    )

    with pytest.raises(PITDataError, match="provider"):
        validate_envelope(kis, ls_payload, status=EvidenceStatus.SUCCESS)
    with pytest.raises(PITDataError, match="endpoint"):
        validate_envelope(kis, _replace(kis_payload, endpoint="t1702"), status=EvidenceStatus.SUCCESS)
    with pytest.raises(PITDataError, match="provider"):
        validate_envelope(ls, _replace(ls_payload, provider="KIS"), status=EvidenceStatus.SUCCESS)


def _replace(payload: bytes, **fields: str) -> bytes:
    document = json.loads(payload)
    document.update(fields)
    return _rebuild(document)


def test_native_page_is_checked_only_where_it_declares_identity() -> None:
    contract = source_contract(KIS_INDUSTRY_SOURCE)

    validate_envelope(contract, b'{"records": []}', status=EvidenceStatus.EMPTY)
    validate_envelope(contract, b'[{"ticker": "005930"}]', status=EvidenceStatus.SUCCESS)
    validate_envelope(contract, b'{"provider": "KIS", "endpoint": "inquire-price"}', status=EvidenceStatus.SUCCESS)
    validate_envelope(contract, b'{"provider": "KIS", "endpoint": "search-stock-info"}', status=EvidenceStatus.SUCCESS)

    with pytest.raises(PITDataError, match="provider"):
        validate_envelope(contract, b'{"provider": "LS"}', status=EvidenceStatus.SUCCESS)
    with pytest.raises(PITDataError, match="endpoint"):
        validate_envelope(contract, b'{"provider": "KIS", "endpoint": "t1702"}', status=EvidenceStatus.SUCCESS)
    with pytest.raises(PITDataError, match="envelope"):
        validate_envelope(contract, b'{"envelope": "raw-rows-v1"}', status=EvidenceStatus.SUCCESS)


def test_invalid_json_fails_closed() -> None:
    with pytest.raises(PITDataError, match="not valid JSON"):
        validate_envelope(source_contract(KIS_INDUSTRY_SOURCE), b"{not json", status=EvidenceStatus.SUCCESS)
    with pytest.raises(PITDataError, match="not a JSON object"):
        validate_envelope(source_contract(LS_FLOW_SOURCE), b"[]", status=EvidenceStatus.EMPTY)
    with pytest.raises(PITDataError, match="does not carry a rows list"):
        validate_envelope(
            source_contract(LS_FLOW_SOURCE), b'{"query":{"symbol":"a","start":"b","end":"c"}}', status=EvidenceStatus.EMPTY
        )
    with pytest.raises(PITDataError, match="query is missing"):
        validate_envelope(
            source_contract(LS_FLOW_SOURCE),
            _rebuild(_ls_envelope(query={"symbol": "005930"})),
            status=EvidenceStatus.SUCCESS,
        )


def test_canonical_bytes_ignore_key_order() -> None:
    contract = source_contract(LS_FLOW_SOURCE)
    query = {"symbol": "005930", "start": "2026-09-04", "end": "2026-09-04"}
    first = raw_rows_envelope(contract, query=query, rows=[_ls_row(tjj0000=1, tjj0001=2)])
    second = raw_rows_envelope(
        contract, query={"end": "2026-09-04", "start": "2026-09-04", "symbol": "005930"}, rows=[_ls_row(tjj0001=2, tjj0000=1)]
    )

    assert first == second
    assert b"\\u" not in first
    assert b", " not in first
    assert b": " not in first


def test_canonical_bytes_keep_non_ascii_unescaped() -> None:
    contract = source_contract(LS_FLOW_SOURCE)
    payload = raw_rows_envelope(
        contract,
        query={"symbol": "005930", "start": "2026-09-04", "end": "2026-09-04"},
        rows=[_ls_row()],
    )
    document = json.loads(payload)
    document["rows"][0]["note"] = "삼성전자"

    from src.data.evidence_sources import canonical_raw_rows_bytes

    assert "삼성전자".encode() in canonical_raw_rows_bytes(contract, _rebuild(document))
    assert canonical_raw_rows_bytes(contract, _rebuild(document)) == _rebuild(document)


def test_endpoint_label_requires_exactly_one_pinned_endpoint() -> None:
    with pytest.raises(PITDataError, match="does not pin exactly one endpoint"):
        source_contract(KIS_INDUSTRY_SOURCE).endpoint_label()
    with pytest.raises(PITDataError, match="does not pin exactly one endpoint"):
        source_contract("krx_daily_market").endpoint_label()


def test_unserializable_row_value_fails_closed() -> None:
    contract = source_contract(LS_FLOW_SOURCE)

    with pytest.raises(PITDataError, match="not JSON-serializable"):
        raw_rows_envelope(
            contract,
            query={"symbol": "005930", "start": "2026-09-04", "end": "2026-09-04"},
            rows=[_ls_row(unserializable={1, 2})],
        )


def test_validated_rows_must_all_be_objects() -> None:
    contract = source_contract(KIS_FLOW_SOURCE)
    payload = _rebuild(
        {
            "envelope": "raw-rows-v1",
            "provider": "KIS",
            "endpoint": "investor-trade-by-stock-daily",
            "query": {"symbol": "005930", "anchor": "20260904"},
            "rows": ["not-a-row"],
        }
    )

    with pytest.raises(PITDataError, match="row 0 of 'kis_investor_flow' is not an object"):
        validate_envelope(contract, payload, status=EvidenceStatus.SUCCESS)


def test_native_payload_always_counts_as_carrying_rows() -> None:
    from src.data.evidence_sources import envelope_carries_rows

    assert envelope_carries_rows(source_contract(KIS_INDUSTRY_SOURCE), b'{"records": []}') is True
    assert envelope_carries_rows(
        source_contract(LS_FLOW_SOURCE), b'{"rows": []}'
    ) is False
    assert envelope_carries_rows(
        source_contract(LS_FLOW_SOURCE), b'{"rows": [{"date": "20260904"}]}'
    ) is True
    with pytest.raises(PITDataError, match="not valid JSON"):
        envelope_carries_rows(source_contract(LS_FLOW_SOURCE), b"{oops")
