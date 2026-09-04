"""Official DART XBRL collection preserving filing identity."""
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

from src.data.schemas import PITDataError

_REQUIRED_FACTS: tuple[str, ...] = (
    "sales",
    "gross_profit",
    "operating_profit",
    "net_income",
    "assets",
    "equity",
    "cash",
    "debt",
    "operating_cash_flow",
    "capex",
)
_REPORT_CODE_BY_KIND = {
    "사업보고서": "11011",
    "반기보고서": "11012",
    "분기보고서": None,
}
_PERIOD = re.compile(r"\((\d{4})\.(\d{2})\)")
_REPRT_QUARTER = {"11013": "Q1", "11012": "Q2", "11014": "Q3", "11011": "Q4"}


class DartXbrlCollector:
    """Filing-identity plus XBRL-facts evidence; missing XBRL blocks certification."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_json: Any | None = None,
        request_bytes: Any | None = None,
    ) -> None:
        key = api_key or os.getenv("OPENDART_API_KEY")
        if not key and request_json is None and request_bytes is None:
            raise ValueError("OPENDART_API_KEY not found in environment variables")
        self._api_key = key
        self._request_json = request_json
        self._request_bytes = request_bytes
        self._client: Any | None = None
        if request_json is None and request_bytes is None and key is not None:
            from src.integrations.dart.client import DartApiClient

            self._client = DartApiClient(api_key=key)
        if request_bytes is not None and key is not None and self._client is None:
            from src.integrations.dart.client import DartApiClient

            self._client = DartApiClient(api_key=key, request_bytes=request_bytes)

    def fetch_disclosures(self, start: date, end: date) -> Iterable[dict[str, Any]]:
        if start > end:
            raise PITDataError("coverage_start must not be after coverage_end")
        if self._client is None:
            raise PITDataError("DART disclosures endpoint is not configured")
        records = self._client.list_disclosures(start, end)
        if not records:
            raise PITDataError("DART disclosures response is empty; certification blocked")
        return ({"records": records, "start": start.isoformat(), "end": end.isoformat()},)

    @staticmethod
    def filing_identities_from_bronze(
        bronze_root: Path | str, *, start: date, end: date
    ) -> tuple[dict[str, str], ...]:
        """Select only periodic financial filings with complete OpenDART account identity."""
        paths = sorted((Path(bronze_root) / "disclosures").glob("*/payload.json"))
        if not paths:
            raise PITDataError("expected exactly one retained disclosure Bronze receipt")
        identities: list[dict[str, str]] = []
        for payload_path in paths:
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise PITDataError("retained disclosure Bronze receipt is unreadable") from exc
            records = payload.get("records") if isinstance(payload, dict) else None
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                receipt_day = str(record.get("rcept_dt") or "").strip()
                if len(receipt_day) != 8 or not receipt_day.isdigit():
                    continue
                receipt_date = date(int(receipt_day[:4]), int(receipt_day[4:6]), int(receipt_day[6:]))
                if not start <= receipt_date <= end:
                    continue
                name = str(record.get("report_nm") or "")
                matched = _PERIOD.search(name)
                corp_code = str(record.get("corp_code") or "").strip()
                filing_id = str(record.get("rcept_no") or "").strip()
                if not matched or not corp_code or not filing_id:
                    continue
                year, month = matched.groups()
                if "사업보고서" in name:
                    report_code = _REPORT_CODE_BY_KIND["사업보고서"]
                elif "반기보고서" in name:
                    report_code = _REPORT_CODE_BY_KIND["반기보고서"]
                elif "분기보고서" in name and month == "03":
                    report_code = "11013"
                elif "분기보고서" in name and month == "09":
                    report_code = "11014"
                else:
                    continue
                try:
                    published_at = date(
                        int(receipt_day[:4]), int(receipt_day[4:6]), int(receipt_day[6:])
                    ).isoformat()
                except ValueError:
                    continue
                identities.append(
                    {
                        "corp_code": corp_code,
                        "filing_id": filing_id,
                        "rcept_no": filing_id,
                        "biz_year": year,
                        "reprt_code": str(report_code),
                        "fs_div": "CFS",
                        "published_at": published_at,
                        "correction_of": str(record.get("rm") or "").strip(),
                    }
                )
        unique: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
        for identity in identities:
            unique[tuple(sorted(identity.items()))] = identity
        return tuple(unique.values())

    def fetch_xbrl_facts(self, filing_ids: tuple[Any, ...]) -> Iterable[dict[str, Any]]:
        if not filing_ids:
            raise PITDataError("DART XBRL facts require filing identities; certification blocked")
        identities: list[dict[str, str]] = []
        for item in filing_ids:
            if isinstance(item, str):
                if not item.strip():
                    raise PITDataError("DART XBRL facts require filing identity with corp_code and filing_id")
                raise PITDataError("DART XBRL facts require filing identity with corp_code and filing_id")
            if not isinstance(item, dict):
                raise PITDataError("DART XBRL facts require filing identity with corp_code and filing_id")
            corp_code = str(item.get("corp_code") or "").strip()
            filing_id = str(item.get("filing_id") or item.get("rcept_no") or "").strip()
            biz_year = str(item.get("biz_year") or item.get("bsns_year") or "").strip()
            reprt_code = str(item.get("reprt_code") or item.get("report_code") or "").strip()
            fs_div = str(item.get("fs_div") or "").strip()
            if not corp_code or not filing_id or not biz_year or not reprt_code or not fs_div:
                raise PITDataError("DART XBRL facts require filing identity with corp_code and filing_id")
            identities.append(
                {"corp_code": corp_code, "filing_id": filing_id, "biz_year": biz_year, "reprt_code": reprt_code, "fs_div": fs_div}
            )
        if not identities:
            raise PITDataError("DART XBRL facts require filing identities; certification blocked")
        if self._client is None and self._request_json is None:
            raise PITDataError("DART XBRL facts endpoint is not configured")
        pages: list[dict[str, Any]] = []
        for identity in identities:
            fid = identity["filing_id"]
            divisions = (identity["fs_div"], "OFS") if identity["fs_div"] == "CFS" else (identity["fs_div"],)
            for fs_div in divisions:
                request_identity = {**identity, "fs_div": fs_div}
                if self._request_json is not None:
                    raw = self._request_json("fnlttSinglAcntAll", dict(request_identity))
                else:
                    assert self._client is not None
                    try:
                        raw = self._client._request_validated(
                            "fnlttSinglAcntAll.json",
                            {
                                "corp_code": identity["corp_code"],
                                "bsns_year": identity["biz_year"],
                                "reprt_code": identity["reprt_code"],
                                "fs_div": fs_div,
                            },
                        )
                    except Exception as exc:
                        raise PITDataError(f"missing XBRL facts for {fid}; certification failure") from exc
                if not isinstance(raw, dict) or not raw or str(raw.get("status") or "") in {"013", "014"}:
                    continue
                facts = raw.get("list", raw.get("records", raw))
                if not facts:
                    continue
                pages.append(
                    {
                        "records": facts if isinstance(facts, list) else [facts],
                        **request_identity,
                        "raw_provenance": dict(raw),
                    }
                )
                break
            else:
                raise PITDataError(f"missing XBRL facts for {fid}; full-statement response is empty")
        if not pages:
            raise PITDataError("DART XBRL facts response is empty; certification blocked")
        return tuple(pages)

    def fetch_financial_fact_sources(
        self, identities: tuple[dict[str, str], ...]
    ) -> Iterable[dict[str, Any]]:
        """Fetch canonical fact sources; 013 alone never proves absence."""
        from src.integrations.dart.legacy_filing import (
            MAPPING_VERSION,
            map_standardized_account,
            parse_legacy_filing_archive,
        )

        if not identities:
            raise PITDataError("DART financial facts require filing identities")
        normalized: list[dict[str, str]] = []
        for item in identities:
            if not isinstance(item, dict):
                raise PITDataError("DART financial facts require filing identity")
            corp_code = str(item.get("corp_code") or "").strip()
            filing_id = str(item.get("filing_id") or item.get("rcept_no") or "").strip()
            biz_year = str(item.get("biz_year") or item.get("bsns_year") or "").strip()
            reprt_code = str(item.get("reprt_code") or item.get("report_code") or "").strip()
            fs_div = str(item.get("fs_div") or "CFS").strip() or "CFS"
            if not corp_code or not filing_id or not biz_year or not reprt_code:
                raise PITDataError("DART financial facts require filing identity")
            normalized.append(
                {
                    "corp_code": corp_code,
                    "filing_id": filing_id,
                    "rcept_no": str(item.get("rcept_no") or filing_id).strip(),
                    "biz_year": biz_year,
                    "reprt_code": reprt_code,
                    "fs_div": fs_div,
                    "published_at": str(item.get("published_at") or "").strip(),
                }
            )
        pages: list[dict[str, Any]] = []
        for identity in normalized:
            fid = identity["filing_id"]
            divisions = (identity["fs_div"], "OFS") if identity["fs_div"] == "CFS" else (identity["fs_div"],)
            standardized_hit: dict[str, Any] | None = None
            last_status = ""
            for fs_div in divisions:
                request_identity = {**identity, "fs_div": fs_div}
                if self._request_json is not None:
                    raw = self._request_json("fnlttSinglAcntAll", dict(request_identity))
                elif self._client is not None:
                    try:
                        raw = self._client._request_validated(
                            "fnlttSinglAcntAll.json",
                            {
                                "corp_code": identity["corp_code"],
                                "bsns_year": identity["biz_year"],
                                "reprt_code": identity["reprt_code"],
                                "fs_div": fs_div,
                            },
                        )
                    except Exception as exc:
                        from src.integrations.dart.client import (
                            DartApiError,
                            DartRetryableError,
                            DartTerminalError,
                        )

                        if isinstance(exc, (DartRetryableError, DartTerminalError, DartApiError, PITDataError)):
                            raise PITDataError(f"DART request failed for {fid}") from exc
                        raise PITDataError(f"DART request failed for {fid}") from exc
                else:
                    raise PITDataError("DART XBRL facts endpoint is not configured")
                if not isinstance(raw, dict) or not raw:
                    raise PITDataError(f"DART request failed for {fid}")
                status = str(raw.get("status") or "")
                last_status = status or last_status
                if status in {"013", "014"}:
                    continue
                if status != "000":
                    raise PITDataError(f"DART request failed for {fid}: status {status}")
                facts = raw.get("list", raw.get("records", []))
                if not facts:
                    continue
                rows = facts if isinstance(facts, list) else [facts]
                canonical: list[dict[str, Any]] = []
                diagnostics: list[str] = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    fact = map_standardized_account(
                        account_id=str(row.get("account_id") or row.get("accountId") or ""),
                        account_nm=str(row.get("account_nm") or row.get("account") or ""),
                    )
                    if fact is None:
                        diagnostics.append(f"unknown_account:{row.get('account_nm') or row.get('account_id')}")
                        continue
                    raw_amount = (
                        row.get("thstrm_amount")
                        if row.get("thstrm_amount") not in (None, "")
                        else row.get("thstrm_add_amount")
                    )
                    if raw_amount in (None, ""):
                        diagnostics.append(f"missing_amount:{fact}")
                        continue
                    try:
                        value = float(str(raw_amount).replace(",", "").strip())
                    except (TypeError, ValueError):
                        diagnostics.append(f"non_finite:{fact}")
                        continue
                    import math as _math

                    if not _math.isfinite(value):
                        diagnostics.append(f"non_finite:{fact}")
                        continue
                    biz_year = str(
                        row.get("bsns_year") or identity.get("biz_year") or ""
                    ).strip()
                    reprt_code = str(
                        row.get("reprt_code") or identity.get("reprt_code") or ""
                    ).strip()
                    quarter = _REPRT_QUARTER.get(reprt_code, "Q4")
                    fiscal_period = f"{biz_year}{quarter}" if biz_year else ""
                    if not fiscal_period:
                        diagnostics.append(f"missing_fiscal_period:{fact}")
                        continue
                    corp_code = str(
                        row.get("corp_code") or identity.get("corp_code") or ""
                    ).strip()
                    filing_id = str(
                        row.get("rcept_no") or identity.get("filing_id") or ""
                    ).strip()
                    fs_div = str(row.get("fs_div") or identity.get("fs_div") or "CFS").strip()
                    canonical.append(
                        {
                            **row,
                            "company_id": corp_code,
                            "corp_code": corp_code,
                            "filing_id": filing_id,
                            "fiscal_period": fiscal_period,
                            "fact": fact,
                            "value": value,
                            "unit": "KRW",
                            "consolidated": fs_div == "CFS",
                            "restatement_id": "r0",
                            "source_kind": "opendart_standard",
                            "mapping_version": MAPPING_VERSION,
                            "raw_document_hash": None,
                        }
                    )
                standardized_hit = {
                    "source_kind": "opendart_standard",
                    "status": "000",
                    "identity": dict(request_identity),
                    "records": canonical,
                    "mapping_version": MAPPING_VERSION,
                    "diagnostics": tuple(diagnostics),
                    "raw_document_hash": None,
                    "raw_provenance": dict(raw),
                    **request_identity,
                }
                break
            if standardized_hit is not None:
                pages.append(standardized_hit)
                continue
            rcept_no = identity.get("rcept_no") or fid
            try:
                if self._request_bytes is not None:
                    archive = self._request_bytes("document.xml", {"rcept_no": rcept_no})
                elif self._client is not None:
                    archive = self._client.fetch_document_archive(rcept_no)
                else:
                    raise PITDataError("DART document archive endpoint is not configured")
            except PITDataError:
                raise
            except Exception as exc:
                raise PITDataError(f"DART document archive failed for {fid}") from exc
            if not isinstance(archive, (bytes, bytearray)) or len(archive) == 0:
                pages.append(
                    {
                        "source_kind": "unavailable",
                        "status": last_status or "013",
                        "identity": dict(identity),
                        "records": [],
                        "mapping_version": MAPPING_VERSION,
                        "diagnostics": ("empty_archive",),
                        "raw_document_hash": None,
                        **identity,
                    }
                )
                continue
            if not zipfile.is_zipfile(io.BytesIO(bytes(archive))):
                pages.append(
                    {
                        "source_kind": "unavailable",
                        "status": last_status or "013",
                        "identity": dict(identity),
                        "records": [],
                        "mapping_version": MAPPING_VERSION,
                        "diagnostics": ("invalid_document_archive",),
                        "raw_document_hash": None,
                        **identity,
                    }
                )
                continue
            import hashlib

            digest = hashlib.sha256(bytes(archive)).hexdigest()
            parsed = parse_legacy_filing_archive(
                archive_bytes=bytes(archive), identity=dict(identity), document_hash=digest
            )
            if parsed.status == "extraction_failed" and not parsed.records:
                pages.append(
                    {
                        "source_kind": "legacy_document",
                        "status": "extraction_failed",
                        "identity": dict(identity),
                        "records": [],
                        "mapping_version": MAPPING_VERSION,
                        "diagnostics": tuple(parsed.diagnostics),
                        "raw_document_hash": digest,
                        "raw_archive": bytes(archive),
                        **identity,
                    }
                )
                continue
            pages.append(
                {
                    "source_kind": "legacy_document",
                    "status": last_status or "013",
                    "identity": dict(identity),
                    "records": list(parsed.records),
                    "mapping_version": MAPPING_VERSION,
                    "diagnostics": tuple(parsed.diagnostics),
                    "raw_document_hash": digest,
                    "raw_archive": bytes(archive),
                    **identity,
                }
            )
        return iter(tuple(pages))
