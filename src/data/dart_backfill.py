"""PIT-safe DART historical fact backfill with KRX ticker bridge."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from src.data.collection import collect_dart_disclosures, collect_dart_financial_facts
from src.data.receipt_catalog import ReceiptCatalog
from src.data.research_scope import PRIMARY_DART_KEY_ENV
from src.data.runtime import DataRuntime
from src.data.schemas import PITDataError
from src.data.scope_coverage import CoverageRequirement
from src.data.scoped_ingestion import FACT_SOURCE, dart_fact_natural_key
from src.integrations.dart.client import DartCorpCodeRecord, dart_quota_provider
from src.integrations.dart.xbrl import DartXbrlCollector
from src.integrations.quota import ProviderQuotaStateStore

__all__ = [
    "DartFactBatchPlan",
    "DartHistoricalBackfillPlan",
    "DartHistoricalBackfillRequest",
    "DartMissingFactsPlan",
    "DartMissingFactsRequest",
    "SingleAccountBackfillRequest",
    "build_dart_historical_backfill_plan",
    "build_scoped_dart_collector",
    "build_scoped_dart_fact_batch",
    "build_single_account_request_plan",
    "run_dart_historical_backfill_batch",
    "run_dart_missing_facts_batch",
    "scoped_dart_request_headroom",
]


@dataclass(frozen=True, slots=True)
class SingleAccountBackfillRequest:
    corp_code: str
    biz_year: int
    reprt_code: str
    fs_div: str

    def __post_init__(self) -> None:
        import re as _re

        if not _re.fullmatch(r"\d{8}", str(self.corp_code)):
            raise ValueError(f"invalid corp_code {self.corp_code!r}: must be 8 digits")
        if not (2000 <= int(self.biz_year) <= 2100):
            raise ValueError(f"invalid fiscal_year {self.biz_year!r}: must be within 2000..2100")
        if str(self.reprt_code) not in ("11011", "11012", "11013", "11014"):
            raise ValueError(f"invalid reprt_code {self.reprt_code!r}")
        if str(self.fs_div) not in ("CFS", "OFS"):
            raise ValueError(f"invalid fs_div {self.fs_div!r}: must be CFS or OFS")


_REPRT_ORDER = ("11013", "11012", "11014", "11011")


def build_single_account_request_plan(
    *,
    corp_codes: tuple[str, ...],
    first_fiscal_year: int,
    last_fiscal_year: int,
    daily_call_budget: int = 20000,
    include_separate_fallback: bool = True,
) -> tuple[tuple[SingleAccountBackfillRequest, ...], ...]:
    """Plan resumable fnlttSinglAcntAll batches under the OpenDART daily quota."""
    import re as _re

    if not corp_codes:
        raise ValueError("corp_codes must be non-empty")
    for code in corp_codes:
        if not _re.fullmatch(r"\d{8}", str(code)):
            raise ValueError(f"invalid corp_code {code!r}: must be 8 digits")
    if first_fiscal_year > last_fiscal_year:
        raise ValueError(f"invalid fiscal_year range {first_fiscal_year} > {last_fiscal_year}")
    if daily_call_budget < 1:
        raise ValueError(f"invalid daily_call_budget {daily_call_budget!r}: must be >= 1")
    unique_codes = sorted(set(corp_codes))
    flat: list[SingleAccountBackfillRequest] = []
    for corp_code in unique_codes:
        for biz_year in range(first_fiscal_year, last_fiscal_year + 1):
            for reprt_code in _REPRT_ORDER:
                flat.append(
                    SingleAccountBackfillRequest(
                        corp_code=corp_code, biz_year=biz_year, reprt_code=reprt_code, fs_div="CFS"
                    )
                )
                if include_separate_fallback:
                    flat.append(
                        SingleAccountBackfillRequest(
                            corp_code=corp_code, biz_year=biz_year, reprt_code=reprt_code, fs_div="OFS"
                        )
                    )
    batches = [
        tuple(flat[start : start + daily_call_budget])
        for start in range(0, len(flat), daily_call_budget)
    ]
    return tuple(batches)


@dataclass(frozen=True, slots=True)
class DartHistoricalBackfillRequest:
    bronze_root: Path
    artifact_root: Path
    silver_root: Path
    validation_start: date
    validation_end: date
    retrieved_at: datetime
    offset: int
    limit: int


@dataclass(frozen=True, slots=True)
class DartHistoricalBackfillPlan:
    plan_id: str
    required_periods: tuple[str, ...]
    ticker_by_corp_code: Mapping[str, str]
    unresolved_tickers: tuple[str, ...]
    identities: tuple[Mapping[str, str], ...]
    corp_code_receipt_hash: str


@dataclass(frozen=True, slots=True)
class DartMissingFactsRequest:
    bronze_root: Path
    artifact_root: Path
    backfill_artifact: Path
    retrieved_at: datetime
    offset: int
    limit: int


@dataclass(frozen=True, slots=True)
class DartMissingFactsPlan:
    plan_id: str
    candidate_count: int
    selected_identities: tuple[Mapping[str, str], ...]
    missing_without_filing_count: int


# 4분기 TTM 윈도우에 전년동기 이익모멘텀 비교 분기(latest - 4)를 더한 5분기
QVEF_FUNDAMENTAL_LOOKBACK_QUARTERS = 5


def _prev_quarter(period: str) -> str:
    year = int(period[:4])
    q = int(period[5])
    total = year * 4 + (q - 1) - 1
    return f"{total // 4}Q{(total % 4) + 1}"


def _quarters_back(latest: str, n: int) -> tuple[str, ...]:
    out = [latest]
    while len(out) < n:
        out.append(_prev_quarter(out[-1]))
    return tuple(reversed(out))


def _publication_cutoff(period: str) -> date:
    year = int(period[:4])
    q = int(period[5])
    if q == 1:
        return date(year, 5, 15)
    if q == 2:
        return date(year, 8, 15)
    if q == 3:
        return date(year, 11, 15)
    return date(year + 1, 3, 30)


def _latest_available_quarter(validation_start: date) -> str:
    y, m = validation_start.year, validation_start.month
    # Start from the quarter containing validation_start, walk back.
    q = (m - 1) // 3 + 1
    cur = f"{y}Q{q}"
    for _ in range(12):
        if _publication_cutoff(cur) < validation_start:
            return cur
        cur = _prev_quarter(cur)
    return cur


def build_dart_historical_backfill_plan(
    *,
    security_master: pl.DataFrame,
    corp_code_records: tuple[DartCorpCodeRecord, ...],
    validation_start: date,
    validation_end: date,
    corp_code_receipt_hash: str,
) -> DartHistoricalBackfillPlan:
    import re as _re

    if security_master.is_empty():
        raise PITDataError("security master is absent; backfill blocked")
    ticker_re = _re.compile(r"^\d{6}$")
    # Only common-share instruments available at validation_start.
    # security_master republishes identical dimension rows per session (one row per
    # instrument per day); deduping on exactly the columns the loop below reads before
    # materializing Python dicts collapses millions of duplicate daily rows to one per
    # distinct (ticker, share_class, validity window) combination without changing which
    # tickers pass the filter.
    rows = security_master.select("ticker", "share_class", "available_at", "valid_from", "valid_to").unique().to_dicts()
    tickers: set[str] = set()
    for row in rows:
        share = str(row.get("share_class") or "").strip().lower()
        if share != "common":
            continue
        avail = row.get("available_at")
        if isinstance(avail, datetime) and avail.date() > validation_start:
            continue
        if isinstance(avail, date) and not isinstance(avail, datetime) and avail > validation_start:
            continue
        vf = row.get("valid_from")
        if isinstance(vf, datetime) and vf.date() > validation_start:
            continue
        if isinstance(vf, date) and not isinstance(vf, datetime) and vf > validation_start:
            continue
        vt = row.get("valid_to")
        if isinstance(vt, datetime) and vt.date() < validation_start:
            continue
        if isinstance(vt, date) and not isinstance(vt, datetime) and vt < validation_start:
            continue
        t = str(row.get("ticker") or "").strip()
        if t and ticker_re.match(t):
            tickers.add(t)
    if not tickers:
        raise PITDataError("security master has no common-share tickers")
    # Exact ticker bridge; reject conflicting mappings.
    code_by_ticker: dict[str, str] = {}
    name_by_ticker: dict[str, str] = {}
    for rec in corp_code_records:
        t = str(rec.ticker).strip()
        c = str(rec.corp_code).strip()
        if not t or not ticker_re.match(t):
            continue
        if not c:
            continue
        prev = code_by_ticker.get(t)
        if prev is not None and prev != c:
            raise PITDataError(f"ticker {t} maps to multiple corp codes")
        code_by_ticker[t] = c
        name_by_ticker[t] = str(rec.corp_name)
    ticker_by_corp: dict[str, str] = {}
    for t in sorted(tickers):
        mapped = code_by_ticker.get(t)
        if mapped is None:
            continue
        if mapped in ticker_by_corp and ticker_by_corp[mapped] != t:
            raise PITDataError(f"corp code {mapped} maps to multiple tickers")
        ticker_by_corp[mapped] = t
    unresolved = tuple(sorted(t for t in sorted(tickers) if t not in code_by_ticker))
    latest = _latest_available_quarter(validation_start)
    required = _quarters_back(latest, QVEF_FUNDAMENTAL_LOOKBACK_QUARTERS)
    canonical = json.dumps(
        {
            "required_periods": list(required),
            "ticker_by_corp_code": {k: ticker_by_corp[k] for k in sorted(ticker_by_corp)},
            "unresolved_tickers": list(unresolved),
            "validation_start": validation_start.isoformat(),
            "validation_end": validation_end.isoformat(),
            "corp_code_receipt_hash": corp_code_receipt_hash,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    plan_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return DartHistoricalBackfillPlan(
        plan_id=plan_id,
        required_periods=required,
        ticker_by_corp_code=dict(ticker_by_corp),
        unresolved_tickers=unresolved,
        identities=(),
        corp_code_receipt_hash=corp_code_receipt_hash,
    )


def _load_security_master(silver_root: Path, *, decision_time: datetime) -> pl.DataFrame:
    """Load exactly the latest certified security_master dataset (PIT-safe, single version).

    ``security_master`` accumulates one immutable dataset directory per publish; loading
    every directory under the table root (as opposed to selecting one by id) silently
    concatenates duplicate historical snapshots of the same reference data, multiplying
    row count and memory use by the number of retained versions with no benefit.
    """
    from src.data.schemas import SilverTable
    from src.data.silver import latest_silver_dataset_path, load_silver_table_by_dataset_id

    root = Path(silver_root) / "security_master"
    if root.exists() and any(root.iterdir()):
        dataset_path = latest_silver_dataset_path(
            root=Path(silver_root), table=SilverTable.SECURITY_MASTER, decision_time=decision_time
        )
        return load_silver_table_by_dataset_id(
            root=Path(silver_root),
            table=SilverTable.SECURITY_MASTER,
            dataset_id=dataset_path.name,
            decision_time=decision_time,
        )
    # Fallback: silver_root directly holds parquet files (used by tests with a flat layout).
    files = list(Path(silver_root).rglob("*.parquet"))
    if files:
        frames = [pl.read_parquet(p) for p in files]
        return pl.concat(frames, how="diagonal_relaxed")
    raise PITDataError("security master is absent; backfill blocked")


def _persist_corp_code_receipt(
    *, records: tuple[DartCorpCodeRecord, ...], artifact_root: Path, retrieved_at: datetime, bronze_root: Path | None = None
) -> str:
    serial = [{"ticker": r.ticker, "corp_code": r.corp_code, "corp_name": r.corp_name} for r in records]
    text = json.dumps(sorted(serial, key=lambda r: r["ticker"]), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out_dir = Path(artifact_root) / "dart_backfill" / "corp_code_receipts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{digest}.json").write_text(
        json.dumps({"retrieved_at": retrieved_at.isoformat(), "records": serial}, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    if bronze_root is not None:
        bronze_dir = Path(bronze_root) / "dart_corp_codes" / digest
        bronze_dir.mkdir(parents=True, exist_ok=True)
        (bronze_dir / "payload.json").write_text(text, encoding="utf-8")
    return digest


def _endpoint_key(identity: Mapping[str, str]) -> tuple[str, str, str, str]:
    return (
        str(identity.get("corp_code") or ""),
        str(identity.get("biz_year") or ""),
        str(identity.get("reprt_code") or ""),
        str(identity.get("fs_div") or "CFS"),
    )


def _dedupe_endpoint_identities(
    identities: tuple[dict[str, str], ...],
) -> tuple[dict[str, str], ...]:
    selected: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for identity in identities:
        key = _endpoint_key(identity)
        current = selected.get(key)
        if current is None or (
            str(identity.get("published_at") or ""), str(identity.get("filing_id") or "")
        ) > (
            str(current.get("published_at") or ""), str(current.get("filing_id") or "")
        ):
            selected[key] = identity
    return tuple(selected[key] for key in sorted(selected))


def _successful_fact_endpoints(bronze_root: Path) -> frozenset[tuple[str, str, str, str]]:
    covered: set[tuple[str, str, str, str]] = set()
    for payload_path in (Path(bronze_root) / "financial_facts").glob("*/payload.json"):
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PITDataError("financial-facts Bronze payload is unreadable") from exc
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip()
        records = payload.get("records")
        if status in {"013", "014", "extraction_failed"} or not isinstance(records, list) or not records:
            continue
        endpoint = _endpoint_key({
            "corp_code": str(payload.get("corp_code") or ""),
            "biz_year": str(payload.get("biz_year") or ""),
            "reprt_code": str(payload.get("reprt_code") or ""),
            "fs_div": str(payload.get("fs_div") or "CFS"),
        })
        if all(endpoint[:3]):
            covered.add(endpoint)
    return frozenset(covered)


def run_dart_missing_facts_batch(
    *, request: DartMissingFactsRequest, dart: DartXbrlCollector
) -> DartMissingFactsPlan:
    """Collect retained periodic filing identities lacking successful fact evidence.

    This reuses Bronze disclosures and therefore never invokes the costly DART
    disclosure-list endpoint. Filing periods absent from retained disclosures
    are reported separately because no fact request can identify them safely.
    """
    if request.retrieved_at.tzinfo is None:
        raise PITDataError("retrieved_at must be timezone-aware")
    if request.offset < 0 or request.limit < 1:
        raise PITDataError("offset must be nonnegative and limit must be positive")
    try:
        artifact = json.loads(Path(request.backfill_artifact).read_text(encoding="utf-8"))
        mapping = {str(code): str(ticker) for code, ticker in artifact["ticker_by_corp_code"].items()}
        periods = frozenset(str(item) for item in artifact["required_periods"])
        validation_start = date.fromisoformat(str(artifact["validation_start"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PITDataError("invalid DART backfill artifact") from exc
    if not mapping or not periods:
        raise PITDataError("invalid DART backfill artifact")
    identities = DartXbrlCollector.filing_identities_from_bronze(
        Path(request.bronze_root), start=date(min(int(period[:4]) for period in periods), 1, 1),
        end=validation_start, ticker_by_corp_code=mapping, required_periods=periods,
        corp_codes=frozenset(mapping),
    )
    identities = _dedupe_endpoint_identities(tuple(dict(item) for item in identities))
    covered = _successful_fact_endpoints(Path(request.bronze_root))
    candidates = tuple(item for item in identities if _endpoint_key(item) not in covered)
    observed = {(str(item.get("ticker") or ""), str(item.get("fiscal_period") or "")) for item in identities}
    missing_without_filing = len({(ticker, period) for ticker in mapping.values() for period in periods} - observed)
    selected = candidates[request.offset : request.offset + request.limit]
    if not selected:
        raise PITDataError("requested missing-facts batch is empty")
    result = collect_dart_financial_facts(
        dart=dart, identities=tuple(dict(item) for item in selected),
        bronze_root=Path(request.bronze_root), retrieved_at=request.retrieved_at,
    )
    digest = hashlib.sha256()
    for item in selected:
        digest.update(json.dumps(item, sort_keys=True).encode("utf-8"))
        digest.update(b"\x00")
    plan_id = f"missing-facts-{digest.hexdigest()[:16]}"
    output = {
        "plan_id": plan_id, "candidate_count": len(candidates),
        "selected_identities": [dict(item) for item in selected],
        "missing_without_filing_count": missing_without_filing,
        "offset": request.offset, "limit": request.limit,
        "financial_fact_content_hash": result.content_hash,
    }
    target = Path(request.artifact_root) / "dart_missing_facts"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{plan_id}.json").write_text(json.dumps(output, sort_keys=True, indent=2), encoding="utf-8")
    return DartMissingFactsPlan(plan_id, len(candidates), selected, missing_without_filing)


def run_dart_historical_backfill_batch(
    *, request: DartHistoricalBackfillRequest, dart: DartXbrlCollector
) -> DartHistoricalBackfillPlan:
    if request.retrieved_at.tzinfo is None:
        raise PITDataError("retrieved_at must be timezone-aware")
    if request.offset < 0 or request.limit < 1:
        raise PITDataError("offset must be nonnegative and limit must be positive")
    master = _load_security_master(request.silver_root, decision_time=request.retrieved_at)
    if master.is_empty():
        raise PITDataError("security master is absent; backfill blocked")
    existing = sorted((Path(request.bronze_root) / "dart_corp_codes").glob("*/payload.json"))
    if existing:
        payload_path = existing[-1]
        receipt_hash = payload_path.parent.name
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        records = tuple(
            DartCorpCodeRecord(
                ticker=str(row.get("ticker") or ""),
                corp_code=str(row.get("corp_code") or ""),
                corp_name=str(row.get("corp_name") or ""),
            )
            for row in payload
            if isinstance(row, dict)
        )
    else:
        records = tuple(dart.fetch_corp_code_records())
        receipt_hash = _persist_corp_code_receipt(
            records=records,
            artifact_root=request.artifact_root,
            bronze_root=request.bronze_root,
            retrieved_at=request.retrieved_at,
        )
    if not records:
        raise PITDataError("corp code map is absent; backfill blocked")
    base = build_dart_historical_backfill_plan(
        security_master=master,
        corp_code_records=records,
        validation_start=request.validation_start,
        validation_end=request.validation_end,
        corp_code_receipt_hash=receipt_hash,
    )
    if not base.ticker_by_corp_code:
        raise PITDataError("ticker bridge has zero mappings; backfill blocked")
    sorted_codes = tuple(sorted(base.ticker_by_corp_code.keys()))
    batch_codes = sorted_codes[request.offset : request.offset + request.limit]
    if not batch_codes:
        raise PITDataError("requested backfill batch is empty")
    from src.data.research_period import OPENDART_FIRST_FISCAL_YEAR

    fetch_start = date(OPENDART_FIRST_FISCAL_YEAR - 1, 1, 1)
    fetch_end = request.retrieved_at.date()
    coverage_start = date(request.validation_start.year - 2, 1, 1)
    coverage_end = request.validation_start
    collect_dart_disclosures(dart=dart, start=fetch_start, end=fetch_end, bronze_root=Path(request.bronze_root), retrieved_at=request.retrieved_at, corp_codes=tuple(sorted(batch_codes)))
    all_identities = DartXbrlCollector.filing_identities_from_bronze(
        Path(request.bronze_root),
        start=coverage_start,
        end=coverage_end,
        ticker_by_corp_code=dict(base.ticker_by_corp_code),
        required_periods=frozenset(base.required_periods),
        corp_codes=frozenset(batch_codes),
    )
    batch_code_set = set(batch_codes)
    identities = tuple(
        identity
        for identity in all_identities
        if str(identity.get("corp_code") or "").strip() in batch_code_set
    )
    identities = _dedupe_endpoint_identities(identities)
    # Freeze the selected mapping and identities before any statement request.
    pending_dir = Path(request.artifact_root) / "dart_backfill"
    pending_dir.mkdir(parents=True, exist_ok=True)
    (pending_dir / f"{base.plan_id}.json").write_text(
        json.dumps(
            {
                "plan_id": base.plan_id,
                "status": "pending_fact_collection",
                "required_periods": list(base.required_periods),
                "ticker_by_corp_code": {
                    k: base.ticker_by_corp_code[k]
                    for k in sorted(base.ticker_by_corp_code)
                },
                "unresolved_tickers": list(base.unresolved_tickers),
                "corp_code_receipt_hash": receipt_hash,
                "identities": [dict(identity) for identity in identities],
                "offset": request.offset,
                "limit": request.limit,
            },
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    fact_artifact = None
    if identities:
        fact_artifact = collect_dart_financial_facts(
            dart=dart,
            identities=tuple(dict(identity) for identity in identities),
            bronze_root=Path(request.bronze_root),
            retrieved_at=request.retrieved_at,
        )
    plan = DartHistoricalBackfillPlan(
        plan_id=base.plan_id,
        required_periods=base.required_periods,
        ticker_by_corp_code=base.ticker_by_corp_code,
        unresolved_tickers=base.unresolved_tickers,
        identities=tuple(dict(i) for i in identities),
        corp_code_receipt_hash=receipt_hash,
    )
    # Bronze receipt hashes for provenance.
    disc_dir = Path(request.bronze_root) / "disclosures"
    receipt_hashes: list[str] = (
        sorted(p.name for p in disc_dir.iterdir() if p.is_dir() and len(p.name) == 64)
        if disc_dir.exists()
        else []
    )
    # Ticker-period coverage (incomplete coverage reported, never promoted).
    covered: dict[str, set[str]] = {}
    for ident in plan.identities:
        t = str(ident.get("ticker") or "")
        p = str(ident.get("fiscal_period") or "")
        if t and p:
            covered.setdefault(t, set()).add(p)
    coverage = {
        t: {p: (p in covered.get(t, set())) for p in plan.required_periods}
        for t in sorted(set(base.ticker_by_corp_code.values()) & set(covered.keys()) | set())
    }
    # Per-ticker incomplete report includes batch tickers even with zero coverage.
    batch_tickers = sorted({base.ticker_by_corp_code[c] for c in batch_codes})
    for t in batch_tickers:
        coverage.setdefault(t, {p: (p in covered.get(t, set())) for p in plan.required_periods})
    payload = {
        "plan_id": plan.plan_id,
        "required_periods": list(plan.required_periods),
        "ticker_by_corp_code": {k: plan.ticker_by_corp_code[k] for k in sorted(plan.ticker_by_corp_code)},
        "unresolved_tickers": list(plan.unresolved_tickers),
        "corp_code_receipt_hash": receipt_hash,
        "identities": [dict(i) for i in plan.identities],
        "offset": request.offset,
        "limit": request.limit,
        "batch_corp_codes": list(batch_codes),
        "bronze_receipt_hashes": sorted(receipt_hashes),
        "financial_fact_content_hash": (
            fact_artifact.content_hash if fact_artifact is not None else None
        ),
        "source_counts": {"identities": len(plan.identities), "batch_corp_codes": len(batch_codes)},
        "ticker_period_coverage": {k: dict(v) for k, v in sorted(coverage.items())},
        "validation_start": request.validation_start.isoformat(),
        "validation_end": request.validation_end.isoformat(),
    }
    out_dir = Path(request.artifact_root) / "dart_backfill"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{plan.plan_id}.json").write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return plan


_REPRT_QUARTER = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}
_FISCAL_PATTERN = re.compile(r"\d{4}Q[1-4]")


def _period_key(period: str) -> int:
    return int(period[:4]) * 4 + int(period[5])


def _identity_fiscal_period(identity: Mapping[str, str]) -> str:
    raw = str(identity.get("fiscal_period") or "").strip()
    if raw:
        if not _FISCAL_PATTERN.fullmatch(raw):
            raise PITDataError(f"invalid fiscal period {raw!r}")
        return raw
    biz_year = str(identity.get("biz_year") or "").strip()
    reprt_code = str(identity.get("reprt_code") or "").strip()
    quarter = _REPRT_QUARTER.get(reprt_code)
    if not biz_year.isdigit() or quarter is None:
        raise PITDataError("DART fact identity is missing a derivable fiscal period")
    return f"{int(biz_year)}Q{quarter}"


@dataclass(frozen=True, slots=True)
class DartFactBatchPlan:
    """Quota-bounded DART fact identities selected from in-scope filing evidence."""

    scope_hash: str
    plan_id: str
    identities: tuple[Mapping[str, str], ...]
    missing_without_filing: tuple[CoverageRequirement, ...]
    estimated_request_ceiling: int
    available_request_headroom: int


def scoped_dart_request_headroom(
    *,
    runtime: DataRuntime,
    quota_store: ProviderQuotaStateStore | None = None,
    now: datetime | None = None,
    key_env: str = PRIMARY_DART_KEY_ENV,
) -> int:
    """Return request capacity for one key after reserving its quota headroom.

    Each key is metered in its own ledger against the budget and reserve its
    scope policy declares, so a run of one key never consumes another's
    headroom.
    """
    policy = runtime.scope.collection.dart_key_policy(key_env)
    store = quota_store or ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    remaining = store.remaining_daily_attempts(
        provider=dart_quota_provider(os.environ.get(key_env)),
        now=now or datetime.now(UTC),
        daily_limit=policy.daily_budget,
    )
    return max(0, remaining - policy.daily_reserve)


def build_scoped_dart_collector(
    *,
    runtime: DataRuntime,
    quota_store: ProviderQuotaStateStore | None = None,
    key_env: str = PRIMARY_DART_KEY_ENV,
) -> DartXbrlCollector:
    """Build the sole DART collector for one declared key from its scope policy.

    Raises:
        ValueError: the key has no declared policy or its environment variable is unset.
    """
    policy = runtime.scope.collection.dart_key_policy(key_env)
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"{key_env} is not set")
    store = quota_store or ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    return DartXbrlCollector(
        api_key=api_key,
        quota_store=store,
        max_workers=policy.max_workers,
        min_interval=policy.min_interval_seconds,
        daily_request_limit=policy.daily_budget,
    )


def build_scoped_dart_fact_batch(
    *,
    runtime: DataRuntime,
    catalog: ReceiptCatalog,
    filing_identities: Collection[Mapping[str, str]],
    offset: int,
    limit: int,
    quota_store: ProviderQuotaStateStore | None = None,
    now: datetime | None = None,
) -> DartFactBatchPlan:
    """Select retained 2019+ filing identities lacking successful fact evidence without disclosure-list discovery."""
    scope = runtime.scope
    if offset < 0 or limit < 1:
        raise PITDataError("offset must be nonnegative and limit must be positive")
    floor = scope.features.fundamental_fiscal_start
    full: dict[tuple[str, str, str], dict[str, str]] = {}
    missing: list[CoverageRequirement] = []
    for raw_identity in filing_identities:
        item = dict(raw_identity)
        corp_code = str(item.get("corp_code") or "").strip()
        biz_year = str(item.get("biz_year") or "").strip()
        reprt_code = str(item.get("reprt_code") or "").strip()
        if not corp_code or not biz_year or not reprt_code:
            raise PITDataError("DART fact identity is missing corp code, business year, or report code")
        filing_id = str(item.get("filing_id") or item.get("rcept_no") or "").strip()
        published_at = str(item.get("published_at") or item.get("available_at") or "").strip()
        natural_key = dart_fact_natural_key(corp_code=corp_code, biz_year=biz_year, reprt_code=reprt_code)
        if not filing_id or not published_at:
            as_of_raw = str(item.get("as_of") or "").strip()
            missing.append(
                CoverageRequirement(
                    source=FACT_SOURCE,
                    natural_key=natural_key,
                    as_of=date.fromisoformat(as_of_raw[:10]) if as_of_raw else None,
                    fiscal_period=_identity_fiscal_period(item),
                    required=True,
                )
            )
            continue
        if _period_key(_identity_fiscal_period(item)) < _period_key(floor):
            continue
        key = (corp_code, biz_year, reprt_code)
        current = full.get(key)
        if current is None or (published_at, filing_id) > (
            str(current.get("published_at") or current.get("available_at") or ""),
            str(current.get("filing_id") or current.get("rcept_no") or ""),
        ):
            full[key] = item
    covered = catalog.successful_keys(source=FACT_SOURCE, fiscal_start=floor)
    candidates = sorted(
        (
            item
            for key, item in full.items()
            if dart_fact_natural_key(corp_code=key[0], biz_year=key[1], reprt_code=key[2]) not in covered
        ),
        key=lambda item: dart_fact_natural_key(
            corp_code=str(item.get("corp_code") or ""),
            biz_year=str(item.get("biz_year") or ""),
            reprt_code=str(item.get("reprt_code") or ""),
        ),
    )
    page = candidates[offset : offset + limit]
    request_headroom = scoped_dart_request_headroom(
        runtime=runtime, quota_store=quota_store, now=now
    )
    allowance = min(
        len(page),
        scope.collection.dart_batch_identities,
        request_headroom // 3,
    )
    selected = tuple(page[:allowance])
    digest = hashlib.sha256()
    digest.update(scope.content_hash.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(json.dumps([dict(item) for item in selected], sort_keys=True).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(f"{offset}:{limit}".encode())
    return DartFactBatchPlan(
        scope_hash=scope.content_hash,
        plan_id=f"dart-facts-{digest.hexdigest()[:16]}",
        identities=selected,
        missing_without_filing=tuple(sorted(missing, key=lambda item: (item.source, item.natural_key))),
        estimated_request_ceiling=len(selected) * 3,
        available_request_headroom=request_headroom,
    )
