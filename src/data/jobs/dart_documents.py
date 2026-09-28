"""Document reparse and fetch jobs for DART filing-document facts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.core.pit import PITDataError
from src.data.collection import dart_fact_scoped_payload
from src.data.dart_documents import DartDocumentStore
from src.data.evidence_sources import DART_DOCUMENT_SOURCE
from src.data.fact_page_meta import FactPageMetaStore
from src.data.jobs.runner import JobContext, JobUnit
from src.data.scoped_ingestion import ScopedRawPayload, dart_fact_natural_key

__all__ = [
    "DartBenchmarkDocumentFetchJob",
    "DartDocumentFetchJob",
    "DartDocumentReparseJob",
    "relevant_fact_identities",
]

_FACT_SOURCE = "financial_facts"
_PENDING_KINDS = frozenset({"legacy_document", "document_verified"})


def _eligible_years_by_ticker(ctx: JobContext) -> dict[str, set[int]]:
    """Map each ticker to the calendar years it was eligible in Silver."""
    import polars as pl

    from src.data.dataset_registry import DatasetRegistry

    registry = DatasetRegistry(ctx.runtime.workspace.state_root)
    dataset_id = registry.require("ordinary_universe")
    dataset_dir = ctx.runtime.workspace.silver_root / dataset_id
    if not dataset_dir.is_dir():  # pragma: no cover - registry points at a removed dataset
        raise PITDataError(f"ordinary universe dataset is missing: {dataset_id}")
    files = sorted(str(path) for path in dataset_dir.rglob("*.parquet"))
    if not files:  # pragma: no cover - empty dataset directory
        raise PITDataError("ordinary universe has no published partitions")
    try:
        frame = pl.scan_parquet(files).select("ticker", "eligible", "session").collect()
    except Exception as exc:  # pragma: no cover - corrupt parquet
        raise PITDataError(f"ordinary universe is unreadable: {dataset_id}") from exc
    out: dict[str, set[int]] = {}
    for row in frame.to_dicts():
        if not row.get("eligible"):
            continue
        ticker = str(row.get("ticker") or "").strip()
        if not ticker:
            continue
        session = row.get("session")
        try:
            year_value = session.year if session is not None and hasattr(session, "year") else int(str(session)[:4])
        except (ValueError, TypeError, AttributeError):
            continue
        out.setdefault(ticker, set()).add(int(year_value))
    return out


def relevant_fact_identities(ctx: JobContext) -> frozenset[str]:
    """Fact natural keys whose corp code maps to a ticker eligible in fiscal year Y or Y+1.

    Documents are fetched only for filings the research scope can consume:
    facts for fiscal year Y are used through year Y+1 because of the
    publication lag.
    """
    from src.data.dart_disclosures import iter_disclosure_records, periodic_filing_identities
    from src.data.jobs.dart import last_completed_kst_day
    from src.data.jobs.universe import read_corp_code_bridge

    bridge, _ = read_corp_code_bridge(ctx.catalog)
    bridge = dict(bridge)
    ticker_years = _eligible_years_by_ticker(ctx)
    scope = ctx.runtime.scope
    end = last_completed_kst_day(ctx.now())
    identities = periodic_filing_identities(
        iter_disclosure_records(ctx.catalog),
        start=scope.evidence_start,
        end=end,
        ticker_by_corp_code=None,
        required_periods=None,
        corp_codes=None,
    )
    out: set[str] = set()
    for item in identities:
        corp_code = str(item.get("corp_code") or "").strip()
        biz_year = str(item.get("biz_year") or "").strip()
        reprt_code = str(item.get("reprt_code") or "").strip()
        if not corp_code or not biz_year or not reprt_code:  # pragma: no cover - producer always fills identity
            continue
        ticker = bridge.get(corp_code)
        if not ticker:
            continue
        try:
            year = int(biz_year)
        except ValueError:  # pragma: no cover - producer emits numeric years
            continue
        years = ticker_years.get(ticker, set())
        if year in years or (year + 1) in years:
            out.add(dart_fact_natural_key(corp_code=corp_code, biz_year=biz_year, reprt_code=reprt_code))
    return frozenset(out)


def _latest_fact_entries(ctx: JobContext) -> dict[str, Any]:
    """Latest catalog entry per fact natural key, streaming one entry at a time."""
    latest: dict[str, Any] = {}
    for entry in ctx.catalog.entries(source=_FACT_SOURCE):
        current = latest.get(entry.natural_key)
        if current is None or entry.retrieved_at > current.retrieved_at:
            latest[entry.natural_key] = entry
    return latest


def _read_page(entry: Any) -> dict[str, Any] | None:
    try:
        raw = Path(str(entry.payload_path)).read_bytes()
    except OSError:  # pragma: no cover - missing payload file
        return None
    try:
        document = json.loads(raw)
    except ValueError:  # pragma: no cover - corrupt payload
        return None
    if not isinstance(document, dict):  # pragma: no cover - unexpected payload shape
        return None
    return document


def _usable_document_hashes(ctx: JobContext) -> set[str]:
    return {blob.content_hash for blob in ctx.catalog.blobs(source=DART_DOCUMENT_SOURCE)}


def _unit_payload(natural_key: str, page: Mapping[str, Any]) -> dict[str, str]:
    parts = natural_key.split(":")
    corp_code = str(page.get("corp_code") or (parts[0] if len(parts) > 0 else "")).strip()
    biz_year = str(page.get("biz_year") or (parts[1] if len(parts) > 1 else "")).strip()
    reprt_code = str(page.get("reprt_code") or (parts[2] if len(parts) > 2 else "")).strip()
    identity = page.get("identity")
    identity_map = dict(identity) if isinstance(identity, Mapping) else {}
    filing_id = str(
        page.get("filing_id") or page.get("rcept_no") or identity_map.get("filing_id") or identity_map.get("rcept_no") or ""
    ).strip()
    return {
        "corp_code": corp_code,
        "biz_year": biz_year,
        "reprt_code": reprt_code,
        "filing_id": filing_id,
        "rcept_no": str(page.get("rcept_no") or identity_map.get("rcept_no") or filing_id),
        "fs_div": str(page.get("fs_div") or identity_map.get("fs_div") or "CFS"),
        "published_at": str(page.get("published_at") or identity_map.get("published_at") or ""),
        "ticker": str(page.get("ticker") or identity_map.get("ticker") or ""),
        "fiscal_period": str(page.get("fiscal_period") or identity_map.get("fiscal_period") or ""),
        "raw_document_hash": str(page.get("raw_document_hash") or ""),
    }


def _is_document_not_found(page: Mapping[str, Any]) -> bool:
    diagnostics = page.get("diagnostics") or ()
    try:
        items = tuple(diagnostics)
    except TypeError:
        return False
    return any(str(item) == "document_not_found" for item in items)


def _archive_bytes_for_hash(ctx: JobContext, content_hash: str) -> bytes | None:
    for blob in ctx.catalog.blobs(source=DART_DOCUMENT_SOURCE):
        if blob.content_hash == content_hash:
            try:
                return Path(str(blob.payload_path)).read_bytes()
            except OSError:  # pragma: no cover - blob file race
                return None
    candidate = ctx.runtime.workspace.bronze_root / DART_DOCUMENT_SOURCE / content_hash / "payload.zip"  # pragma: no cover - legacy files without a catalog blob
    try:  # pragma: no cover - legacy files without a catalog blob
        return candidate.read_bytes()
    except OSError:  # pragma: no cover - no stored archive
        return None


class DartDocumentReparseJob:
    """Re-derive document fact pages from archives already in Bronze (no API).

    Streams ``catalog.entries(source="financial_facts")`` one entry at a time.
    A page is pending when its ``source_kind`` is ``legacy_document`` or a
    ``document_verified`` page with an older ``parser_version``, and its
    ``raw_document_hash`` is a usable ``dart_documents`` blob.
    """

    name = "dart_document_reparse"

    def __init__(self) -> None:
        self._archive_cache: dict[str, Path] | None = None

    def _archive_bytes(self, ctx: JobContext, content_hash: str) -> bytes | None:
        """Read the stored archive of one document hash.

        The runner calls ``fetch`` once per unit, so the hash-to-file mapping is built once per job
        instance instead of once per call. A new job instance (a new run) always rebuilds it, so blobs
        registered between runs are seen.

        Args:
            ctx: Job context whose catalog holds the usable ``dart_documents`` blobs.
            content_hash: Content hash named by a fact page's ``raw_document_hash``.

        Returns:
            The archive bytes, or None when the hash has no usable blob or its file is unreadable.
        """
        if self._archive_cache is None:
            self._archive_cache = {
                blob.content_hash: Path(str(blob.payload_path))
                for blob in ctx.catalog.blobs(source=DART_DOCUMENT_SOURCE)
            }
        path = self._archive_cache.get(content_hash)
        if path is not None:
            try:
                return path.read_bytes()
            except OSError:  # pragma: no cover - blob removed between plan and fetch
                return None
        candidate = ctx.runtime.workspace.bronze_root / DART_DOCUMENT_SOURCE / content_hash / "payload.zip"
        try:
            return candidate.read_bytes()
        except OSError:
            return None

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        from src.integrations.dart.document_statements import PARSER_VERSION

        usable = _usable_document_hashes(ctx)
        latest = _latest_fact_entries(ctx)
        store = FactPageMetaStore(ctx.catalog.root)
        units: list[JobUnit] = []
        for entry, meta in store.resolve(latest.values()):
            natural_key = entry.natural_key
            if meta.source_kind not in _PENDING_KINDS:
                continue
            if meta.document_not_found:
                continue
            if meta.source_kind == "document_verified" and meta.parser_version == PARSER_VERSION:
                continue
            if not meta.raw_document_hash or meta.raw_document_hash not in usable:
                continue
            page = _read_page(entry)
            if page is None:  # pragma: no cover - payload removed between plan and read
                continue
            units.append(
                JobUnit(
                    source=_FACT_SOURCE,
                    natural_key=natural_key,
                    payload=_unit_payload(natural_key, page),
                    max_requests=1,
                )
            )
        units.sort(key=lambda unit: unit.natural_key)
        return units

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        from src.integrations.dart.document_statements import document_verified_page, parse_filing_document

        out: list[ScopedRawPayload] = []
        for unit in units:
            identity = dict(unit.payload)
            raw_hash = str(identity.get("raw_document_hash") or "")
            if not raw_hash:
                continue
            archive = self._archive_bytes(ctx, raw_hash)
            if archive is None:
                continue
            parsed = parse_filing_document(
                bytes(archive),
                reprt_code=str(identity.get("reprt_code") or ""),
                biz_year=str(identity.get("biz_year") or ""),
            )
            page = dict(
                document_verified_page(identity=identity, result=parsed, document_hash=raw_hash)
            )
            out.append(dart_fact_scoped_payload(page=page, retrieved_at=ctx.now()))
        return out

    def health_check(self, ctx: JobContext) -> None:
        _ = ctx


class DartDocumentFetchJob:
    """Fetch ``document.xml`` for relevant document-path identities lacking a stored archive."""

    name = "dart_document_fetch"

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        from src.integrations.dart.document_statements import PARSER_VERSION

        try:
            relevant = relevant_fact_identities(ctx)
        except PITDataError:
            return ()
        usable = _usable_document_hashes(ctx)
        latest = _latest_fact_entries(ctx)
        store = FactPageMetaStore(ctx.catalog.root)
        units: list[JobUnit] = []
        for entry, meta in store.resolve(latest.values()):
            natural_key = entry.natural_key
            if natural_key not in relevant:
                continue
            if meta.source_kind not in _PENDING_KINDS:
                continue
            if meta.document_not_found:
                continue
            if meta.source_kind == "document_verified" and meta.parser_version == PARSER_VERSION:
                continue
            if meta.raw_document_hash and meta.raw_document_hash in usable:
                continue
            page = _read_page(entry)
            if page is None:  # pragma: no cover - payload removed between plan and read
                continue
            units.append(
                JobUnit(
                    source=_FACT_SOURCE,
                    natural_key=natural_key,
                    payload=_unit_payload(natural_key, page),
                    max_requests=1,
                )
            )
        units.sort(key=lambda unit: unit.natural_key)
        return units

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        from src.integrations.dart.accounts import MAPPING_VERSION
        from src.integrations.dart.document_statements import document_verified_page, parse_filing_document

        documents = DartDocumentStore(ctx.runtime.workspace.bronze_root, catalog=ctx.catalog)
        retrieved_at = ctx.now()
        out: list[ScopedRawPayload] = []
        for unit in units:
            identity = dict(unit.payload)
            rcept_no = str(identity.get("rcept_no") or identity.get("filing_id") or "").strip()
            if not rcept_no:
                continue
            archive = bytes(ctx.collector.fetch_document_archive(rcept_no))
            if b"<status>014</status>" in bytes(archive[:600]):
                page: dict[str, Any] = {
                    "source_kind": "legacy_document",
                    "status": "013",
                    "identity": dict(identity),
                    "records": [],
                    "mapping_version": MAPPING_VERSION,
                    "diagnostics": ("document_not_found",),
                    "raw_document_hash": None,
                    **identity,
                }
                out.append(dart_fact_scoped_payload(page=page, retrieved_at=retrieved_at))
                continue
            receipt = documents.store_archive(archive, rcept_no=rcept_no, retrieved_at=retrieved_at)
            parsed = parse_filing_document(
                bytes(archive),
                reprt_code=str(identity.get("reprt_code") or ""),
                biz_year=str(identity.get("biz_year") or ""),
            )
            verified = dict(
                document_verified_page(identity=identity, result=parsed, document_hash=receipt.content_hash)
            )
            out.append(dart_fact_scoped_payload(page=verified, retrieved_at=retrieved_at))
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


def _benchmark_receipt_hashes(ctx: JobContext) -> set[str]:
    return {blob.content_hash for blob in ctx.catalog.blobs(source=DART_DOCUMENT_SOURCE)}


def _stored_archive_receipts(ctx: JobContext) -> set[str]:
    """Receipt numbers with a stored ``dart_documents`` archive blob."""
    receipts: set[str] = set()
    for blob in ctx.catalog.blobs(source=DART_DOCUMENT_SOURCE):
        receipt_path = Path(str(blob.payload_path)).parent / "receipt.json"
        try:
            meta = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # pragma: no cover - blob removed between plan and read
            continue
        if not isinstance(meta, Mapping):  # pragma: no cover - foreign blob layout
            continue
        rcept_no = str(meta.get("rcept_no") or "").strip()
        if len(rcept_no) == 14 and rcept_no.isdigit():
            receipts.add(rcept_no)
    return receipts


class DartBenchmarkDocumentFetchJob:
    """Fetch ``document.xml`` for benchmark filings lacking a stored archive."""

    name = "dart_benchmark_documents"

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        from src.data.dart_document_benchmark import BENCHMARK_SEED, select_benchmark_filings

        size = int(ctx.provider.dart.document_parser.benchmark_sample)
        filings = select_benchmark_filings(ctx.catalog, size=size, seed=BENCHMARK_SEED)
        if not filings:
            return ()
        usable = _benchmark_receipt_hashes(ctx)
        stored = _stored_archive_receipts(ctx)
        wanted = set(filings)
        pages: dict[str, dict[str, Any]] = {}
        for natural_key, entry in _latest_fact_entries(ctx).items():
            if natural_key not in wanted:
                continue
            page = _read_page(entry)
            if page is None:  # pragma: no cover - payload removed between select and plan
                continue
            pages[natural_key] = page
        units: list[JobUnit] = []
        for natural_key in filings:
            page = pages.get(natural_key)
            if page is None:  # pragma: no cover - payload removed between select and plan
                continue
            raw_hash = str(page.get("raw_document_hash") or "")
            if raw_hash and raw_hash in usable:
                continue
            payload = _unit_payload(natural_key, page)
            if str(payload.get("rcept_no") or "") in stored:
                continue
            units.append(
                JobUnit(
                    source=_FACT_SOURCE,
                    natural_key=natural_key,
                    payload=payload,
                    max_requests=1,
                )
            )
        return units

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        """Store each unit's archive only.

        The benchmark compares a parsed document against the *same filing's*
        standard-API page (``run_benchmark`` parses the archive itself). Never
        persist a ``financial_facts`` page here: doing so would supersede that
        identity's standard page as the catalog's latest entry, destroying the
        very label the benchmark needs to compare against. A ``014`` (document
        not found) answer is therefore recorded nowhere; the identity simply
        stays without an archive and is skipped by the benchmark.
        """
        documents = DartDocumentStore(ctx.runtime.workspace.bronze_root, catalog=ctx.catalog)
        retrieved_at = ctx.now()
        for unit in units:
            identity = dict(unit.payload)
            rcept_no = str(identity.get("rcept_no") or identity.get("filing_id") or "").strip()
            if not rcept_no:
                continue
            archive = bytes(ctx.collector.fetch_document_archive(rcept_no))
            if b"<status>014</status>" in bytes(archive[:600]):
                continue
            documents.store_archive(archive, rcept_no=rcept_no, retrieved_at=retrieved_at)
        return ()

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()
