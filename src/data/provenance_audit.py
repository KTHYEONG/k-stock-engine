"""Fail-closed production provenance audit for retained stock evidence."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InvestorFlowResolution:
    symbol: str
    providers: tuple[str, ...]
    state: str


@dataclass(frozen=True, slots=True)
class ProductionProvenanceAudit:
    investor_flow: tuple[InvestorFlowResolution, ...]
    fixture_tables: tuple[str, ...]
    artifact_path: Path


def _flow_resolutions(bronze_root: Path) -> tuple[InvestorFlowResolution, ...]:
    statuses: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for payload_path in (bronze_root / "investor_flow").glob("*/payload.json"):
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        status = payload.get("status")
        if status not in {"source_unavailable", "provider_error"}:
            continue
        symbol = str(payload.get("symbol") or "").strip()
        provider = str(payload.get("provider") or "").strip().lower()
        if not symbol or not provider:
            raise ValueError(f"invalid investor-flow negative receipt: {payload_path}")
        statuses[symbol][provider].add(str(status))
    out: list[InvestorFlowResolution] = []
    for symbol, providers in sorted(statuses.items()):
        provider_names = tuple(sorted(providers))
        # The collectors only know that the requested session was absent from
        # the returned rows.  They do not receive a provider-level "no
        # history" assertion, so even two empty responses cannot prove that
        # the underlying source never existed.
        empty_response = all(values == {"source_unavailable"} for values in providers.values())
        out.append(
            InvestorFlowResolution(
                symbol=symbol,
                providers=provider_names,
                state="unverified_empty_response" if empty_response else "retry_required",
            )
        )
    return tuple(out)


def _fixture_tables(silver_root: Path) -> tuple[str, ...]:
    fixtures: set[str] = set()
    for manifest_path in silver_root.glob("*/*/dataset_manifest.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("provider_version") == "fixture":
            fixtures.add(manifest_path.parent.parent.name)
    return tuple(sorted(fixtures))


def audit_production_provenance(
    *, bronze_root: Path, silver_root: Path, artifact_root: Path
) -> ProductionProvenanceAudit:
    """Classify retained failures without changing source evidence or Silver datasets."""
    investor_flow = _flow_resolutions(Path(bronze_root))
    fixture_tables = _fixture_tables(Path(silver_root))
    body = {
        "fixture_tables": list(fixture_tables),
        "investor_flow": [asdict(item) for item in investor_flow],
    }
    canonical = json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    directory = Path(artifact_root) / "provenance_audit"
    directory.mkdir(parents=True, exist_ok=True)
    artifact_path = directory / f"{digest}.json"
    artifact_path.write_text(json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return ProductionProvenanceAudit(
        investor_flow=investor_flow,
        fixture_tables=fixture_tables,
        artifact_path=artifact_path,
    )
