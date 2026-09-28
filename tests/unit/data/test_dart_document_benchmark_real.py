"""Real-data benchmark gate for filing-document facts (slow)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def _archive_present(bronze_root: Path) -> bool:
    documents = bronze_root / "dart_documents"
    if not documents.is_dir():
        return False
    return any(documents.rglob("payload.zip"))


def test_benchmark_precision_on_real_scope() -> None:
    """Run the benchmark on the real scope sample and gate on min_precision."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.dart_document_benchmark import BENCHMARK_SEED, run_benchmark, select_benchmark_filings
    from src.data.receipt_catalog import ReceiptCatalog
    from src.data.runtime import resolve_data_runtime

    runtime = resolve_data_runtime()
    bronze_root = runtime.workspace.bronze_root
    catalog = ReceiptCatalog(bronze_root / "catalog")
    provider = load_provider_policy(load_runtime_config())
    policy = provider.dart.document_parser
    filings = select_benchmark_filings(catalog, size=policy.benchmark_sample, seed=BENCHMARK_SEED)
    if not filings or not _archive_present(bronze_root):
        pytest.skip("benchmark archives are absent")
    report = run_benchmark(runtime, filings=filings)
    sys.stdout.write(json.dumps(report.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    assert report.precision >= policy.min_precision
