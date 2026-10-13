"""Exercise relay CLI wiring with real assembly and encoding."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.data.kind_relay import decode_relay_bundle
from src.integrations.krx.kind import KindNoticeDocument
from tests.fixtures.kind_html import kind_row_html, kind_search_html


def test_produce_assembles_evidence_with_ephemeral_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.ops.kind_relay as cli

    today = (datetime.now(UTC) + timedelta(hours=9)).date()
    receipt = "20261012000001"
    quota_roots: list[Path] = []
    collections: list[datetime] = []
    original_loader = cli.load_runtime_config

    def read_only_loader(*, create_directories):  # type: ignore[no-untyped-def]
        assert create_directories is False
        return original_loader(create_directories=False)

    class Collector:
        def search_page(self, keyword, start, end, *, page_index):  # type: ignore[no-untyped-def]
            rows = [kind_row_html(receipt, f"{today} 09:00", "상장폐지", "005930", "유가증권시장본부")]
            collections.append(datetime.now(UTC))
            return kind_search_html(rows, 1) if end == today else kind_search_html([], 0)

        def fetch_document(self, acptno: str) -> KindNoticeDocument:
            collections.append(datetime.now(UTC))
            return KindNoticeDocument(acptno, "7", "a.htm", "<html>BODY-MARKER</html>")

    def build_client(*, policy, quota_store):  # type: ignore[no-untyped-def]
        quota_roots.append(quota_store._root)
        return Collector()

    monkeypatch.setattr(cli, "load_runtime_config", read_only_loader)
    monkeypatch.setattr(cli, "build_scoped_kind_client", build_client)
    monkeypatch.setenv("GITHUB_RUN_ID", "isolated-run")
    monkeypatch.setenv("GITHUB_SHA", "isolated-commit")
    out = tmp_path / "nested" / "bundle.json"
    assert cli.main(["produce", "--out", str(out)]) == 0
    bundle = decode_relay_bundle(out.read_bytes())
    assert bundle.producer == {"run_id": "isolated-run", "commit": "isolated-commit", "document_failures": "0"}
    assert sum(item.source == "kind_notice_documents" for item in bundle.items) == 1
    assert bundle.fetched_at >= max(collections)
    assert all(item.retrieved_at <= bundle.fetched_at for item in bundle.items)
    document = next(item for item in bundle.items if item.source == "kind_notice_documents")
    assert document.retrieved_at >= collections[-1]
    assert quota_roots
    assert all(not root.parent.exists() for root in quota_roots)
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["documents"] == 1
    assert "BODY-MARKER" not in captured.out + captured.err
