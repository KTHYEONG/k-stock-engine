"""Off-network KIND relay CLI: collect a verified bundle and store it atomically."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from src.config.providers import load_provider_policy
from src.config.runtime import load_runtime_config
from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE, KIND_NOTICE_SEARCH_SOURCE
from src.data.kind_relay import decode_relay_bundle, encode_relay_bundle
from src.data.kind_relay_producer import build_relay_bundle
from src.data.research_scope import load_research_scope
from src.integrations.krx.kind import build_scoped_kind_client
from src.integrations.quota import ProviderQuotaStateStore

__all__ = ["main"]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="src.ops.kind_relay")
    sub = parser.add_subparsers(dest="command", required=True)
    produce = sub.add_parser("produce")
    produce.add_argument("--out", type=Path, required=True)
    produce.add_argument("--scope-config", type=Path, required=False, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``produce --out PATH [--scope-config PATH]`` writes one verified bundle; returns the exit code."""
    args = _parse_args(argv)
    out = Path(args.out)
    staged: Path | None = None
    try:
        runtime_config = load_runtime_config(create_directories=False)
        provider = load_provider_policy(runtime_config)
        scope_path = Path(args.scope_config) if args.scope_config is not None else runtime_config.default_scope
        scope = load_research_scope(scope_path)
        with tempfile.TemporaryDirectory(prefix="kind-relay-") as tmp:
            client = build_scoped_kind_client(
                policy=provider.kind, quota_store=ProviderQuotaStateStore(Path(tmp) / "quota")
            )
            bundle = build_relay_bundle(
                client,
                policy=provider.kind,
                evidence_start=scope.evidence_start,
                now=datetime.now(UTC),
                producer={
                    "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
                    "commit": os.environ.get("GITHUB_SHA", ""),
                },
            )
        raw = encode_relay_bundle(bundle)
        decode_relay_bundle(raw)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=out.parent, prefix=".kind-bundle.", suffix=".tmp", delete=False
        ) as handle:
            staged = Path(handle.name)
            handle.write(raw)
            handle.flush()
        os.replace(staged, out)
        summary = {
            "fetched_at": bundle.fetched_at.astimezone(UTC).isoformat(),
            "fetch_date_kst": bundle.fetch_date_kst.isoformat(),
            "items": len(bundle.items),
            "search_windows": sum(1 for item in bundle.items if item.source == KIND_NOTICE_SEARCH_SOURCE),
            "documents": sum(1 for item in bundle.items if item.source == KIND_NOTICE_DOCUMENT_SOURCE),
            "document_failures": int(bundle.producer.get("document_failures", "0")),
            "bytes": len(raw),
            "items_sha256": str(json.loads(raw.decode("utf-8"))["items_sha256"]),
        }
        sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
        return 0
    except Exception:
        with contextlib.suppress(OSError):
            out.unlink(missing_ok=True)
        sys.stderr.write("kind_relay produce failed\n")
        return 1
    finally:
        if staged is not None:
            with contextlib.suppress(OSError):
                staged.unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover - entrypoint glue
    raise SystemExit(main())
