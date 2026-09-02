"""Initial production import/materialization entry point."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from src.data.bronze import BronzeStore, import_retained_stock_evidence
from src.data.schemas import PITDataError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIT dataset foundation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import-retained", help="Bronze-only import")
    p_import.add_argument("--source-root", type=Path, required=True)
    p_import.add_argument("--bronze-root", type=Path, required=True)
    p_import.add_argument("--retrieved-at", type=str, required=False, default=None)

    p_mat = sub.add_parser("materialize", help="Materialize Silver")
    p_mat.add_argument("--source-root", type=Path, required=True)
    p_mat.add_argument("--bronze-root", type=Path, required=True)
    p_mat.add_argument("--silver-root", type=Path, required=True)
    p_mat.add_argument("--decision-time", type=str, required=True)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "import-retained":
        retrieved_at = datetime.now(UTC)
        if args.retrieved_at:
            retrieved_at = datetime.fromisoformat(args.retrieved_at)
            if retrieved_at.tzinfo is None:
                retrieved_at = retrieved_at.replace(tzinfo=UTC)
        store = BronzeStore(Path(args.bronze_root))
        try:
            import_retained_stock_evidence(Path(args.source_root), store=store, retrieved_at=retrieved_at)
        except PITDataError:
            return 1
        return 0
    if args.command == "materialize":
        # Materialization requires explicit evidence registry and reports certification failure
        retrieved_at = datetime.now(UTC)
        store = BronzeStore(Path(args.bronze_root))
        try:
            import_retained_stock_evidence(Path(args.source_root), store=store, retrieved_at=retrieved_at)
        except PITDataError:
            return 1
        # Further silver materialization would go here; for now report failure if incomplete
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
