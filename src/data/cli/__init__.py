"""Area-based data CLI: parser built from the command registry, uniform envelope."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from src.config.errors import ConfigError
from src.core.pit import PITDataError
from src.data.cli.common import CommandFailed, emit
from src.data.cli.registry import build_parser, commands, register
from src.integrations.errors import ProviderError

__all__ = ["main"]

_LOG = logging.getLogger(__name__)

_REGISTERED = False


def _ensure_registered() -> None:
    """Register area commands once, preserving the pre-split command order."""
    global _REGISTERED
    if _REGISTERED:
        return
    from src.data.cli import build as _build
    from src.data.cli import collect as _collect
    from src.data.cli import maintain as _maintain

    by_name = {command.name: command for command in (
        *_collect.COLLECT_COMMANDS, *_build.BUILD_COMMANDS, *_maintain.MAINTAIN_COMMANDS,
    )}
    for name in (
        "normalize-dart-facts",
        "audit-ordinary-universe-prices",
        "scope-info",
        "init-workspace",
        "index-bronze",
        "collect-scoped",
        "collect-krx-daily-market",
        "collect-krx-security-master",
        "collect-kind-notices",
        "collect-kind-documents",
        "collect-ls-investor-flow",
        "collect-kis-investor-flow",
        "collect-dart-disclosures",
        "collect-dart-corp-codes",
        "collect-dart-facts",
        "collect-dividend-decisions",
        "collect-earnings-releases",
        "reparse-dart-documents",
        "collect-dart-documents",
        "collect-dart-benchmark-documents",
        "benchmark-dart-documents",
        "build-investor-flow-silver",
        "build-daily-market-silver",
        "build-ordinary-universe",
        "build-market-panel",
        "build-reference-benchmarks",
        "verify-datasets",
        "prune-datasets",
        "refresh-scope",
        "build-investor-flow-kis-supplement",
        "build-investor-flow-union",
        "build-financial-quality",
        "build-dividend-events",
        "build-earnings-releases",
        "benchmark-earnings-releases",
        "build-market-actions",
        "collect-industry-classification",
        "collect-stock-classification",
        "build-industry-classification-silver",
    ):
        register(by_name[name])
    _REGISTERED = True


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments from the registry (kept for caller compatibility)."""
    _ensure_registered()
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one data CLI command with the uniform JSON-line envelope."""
    _ensure_registered()
    args = build_parser().parse_args(argv)
    lookup = {command.name: command for command in commands()}
    command = lookup[args.command]
    try:
        payload = command.run(args)
    except CommandFailed as exc:
        if exc.payload is not None:
            emit(exc.payload)
        _LOG.info("[DATA] command=%s status=failed exit=%d", command.name, exc.exit_code)
        return exc.exit_code
    except ProviderError as exc:
        _LOG.error("[DATA] command=%s status=failed error=%s", command.name, exc)
        emit({"error": str(exc)})
        return 3
    except (PITDataError, ConfigError, ValueError, OSError) as exc:
        _LOG.error("[DATA] command=%s status=failed error=%s", command.name, exc)
        emit({"error": str(exc)})
        return 2
    emit(dict(payload))
    _LOG.info("[DATA] command=%s status=ok", command.name)
    return 0


_LAZY_ALIASES: Mapping[str, tuple[str, str]] = {
    "DatasetRegistry": ("src.data.dataset_registry", "DatasetRegistry"),
    "normalize_dart_facts": ("src.data.incremental_normalization", "normalize_dart_facts"),
    "load_data_runtime": ("src.data.runtime", "load_data_runtime"),
    "_collect_classification_with_isolation": (
        "src.data.industry_collection", "collect_classification_with_isolation",
    ),
    "_parse_quality_decision_time": ("src.data.financial_quality", "_parse_quality_decision_time"),
    "_parse_quality_event_timestamp": ("src.data.financial_quality", "_parse_quality_event_timestamp"),
    "_parse_decision_time": ("src.data.cli.build", "_parse_decision_time"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve moved helpers so existing callers keep working without heavy imports."""
    target = _LAZY_ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    import importlib

    return getattr(importlib.import_module(module_name), attr)
