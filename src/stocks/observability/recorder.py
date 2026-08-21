"""JSONL diagnostic recorder and no-op sink.

``JsonlRunDiagnostics`` streams events to category-separated files under
``logs/runs/<run_id>/``.  ``NullRunDiagnostics`` discards every event and
is the default when diagnostics are disabled.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from src.stocks.observability.contracts import (
    DiagnosticEvent,
    RunDiagnostics,
    RunIdentity,
)

_CATEGORY_SUFFIX: dict[str, str] = {
    "SYS": "sys.jsonl",
    "DATA": "data.jsonl",
    "ALGO": "algo.jsonl",
    "EVAL": "eval.jsonl",
}

_MAX_STREAM_BYTES_DEFAULT = 64 * 1024 * 1024


class NullRunDiagnostics:
    """No-op sink: discards every event.  Used when diagnostics are disabled."""

    def emit(self, event: DiagnosticEvent) -> None:
        pass

    def close(self, status: str) -> None:
        pass


class JsonlRunDiagnostics:
    """Streaming JSONL recorder writing to category-separated files.

    Parameters
    ----------
    identity:
        Stable run identity (run_id + project).
    root:
        Parent directory for this run's diagnostic files.  Created on init.
    max_stream_bytes:
        Per-file soft rotation limit.
    required:
        When ``True``, recorder failures raise; when ``False``, they are
        surfaced at terminal output only.
    """

    def __init__(
        self,
        identity: RunIdentity,
        root: Path,
        *,
        max_stream_bytes: int = _MAX_STREAM_BYTES_DEFAULT,
        required: bool = False,
    ) -> None:
        self._identity = identity
        self._root = root / identity.run_id
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_stream_bytes = max_stream_bytes
        self._required = required
        self._sequence = 0
        self._handles: dict[str, TextIO] = {}
        self._byte_counts: dict[str, int] = {}
        self._event_counts: dict[str, int] = {}
        self._truncation_counts: dict[str, int] = {}
        self._closed = False

    def emit(self, event: DiagnosticEvent) -> None:
        if self._closed:
            return
        self._sequence += 1
        suffix = _CATEGORY_SUFFIX.get(event.category)
        if suffix is None:
            return
        path = self._root / suffix
        record = {
            "run_id": event.run_id,
            "sequence": self._sequence,
            "category": event.category,
            "component": event.component,
            "stage": event.stage,
            "event": event.event,
            "status": event.status,
            "elapsed_ms": event.elapsed_ms,
        }
        if event.payload:
            record["payload"] = event.payload
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        line_bytes = (line + "\n").encode("utf-8")
        try:
            fh = self._handles.get(suffix)
            if fh is None:
                fh = open(path, "a", encoding="utf-8")  # noqa: SIM115
                self._handles[suffix] = fh
            fh.write(line_bytes.decode("utf-8"))
            self._byte_counts[suffix] = self._byte_counts.get(suffix, 0) + len(
                line_bytes
            )
            self._event_counts[suffix] = self._event_counts.get(suffix, 0) + 1
        except OSError:
            if self._required:
                raise
            self._truncation_counts[suffix] = (
                self._truncation_counts.get(suffix, 0) + 1
            )

    def close(self, status: str) -> None:
        if self._closed:
            return
        self._closed = True
        for fh in self._handles.values():
            fh.close()
        manifest = {
            "run_id": self._identity.run_id,
            "project": self._identity.project,
            "status": status,
            "event_counts": dict(self._event_counts),
            "byte_counts": dict(self._byte_counts),
            "truncation_counts": dict(self._truncation_counts),
        }
        manifest_path = self._root / "manifest.json"
        try:
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, indent=2)
        except OSError:
            if self._required:
                raise


def open_run_diagnostics(
    identity: RunIdentity,
    settings: Mapping[str, object] | None = None,
) -> RunDiagnostics:
    """Factory: returns a ``JsonlRunDiagnostics`` or ``NullRunDiagnostics``.

    Parameters
    ----------
    identity:
        Stable run identity.
    settings:
        Optional mapping with ``diagnostics_enabled`` (bool, default True)
        and ``diagnostics_required`` (bool, default False) keys.
    """
    if settings is None:
        settings = {}
    enabled = settings.get("diagnostics_enabled", True)
    if not enabled:
        return NullRunDiagnostics()
    required = bool(settings.get("diagnostics_required", False))
    from src.core.paths import RUN_DIAGNOSTIC_ROOT

    return JsonlRunDiagnostics(
        identity, RUN_DIAGNOSTIC_ROOT, required=required
    )
