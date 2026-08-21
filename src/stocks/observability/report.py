"""Deterministic AI-readable diagnostic report.

``analyze_run`` performs a single O(E) streaming pass over a run's JSONL
bundles and returns a ``DiagnosticReport`` containing the first failed
checkpoint, stage time/RSS ranking, candidate/order/fill funnels,
base/stress cost attribution, parameter differences, and missing checkpoints.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class StageSummary:
    stage: str
    elapsed_ms: float = 0.0
    event_count: int = 0
    rss_mib: float | None = None


@dataclass(frozen=True, slots=True)
class CostAttribution:
    commission: float = 0.0
    tax: float = 0.0
    spread: float = 0.0
    impact: float = 0.0
    other: float = 0.0
    total: float = 0.0


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    run_id: str
    status: str
    first_fail_sequence: int | None = None
    first_fail_event: str | None = None
    first_fail_component: str | None = None
    stage_summaries: list[StageSummary] = field(default_factory=list)
    candidate_funnel: dict[str, int] = field(default_factory=dict)
    order_funnel: dict[str, int] = field(default_factory=dict)
    fill_funnel: dict[str, int] = field(default_factory=dict)
    cost_attribution: CostAttribution | None = None
    parameter_differences: list[str] = field(default_factory=list)
    missing_checkpoints: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "status": self.status,
                "first_fail_sequence": self.first_fail_sequence,
                "first_fail_event": self.first_fail_event,
                "first_fail_component": self.first_fail_component,
                "stage_summaries": [
                    {
                        "stage": s.stage,
                        "elapsed_ms": s.elapsed_ms,
                        "event_count": s.event_count,
                        "rss_mib": s.rss_mib,
                    }
                    for s in self.stage_summaries
                ],
                "candidate_funnel": dict(self.candidate_funnel),
                "order_funnel": dict(self.order_funnel),
                "fill_funnel": dict(self.fill_funnel),
                "parameter_differences": list(self.parameter_differences),
                "missing_checkpoints": list(self.missing_checkpoints),
            },
            ensure_ascii=False,
            indent=2,
        )


_REQUIRED_STAGES = [
    "input",
    "data",
    "split_fit",
    "calibration",
    "selection",
    "allocation",
    "execution",
    "costs",
    "settlement",
    "terminal",
]


def analyze_run(runs_root: Path, run_id: str) -> DiagnosticReport:
    """Analyze a run's diagnostic bundles with a single O(E) streaming pass.

    Parameters
    ----------
    runs_root:
        Root directory containing run subdirectories.
    run_id:
        The run identifier to analyze.

    Returns
    -------
    DiagnosticReport
        Aggregated diagnostic report.
    """
    run_dir = runs_root / run_id
    if not run_dir.exists():
        return DiagnosticReport(
            run_id=run_id,
            status="UNKNOWN",
            missing_checkpoints=list(_REQUIRED_STAGES),
        )

    events: list[dict[str, Any]] = []
    for suffix in ("sys.jsonl", "data.jsonl", "algo.jsonl", "eval.jsonl"):
        path = run_dir / suffix
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    events.sort(key=lambda e: int(cast(Any, e.get("sequence", 0))))

    first_fail_seq: int | None = None
    first_fail_event: str | None = None
    first_fail_component: str | None = None
    stage_events: dict[str, list[dict[str, object]]] = defaultdict(list)
    candidate_funnel: dict[str, int] = defaultdict(int)
    order_funnel: dict[str, int] = defaultdict(int)
    fill_funnel: dict[str, int] = defaultdict(int)
    seen_stages: set[str] = set()
    total_elapsed: dict[str, float] = defaultdict(float)
    stage_counts: dict[str, int] = defaultdict(int)

    for ev in events:
        stage = str(ev.get("stage", ""))
        status = str(ev.get("status", ""))
        event_name = str(ev.get("event", ""))
        component = str(ev.get("component", ""))
        elapsed_ms_val = ev.get("elapsed_ms", 0.0)
        elapsed = float(cast(Any, elapsed_ms_val))

        seen_stages.add(stage)
        stage_events[stage].append(ev)
        total_elapsed[stage] += elapsed
        stage_counts[stage] += 1

        if status == "FAIL" and first_fail_seq is None:
            seq_val = ev.get("sequence", 0)
            first_fail_seq = int(cast(Any, seq_val))
            first_fail_event = event_name
            first_fail_component = component

        if "candidate" in event_name or "horizon" in event_name:
            candidate_funnel[event_name] += 1
        if "order" in event_name or "fill" in event_name or "reject" in event_name:
            order_funnel[event_name] += 1
        if "filled" in event_name or "unfilled" in event_name:
            fill_funnel[event_name] += 1

    stage_summaries = [
        StageSummary(
            stage=s,
            elapsed_ms=total_elapsed[s],
            event_count=stage_counts[s],
        )
        for s in sorted(total_elapsed.keys())
    ]

    missing = [s for s in _REQUIRED_STAGES if s not in seen_stages]

    run_status = "PASS"
    if first_fail_seq is not None:
        run_status = "FAIL"
    elif missing:
        run_status = "INCOMPLETE"

    # Load manifest for additional context
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            if manifest.get("status") == "FAIL" and run_status != "FAIL":
                run_status = "FAIL"
        except (json.JSONDecodeError, OSError):
            pass

    return DiagnosticReport(
        run_id=run_id,
        status=run_status,
        first_fail_sequence=first_fail_seq,
        first_fail_event=first_fail_event,
        first_fail_component=first_fail_component,
        stage_summaries=stage_summaries,
        candidate_funnel=dict(candidate_funnel),
        order_funnel=dict(order_funnel),
        fill_funnel=dict(fill_funnel),
        missing_checkpoints=missing,
    )
