"""Human-readable terminal report for ML training and research runs."""
from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from src.stocks.ml.result_ledger import MlRunContext, peak_rss_mib
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.models import ModelManifest

_REPORT_FILENAME = "ml-cmp.md"
_MAX_TEXT_LENGTH = 512


class MlComparisonReport:
    """Atomically overwrite the latest bounded ML comparison report."""

    def __init__(
        self,
        results_root: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._path = Path(results_root) / _REPORT_FILENAME
        self._clock = clock or (lambda: datetime.now(UTC))

    def record_completed(
        self,
        context: MlRunContext,
        manifest: ModelManifest,
        registry: ModelArtifactRegistry,
        telemetry: Mapping[str, object] | None = None,
        diagnostic_report: object | None = None,
    ) -> None:
        """Write the terminal result of a completed training run."""
        del telemetry, diagnostic_report
        try:
            metrics = registry.read_metrics(context.artifact_id)
        except (FileNotFoundError, ValueError):
            metrics = {}
        promoted = metrics.get("promoted") is True
        reasons = _as_text_list(metrics.get("promotion_reasons"))
        if not reasons:
            reasons = _as_text_list(metrics.get("rejection_reasons"))
        growth = _as_mapping(metrics.get("growth_route"))
        wealth_evidence = growth.get('wealth_evidence') if isinstance(growth.get('wealth_evidence'), Mapping) else None
        base_rows: list[tuple[str, str]] = [
            ("실행 ID", context.artifact_id),
            ("상태", "PROMOTED" if promoted else "NO_TRADE"),
            ("모델", manifest.model_type),
            ("입력", f"{context.feature_rows:,} rows / {context.instrument_count:,} instruments / {context.session_count:,} sessions"),
            ("레이블 horizon", str(context.label_horizon_sessions)),
            ("승격", str(promoted).lower()),
            ("실행 시간", _elapsed_seconds(context.started_at, self._clock())),
            ("Peak RSS", _format_number(peak_rss_mib(), " MiB")),
            ("Base lower CAGR", _format_number(growth.get("base_lower_cagr"), "")),
            ("Stress lower CAGR", _format_number(growth.get("stress_lower_cagr"), "")),
            ("MDD", _format_number(growth.get("mdd"), "")),
        ]
        if isinstance(wealth_evidence, Mapping):
            def _fmt_krw(v: object) -> str:
                if isinstance(v, (int, float)):
                    try:
                        fv = float(v)
                        import math as _math
                        if _math.isfinite(fv):
                            return f"{fv:,.3f}"
                    except Exception:  # noqa: S110
                        pass
                return "-"
            def _fmt_ret(v: object) -> str:
                if isinstance(v, (int, float)):
                    try:
                        fv = float(v)
                        import math as _math
                        if _math.isfinite(fv):
                            return f"{fv:.3f}"
                    except Exception:  # noqa: S110
                        pass
                return "-"
            base_rows.extend([
                ("초기 자본", _fmt_krw(wealth_evidence.get("initial_cash_krw"))),
                ("Base 관측 자산", _fmt_krw(wealth_evidence.get("base_terminal_wealth_krw"))),
                ("Stress 관측 자산", _fmt_krw(wealth_evidence.get("stress_terminal_wealth_krw"))),
                ("Base 관측 수익률", _fmt_ret(wealth_evidence.get("base_observed_return"))),
                ("관측 성장(인증 아님)", "true" if wealth_evidence.get("observed_base_growth_positive") is True else "false" if wealth_evidence.get("observed_base_growth_positive") is False else "-"),
            ])
        base_rows.append(("사유", ", ".join(reasons) if reasons else "-"))
        self._write(
            title="ML 비교 결과 (최신 실행)",
            rows=tuple(base_rows),
        )

    def record_failed(
        self,
        context: MlRunContext,
        phase: str,
        exc: BaseException,
        telemetry: Mapping[str, object] | None = None,
    ) -> None:
        """Write a bounded terminal failure report without suppressing the error."""
        del telemetry
        self._write(
            title="ML 비교 결과 (최신 실행)",
            rows=(
                ("실행 ID", context.artifact_id),
                ("상태", "FAILED"),
                ("단계", _bounded(phase)),
                ("실행 시간", _elapsed_seconds(context.started_at, self._clock())),
                ("오류", _bounded(str(exc) or type(exc).__name__)),
            ),
        )

    def record_research_outcome(
        self,
        *,
        run_id: str,
        status: str,
        data_inputs: Mapping[str, object],
        readiness: Mapping[str, object],
        outcome: Mapping[str, object],
        started_at: datetime,
        failure: BaseException | None = None,
    ) -> None:
        """Write a bounded terminal result for a read-only research study."""
        reasons = _as_text_list(outcome.get("rejection_reasons"))
        reason_counts = _as_mapping(outcome.get("rejection_reason_counts"))
        if not reasons and reason_counts:
            reasons = [f"{key}={value}" for key, value in sorted(reason_counts.items())]
        self._write(
            title="ML 비교 결과 (최신 실행)",
            rows=(
                ("실행 ID", _bounded(run_id)),
                ("상태", _bounded(status)),
                ("결과", _bounded(outcome.get("next_action") or outcome.get("status") or "-")),
                ("후보 수", _bounded(outcome.get("candidate_count") or "-")),
                ("입력 dataset", _bounded(data_inputs.get("feature_dataset_id") or "-")),
                ("Readiness", _bounded(readiness.get("passed"))),
                ("실행 시간", _elapsed_seconds(started_at, self._clock())),
                ("사유", ", ".join(reasons) if reasons else "-"),
                ("오류", _bounded(str(failure)) if failure is not None else "-"),
            ),
        )

    def _write(self, *, title: str, rows: tuple[tuple[str, str], ...]) -> None:
        lines = [f"# {title}", "", "| 항목 | 값 |", "|---|---:|"]
        lines.extend(f"| {key} | {_markdown(value)} |" for key, value in rows)
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self._path.parent,
            prefix=".ml-cmp-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._path)


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_text_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_bounded(item) for item in value if _bounded(item)]


def _bounded(value: object) -> str:
    return " ".join(str(value).splitlines())[:_MAX_TEXT_LENGTH]


def _format_number(value: object, suffix: str) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}{suffix}"
    return "-"


def _elapsed_seconds(started_at: datetime, finished_at: datetime) -> str:
    return f"{max(0.0, (finished_at - started_at).total_seconds()):.3f} sec"


def _markdown(value: str) -> str:
    return value.replace("|", "\\|")
