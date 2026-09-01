"""ETF research: stability analysis and go/no-go decision.

Reimplements the legacy ETF go/no-go and stability checks with modern typed
result objects and invariant-based thresholds, replacing ad-hoc metric floats.
"""
from __future__ import annotations

from dataclasses import dataclass

from legacy.etfs.backtesting.results import EtfBacktestResult
from legacy.etfs.research.walk_forward import WalkForwardReport


@dataclass(frozen=True, slots=True)
class StabilityReport:
    """Stability assessment across walk-forward folds.

    Criteria are structural invariants: positive out-of-sample growth,
    bounded drawdown, a minimum profit factor, and a minimum trade count.
    """

    mean_return_pct: float
    mean_mdd_pct: float
    mean_profit_factor: float
    total_trades: int
    passed: bool
    reasons: dict[str, bool]

    def __post_init__(self) -> None:
        required = {
            "out_of_sample_growth": self.mean_return_pct > 0.0,
            "max_drawdown_bounded": abs(self.mean_mdd_pct) <= 25.0,
            "profit_factor_above_floor": self.mean_profit_factor >= 1.10,
            "sufficient_trades": self.total_trades >= 5,
        }
        if self.passed != all(required.values()):
            raise ValueError("stability passed flag must equal the invariant set")


def assess_stability(report: WalkForwardReport) -> StabilityReport:
    """Evaluate a walk-forward report against structural stability invariants."""
    if not report.results:
        return StabilityReport(
            mean_return_pct=0.0,
            mean_mdd_pct=0.0,
            mean_profit_factor=0.0,
            total_trades=0,
            passed=False,
            reasons={},
        )
    mean_pf = sum(r.result.profit_factor for r in report.results) / len(report.results)
    total_trades = sum(r.result.total_trades for r in report.results)
    reasons = {
        "out_of_sample_growth": report.mean_return_pct > 0.0,
        "max_drawdown_bounded": abs(report.mean_mdd_pct) <= 25.0,
        "profit_factor_above_floor": mean_pf >= 1.10,
        "sufficient_trades": total_trades >= 5,
    }
    return StabilityReport(
        mean_return_pct=report.mean_return_pct,
        mean_mdd_pct=report.mean_mdd_pct,
        mean_profit_factor=mean_pf,
        total_trades=total_trades,
        passed=all(reasons.values()),
        reasons=reasons,
    )


def summarize_result(result: EtfBacktestResult) -> dict[str, float]:
    """Extract scalar stability inputs from one backtest result."""
    return {
        "total_return_pct": result.total_return_pct,
        "mdd_pct": result.mdd_pct,
        "profit_factor": result.profit_factor,
        "total_trades": float(result.total_trades),
    }
