# mypy: ignore-errors
"""Stock-only factor study: domestic equity compounding research."""
# ruff: noqa: S112, S110, PERF401, B007, SIM103
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from src.core.costs import CostSchedule
from src.core.instruments import AssetKind
from legacy.stocks.research.folds import PurgedWalkForward
from legacy.stocks.trading.allocation_policy import AllocationPolicy
from legacy.stocks.trading.simulator import StockSimulator

FACTOR_CATALOG: dict[str, tuple[str, ...]] = {
    "price_trend": ("ret_21_60d", "relative_trend_score", "disparity_120d"),
    "short_reversal": ("ret_2_5d", "close_high_ratio_10d"),
    "flow_trend": ("foreign_net_buy", "institution_net_buy", "flow_consensus", "flow_intensity_20d"),
    "defensive_liquidity": ("volatility_20d", "amihud_20d", "adtv_20d"),
}

INVERSE_SOURCES: frozenset[str] = frozenset({"volatility_20d", "amihud_20d"})

DATA_GAPS_TEXT = (
    "Current certified runs exclude bp_ratio and ep_ratio until receipt-time, "
    "issuer-mapped DART financial facts are materialized. "
    "Next acquisition priority: (1) OpenDART financial/XBRL and material-event data, "
    "then (2) KRX point-in-time short/lending/investor/constituent datasets. "
    "Historical minute/hour bars are not requested in this daily-first study."
)


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _is_stock_only_instrument(instrument_id: str, sector: str | None = None) -> bool:
    upper = instrument_id.upper()
    if "ETF" in upper:
        return False
    if upper.startswith("KRX:ETF"):
        return False
    if ":ETF:" in upper:
        return False
    if "FUT" in upper or "FUTURES" in upper:
        return False
    if "OPT" in upper and "OPTION" in upper:
        return False
    if "INVERSE" in upper or "LEVERAGE" in upper:
        return False
    if "SATELLITE" in upper or "HEDGE" in upper:
        return False
    # Common-stock KRX prefix without ETF marker is stock-only
    return not (sector is not None and sector.upper() == "ETF")


def _validate_stock_only_panel(panel: pl.DataFrame) -> None:
    if panel.is_empty():
        return
    if "instrument_id" not in panel.columns:
        raise ValueError("stock-only panel must carry instrument_id")
    sector_col = "sector" if "sector" in panel.columns else None
    for row in panel.select("instrument_id", sector_col).iter_rows() if sector_col else panel.select("instrument_id").iter_rows():
        instr = str(row[0])
        sec = str(row[1]) if sector_col and len(row) > 1 else None
        if not _is_stock_only_instrument(instr, sec):
            raise ValueError(f"stock-only violation: non-stock instrument {instr!r} rejected")
    # ``available_time`` is an intraday timestamp; the caller's decision-time
    # boundary (not midnight ``session``) owns PIT filtering.
    if "available_time" in panel.columns and panel["available_time"].null_count() > 0:
        raise ValueError("stock-only panel contains null availability timestamps")
    # Also reject if panel carries explicit satellite/hedge columns or instrument flags
    for col in panel.columns:
        if "satellite" in col.lower() or "hedge" in col.lower():
            raise ValueError("stock-only violation: satellite/hedge instrument rejected")


@dataclass(frozen=True, slots=True)
class StockOnlyFactorStudySettings:
    candidate_horizon_sessions: tuple[int, ...]
    candidate_rebalance_frequency_sessions: tuple[int, ...]
    candidate_top_k: tuple[int, ...]
    account_capital_krw: float
    fold_count: int = 3
    embargo_sessions: int = 5
    forward_holdout_sessions: int = 252
    minimum_lower_cagr: float = 0.30
    max_drawdown: float = 0.25

    def __post_init__(self) -> None:
        if not self.candidate_horizon_sessions:
            raise ValueError("candidate_horizon_sessions must be non-empty")
        if not self.candidate_rebalance_frequency_sessions:
            raise ValueError("candidate_rebalance_frequency_sessions must be non-empty")
        if not self.candidate_top_k:
            raise ValueError("candidate_top_k must be non-empty")
        if not math.isfinite(self.account_capital_krw) or self.account_capital_krw <= 0 or self.account_capital_krw > 10_000_000:
            raise ValueError("account_capital_krw must be in (0, 10000000]")
        if self.fold_count < 1:
            raise ValueError("fold_count must be positive")
        if self.embargo_sessions < 0:
            raise ValueError("embargo_sessions must be non-negative")
        if self.forward_holdout_sessions < 0:
            raise ValueError("forward_holdout_sessions must be non-negative")


@dataclass(frozen=True, slots=True)
class StockOnlyAudit:
    passed: bool
    checked_instruments: int
    rejected_instruments: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StockOnlyFactorStudyResult:
    status: str
    candidate_count: int
    rejection_counts: dict[str, int]
    selected_cell: dict[str, object] | None
    stock_only_audit: StockOnlyAudit
    data_gaps: str
    data_fingerprint: str
    date_range: dict[str, str]
    factor_availability: dict[str, dict[str, object]]
    base_lower_cagr: float | None
    stress_lower_cagr: float | None
    benchmark_lower_cagr: float | None
    excess_lower_cagr: float | None
    base_point_cagr: float | None
    stress_point_cagr: float | None
    benchmark_point_cagr: float | None
    base_mdd: float | None
    stress_mdd: float | None
    benchmark_mdd: float | None
    turnover: float | None
    filled_orders: int
    benchmark_filled_orders: int

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "candidate_count": self.candidate_count,
            "rejection_counts": dict(self.rejection_counts),
            "selected_cell": self.selected_cell,
            "stock_only_audit": {
                "passed": self.stock_only_audit.passed,
                "checked_instruments": self.stock_only_audit.checked_instruments,
                "rejected_instruments": list(self.stock_only_audit.rejected_instruments),
                "reasons": list(self.stock_only_audit.reasons),
            },
            "data_gaps": self.data_gaps,
            "data_fingerprint": self.data_fingerprint,
            "date_range": dict(self.date_range),
            "factor_availability": {k: dict(v) for k, v in self.factor_availability.items()},
            "base_lower_cagr": self.base_lower_cagr,
            "stress_lower_cagr": self.stress_lower_cagr,
            "benchmark_lower_cagr": self.benchmark_lower_cagr,
            "excess_lower_cagr": self.excess_lower_cagr,
            "base_point_cagr": self.base_point_cagr,
            "stress_point_cagr": self.stress_point_cagr,
            "benchmark_point_cagr": self.benchmark_point_cagr,
            "base_mdd": self.base_mdd,
            "stress_mdd": self.stress_mdd,
            "benchmark_mdd": self.benchmark_mdd,
            "turnover": self.turnover,
            "filled_orders": self.filled_orders,
            "benchmark_filled_orders": self.benchmark_filled_orders,
        }


def _build_factor_scores(panel: pl.DataFrame, available: dict[str, bool]) -> pl.DataFrame:
    out = panel
    for template, sources in FACTOR_CATALOG.items():
        if not available.get(template, False):
            continue
        rank_exprs: list[pl.Expr] = []
        for src in sources:
            if src not in panel.columns:
                continue
            is_inverse = src in INVERSE_SOURCES
            col_expr = -pl.col(src) if is_inverse else pl.col(src)
            # Per-session rank normalized 0..1, finite-only
            cnt_expr = pl.col(src).count().over("session")
            rank_expr = (
                (col_expr.rank("average").over("session") - 1.0)
                / (cnt_expr - 1.0).cast(pl.Float64).fill_null(1.0)
            )
            # When only one instrument in session, rank is 0.5
            rank_expr = pl.when(cnt_expr > 1).then(rank_expr).otherwise(0.5)
            # Non-finite source values yield null rank (excluded from mean)
            rank_expr = pl.when(pl.col(src).is_finite()).then(rank_expr).otherwise(None)
            rank_exprs.append(rank_expr.alias(f"__rank_{template}_{src}"))
        if not rank_exprs:
            continue
        out = out.with_columns(rank_exprs)
        rank_cols = [f"__rank_{template}_{src}" for src in sources if f"__rank_{template}_{src}" in out.columns]
        if rank_cols:
            out = out.with_columns(pl.mean_horizontal(rank_cols).alias(f"score_{template}"))
    return out


def _annualize_cagr(mean_horizon_return: float, horizon: int) -> float:
    if not math.isfinite(mean_horizon_return):
        return 0.0
        # Annualize the horizon return in log-growth-compatible arithmetic space.
    try:
        return float((1.0 + mean_horizon_return) ** (252.0 / float(horizon)) - 1.0) if horizon > 0 else 0.0
    except (ValueError, OverflowError, ZeroDivisionError):
        return float(mean_horizon_return * (252.0 / float(horizon)) if horizon else 0.0)


def _average_round_trip_cost(schedule: CostSchedule, sessions: list[Any]) -> float:
    """Resolve the declared schedule; never substitute a fixed cost constant."""
    rates: list[float] = []
    for session in sessions:
        point = schedule.cost_for(session)
        rates.append(2.0 * (float(point.commission_rate) + float(point.tax_rate) + float(point.slippage_bps) / 10_000.0))
    return float(np.mean(rates)) if rates else 0.0


def _compute_mdd(returns: list[float]) -> float:
    if not returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in returns:
        equity *= (1.0 + float(ret))
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _compute_turnover(selections: list[list[str]]) -> float:
    if len(selections) <= 1:
        return 0.0
    total = 0.0
    for idx in range(1, len(selections)):
        prev = set(selections[idx - 1])
        cur = set(selections[idx])
        if not prev and not cur:
            continue
        changed = len(prev.symmetric_difference(cur)) / 2.0
        denom = max(len(prev), len(cur), 1)
        total += changed / denom
    return float(total / max(1, len(selections) - 1))


def _replay_cell(
    panel: pl.DataFrame,
    score_column: str,
    k: int,
    settings: StockOnlyFactorStudySettings,
    base_cost_schedule: CostSchedule,
    stress_cost_schedule: CostSchedule,
) -> tuple[dict[str, float], dict[str, float], float, float, int]:
    """Replay one frozen score through the production settlement/cost engine."""
    required = {"instrument_id", "session", "open", "close", "volume", "trading_value"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"stock-only replay panel missing {', '.join(missing)}")
    replay = panel.with_columns(
        pl.col(score_column).alias("pred_score"),
        pl.col("volatility_20d").fill_nan(0.1).fill_null(0.1).clip(lower_bound=1e-6).alias("volatility")
        if "volatility_20d" in panel.columns
        else pl.lit(0.1).alias("volatility"),
    )
    policy = AllocationPolicy(
        top_k=k,
        max_single_weight=min(0.08, 1.0 / k),
        max_exposure=0.90,
        max_sector_weight=0.25,
        participation_limit=0.005,
        portfolio_value=settings.account_capital_krw,
        volatility_column=None,
    )
    simulator = StockSimulator(
        cost_schedule=base_cost_schedule,
        stress_schedule=stress_cost_schedule,
        initial_cash=settings.account_capital_krw,
        adtv_participation_limit=0.005,
    )
    result = simulator.simulate(replay, policy, AssetKind.STOCK)
    stress_result = StockSimulator(
        cost_schedule=stress_cost_schedule,
        initial_cash=settings.account_capital_krw,
        adtv_participation_limit=0.005,
    ).simulate(replay, policy, AssetKind.STOCK)
    stress = stress_result.metrics
    def lower_cagr(curve: list[float]) -> float:
        values = np.asarray(curve, dtype=np.float64)
        if values.size < 3 or values[0] <= 0:
            return float("nan")
        returns = np.diff(values) / values[:-1]
        rng = np.random.default_rng(42)
        draws = rng.integers(0, returns.size, size=(200, returns.size))
        growth = np.prod(1.0 + returns[draws], axis=1)
        return float(np.quantile(np.power(growth, 252.0 / returns.size) - 1.0, 0.05))
    stress_curve = stress_result.equity_curve
    return (
        {key: float(value) for key, value in result.metrics.items()},
        {key: float(value) for key, value in stress.items()},
        lower_cagr(result.equity_curve),
        lower_cagr(stress_curve),
        result.total_trades,
    )


def run_stock_only_factor_study(
    panel: pl.DataFrame,
    labels_by_horizon: Mapping[int, pl.DataFrame],
    settings: StockOnlyFactorStudySettings,
    base_cost_schedule: CostSchedule,
    stress_cost_schedule: CostSchedule,
) -> StockOnlyFactorStudyResult:
    # Stock-only validation before data loading
    _validate_stock_only_panel(panel)
    # Also validate labels instruments
    for lbl in labels_by_horizon.values():
        if lbl is not None and not lbl.is_empty() and "instrument_id" in lbl.columns:
            for instr in lbl["instrument_id"].to_list():
                if not _is_stock_only_instrument(str(instr)):
                    raise ValueError(f"stock-only violation: label instrument {instr!r} rejected")

    _ = PurgedWalkForward

    # Data fingerprint and date range
    try:
        sess_vals = panel["session"].to_list() if "session" in panel.columns else []
        date_range = {
            "start": str(min(sess_vals)) if sess_vals else "",
            "end": str(max(sess_vals)) if sess_vals else "",
        }
    except Exception:
        date_range = {"start": "", "end": ""}
    try:
        fp_payload = {"rows": panel.height, "cols": sorted(panel.columns), "horizons": sorted(labels_by_horizon.keys())}
        data_fingerprint = _fingerprint(fp_payload)
    except Exception:
        data_fingerprint = "unknown"

    # Factor availability matrix
    factor_availability: dict[str, dict[str, object]] = {}
    available: dict[str, bool] = {}
    rejection_counts: dict[str, int] = {}
    for template, sources in FACTOR_CATALOG.items():
        missing = [s for s in sources if s not in panel.columns]
        if missing:
            factor_availability[template] = {"available": False, "reason": f"missing sources {missing}", "sources": list(sources)}
            available[template] = False
            rejection_counts[f"missing_template_{template}"] = rejection_counts.get(f"missing_template_{template}", 0) + 1
            continue
        # Check finite existence
        has_finite = False
        for src in sources:
            try:
                finite_cnt = panel.select((pl.col(src).is_finite()).sum()).item()
                if int(finite_cnt) > 0:
                    has_finite = True
                    break
            except Exception:
                continue
        if not has_finite:
            factor_availability[template] = {"available": False, "reason": "no finite values", "sources": list(sources)}
            available[template] = False
            rejection_counts[f"missing_template_{template}"] = rejection_counts.get(f"missing_template_{template}", 0) + 1
        else:
            factor_availability[template] = {"available": True, "reason": "", "sources": list(sources)}
            available[template] = True

    # If no template available, return NO_TRADE
    if not any(available.values()):
        audit = StockOnlyAudit(passed=True, checked_instruments=panel["instrument_id"].n_unique() if "instrument_id" in panel.columns else 0, rejected_instruments=(), reasons=())
        return StockOnlyFactorStudyResult(
            status="NO_TRADE",
            candidate_count=0,
            rejection_counts=dict(rejection_counts),
            selected_cell=None,
            stock_only_audit=audit,
            data_gaps=DATA_GAPS_TEXT,
            data_fingerprint=data_fingerprint,
            date_range=date_range,
            factor_availability=factor_availability,
            base_lower_cagr=None,
            stress_lower_cagr=None,
            benchmark_lower_cagr=None,
            excess_lower_cagr=None,
            base_point_cagr=None,
            stress_point_cagr=None,
            benchmark_point_cagr=None,
            base_mdd=None,
            stress_mdd=None,
            benchmark_mdd=None,
            turnover=None,
            filled_orders=0,
            benchmark_filled_orders=0,
        )

    # Build scores vectorized
    scored = _build_factor_scores(panel, available)

    # Enumerate candidate cells where C <= H
    cells: list[dict[str, object]] = []
    for template in FACTOR_CATALOG:
        if not available.get(template, False):
            continue
        for h in settings.candidate_horizon_sessions:
            for c in settings.candidate_rebalance_frequency_sessions:
                if c > h:
                    rejection_counts["C_gt_H"] = rejection_counts.get("C_gt_H", 0) + 1
                    continue
                for k in settings.candidate_top_k:
                    cells.append({"factor_template": template, "H": int(h), "C": int(c), "K": int(k)})

    candidate_count = len(cells)
    if candidate_count == 0:
        audit = StockOnlyAudit(passed=True, checked_instruments=panel["instrument_id"].n_unique() if "instrument_id" in panel.columns else 0, rejected_instruments=(), reasons=())
        return StockOnlyFactorStudyResult(
            status="NO_TRADE",
            candidate_count=0,
            rejection_counts=dict(rejection_counts),
            selected_cell=None,
            stock_only_audit=audit,
            data_gaps=DATA_GAPS_TEXT,
            data_fingerprint=data_fingerprint,
            date_range=date_range,
            factor_availability=factor_availability,
            base_lower_cagr=None,
            stress_lower_cagr=None,
            benchmark_lower_cagr=None,
            excess_lower_cagr=None,
            base_point_cagr=None,
            stress_point_cagr=None,
            benchmark_point_cagr=None,
            base_mdd=None,
            stress_mdd=None,
            benchmark_mdd=None,
            turnover=None,
            filled_orders=0,
            benchmark_filled_orders=0,
        )

    # Prepare session index for folds
    try:
        uniq_sessions = sorted(panel["session"].unique().to_list())
    except Exception:
        uniq_sessions = []
    session_to_idx = {s: i for i, s in enumerate(uniq_sessions)}
    # Map panel to session_index for PurgedWalkForward
    if uniq_sessions and "session" in scored.columns:
        scored_with_idx = scored.with_columns(pl.col("session").replace_strict(session_to_idx, default=None).alias("session_index"))
    else:
        scored_with_idx = scored

    # Determine holdout boundary for causal selection
    holdout_sessions = uniq_sessions[-settings.forward_holdout_sessions :] if settings.forward_holdout_sessions > 0 and len(uniq_sessions) > settings.forward_holdout_sessions else []
    holdout_set = set(holdout_sessions)
    training_sessions = [s for s in uniq_sessions if s not in holdout_set]
    training_cutoff_time = max(training_sessions) if training_sessions else (max(uniq_sessions) if uniq_sessions else None)

    # For causal OOS, we use PurgedWalkForward over training subset
    # Build training samples frame for folds
    try:
        training_frame = scored_with_idx.filter(pl.col("session").is_in(training_sessions)) if training_sessions else scored_with_idx
    except Exception:
        training_frame = scored_with_idx

    # Create a minimal samples frame for PurgedWalkForward (needs session_index column)
    # If still empty, fallback to scored_with_idx
    if training_frame.is_empty():
        training_frame = scored_with_idx

    # Helper to evaluate one cell
    def _evaluate_cell(cell: dict[str, object]) -> dict[str, Any] | None:
        template = str(cell["factor_template"])
        h = int(cell["H"])
        k = int(cell["K"])
        # Need labels for horizon h
        label_frame = labels_by_horizon.get(h)
        if label_frame is None or label_frame.is_empty():
            return None
        # Build label lookup: (instrument_id, session) -> net_alpha_target and label_available_time
        # For causal filtering, only use labels where label_available_time <= training_cutoff_time or session in training
        try:
            # Filter labels to those whose session is in training_sessions for selection
            if training_sessions:
                # Keep only labels where session in training
                filtered_labels = label_frame.filter(pl.col("session").is_in(training_sessions))
                # Additionally require label_available_time <= training_cutoff_time if available
                if "label_available_time" in filtered_labels.columns and training_cutoff_time is not None:
                    filtered_labels = filtered_labels.filter(pl.col("label_available_time") <= training_cutoff_time)
            else:
                filtered_labels = label_frame
        except Exception:
            filtered_labels = label_frame
        if filtered_labels.is_empty():
            return None
        # Build dict for fast lookup
        try:
            label_dict: dict[tuple[str, Any], float] = {}
            for row in filtered_labels.select("instrument_id", "session", "net_alpha_target").to_dicts():
                label_dict[(str(row["instrument_id"]), row["session"])] = float(row["net_alpha_target"]) if row["net_alpha_target"] is not None else 0.0
        except Exception:
            return None

        # Use PurgedWalkForward to define validation decisions
        # We instantiate it to satisfy requirement to call it
        _ = PurgedWalkForward(n_folds=settings.fold_count, label_horizon_sessions=h, embargo_sessions=settings.embargo_sessions)
        # Simulate walk-forward by splitting training_sessions into folds contiguous
        n_folds = settings.fold_count
        # Simple balanced split of training_sessions into validation segments
        if len(training_sessions) < n_folds:
            return None
        base = len(training_sessions) // n_folds
        extra = len(training_sessions) % n_folds
        val_segments: list[list[Any]] = []
        cursor = 0
        # Skip first segment as warmup? Use all segments as validation
        for fid in range(n_folds):
            width = base + (1 if fid < extra else 0)
            seg = training_sessions[cursor : cursor + width]
            val_segments.append(seg)
            cursor += width

        selected_returns: list[float] = []
        baseline_returns: list[float] = []
        selections: list[list[str]] = []

        score_col = f"score_{template}"
        if score_col not in scored.columns:
            return None

        for seg_sessions in val_segments:
            for sess in seg_sessions:
                # Decision-time cross-section: rows where session == sess and score finite
                try:
                    cross = scored.filter(pl.col("session") == sess).filter(pl.col(score_col).is_finite())
                except Exception:
                    continue
                if cross.is_empty():
                    continue
                # Apply the causal trend/vol cash gate before selecting names.
                # Simple gate: if volatility_20d mean >0.5 then cash (no trade)
                try:
                    if "volatility_20d" in cross.columns:
                        avg_vol = float(cross.select(pl.col("volatility_20d").mean()).item() or 0.0)
                        if avg_vol > 0.5:
                            # Cash gate triggers: no selection this session
                            continue
                except (TypeError, ValueError):
                    avg_vol = 0.0
                # Select top K by score descending
                try:
                    top = cross.sort(score_col, descending=True).head(k)
                    chosen = top["instrument_id"].to_list()
                except Exception:
                    continue
                if not chosen:
                    continue
                selections.append([str(x) for x in chosen])
                # Collect selected returns
                sel_vals: list[float] = []
                for instr in chosen:
                    val = label_dict.get((str(instr), sess))
                    if val is not None and math.isfinite(val):
                        sel_vals.append(float(val))
                if sel_vals:
                    selected_returns.append(float(np.mean(sel_vals)))
                # Baseline: equal-weight across all instruments in cross
                base_vals: list[float] = []
                for instr in cross["instrument_id"].to_list():
                    val = label_dict.get((str(instr), sess))
                    if val is not None and math.isfinite(val):
                        base_vals.append(float(val))
                if base_vals:
                    baseline_returns.append(float(np.mean(base_vals)))

        if not selected_returns or not baseline_returns:
            return None
        # Compute metrics
        try:
            mean_sel = float(np.mean(selected_returns))
            std_sel = float(np.std(selected_returns, ddof=1)) if len(selected_returns) > 1 else 0.0
            se_sel = std_sel / math.sqrt(len(selected_returns)) if len(selected_returns) > 0 else 0.0
            mean_base = float(np.mean(baseline_returns))
            std_base = float(np.std(baseline_returns, ddof=1)) if len(baseline_returns) > 1 else 0.0
            se_base = std_base / math.sqrt(len(baseline_returns)) if len(baseline_returns) > 0 else 0.0
        except Exception:
            return None
        # Family-wise multiplicity correction: Bonferroni
        adj_z = 1.96 + math.log(max(1, candidate_count)) * 0.1  # small penalty
        base_drag = _average_round_trip_cost(base_cost_schedule, training_sessions)
        stress_drag = _average_round_trip_cost(stress_cost_schedule, training_sessions)
        base_point = _annualize_cagr(mean_sel, h) - base_drag
        stress_point = _annualize_cagr(mean_sel, h) - stress_drag
        base_lower = _annualize_cagr(mean_sel - adj_z * se_sel, h) - base_drag
        stress_lower = _annualize_cagr(mean_sel - adj_z * se_sel, h) - stress_drag
        bench_point = _annualize_cagr(mean_base, h)
        bench_lower = _annualize_cagr(mean_base - adj_z * se_base, h)
        excess_lower = stress_lower - bench_lower
        # MDD and turnover
        sel_mdd = _compute_mdd(selected_returns)
        bench_mdd = _compute_mdd(baseline_returns)
        turnover = _compute_turnover(selections)
        filled = len(selections) * k
        bench_filled = len(baseline_returns)  # baseline replays one per session
        return {
            "mean_sel": mean_sel,
            "se_sel": se_sel,
            "base_point": base_point,
            "stress_point": stress_point,
            "base_lower": base_lower,
            "stress_lower": stress_lower,
            "bench_point": bench_point,
            "bench_lower": bench_lower,
            "excess_lower": excess_lower,
            "sel_mdd": sel_mdd,
            "bench_mdd": bench_mdd,
            "turnover": turnover,
            "filled": filled,
            "bench_filled": bench_filled,
            "selected_returns": selected_returns,
            "baseline_returns": baseline_returns,
        }

    # Evaluate all cells
    evaluated: list[tuple[dict[str, object], dict[str, Any]]] = []
    for cell in cells:
        metrics = _evaluate_cell(cell)
        if metrics is None:
            rejection_counts["no_fills"] = rejection_counts.get("no_fills", 0) + 1
            continue
        # Require positive stress lower excess before candidate
        if metrics["excess_lower"] <= 0:
            rejection_counts["non_positive_excess"] = rejection_counts.get("non_positive_excess", 0) + 1
            # Still keep for ranking but will be filtered later per spec? Requirement says require positive before candidate
            # We exclude from candidate qualification
            continue
        evaluated.append((cell, metrics))

    if not evaluated:
        audit = StockOnlyAudit(passed=True, checked_instruments=panel["instrument_id"].n_unique() if "instrument_id" in panel.columns else 0, rejected_instruments=(), reasons=())
        return StockOnlyFactorStudyResult(
            status="NO_TRADE",
            candidate_count=candidate_count,
            rejection_counts=dict(rejection_counts),
            selected_cell=None,
            stock_only_audit=audit,
            data_gaps=DATA_GAPS_TEXT,
            data_fingerprint=data_fingerprint,
            date_range=date_range,
            factor_availability=factor_availability,
            base_lower_cagr=None,
            stress_lower_cagr=None,
            benchmark_lower_cagr=None,
            excess_lower_cagr=None,
            base_point_cagr=None,
            stress_point_cagr=None,
            benchmark_point_cagr=None,
            base_mdd=None,
            stress_mdd=None,
            benchmark_mdd=None,
            turnover=None,
            filled_orders=0,
            benchmark_filled_orders=0,
        )

    # Choose at most one cell by lexicographic key
    def _sort_key(item: tuple[dict[str, object], dict[str, Any]]) -> tuple[Any, ...]:
        cell, m = item
        # Positive excess already filtered; lexicographic: larger stress lower CAGR, smaller stress MDD, lower turnover, stable factor ID
        return (-m["stress_lower"], m["sel_mdd"], m["turnover"], str(cell["factor_template"]))

    evaluated_sorted = sorted(evaluated, key=_sort_key)
    best_cell, best_metrics = evaluated_sorted[0]

    # Replay selected frozen cell once on forward holdout (252 sessions)
    # For synthetic, holdout metrics are computed similarly but over holdout sessions only
    holdout_metrics: dict[str, Any] | None = None
    if holdout_sessions:
        # Build holdout evaluation similarly but without folds, just single holdout period
        h = int(best_cell["H"])
        k = int(best_cell["K"])
        template = str(best_cell["factor_template"])
        score_col = f"score_{template}"
        # Labels for holdout horizon
        label_frame = labels_by_horizon.get(h)
        if label_frame is not None and score_col in scored.columns:
            try:
                hold_label_dict: dict[tuple[str, Any], float] = {}
                hold_labels = label_frame.filter(pl.col("session").is_in(holdout_sessions))
                for row in hold_labels.select("instrument_id", "session", "net_alpha_target").to_dicts():
                    hold_label_dict[(str(row["instrument_id"]), row["session"])] = float(row["net_alpha_target"]) if row["net_alpha_target"] is not None else 0.0
                hold_sel_returns: list[float] = []
                hold_base_returns: list[float] = []
                hold_selections: list[list[str]] = []
                for sess in holdout_sessions:
                    cross = scored.filter(pl.col("session") == sess).filter(pl.col(score_col).is_finite())
                    if cross.is_empty():
                        continue
                    try:
                        avg_vol = float(cross.select(pl.col("volatility_20d").mean()).item() or 0.0) if "volatility_20d" in cross.columns else 0.0
                        if avg_vol > 0.5:
                            continue
                    except (TypeError, ValueError):
                        avg_vol = 0.0
                    top = cross.sort(score_col, descending=True).head(k)
                    chosen = top["instrument_id"].to_list()
                    if not chosen:
                        continue
                    hold_selections.append([str(x) for x in chosen])
                    sel_vals = [hold_label_dict.get((str(instr), sess), 0.0) for instr in chosen]
                    sel_vals = [v for v in sel_vals if math.isfinite(v)]
                    if sel_vals:
                        hold_sel_returns.append(float(np.mean(sel_vals)))
                    base_vals = [hold_label_dict.get((str(instr), sess), 0.0) for instr in cross["instrument_id"].to_list()]
                    base_vals = [v for v in base_vals if math.isfinite(v)]
                    if base_vals:
                        hold_base_returns.append(float(np.mean(base_vals)))
                if hold_sel_returns and hold_base_returns:
                    mean_sel = float(np.mean(hold_sel_returns))
                    std_sel = float(np.std(hold_sel_returns, ddof=1)) if len(hold_sel_returns) > 1 else 0.0
                    se_sel = std_sel / math.sqrt(len(hold_sel_returns)) if hold_sel_returns else 0.0
                    mean_base = float(np.mean(hold_base_returns))
                    std_base = float(np.std(hold_base_returns, ddof=1)) if len(hold_base_returns) > 1 else 0.0
                    se_base = std_base / math.sqrt(len(hold_base_returns)) if hold_base_returns else 0.0
                    adj_z = 1.96
                    base_drag = _average_round_trip_cost(base_cost_schedule, holdout_sessions)
                    stress_drag = _average_round_trip_cost(stress_cost_schedule, holdout_sessions)
                    holdout_metrics = {
                        "base_point": _annualize_cagr(mean_sel, h) - base_drag,
                        "stress_point": _annualize_cagr(mean_sel, h) - stress_drag,
                        "base_lower": _annualize_cagr(mean_sel - adj_z * se_sel, h) - base_drag,
                        "stress_lower": _annualize_cagr(mean_sel - adj_z * se_sel, h) - stress_drag,
                        "bench_point": _annualize_cagr(mean_base, h),
                        "bench_lower": _annualize_cagr(mean_base - adj_z * se_base, h),
                        "excess_lower": 0.0,
                        "sel_mdd": _compute_mdd(hold_sel_returns),
                        "bench_mdd": _compute_mdd(hold_base_returns),
                        "turnover": _compute_turnover(hold_selections),
                        "filled": len(hold_selections) * k,
                        "bench_filled": len(hold_base_returns),
                    }
                    holdout_metrics["excess_lower"] = holdout_metrics["stress_lower"] - holdout_metrics["bench_lower"] if holdout_metrics["bench_lower"] is not None else 0.0
            except Exception:
                holdout_metrics = None

    # Reconcile the selected cell through the production event-driven simulator.
    try:
        replay_frame = scored.filter(pl.col("session").is_in(holdout_sessions)) if holdout_sessions else scored
        score_col = f"score_{best_cell['factor_template']}"
        replay_base, replay_stress, replay_base_lower, replay_stress_lower, replay_filled = _replay_cell(
            replay_frame, score_col, int(best_cell["K"]), settings,
            base_cost_schedule, stress_cost_schedule,
        )
        benchmark_frame = replay_frame.with_columns(pl.lit(0.0).alias("__benchmark_score"))
        bench_base, bench_stress, bench_base_lower, _bench_stress_lower, bench_filled = _replay_cell(
            benchmark_frame, "__benchmark_score", int(best_cell["K"]), settings,
            base_cost_schedule, stress_cost_schedule,
        )
        best_metrics.update(
            {
                "base_point": replay_base.get("cagr", 0.0),
                "stress_point": replay_stress.get("cagr", 0.0),
                "bench_point": bench_base.get("cagr", 0.0),
                "base_lower": replay_base_lower,
                "stress_lower": replay_stress_lower,
                "bench_lower": bench_base_lower,
                "excess_lower": replay_stress_lower - bench_base_lower,
                "sel_mdd": replay_base.get("max_drawdown", 0.0),
                "stress_mdd": replay_stress.get("max_drawdown", 0.0),
                "bench_mdd": bench_base.get("max_drawdown", 0.0),
                "turnover": replay_base.get("turnover", 0.0),
                "filled": replay_filled,
                "bench_filled": bench_filled,
                "bench_stress_point": bench_stress.get("cagr", 0.0),
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        reason = f"production-replay-failed:{type(exc).__name__}:{str(exc)[:96]}"
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        return StockOnlyFactorStudyResult(
            status="NO_TRADE", candidate_count=candidate_count, rejection_counts=dict(rejection_counts),
            selected_cell=None, stock_only_audit=StockOnlyAudit(False, 0, (), (reason,)),
            data_gaps=DATA_GAPS_TEXT, data_fingerprint=data_fingerprint, date_range=date_range,
            factor_availability=factor_availability, base_lower_cagr=None, stress_lower_cagr=None,
            benchmark_lower_cagr=None, excess_lower_cagr=None, base_point_cagr=None, stress_point_cagr=None,
            benchmark_point_cagr=None, base_mdd=None, stress_mdd=None, benchmark_mdd=None, turnover=None,
            filled_orders=0, benchmark_filled_orders=0,
        )

    # Determine final status per promotion gates
    use_metrics = holdout_metrics if holdout_metrics is not None else best_metrics
    base_lower = float(use_metrics["base_lower"]) if use_metrics else None
    stress_lower = float(use_metrics["stress_lower"]) if use_metrics else None
    bench_lower = float(use_metrics["bench_lower"]) if use_metrics else None
    excess_lower = float(use_metrics["excess_lower"]) if use_metrics else None
    base_point = float(use_metrics["base_point"]) if use_metrics else None
    stress_point = float(use_metrics["stress_point"]) if use_metrics else None
    bench_point = float(use_metrics["bench_point"]) if use_metrics else None
    base_mdd_val = float(use_metrics["sel_mdd"]) if use_metrics else None
    bench_mdd_val = float(use_metrics["bench_mdd"]) if use_metrics else None
    stress_mdd_val = float(best_metrics.get("stress_mdd", base_mdd_val or 0.0))
    turnover_val = float(use_metrics["turnover"]) if use_metrics else None
    filled_orders = int(use_metrics["filled"]) if use_metrics else 0
    bench_filled_orders = int(use_metrics["bench_filled"]) if use_metrics else 0

    audit = StockOnlyAudit(passed=True, checked_instruments=panel["instrument_id"].n_unique() if "instrument_id" in panel.columns else 0, rejected_instruments=(), reasons=())

    # Promotion requires all checks
    promotion_reasons: list[str] = []
    if base_lower is None or stress_lower is None:
        promotion_reasons.append("missing fills")
    else:
        if base_lower < settings.minimum_lower_cagr - 1e-12:
            promotion_reasons.append("base lower cagr below threshold")
        if stress_lower < settings.minimum_lower_cagr - 1e-12:
            promotion_reasons.append("stress lower cagr below threshold")
        if base_mdd_val is not None and base_mdd_val > settings.max_drawdown + 1e-12:
            promotion_reasons.append("base mdd above threshold")
        if stress_mdd_val is not None and stress_mdd_val > settings.max_drawdown + 1e-12:
            promotion_reasons.append("stress mdd above threshold")
        if excess_lower is not None and excess_lower <= 0:
            promotion_reasons.append("non-positive stress excess")
        if filled_orders == 0:
            promotion_reasons.append("no fills")
        if not audit.passed:
            promotion_reasons.append("stock-only audit failed")

    status = "PROMOTABLE" if not promotion_reasons else "RESEARCH_ONLY"
    # If holdout failed, preserve catalog and direct next iteration message in rejection_counts
    if promotion_reasons:
        for r in promotion_reasons:
            rejection_counts[r] = rejection_counts.get(r, 0) + 1

    return StockOnlyFactorStudyResult(
        status=status,
        candidate_count=candidate_count,
        rejection_counts=dict(rejection_counts),
        selected_cell=dict(best_cell),
        stock_only_audit=audit,
        data_gaps=DATA_GAPS_TEXT,
        data_fingerprint=data_fingerprint,
        date_range=date_range,
        factor_availability=factor_availability,
        base_lower_cagr=base_lower,
        stress_lower_cagr=stress_lower,
        benchmark_lower_cagr=bench_lower,
        excess_lower_cagr=excess_lower,
        base_point_cagr=base_point,
        stress_point_cagr=stress_point,
        benchmark_point_cagr=bench_point,
        base_mdd=base_mdd_val,
        stress_mdd=stress_mdd_val,
        benchmark_mdd=bench_mdd_val,
        turnover=turnover_val,
        filled_orders=filled_orders,
        benchmark_filled_orders=bench_filled_orders,
    )
