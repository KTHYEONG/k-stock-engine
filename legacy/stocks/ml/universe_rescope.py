"""Deterministic point-in-time universe re-scoping for direct ML loads.

The kernel applies one causal row mask to the base panel scan before the
feature join, so fitting, calibration, replay, and the exposure-matched
benchmark all observe the same restricted opportunity set. Membership at a
decision session depends only on trailing values observable at that session.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from legacy.stocks.ml.contracts import UniverseRescopeSettings

__all__ = ["apply_universe_rescope"]


def apply_universe_rescope(
    base_lf: pl.LazyFrame,
    settings: UniverseRescopeSettings | None,
) -> tuple[pl.LazyFrame, dict[str, object]]:
    """Mask one base scan to the declared trailing market-cap band.

    Args:
        base_lf: lazy base-panel projection carrying ``instrument_id``,
            ``session``, and ``market_cap`` (plus ``trading_value`` when
            ``max_adtv_quantile`` is set).
        settings: pre-registered rescope policy; ``None`` disables the mask
            and returns the input untouched with empty diagnostics.

    Returns:
        The masked lazy plan plus bounded aggregate diagnostics (counts,
        fractions, per-session kept-instrument percentiles, and the policy
        fingerprint). Raw rows and per-instrument values never enter the
        diagnostics. Raises ``ValueError`` when a required column is absent
        so an enabled policy can never silently degrade into a no-op.
    """
    if settings is None:
        return base_lf, {}

    schema_names = set(base_lf.collect_schema().names())
    missing = [
        column
        for column in ("instrument_id", "session", "market_cap")
        if column not in schema_names
    ]
    if missing:
        raise ValueError(
            f"universe rescope requires column(s) {missing} in the base panel"
        )
    if settings.max_adtv_quantile is not None and "trading_value" not in schema_names:
        raise ValueError(
            "universe rescope requires column 'trading_value' for max_adtv_quantile"
        )

    def _rank_fraction(column: str) -> pl.Expr:
        return (
            pl.col(column).rank("ordinal").over("session") - 1
        ) / pl.col(column).len().over("session")

    keep = _rank_fraction("market_cap") >= settings.market_cap_quantile_lo
    if settings.market_cap_quantile_hi < 1.0:
        keep = keep & (_rank_fraction("market_cap") < settings.market_cap_quantile_hi)
    # Null trailing metrics are not point-in-time membership evidence.
    keep = keep & pl.col("market_cap").is_not_null()
    if settings.min_market_cap_krw is not None:
        keep = keep & (pl.col("market_cap") >= settings.min_market_cap_krw)
    if settings.max_adtv_quantile is not None:
        keep = keep & (
            _rank_fraction("trading_value") <= settings.max_adtv_quantile
        ) & pl.col("trading_value").is_not_null()

    total = int(base_lf.select(pl.len()).collect().item() or 0)
    masked = base_lf.with_columns(keep.alias("__rescope_keep")).filter(
        pl.col("__rescope_keep")
    ).drop("__rescope_keep")
    summary = (
        masked.group_by("session")
        .agg(pl.len().alias("__kept"))
        .select(
            pl.col("__kept").sum().alias("__sum"),
            pl.col("__kept").quantile(0.10, interpolation="nearest").alias("__p10"),
            pl.col("__kept").quantile(0.50, interpolation="nearest").alias("__p50"),
            pl.col("__kept").quantile(0.90, interpolation="nearest").alias("__p90"),
        )
        .collect()
    )
    kept = int(summary["__sum"][0] or 0)

    def _percentile(name: str) -> float | None:
        value = summary[name][0]
        return None if value is None else round(float(value), 12)

    diagnostics: dict[str, object] = {
        "fingerprint": settings.fingerprint,
        "kept_row_count": kept,
        "dropped_row_count": total - kept,
        "kept_row_fraction": round(kept / total, 12) if total > 0 else 0.0,
        "kept_session_instrument_p10": _percentile("__p10"),
        "kept_session_instrument_p50": _percentile("__p50"),
        "kept_session_instrument_p90": _percentile("__p90"),
    }
    return masked, diagnostics
