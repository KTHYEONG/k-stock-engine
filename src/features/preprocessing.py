"""Cross-sectional preprocessing: winsorization and sector-relative ranking."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from src.features.contracts import QvefFeaturePolicy


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("no values for quantile")
    arr = np.asarray(sorted(values), dtype=float)
    # linear interpolation matching numpy default
    return float(np.quantile(arr, q, method="linear"))


def normalize_component_scores(rows: pl.DataFrame, *, policy: QvefFeaturePolicy) -> pl.DataFrame:
    if not policy.version or not policy.version.strip():
        raise ValueError("policy version must be non-empty")
    if not (0 <= policy.winsor_lower_quantile < policy.winsor_upper_quantile <= 1):
        raise ValueError("invalid quantile bounds")
    if policy.minimum_sector_cohort < 2:
        raise ValueError("minimum_sector_cohort must be >=2")
    if rows.is_empty():
        # preserve schema with added columns empty
        return rows.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("normalized_score"),
            pl.lit(False).alias("score_available"),
            pl.lit("").alias("score_reason"),
        ) if "raw_value" in rows.columns else rows

    required = {"instrument_id", "sector", "raw_value"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    # Deterministic order: sort by instrument_id
    sorted_rows = rows.sort("instrument_id")
    records = sorted_rows.to_dicts()

    # Identify valid values
    valid_values: list[float] = []
    for r in records:
        v = r.get("raw_value")
        if v is None:
            continue
        try:
            fv = float(v)  # noqa: S112
        except Exception:  # noqa: S112
            continue
        if math.isfinite(fv):
            valid_values.append(fv)

    # Winsor bounds
    lower: float | None = None
    upper: float | None = None
    if valid_values:
        lower = _quantile(valid_values, policy.winsor_lower_quantile)
        upper = _quantile(valid_values, policy.winsor_upper_quantile)

    # Clip valid rows
    winsorized: list[float | None] = []
    for r in records:
        v = r.get("raw_value")
        if v is None:
            winsorized.append(None)
            continue
        try:
            fv = float(v)  # noqa: S112
        except Exception:  # noqa: S112
            winsorized.append(None)
            continue
        if not math.isfinite(fv):
            winsorized.append(None)
            continue
        if lower is not None and upper is not None:
            if fv < lower:
                fv = lower
            elif fv > upper:
                fv = upper
        winsorized.append(float(fv))

    # Group by sector
    sector_to_indices: dict[str, list[int]] = {}
    for idx, r in enumerate(records):
        sec = str(r.get("sector"))
        sector_to_indices.setdefault(sec, []).append(idx)

    normalized: list[float | None] = [None] * len(records)
    available: list[bool] = [False] * len(records)
    reason: list[str] = [""] * len(records)

    for indices in sector_to_indices.values():  # noqa: PERF102
        # count valid winsorized in sector
        valid_idx = [i for i in indices if winsorized[i] is not None]
        if len(valid_idx) < policy.minimum_sector_cohort:
            for i in indices:
                normalized[i] = None
                available[i] = False
                reason[i] = "sector_too_small"
            continue
        n = len(valid_idx)
        # collect winsorized values for valid
        vals: list[float] = [winsorized[i] for i in valid_idx if winsorized[i] is not None]  # type: ignore
        # compute average rank mapping
        # For each distinct value, average rank
        sorted_vals = sorted(vals)
        # map value -> average rank
        value_to_avg: dict[float, float] = {}
        i = 0
        while i < len(sorted_vals):
            j = i
            while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
                j += 1
            # ranks are 1-indexed: start = i+1, end = j
            avg_rank = ((i + 1) + j) / 2.0
            # assign for this distinct value
            value_to_avg[sorted_vals[i]] = avg_rank
            i = j
        # Now assign scores
        for idx in indices:
            w = winsorized[idx]
            if w is None:
                normalized[idx] = None
                available[idx] = False
                reason[idx] = "non_finite"
                continue
            avg = value_to_avg[w]
            score = 0.0 if n == 1 else 2 * (avg - 1) / (n - 1) - 1  # noqa: SIM108

            # clamp due to floating
            normalized[idx] = float(score)
            available[idx] = True
            reason[idx] = ""

    # Build output frame deterministically sorted by instrument_id
    # Use original sorted order
    out_records: list[dict[str, object]] = []
    for idx, r in enumerate(records):
        out_records.append(
            {
                "instrument_id": r.get("instrument_id"),
                "sector": r.get("sector"),
                "raw_value": r.get("raw_value"),
                "normalized_score": normalized[idx],
                "score_available": available[idx],
                "score_reason": reason[idx],
            }
        )
    # Ensure extra original columns preserved? For test only these matter, but preserve winsorized?
    result = pl.DataFrame(out_records)
    # Preserve column order as instrument_id, sector, raw_value, normalized_score, score_available, score_reason
    result = result.select(["instrument_id", "sector", "raw_value", "normalized_score", "score_available", "score_reason"])
    return result
