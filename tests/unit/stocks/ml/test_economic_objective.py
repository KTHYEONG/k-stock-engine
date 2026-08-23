"""Economic tail-objective contract: exact-K labels and tail-capture arithmetic."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.stocks.ml.economic_objective import (
    InvalidOofEconomicUtilityError,
    build_tail_relevance,
    measure_tail_capture,
)

ECONOMIC_FAMILY_01_EXACT_TAIL_LABEL = "ECONOMIC_FAMILY_01_EXACT_TAIL_LABEL"
ECONOMIC_FAMILY_02_TAIL_CAPTURE_NOT_RANK_IC = "ECONOMIC_FAMILY_02_TAIL_CAPTURE_NOT_RANK_IC"

_ID = "instrument_id"
_SESSION = "session"


def _labels(rows: list[tuple[str, int, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            _ID: [r[0] for r in rows],
            _SESSION: [r[1] for r in rows],
            "risk_residual": [r[2] for r in rows],
            "reference_cost": [r[3] for r in rows],
        }
    )


def test_economic_family_01_exact_tail_label() -> None:
    """ECONOMIC_FAMILY_01_EXACT_TAIL_LABEL.

    Exactly top_k rows carry relevance=1 per session; a tied utility resolves
    by ascending instrument_id; non-finite utility fails closed.
    """
    frame = _labels(
        [
            ("A", 1, 0.10, 0.0),
            ("B", 1, 0.30, 0.0),
            ("C", 1, 0.20, 0.0),
            ("D", 1, 0.00, 0.0),
            ("E", 1, -0.05, 0.0),
            ("Z", 2, 0.50, 0.0),
            ("C", 2, 0.30, 0.0),
            ("B", 2, 0.30, 0.0),
            ("A", 2, 0.10, 0.0),
            ("D", 2, -0.10, 0.0),
        ]
    )
    result = build_tail_relevance(frame, top_k=2)

    assert result.height == frame.height
    selected = result.filter(pl.col("relevance") == 1)
    assert selected.height == 4
    session_one = set(
        selected.filter(pl.col(_SESSION) == 1)[_ID].to_list()
    )
    assert session_one == {"B", "C"}
    # Session 2 ties B/C at 0.30 straddling the K boundary; ascending
    # instrument_id selects B, never C.
    session_two = set(
        selected.filter(pl.col(_SESSION) == 2)[_ID].to_list()
    )
    assert session_two == {"Z", "B"}
    assert result["relevance"].dtype == pl.Int8


def test_economic_family_01_exact_tail_label_fail_closed() -> None:
    base = [
        ("A", 1, 0.10, 0.0),
        ("B", 1, 0.30, 0.0),
        ("C", 1, 0.20, 0.0),
        ("D", 1, 0.00, 0.0),
        ("E", 1, -0.05, 0.0),
    ]
    with pytest.raises(ValueError, match="finite"):
        build_tail_relevance(
            _labels(base[:2])
            .vstack(
                pl.DataFrame(
                    {
                        _ID: ["C"],
                        _SESSION: [1],
                        "risk_residual": [float("inf")],
                        "reference_cost": [0.0],
                    }
                )
            )
            .extend(_labels(base[3:])),
            top_k=2,
        )
    with pytest.raises(ValueError, match="undersized"):
        build_tail_relevance(
            _labels([("A", 1, 0.10, 0.0), ("B", 1, 0.30, 0.0)]),
            top_k=3,
        )
    with pytest.raises(ValueError, match="null"):
        build_tail_relevance(
            pl.DataFrame(
                {
                    _ID: ["A", "B", "C"],
                    _SESSION: [1, 1, 1],
                    "risk_residual": [0.1, None, 0.3],
                    "reference_cost": [0.0, 0.0, 0.0],
                }
            ),
            top_k=2,
        )


def _study_frames(
    utilities_by_name: dict[str, float],
    scores_by_name: dict[str, float],
    sessions: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ids = list(utilities_by_name)
    label_rows = []
    scored_rows = []
    for session in range(1, sessions + 1):
        for name in ids:
            label_rows.append((name, session, utilities_by_name[name], 0.0))
            scored_rows.append(
                {
                    _ID: name,
                    _SESSION: session,
                    "predicted_net_alpha": scores_by_name[name],
                }
            )
    return _labels(label_rows), pl.DataFrame(scored_rows)


def test_economic_family_02_tail_capture_not_rank_ic() -> None:
    """ECONOMIC_FAMILY_02_TAIL_CAPTURE_NOT_RANK_IC.

    A score vector with strongly positive global rank correlation whose
    top-K path still lands below the universe mean is ineligible
    (tail_excess_lower_bound <= 0); a strictly superior top-K path passes.
    """
    # Utilities: two stars, two middling names, four laggards. The bad score
    # vector promotes the middling pair above the stars yet keeps near-sorted
    # order everywhere else, so global rank correlation stays positive.
    utilities = {"A": 0.90, "B": 0.80, "C": 0.05, "D": 0.04,
                 "E": -0.03, "F": -0.03, "G": -0.03, "H": -0.03}
    bad_scores = {"C": 9.0, "D": 8.0, "A": 7.0, "B": 6.0,
                  "E": -0.3, "F": -0.3, "G": -0.3, "H": -0.3}
    good_scores = {"A": 9.0, "B": 8.0, "C": 7.0, "D": 6.0,
                   "E": -0.3, "F": -0.3, "G": -0.3, "H": -0.3}
    labels, bad_scored = _study_frames(utilities, bad_scores, sessions=12)
    _, good_scored = _study_frames(utilities, good_scores, sessions=12)

    ineligible = measure_tail_capture(
        bad_scored,
        labels,
        top_k=2,
        bootstrap_alpha=0.05,
        bootstrap_resamples=400,
        seed=42,
    )
    eligible = measure_tail_capture(
        good_scored,
        labels,
        top_k=2,
        bootstrap_alpha=0.05,
        bootstrap_resamples=400,
        seed=42,
    )

    assert ineligible.model_excess_utility <= 0.0
    assert ineligible.tail_excess_lower_bound <= 0.0
    assert ineligible.tail_gate_ok is False
    assert ineligible.oracle_capacity_ok is True
    assert ineligible.oracle_excess_utility > 0.0
    assert eligible.tail_excess_lower_bound > 0.0
    assert eligible.tail_gate_ok is True
    assert eligible.tail_capture_ratio is not None
    assert eligible.tail_capture_ratio > 0.0
    assert eligible.positive_session_fraction == pytest.approx(1.0)


def test_economic_family_02_tail_capture_segment_clusters() -> None:
    """Bootstrap clusters by oof_segment_id when the scored frame carries it."""
    utilities = {"A": 0.90, "B": 0.80, "C": 0.05, "D": 0.04}
    scores = {"A": 9.0, "B": 8.0, "C": 7.0, "D": 6.0}
    labels, scored = _study_frames(utilities, scores, sessions=6)
    scored = scored.with_columns(
        ((pl.col(_SESSION) - 1) // 3).cast(pl.Int64).alias("oof_segment_id")
    )

    evidence = measure_tail_capture(
        scored,
        labels,
        top_k=2,
        bootstrap_alpha=0.05,
        bootstrap_resamples=200,
        seed=7,
    )

    assert len(evidence.segments) == 2
    assert {segment.segment_id for segment in evidence.segments} == {0, 1}
    assert evidence.session_count == 6


def test_ECONOMIC_UTILITY_04_residual_not_log_growth() -> None:
    """ECONOMIC_UTILITY_04_RESIDUAL_NOT_LOG_GROWTH.

    Finite residual utility below -1 stays rankable arithmetic utility: no
    log-domain operation exists, model/oracle excess evidence is finite and
    arithmetically exact, and the tail gate still requires a strictly
    positive bootstrapped lower excess utility.
    """
    # Four laggards sit at or below the invalid log domain (-1), including a
    # split-distorted -3.93; they must never break ranking or evidence.
    utilities = {"A": 0.90, "B": 0.80, "C": 0.05, "D": -3.93,
                 "E": -2.50, "F": -1.50, "G": -1.20, "H": -0.95}
    scores = {"A": 9.0, "B": 8.0, "C": 7.0, "D": 6.0,
              "E": 5.0, "F": 4.0, "G": 3.0, "H": 2.0}
    labels, scored = _study_frames(utilities, scores, sessions=12)

    relevance = build_tail_relevance(labels, top_k=2)
    selected = relevance.filter(pl.col("relevance") == 1)
    assert set(selected[_ID].unique().to_list()) == {"A", "B"}

    evidence = measure_tail_capture(
        scored,
        labels,
        top_k=2,
        bootstrap_alpha=0.05,
        bootstrap_resamples=400,
        seed=11,
    )
    universe_mean = float(np.mean(list(utilities.values())))
    oracle_mean = float(np.mean(sorted(utilities.values(), reverse=True)[:2]))
    assert np.isfinite(evidence.model_excess_utility)
    assert np.isfinite(evidence.oracle_excess_utility)
    assert np.isfinite(evidence.tail_excess_lower_bound)
    assert evidence.oracle_excess_utility == pytest.approx(oracle_mean - universe_mean)
    # The score order coincides with the oracle here, so excesses coincide.
    assert evidence.model_excess_utility == pytest.approx(oracle_mean - universe_mean)
    assert evidence.tail_gate_ok is True


def test_ECONOMIC_UTILITY_04_invalid_oof_join_fails_closed() -> None:
    """Null or non-finite joined utility raises the typed integrity error."""
    labels, scored = _study_frames({"A": 0.10, "B": -0.20}, {"A": 1.0, "B": 0.5}, sessions=2)
    with pytest.raises(InvalidOofEconomicUtilityError, match="null"):
        measure_tail_capture(
            scored,
            labels.with_columns(
                pl.when(pl.col(_ID) == "A")
                .then(None)
                .otherwise(pl.col("risk_residual"))
                .alias("risk_residual")
            ),
            top_k=1,
            bootstrap_alpha=0.05,
            bootstrap_resamples=10,
            seed=1,
        )
    with pytest.raises(InvalidOofEconomicUtilityError, match="finite"):
        measure_tail_capture(
            scored,
            labels.with_columns(
                pl.when(pl.col(_ID) == "B")
                .then(float("inf"))
                .otherwise(pl.col("risk_residual"))
                .alias("risk_residual")
            ),
            top_k=1,
            bootstrap_alpha=0.05,
            bootstrap_resamples=10,
            seed=1,
        )
