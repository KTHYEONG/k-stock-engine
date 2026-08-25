"""Hot-loop cycle frame cache scenarios: SCENARIO_HOTLOOP_GOLDEN_PARITY."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from src.stocks.workflows.trading_cycle import build_cycle_frame_cache


def _segment_frame() -> pl.DataFrame:
    rows = []
    for day in range(1, 4):
        session = datetime(2026, 2, day, tzinfo=UTC)
        for idx, instrument in enumerate(("KRX:A", "KRX:B", "KRX:C", "KRX:BAD")):
            rows.append(
                {
                    "instrument_id": instrument,
                    "session": session,
                    "observation_time": session,
                    # KRX:BAD becomes visible only on day 2+ to exercise cutoffs.
                    "available_time": session if instrument != "KRX:BAD" or day >= 2
                    else datetime(2026, 1, 1, tzinfo=UTC),
                    "open": 1000.0 + idx + day,
                    "close": 1010.0 + idx + day,
                    "data_quality_status": (
                        "ineligible" if instrument == "KRX:BAD" else "eligible"
                    ),
                    "is_universe": instrument != "KRX:C",
                    "tradable": True,
                    "pred_score": 0.5 + idx * 0.1,
                    "net_alpha_lower_bound": 0.01 * (day + idx),
                    "sector": f"S{idx % 2}",
                }
            )
    return pl.DataFrame(rows)


def test_cache_build_and_slice_matches_legacy_transforms() -> None:
    """SCENARIO_HOTLOOP_GOLDEN_PARITY (structural core of the golden path)."""
    from src.stocks.workflows.trading_cycle import (
        _adapt_score_column,
        _drop_label_columns,
        _universe_gate,
        research_eligible_frame,
    )

    frame = _segment_frame()
    cache = build_cycle_frame_cache(frame)
    assert cache is not None

    for day in (1, 2, 3):
        decision_time = datetime(2026, 2, day, tzinfo=UTC)
        stop_index = day * 4 - (
            1 if day == 1 else 0
        )  # BAD hidden on day 1 shifts visibility
        visible = frame.slice(0, min(stop_index + 4, len(frame)))
        visible = visible.filter(pl.col("available_time") <= decision_time)

        legacy = _adapt_score_column(
            _drop_label_columns(research_eligible_frame(_universe_gate(visible))),
            "stock_net_alpha_v1",
        )
        cached = cache.cross_section_for(decision_time)

        assert cached is not None
        assert not cached.is_empty()
        assert cached.columns == legacy.columns
        assert sorted(cached["instrument_id"].to_list()) == sorted(
            legacy["instrument_id"].to_list()
        )
        assert (
            cached["pred_score"].to_list() == legacy["pred_score"].to_list()
        )


def test_cache_absent_visibility_returns_empty() -> None:
    cache = build_cycle_frame_cache(_segment_frame())
    early = cache.cross_section_for(datetime(2025, 1, 1, tzinfo=UTC))
    assert (early is not None and early.is_empty()) or early is None


def test_planner_accepts_optional_cache_argument() -> None:
    """The planner signature accepts cycle_cache=None preserving legacy flow."""
    import inspect

    from src.stocks.workflows.trading_cycle import plan_prepared_scored_cycle

    signature = inspect.signature(plan_prepared_scored_cycle)
    assert "cycle_cache" in signature.parameters
    assert signature.parameters["cycle_cache"].default is None
