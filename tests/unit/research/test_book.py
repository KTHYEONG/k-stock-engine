"""Sleeve book and optional trend-leg invariants."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.research.book import (
    BookSpec,
    build_sleeve_targets,
    combine_sleeve_targets,
    mean_sleeve_returns,
    sleeve_capital_krw,
)
from src.research.cube import ResearchCube
from src.research.model import ScoreMatrix
from src.research.panel import FeaturePanel
from src.research.policy import TrendCashPolicy, build_targets, decision_rows

S = 24
N = 6
CAP = 10_000_000


def _sessions(n: int = S) -> tuple[date, ...]:
    start = date(2020, 1, 2)
    return tuple(start + timedelta(days=i) for i in range(n))


def _cube(close: np.ndarray | None = None) -> ResearchCube:
    px = np.full((S, N), 5000.0) if close is None else np.asarray(close, dtype=np.float64)
    return ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=list(_sessions()),
        instrument_ids=[f"KRX:{i:06d}" for i in range(N)],
        arrays={"close": np.ascontiguousarray(px)},
        exit_at=np.full(N, -1, dtype=np.int64),
        exit_halted=np.zeros(N, dtype=np.bool_),
    )


def _panel(dev: float = 0.05, ret: float = 0.05) -> FeaturePanel:
    return FeaturePanel(
        features={
            "dev_ma20": np.full((S, N), dev, dtype=np.float32),
            "ret_21": np.full((S, N), ret, dtype=np.float32),
        },
        labels={},
        last_row=S - 1,
    )


def _scores(mat: np.ndarray) -> ScoreMatrix:
    arr = np.ascontiguousarray(np.asarray(mat, dtype=np.float64), dtype=np.float32)
    return ScoreMatrix(scores=arr, test_years=(2020,), config_hash="test", last_row=S - 1)


def test_disabled_dev_leg_ignores_nan() -> None:
    mat = np.full((S, N), 0.0)
    mat[5] = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]
    panel = _panel()
    panel.features["dev_ma20"][5, 0] = np.float32(np.nan)
    uni = np.ones((S, N), dtype=bool)
    off = TrendCashPolicy(n=2, trend_min_dev_ma20=None, trend_min_ret21=0.0)
    assert np.asarray(build_targets(off, _cube(), panel, _scores(mat), uni, rows=[5], capital_krw=CAP, cash_buffer=0.0)[5]) == pytest.approx([0.5, 0.5, 0, 0, 0, 0])
    on = TrendCashPolicy(n=2, trend_min_dev_ma20=0.0, trend_min_ret21=0.0)
    assert build_targets(on, _cube(), panel, _scores(mat), uni, rows=[5], capital_krw=CAP, cash_buffer=0.0)[5][0] == 0.0


def test_both_legs_disabled_keep_pure_top_n() -> None:
    mat = np.full((S, N), 0.0)
    mat[5] = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1]
    panel = _panel(dev=float("nan"), ret=float("nan"))
    policy = TrendCashPolicy(n=3, trend_min_dev_ma20=None, trend_min_ret21=None)
    out = build_targets(policy, _cube(), panel, _scores(mat), np.ones((S, N), dtype=bool), rows=[5], capital_krw=CAP, cash_buffer=0.0)[5]
    assert np.asarray(out) == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0, 0, 0])


def test_ret_only_leg() -> None:
    mat = np.full((S, N), 0.0)
    mat[5] = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]
    panel = _panel()
    panel.features["ret_21"][5, 0] = np.float32(-0.5)
    policy = TrendCashPolicy(n=2, trend_min_dev_ma20=None, trend_min_ret21=0.0)
    out = build_targets(policy, _cube(), panel, _scores(mat), np.ones((S, N), dtype=bool), rows=[5], capital_krw=CAP, cash_buffer=0.0)[5]
    assert np.asarray(out) == pytest.approx([0.0, 0.5, 0, 0, 0, 0])


def test_dev_only_leg() -> None:
    mat = np.full((S, N), 0.0)
    mat[5] = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]
    panel = _panel()
    panel.features["dev_ma20"][5, 0] = np.float32(-0.5)
    policy = TrendCashPolicy(n=2, trend_min_dev_ma20=0.0, trend_min_ret21=None)
    out = build_targets(policy, _cube(), panel, _scores(mat), np.ones((S, N), dtype=bool), rows=[5], capital_krw=CAP, cash_buffer=0.0)[5]
    assert np.asarray(out) == pytest.approx([0.0, 0.5, 0, 0, 0, 0])


def test_enabled_feature_shape_mismatch() -> None:
    mat = np.full((S, N), 0.1)
    bad = FeaturePanel(
        features={
            "dev_ma20": np.full((S, N), 0.05, dtype=np.float32),
            "ret_21": np.full((S, 2), 0.05, dtype=np.float32),
        },
        labels={},
        last_row=S - 1,
    )
    with pytest.raises(ValueError, match="match the scores shape"):
        build_targets(
            TrendCashPolicy(n=2, trend_min_dev_ma20=None, trend_min_ret21=0.0),
            _cube(), bad, _scores(mat), np.ones((S, N), dtype=bool),
            rows=[5], capital_krw=CAP, cash_buffer=0.0,
        )
    bad2 = FeaturePanel(
        features={
            "dev_ma20": np.full((S, 2), 0.05, dtype=np.float32),
            "ret_21": np.full((S, N), 0.05, dtype=np.float32),
        },
        labels={},
        last_row=S - 1,
    )
    with pytest.raises(ValueError, match="match the scores shape"):
        build_targets(
            TrendCashPolicy(n=2, trend_min_dev_ma20=0.0, trend_min_ret21=None),
            _cube(), bad2, _scores(mat), np.ones((S, N), dtype=bool),
            rows=[5], capital_krw=CAP, cash_buffer=0.0,
        )


def test_policy_threshold_validation() -> None:
    with pytest.raises(ValueError, match="trend threshold"):
        TrendCashPolicy(trend_min_ret21=float("inf"))
    assert TrendCashPolicy(trend_min_dev_ma20=None).trend_min_dev_ma20 is None
    assert TrendCashPolicy(trend_min_ret21=None).spec_hash != TrendCashPolicy(trend_min_ret21=0.0).spec_hash


def test_sleeve_capital_arithmetic() -> None:
    assert sleeve_capital_krw(100_000_000, BookSpec(sleeves=5, stock_capital_fraction=0.75)) == 15_000_000
    assert BookSpec(sleeves=5, stock_capital_fraction=0.75).canonical_json() != BookSpec(
        sleeves=5, stock_capital_fraction=0.5
    ).canonical_json()
    with pytest.raises(ValueError, match="total_capital"):
        sleeve_capital_krw(0, BookSpec(sleeves=5, stock_capital_fraction=0.75))
    with pytest.raises(ValueError, match="below 1 KRW"):
        sleeve_capital_krw(1, BookSpec(sleeves=5, stock_capital_fraction=0.01))
    with pytest.raises(ValueError, match="sleeves"):
        BookSpec(sleeves=0, stock_capital_fraction=0.75)
    with pytest.raises(ValueError, match="stock_capital_fraction"):
        BookSpec(sleeves=5, stock_capital_fraction=0.0)


def test_sleeves_tile_and_mismatch() -> None:
    sess = _sessions(30)
    union = set()
    for phase in range(5):
        union.update(decision_rows(sess, lo=5, hi=25, every=5, phase=phase))
    assert union == set(range(4, 25))
    policy = TrendCashPolicy(n=2, rebalance_every_sessions=5, trend_min_dev_ma20=None, trend_min_ret21=None)
    with pytest.raises(ValueError, match="sleeves"):
        build_sleeve_targets(
            policy, _cube(), _panel(), _scores(np.full((S, N), 0.1)), np.ones((S, N), dtype=bool),
            sessions=_sessions(), sleeves=4, lo=5, hi=20, sleeve_capital_krw=CAP, cash_buffer=0.0,
        )
    with pytest.raises(ValueError, match="sleeves"):
        build_sleeve_targets(
            policy, _cube(), _panel(), _scores(np.full((S, N), 0.1)), np.ones((S, N), dtype=bool),
            sessions=_sessions(), sleeves=0, lo=5, hi=20, sleeve_capital_krw=CAP, cash_buffer=0.0,
        )


def test_build_sleeve_targets_use_sleeve_capital() -> None:
    policy = TrendCashPolicy(n=2, rebalance_every_sessions=5, trend_min_dev_ma20=None, trend_min_ret21=None)
    mat = np.full((S, N), 0.1)
    mat[:, 0] = 0.9
    sleeves = build_sleeve_targets(
        policy, _cube(), _panel(), _scores(mat), np.ones((S, N), dtype=bool),
        sessions=_sessions(), sleeves=5, lo=5, hi=20, sleeve_capital_krw=100_000, cash_buffer=0.0,
    )
    assert len(sleeves) == 5
    close = np.full((S, N), 5000.0)
    close[:, 0] = 60_000.0
    poor = build_sleeve_targets(
        TrendCashPolicy(n=1, rebalance_every_sessions=5, trend_min_dev_ma20=None, trend_min_ret21=None),
        _cube(close), _panel(), _scores(mat), np.ones((S, N), dtype=bool),
        sessions=_sessions(), sleeves=5, lo=5, hi=20, sleeve_capital_krw=100_000, cash_buffer=0.0,
    )
    assert all(0 not in np.flatnonzero(w > 0) for m in poor for w in m.values() if w.sum() > 0)


def test_combine_forward_filled_mean() -> None:
    thirds = combine_sleeve_targets(
        ({0: np.array([0.5, 0.5]), 2: np.array([1.0, 0.0])}, {1: np.array([0.0, 1.0])}, {}), lo=1, hi=4,
    )
    assert np.asarray(thirds[0]) == pytest.approx([0.5 / 3, 0.5 / 3])
    assert np.asarray(thirds[1]) == pytest.approx([0.5 / 3, 1.5 / 3])
    assert np.asarray(thirds[2]) == pytest.approx([1.0 / 3, 1.0 / 3])
    assert all(float(v.sum()) <= 1 + 1e-12 for v in thirds.values())
    single = combine_sleeve_targets(({2: np.array([1.0, 0.0])},), lo=1, hi=5)
    assert np.asarray(single[0]) == pytest.approx([0.0, 0.0])
    assert np.asarray(single[4]) == pytest.approx([1.0, 0.0])
    with pytest.raises(ValueError, match="non-empty"):
        combine_sleeve_targets([], lo=1, hi=2)
    with pytest.raises(ValueError, match="lo must"):
        combine_sleeve_targets(({0: np.array([1.0])},), lo=0, hi=2)
    with pytest.raises(ValueError, match="hi must"):
        combine_sleeve_targets(({0: np.array([1.0])},), lo=3, hi=2)
    with pytest.raises(ValueError, match="1-D"):
        combine_sleeve_targets(({0: np.array([[1.0]])},), lo=1, hi=2)
    with pytest.raises(ValueError, match="must match"):
        combine_sleeve_targets(({0: np.array([1.0, 0.0])}, {0: np.array([1.0])}), lo=1, hi=2)
    with pytest.raises(ValueError, match="no weights"):
        combine_sleeve_targets(({},), lo=1, hi=2)
    with pytest.raises(ValueError, match="must be an int"):
        combine_sleeve_targets(({True: np.array([1.0])},), lo=1, hi=2)


def test_mean_sleeve_returns() -> None:
    a = np.array([0.01, -0.02, 0.03])
    b = np.array([0.05, 0.0, -0.01])
    assert mean_sleeve_returns([a, b]) == pytest.approx((np.expm1(a) + np.expm1(b)) / 2)
    assert mean_sleeve_returns([a, a]) == pytest.approx(np.expm1(a))
    with pytest.raises(ValueError, match="equal lengths"):
        mean_sleeve_returns([a, np.array([0.1])])
    with pytest.raises(ValueError, match="finite"):
        mean_sleeve_returns([np.array([np.nan])])
    with pytest.raises(ValueError, match="non-empty"):
        mean_sleeve_returns([])
    with pytest.raises(ValueError, match="1-D"):
        mean_sleeve_returns([np.array([[0.01]])])
