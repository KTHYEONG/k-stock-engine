"""Trend-cash selection policy invariants."""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

from src.research.cube import ResearchCube
from src.research.model import ScoreMatrix
from src.research.panel import FeaturePanel
from src.research.policy import TrendCashPolicy, UniverseRule, build_targets, decision_rows, universe_mask

S = 24
N = 6
CAP = 10_000_000


def _sessions(n: int = S) -> tuple[date, ...]:
    start = date(2020, 1, 2)
    return tuple(start + timedelta(days=i) for i in range(n))


def _cube(close: np.ndarray | None = None) -> ResearchCube:
    shape = (S, N)
    px = np.full(shape, 5000.0) if close is None else np.asarray(close, dtype=np.float64)
    return ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=list(_sessions()),
        instrument_ids=[f"KRX:{i:06d}" for i in range(N)],
        arrays={"close": np.ascontiguousarray(px)},
        exit_at=np.full(N, -1, dtype=np.int64),
        exit_halted=np.zeros(N, dtype=np.bool_),
    )


def _panel(dev: float = 0.05, ret: float = 0.05) -> FeaturePanel:
    feats = {
        "dev_ma20": np.full((S, N), dev, dtype=np.float32),
        "ret_21": np.full((S, N), ret, dtype=np.float32),
    }
    return FeaturePanel(features=feats, labels={}, last_row=S - 1)


def _scores(matrix: np.ndarray) -> ScoreMatrix:
    arr = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64), dtype=np.float32)
    return ScoreMatrix(scores=arr, test_years=(2020,), config_hash="test", last_row=S - 1)


def _policy(**overrides: object) -> TrendCashPolicy:
    kwargs: dict[str, object] = {
        "family": "ml_trend_cash",
        "n": 3,
        "keep_rank_multiple": 3.0,
        "rebalance_every_sessions": 5,
        "trend_min_dev_ma20": 0.0,
        "trend_min_ret21": 0.0,
    }
    kwargs.update(overrides)
    return TrendCashPolicy(**kwargs)  # type: ignore[arg-type]


def _full_scores(value: float = 0.0) -> np.ndarray:
    return np.full((S, N), value, dtype=np.float64)


def test_top_n_by_score_with_stable_ties() -> None:
    mat = _full_scores()
    mat[5] = [0.5, 0.9, 0.9, 0.1, 0.2, 0.3]
    out = build_targets(
        _policy(n=3), _cube(), _panel(), _scores(mat), np.ones((S, N), dtype=bool),
        rows=[5], capital_krw=CAP, cash_buffer=0.0,
    )
    assert list(out) == [5]
    assert np.asarray(out[5]) == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0])


def test_failed_trend_moves_weight_to_cash() -> None:
    mat = _full_scores()
    mat[5] = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
    panel = _panel()
    panel.features["ret_21"][5, 0] = np.float32(-0.01)
    out = build_targets(
        _policy(n=3), _cube(), panel, _scores(mat), np.ones((S, N), dtype=bool),
        rows=[5], capital_krw=CAP, cash_buffer=0.0,
    )
    assert np.asarray(out[5]) == pytest.approx([0.0, 1 / 3, 1 / 3, 0.0, 0.0, 0.0])
    assert float(np.asarray(out[5]).sum()) == pytest.approx(2 / 3)


def test_nan_trend_input_fails_the_rule() -> None:
    mat = _full_scores()
    mat[5] = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]
    panel = _panel()
    panel.features["dev_ma20"][5, 0] = np.float32(np.nan)
    out = build_targets(
        _policy(n=2), _cube(), panel, _scores(mat), np.ones((S, N), dtype=bool),
        rows=[5], capital_krw=CAP, cash_buffer=0.0,
    )
    assert np.asarray(out[5]) == pytest.approx([0.0, 1 / 2, 0.0, 0.0, 0.0, 0.0])


def test_incumbency_uses_pre_trend_selection() -> None:
    mat = _full_scores()
    mat[4] = [0.9, 0.8, 0.7, 0.6, 0.1, 0.1]
    mat[5] = [0.5, 0.85, 0.9, 0.1, 0.1, 0.1]
    panel = _panel()
    panel.features["ret_21"][4, 0] = np.float32(-0.01)
    out = build_targets(
        _policy(n=2), _cube(), panel, _scores(mat), np.ones((S, N), dtype=bool),
        rows=[4, 5], capital_krw=CAP, cash_buffer=0.0,
    )
    assert np.asarray(out[4]) == pytest.approx([0.0, 1 / 2, 0.0, 0.0, 0.0, 0.0])
    assert np.asarray(out[5]) == pytest.approx([1 / 2, 1 / 2, 0.0, 0.0, 0.0, 0.0])


def test_keep_rank_multiple_boundary() -> None:
    mat = _full_scores()
    mat[4] = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    mat[5] = [0.1, 0.9, 0.8, 0.7, 0.6, 0.5]
    uni = np.ones((S, N), dtype=bool)
    out = build_targets(
        _policy(n=1), _cube(), _panel(), _scores(mat), uni,
        rows=[4, 5], capital_krw=CAP, cash_buffer=0.0,
    )
    assert np.asarray(out[5]) == pytest.approx([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    mat2 = _full_scores()
    mat2[4] = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    mat2[5] = [0.55, 0.9, 0.8, 0.1, 0.1, 0.1]
    out2 = build_targets(
        _policy(n=1), _cube(), _panel(), _scores(mat2), uni,
        rows=[4, 5], capital_krw=CAP, cash_buffer=0.0,
    )
    assert np.asarray(out2[5]) == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_thin_universe_resets() -> None:
    mat = _full_scores()
    mat[4] = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]
    mat[5, 0] = 0.9
    mat[5, 1:] = np.nan
    mat[6] = [0.5, 0.9, 0.8, 0.1, 0.1, 0.1]
    out = build_targets(
        _policy(n=2), _cube(), _panel(), _scores(mat), np.ones((S, N), dtype=bool),
        rows=[4, 5, 6], capital_krw=CAP, cash_buffer=0.0,
    )
    assert bool(np.all(np.asarray(out[5]) == 0.0))
    assert np.asarray(out[6]) == pytest.approx([0.0, 1 / 2, 1 / 2, 0.0, 0.0, 0.0])


def test_unit_price_cap() -> None:
    mat = _full_scores()
    mat[5] = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1]
    close = np.full((S, N), 5000.0)
    close[:, 0] = 2_000_000.0
    out = build_targets(
        _policy(n=2), _cube(close), _panel(), _scores(mat), np.ones((S, N), dtype=bool),
        rows=[5], capital_krw=CAP, cash_buffer=0.0,
    )
    assert np.asarray(out[5]) == pytest.approx([0.0, 1 / 2, 1 / 2, 0.0, 0.0, 0.0])
    admitted = build_targets(
        _policy(n=2, min_units_per_slot=0), _cube(close), _panel(), _scores(mat),
        np.ones((S, N), dtype=bool), rows=[5], capital_krw=CAP, cash_buffer=0.0,
    )
    assert np.asarray(admitted[5]) == pytest.approx([1 / 2, 1 / 2, 0.0, 0.0, 0.0, 0.0])


def test_decision_rows_by_phase() -> None:
    sess = _sessions(30)
    assert decision_rows(sess, lo=5, hi=25, every=5, phase=2) == (6, 11, 16, 21)
    assert decision_rows(sess, lo=0, hi=10, every=5, phase=0) == (4, 9)
    with pytest.raises(ValueError, match="phase"):
        decision_rows(sess, lo=5, hi=25, every=5, phase=5)
    with pytest.raises(ValueError, match="every"):
        decision_rows(sess, lo=5, hi=25, every=0, phase=0)
    with pytest.raises(ValueError, match="window"):
        decision_rows(sess, lo=20, hi=10, every=5, phase=0)
    with pytest.raises(ValueError, match="window"):
        decision_rows(sess, lo=0, hi=31, every=5, phase=0)


def test_spec_identity() -> None:
    base = _policy(n=20)
    other_n = _policy(n=10)
    assert base.spec_hash != other_n.spec_hash
    assert base.spec_hash == _policy(n=20).spec_hash
    assert "capital" not in base.canonical_json()


def _wide_row(n: int, n_pass: int, *, row: int = 5) -> tuple[ResearchCube, FeaturePanel, ScoreMatrix, np.ndarray]:
    """Cube/panel/scores of ``n`` names where the first ``n_pass`` pass the trend rule, and a full universe."""
    shape = (S, n)
    cube = ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=list(_sessions()),
        instrument_ids=[f"KRX:{i:06d}" for i in range(n)],
        arrays={"close": np.full(shape, 5000.0)},
        exit_at=np.full(n, -1, dtype=np.int64),
        exit_halted=np.zeros(n, dtype=np.bool_),
    )
    ret = np.full(shape, 0.05, dtype=np.float32)
    ret[row, n_pass:] = np.float32(-0.01)
    panel = FeaturePanel(
        features={"dev_ma20": np.full(shape, 0.05, dtype=np.float32), "ret_21": ret},
        labels={},
        last_row=S - 1,
    )
    scores = np.full(shape, 0.0, dtype=np.float64)
    scores[row] = np.arange(n, 0, -1, dtype=np.float64)
    return cube, panel, ScoreMatrix(
        scores=np.ascontiguousarray(scores, dtype=np.float32),
        test_years=(2020,),
        config_hash="test",
        last_row=S - 1,
    ), np.ones(shape, dtype=bool)


def _redist_weights(n: int, n_pass: int, cap: float | None) -> np.ndarray:
    cube, panel, scores, uni = _wide_row(n, n_pass)
    out = build_targets(
        _policy(n=n, redistribute_cap_multiple=cap), cube, panel, scores, uni,
        rows=[5], capital_krw=CAP, cash_buffer=0.0,
    )
    return np.asarray(out[5])


def test_cap_one_reproduces_legacy_weights() -> None:
    for n_pass in (0, 1, 8, 19, 20):
        legacy = _redist_weights(20, n_pass, None)
        capped = _redist_weights(20, n_pass, 1.0)
        assert np.array_equal(capped, legacy)


def test_failed_slots_spread_under_the_cap() -> None:
    w = _redist_weights(20, 8, 2.0)
    assert np.asarray(w[:8]) == pytest.approx([0.1] * 8)
    assert np.asarray(w[8:]) == pytest.approx([0.0] * 12)
    assert float(w.sum()) == pytest.approx(0.8)


def test_enough_passing_names_fill_the_book() -> None:
    w = _redist_weights(20, 15, 2.0)
    assert np.asarray(w[:15]) == pytest.approx([1 / 15] * 15)
    assert np.asarray(w[15:]) == pytest.approx([0.0] * 5)
    assert float(w.sum()) == pytest.approx(1.0)


def test_no_passing_names_stays_cash() -> None:
    assert np.asarray(_redist_weights(20, 0, 2.0)) == pytest.approx([0.0] * 20)


def test_uncapped_split_divides_the_whole_book() -> None:
    w = _redist_weights(20, 3, 100.0)
    assert np.asarray(w[:3]) == pytest.approx([1 / 3] * 3)
    assert float(w.sum()) == pytest.approx(1.0)


def test_redistribution_ignores_disabled_trend_legs() -> None:
    cube, panel, scores, uni = _wide_row(20, 0)
    plain = build_targets(
        _policy(n=20, trend_min_ret21=None, redistribute_cap_multiple=None),
        cube, panel, scores, uni, rows=[5], capital_krw=CAP, cash_buffer=0.0,
    )
    capped = build_targets(
        _policy(n=20, trend_min_ret21=None, redistribute_cap_multiple=2.0),
        cube, panel, scores, uni, rows=[5], capital_krw=CAP, cash_buffer=0.0,
    )
    assert np.array_equal(np.asarray(capped[5]), np.asarray(plain[5]))
    assert float(np.asarray(capped[5]).sum()) == pytest.approx(1.0)


def test_incumbency_survives_redistribution() -> None:
    mat = _full_scores()
    mat[4] = [0.9, 0.8, 0.7, 0.6, np.nan, np.nan]
    mat[5] = [0.95, 0.05, 0.94, 0.93, np.nan, np.nan]
    panel = _panel()
    panel.features["ret_21"][4:, 1] = np.float32(-0.01)
    uni = np.ones((S, N), dtype=bool)
    kwargs: dict[str, object] = {"rows": [4, 5], "capital_krw": CAP, "cash_buffer": 0.0}
    base = build_targets(
        _policy(n=2, redistribute_cap_multiple=None), _cube(), panel, _scores(mat), uni, **kwargs
    )
    spread = build_targets(
        _policy(n=2, redistribute_cap_multiple=2.0), _cube(), panel, _scores(mat), uni, **kwargs
    )
    # Only the weight differs; the selection set (name 1 kept at 0, name 2 crowded out) is identical.
    assert np.asarray(base[5]) == pytest.approx([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.asarray(spread[5]) == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.asarray(spread[4]) == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_redistribute_cap_validation() -> None:
    assert _policy(redistribute_cap_multiple=None).redistribute_cap_multiple is None
    assert _policy(redistribute_cap_multiple=1).redistribute_cap_multiple == 1.0
    with pytest.raises(ValueError, match="redistribute_cap_multiple"):
        TrendCashPolicy(redistribute_cap_multiple=0.5)
    with pytest.raises(ValueError, match="redistribute_cap_multiple"):
        TrendCashPolicy(redistribute_cap_multiple=float("nan"))
    with pytest.raises(ValueError, match="redistribute_cap_multiple"):
        TrendCashPolicy(redistribute_cap_multiple=float("inf"))


def test_redistribute_cap_is_part_of_the_identity() -> None:
    base = _policy(n=20)
    assert base.spec_hash != _policy(n=20, redistribute_cap_multiple=2.0).spec_hash
    assert json.loads(base.canonical_json())["redistribute_cap_multiple"] is None


def test_rule_and_policy_validation() -> None:
    with pytest.raises(ValueError, match="non-negative int"):
        UniverseRule(min_adtv20_krw=-1, min_price_krw=1000)
    with pytest.raises(ValueError, match="family"):
        TrendCashPolicy(family="  ")
    with pytest.raises(ValueError, match="n must"):
        TrendCashPolicy(n=0)
    with pytest.raises(ValueError, match="keep_rank_multiple"):
        TrendCashPolicy(keep_rank_multiple=-1.0)
    with pytest.raises(ValueError, match="rebalance_every_sessions"):
        TrendCashPolicy(rebalance_every_sessions=0)
    with pytest.raises(ValueError, match="min_units_per_slot"):
        TrendCashPolicy(min_units_per_slot=-1)
    with pytest.raises(ValueError, match="trend threshold"):
        TrendCashPolicy(trend_min_ret21=float("nan"))


def test_build_targets_guards() -> None:
    uni = np.ones((S, N), dtype=bool)
    policy = _policy(n=2)
    panel = _panel()
    mat = _full_scores(0.5)
    scores = _scores(mat)
    cube = _cube()
    with pytest.raises(ValueError, match="no score row"):
        build_targets(policy, cube, panel, scores, uni, rows=[S], capital_krw=CAP, cash_buffer=0.0)
    with pytest.raises(ValueError, match="no score row"):
        build_targets(policy, cube, panel, scores, uni, rows=[-1], capital_krw=CAP, cash_buffer=0.0)
    with pytest.raises(ValueError, match="capital_krw"):
        build_targets(policy, cube, panel, scores, uni, rows=[5], capital_krw=0, cash_buffer=0.0)
    with pytest.raises(ValueError, match="cash_buffer"):
        build_targets(policy, cube, panel, scores, uni, rows=[5], capital_krw=CAP, cash_buffer=1.0)
    with pytest.raises(ValueError, match="match the scores shape"):
        build_targets(
            policy, cube, panel, scores, np.ones((S, 2), dtype=bool),
            rows=[5], capital_krw=CAP, cash_buffer=0.0,
        )


def test_universe_mask_contract() -> None:
    shape = (5, 5)
    good = np.ones(shape)
    present = good.copy()
    present[:, 1] = 0.0
    blocked = np.zeros(shape)
    blocked[:, 2] = 1.0
    adtv = np.full(shape, 1e9)
    adtv[:, 3] = 1e6
    close = np.full(shape, 5000.0)
    close[:, 4] = 500.0
    vol = np.full(shape, 0.02)
    vol[:, 3] = np.nan
    cube = ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=list(_sessions(5)),
        instrument_ids=[f"KRX:{i:06d}" for i in range(5)],
        arrays={
            "present": present.astype(bool),
            "eligible": np.ones(shape, dtype=bool),
            "entry_blocked": blocked.astype(bool),
            "volume": np.full(shape, 1e5),
            "adtv20": adtv,
            "close": close,
            "ret_vol60": vol,
        },
        exit_at=np.full(5, -1, dtype=np.int64),
        exit_halted=np.zeros(5, dtype=np.bool_),
    )
    mask = universe_mask(cube, UniverseRule(min_adtv20_krw=500_000_000, min_price_krw=1000))
    assert bool(np.all(mask[:, 0]))
    assert bool(np.all(~mask[:, 1:]))


def test_row_uses_only_its_own_inputs() -> None:
    mat = _full_scores()
    mat[5] = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]
    mat[6] = [0.1, 0.1, 0.1, 0.1, 0.9, 0.8]
    uni = np.ones((S, N), dtype=bool)
    policy = _policy(n=2)
    panel = _panel()
    first = build_targets(policy, _cube(), panel, _scores(mat), uni, rows=[5], capital_krw=CAP, cash_buffer=0.0)
    other_mat = mat.copy()
    other_mat[6] = [0.8, 0.9, 0.1, 0.1, 0.1, 0.1]
    other_panel = _panel()
    other_panel.features["dev_ma20"][6, :] = np.float32(-0.5)
    second = build_targets(
        policy, _cube(), other_panel, _scores(other_mat), uni, rows=[5], capital_krw=CAP, cash_buffer=0.0
    )
    assert np.array_equal(np.asarray(first[5]), np.asarray(second[5]))
