"""Research multiplicity statistics invariants."""
from __future__ import annotations

import math

import numpy as np

import pytest


def test_bootstrap_reproducible() -> None:
    from src.research.stats import block_bootstrap_annualized_means

    rng = np.random.default_rng(23)
    x = rng.normal(size=252)
    first = block_bootstrap_annualized_means(x, block=21, draws=50, seed=99, sessions_per_year=252)
    second = block_bootstrap_annualized_means(x, block=21, draws=50, seed=99, sessions_per_year=252)
    third = block_bootstrap_annualized_means(x, block=21, draws=50, seed=100, sessions_per_year=252)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_max_drawdown_of_known_path() -> None:
    from src.research.stats import max_drawdown

    x = np.array([math.log(2.0), math.log(0.25), math.log(2.0)])
    assert max_drawdown(x) == pytest.approx(-0.75)


def test_annualized_growth_and_validation() -> None:
    from src.research.stats import annualized_log_growth

    x = np.array([0.001, 0.002, -0.001])
    assert annualized_log_growth(x, sessions_per_year=252) == pytest.approx(float(np.mean(x) * 252))
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        annualized_log_growth(np.array([]), sessions_per_year=252)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        annualized_log_growth(np.zeros((2, 2)), sessions_per_year=252)


def test_max_drawdown_rejects_empty() -> None:
    from src.research.stats import max_drawdown

    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        max_drawdown(np.array([]))


def test_bootstrap_validation() -> None:
    from src.research.stats import block_bootstrap_annualized_means

    x = np.ones(10)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        block_bootstrap_annualized_means(np.array([]), block=2, draws=5, seed=1, sessions_per_year=252)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        block_bootstrap_annualized_means(x, block=11, draws=5, seed=1, sessions_per_year=252)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        block_bootstrap_annualized_means(x, block=2, draws=0, seed=1, sessions_per_year=252)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        block_bootstrap_annualized_means(x, block=2, draws=5, seed=1, sessions_per_year=252, horizon=0)
    out = block_bootstrap_annualized_means(x, block=3, draws=4, seed=1, sessions_per_year=252, horizon=5)
    assert out.shape == (4,)


def test_point_metrics_on_known_path() -> None:
    from src.research.stats import point_metrics

    x = np.log(np.array([1.1, 0.8, 1.3]))
    metrics = point_metrics(x, sessions_per_year=252)
    assert metrics.max_drawdown == pytest.approx(-0.2)
    assert metrics.calmar == pytest.approx(metrics.cagr / 0.2)
    assert metrics.underwater_sessions == 1
    assert metrics.log_growth == pytest.approx(float(np.mean(x) * 252))
    assert metrics.cagr == pytest.approx(math.exp(metrics.log_growth) - 1.0)
    with pytest.raises(ValueError, match="non-empty"):
        point_metrics(np.array([]), sessions_per_year=252)
    with pytest.raises(ValueError, match="non-empty"):
        point_metrics(np.zeros((2, 2)), sessions_per_year=252)


def test_zero_drawdown_calmar_is_infinite() -> None:
    from src.research.stats import point_metrics

    metrics = point_metrics(np.full(100, 0.001), sessions_per_year=252)
    assert metrics.calmar == math.inf
    assert metrics.max_drawdown == 0.0
    assert metrics.underwater_sessions == 0


def test_growth_profile_deterministic_and_reuses_draws() -> None:
    from src.research.stats import growth_profile

    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    kwargs = {"block": 21, "draws": 50, "seed": 7, "horizon": 126, "sessions_per_year": 252}
    first = growth_profile(x, **kwargs, quantile=0.1, mdd_limits=(-0.3, -0.5))
    second = growth_profile(x, **kwargs, quantile=0.1, mdd_limits=(-0.3, -0.5))
    assert first == second
    other_q = growth_profile(x, **kwargs, quantile=0.5, mdd_limits=(-0.3, -0.5))
    assert other_q.g_median == first.g_median
    assert other_q.g_point == first.g_point
    assert other_q.p_growth_le_zero == first.p_growth_le_zero


def test_growth_profile_quantile_below_median() -> None:
    from src.research.stats import growth_profile

    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.001, scale=0.02, size=1500)
    profile = growth_profile(
        x, block=63, draws=300, seed=3, horizon=1260,
        sessions_per_year=252, quantile=0.1, mdd_limits=(-0.5,),
    )
    assert profile.g_quantile < profile.g_median


def test_growth_profile_validation() -> None:
    from src.research.stats import growth_profile

    x = np.ones(10)
    base = {"block": 3, "draws": 4, "seed": 1, "horizon": 5,
            "sessions_per_year": 252, "quantile": 0.1, "mdd_limits": (-0.5,)}
    with pytest.raises(ValueError, match=r"non-empty|1-D"):
        growth_profile(np.array([]), **base)
    with pytest.raises(ValueError, match=r"non-empty|1-D"):
        growth_profile(np.zeros((2, 2)), **base)
    with pytest.raises(ValueError, match="block"):
        growth_profile(x, **{**base, "block": 0})
    with pytest.raises(ValueError, match="block"):
        growth_profile(x, **{**base, "block": 11})
    with pytest.raises(ValueError, match="draws"):
        growth_profile(x, **{**base, "draws": 0})
    with pytest.raises(ValueError, match="horizon"):
        growth_profile(x, **{**base, "horizon": 0})
    with pytest.raises(ValueError, match="sessions_per_year"):
        growth_profile(x, **{**base, "sessions_per_year": 0})
    with pytest.raises(ValueError, match="quantile"):
        growth_profile(x, **{**base, "quantile": 0.0})
    with pytest.raises(ValueError, match="quantile"):
        growth_profile(x, **{**base, "quantile": 1.0})
    with pytest.raises(ValueError, match="mdd limit"):
        growth_profile(x, **{**base, "mdd_limits": (-1.5,)})
    with pytest.raises(ValueError, match="mdd limit"):
        growth_profile(x, **{**base, "mdd_limits": (0.0,)})


def test_paired_delta_cancels_shared_noise() -> None:
    from src.research.stats import paired_growth_delta

    rng = np.random.default_rng(2)
    b = rng.normal(size=1500)
    a = b + 0.0004
    delta = paired_growth_delta(
        a, b, block=21, draws=300, seed=5, horizon=1260, sessions_per_year=252, alpha=0.05,
    )
    assert delta.lower == pytest.approx(0.0004 * 252, abs=1e-9)
    assert delta.p_positive == 1.0
    assert delta.sessions == 1500


def test_paired_delta_identity() -> None:
    from src.research.stats import paired_growth_delta

    rng = np.random.default_rng(3)
    b = rng.normal(size=200)
    delta = paired_growth_delta(
        b, b, block=10, draws=20, seed=5, horizon=50, sessions_per_year=252, alpha=0.05,
    )
    assert delta.mean == 0.0
    assert delta.lower == 0.0
    assert delta.upper == 0.0


def test_paired_delta_validation() -> None:
    from src.research.stats import paired_growth_delta

    a = np.ones(10)
    b = np.ones(9)
    base = {"block": 3, "draws": 4, "seed": 1, "horizon": 5, "sessions_per_year": 252, "alpha": 0.05}
    with pytest.raises(ValueError, match=r"non-empty|1-D"):
        paired_growth_delta(np.array([]), a, **base)
    with pytest.raises(ValueError, match=r"share length"):
        paired_growth_delta(a, b, **base)
    with pytest.raises(ValueError, match="alpha"):
        paired_growth_delta(a, a, **{**base, "alpha": 0.0})
    with pytest.raises(ValueError, match="alpha"):
        paired_growth_delta(a, a, **{**base, "alpha": 0.5})
    with pytest.raises(ValueError, match="block"):
        paired_growth_delta(a, a, **{**base, "block": 11})
    with pytest.raises(ValueError, match="draws"):
        paired_growth_delta(a, a, **{**base, "draws": 0})
    with pytest.raises(ValueError, match="horizon"):
        paired_growth_delta(a, a, **{**base, "horizon": 0})
    with pytest.raises(ValueError, match="sessions_per_year"):
        paired_growth_delta(a, a, **{**base, "sessions_per_year": 0})


def test_breakeven_interpolation_and_edges() -> None:
    from src.research.stats import breakeven_slippage_ticks

    assert breakeven_slippage_ticks({0: 0.20, 0.5: 0.10, 1.0: -0.02}) == pytest.approx(
        0.5 + 0.5 * (0.10 / 0.12)
    )
    assert breakeven_slippage_ticks({0.0: 0.2, 1.0: 0.1}) == math.inf
    assert breakeven_slippage_ticks({0.0: -0.1, 1.0: 0.2}) == 0.0
    assert breakeven_slippage_ticks({0.0: 0.0, 1.0: -0.1}) == 0.0
    with pytest.raises(ValueError, match="at least two"):
        breakeven_slippage_ticks({0.0: 0.1})
    with pytest.raises(ValueError, match="finite"):
        breakeven_slippage_ticks({0.0: 0.1, 1.0: math.inf})
    with pytest.raises(ValueError, match="finite"):
        breakeven_slippage_ticks({0.0: 0.1, math.nan: 0.2})


def test_paired_growth_tail_delta_growth_matches_legacy() -> None:
    from src.research.stats import paired_growth_delta, paired_growth_tail_delta

    rng = np.random.default_rng(42)
    b = rng.normal(size=500)
    a = b + 0.0003
    kwargs = {
        "block": 21,
        "draws": 100,
        "seed": 999,
        "horizon": 252,
        "sessions_per_year": 252,
        "alpha": 0.05,
    }
    legacy = paired_growth_delta(a, b, **kwargs)
    combined = paired_growth_tail_delta(a, b, **kwargs, tail_block_sessions=21, tail_quantile=0.05)
    assert combined.growth == legacy


def test_paired_growth_tail_delta_tail_improves_when_crashes_shrink() -> None:
    from src.research.stats import paired_growth_tail_delta

    rng = np.random.default_rng(123)
    b = rng.normal(loc=0.0005, scale=0.002, size=2520)
    # create severe crash months in b
    m = b.size // 21
    sums = [b[i * 21 : (i + 1) * 21].sum() for i in range(m)]
    worst_indices = np.argsort(sums)[: max(1, round(0.05 * m))]
    for idx in worst_indices:
        b[idx * 21 : (idx + 1) * 21] -= 0.006

    # halve the worst monthly blocks in a
    a = b.copy()
    for idx in worst_indices:
        a[idx * 21 : (idx + 1) * 21] = b[idx * 21 : (idx + 1) * 21] * 0.5

    # redistribute difference across other sessions to keep |growth.mean| < 1%p
    diff_total = float((a - b).sum())
    other_indices = [i for i in range(b.size) if i // 21 not in worst_indices]
    for i in other_indices:
        a[i] -= diff_total / len(other_indices)

    result = paired_growth_tail_delta(
        a, b,
        block=21, draws=500, seed=77, horizon=2520, sessions_per_year=252,
        alpha=0.05, tail_block_sessions=21, tail_quantile=0.05,
    )
    assert result.tail.lower > 0.0
    assert abs(result.growth.mean) < 0.01


def test_paired_growth_tail_delta_argument_validation() -> None:
    from src.research.stats import paired_growth_tail_delta

    a = np.ones(50)
    b = np.ones(50)
    base = {
        "block": 10,
        "draws": 20,
        "seed": 1,
        "horizon": 30,
        "sessions_per_year": 252,
        "alpha": 0.05,
    }
    with pytest.raises(ValueError, match="tail_quantile"):
        paired_growth_tail_delta(a, b, **base, tail_block_sessions=10, tail_quantile=0.0)
    with pytest.raises(ValueError, match="tail_quantile"):
        paired_growth_tail_delta(a, b, **base, tail_block_sessions=10, tail_quantile=0.6)
    with pytest.raises(ValueError, match="tail_block_sessions"):
        paired_growth_tail_delta(a, b, **base, tail_block_sessions=0, tail_quantile=0.05)
    with pytest.raises(ValueError, match="tail_block_sessions"):
        paired_growth_tail_delta(a, b, **base, tail_block_sessions=35, tail_quantile=0.05)
    with pytest.raises(ValueError, match=r"non-empty|1-D"):
        paired_growth_tail_delta(np.array([]), a, **base, tail_block_sessions=10, tail_quantile=0.05)
    with pytest.raises(ValueError, match=r"share length"):
        paired_growth_tail_delta(a, np.ones(40), **base, tail_block_sessions=10, tail_quantile=0.05)
    with pytest.raises(ValueError, match="alpha"):
        paired_growth_tail_delta(a, b, **{**base, "alpha": 0.0}, tail_block_sessions=10, tail_quantile=0.05)
    with pytest.raises(ValueError, match="alpha"):
        paired_growth_tail_delta(a, b, **{**base, "alpha": 0.5}, tail_block_sessions=10, tail_quantile=0.05)

