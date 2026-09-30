"""Research multiplicity statistics invariants."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

import pytest


def test_effective_n_bounds() -> None:
    from src.research.stats import effective_trial_count

    rng = np.random.default_rng(7)
    identical = np.repeat(rng.normal(size=(5000, 1)), 5, axis=1)
    assert effective_trial_count(identical) == pytest.approx(1.0, abs=1e-9)
    noise = rng.normal(size=(5000, 5))
    assert 4.5 <= effective_trial_count(noise) <= 5.0


def test_dsr_decreases_with_trial_count() -> None:
    from src.research.stats import deflated_sharpe_ratio

    rng = np.random.default_rng(11)
    candidate = rng.normal(loc=0.001, scale=0.01, size=750)
    trials = rng.normal(loc=0.4, scale=0.3, size=200)
    values = [deflated_sharpe_ratio(candidate, trial_sharpes=trials, n_trials=n) for n in (2, 50, 500)]
    assert all(0.0 < value < 1.0 for value in values)
    assert values[0] > values[1] > values[2]


def test_dsr_degenerate_n() -> None:
    from src.research.stats import deflated_sharpe_ratio

    rng = np.random.default_rng(13)
    candidate = rng.normal(loc=0.001, scale=0.01, size=500)
    trials = np.array([0.5])
    value = deflated_sharpe_ratio(candidate, trial_sharpes=trials, n_trials=1)
    mean, std = float(np.mean(candidate)), float(np.std(candidate, ddof=1))
    sharpe = mean / std
    centered = candidate - mean
    moment2 = float(np.mean(centered**2))
    skew = float(np.mean(centered**3) / moment2**1.5)
    kurt = float(np.mean(centered**4) / moment2**2)
    denom = math.sqrt(1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2)
    assert value == pytest.approx(float(norm.cdf(sharpe * math.sqrt(len(candidate) - 1) / denom)))


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


def test_effective_n_validation() -> None:
    from src.research.stats import effective_trial_count

    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        effective_trial_count(np.ones(10))
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        effective_trial_count(np.ones((1, 3)))
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        effective_trial_count(np.column_stack([np.ones(10), np.arange(10, dtype=float)]))
    single = np.arange(10, dtype=float).reshape(10, 1)
    assert effective_trial_count(single) == pytest.approx(1.0)


def test_dsr_validation() -> None:
    from src.research.stats import deflated_sharpe_ratio

    good = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
    trials = np.array([0.3, 0.5, 0.4])
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        deflated_sharpe_ratio(np.array([0.1]), trial_sharpes=trials, n_trials=5)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        deflated_sharpe_ratio(good, trial_sharpes=np.array([]), n_trials=5)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        deflated_sharpe_ratio(good, trial_sharpes=trials, n_trials=0)
    with pytest.raises(ValueError, match=r"candidate must have positive variance"):
        deflated_sharpe_ratio(np.full(10, 0.5), trial_sharpes=trials, n_trials=5)


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


def test_bootstrap_profile_determinism() -> None:
    from src.research.stats import bootstrap_profile

    rng = np.random.default_rng(31)
    x = rng.normal(loc=0.0005, scale=0.01, size=252)
    kwargs = {"block": 21, "draws": 30, "seed": 7, "horizon": 60, "mdd_limit": -0.5, "sessions_per_year": 252}
    first = bootstrap_profile(x, **kwargs)
    second = bootstrap_profile(x, **kwargs)
    assert first == second
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_profile(np.array([]), **kwargs)
    with pytest.raises(ValueError, match="bootstrap requires"):
        bootstrap_profile(x, **{**kwargs, "block": 500})
    with pytest.raises(ValueError, match="bootstrap requires"):
        bootstrap_profile(x, **{**kwargs, "draws": 0})


def test_bootstrap_extremes() -> None:
    from src.research.stats import bootstrap_profile

    rng = np.random.default_rng(37)
    rising = rng.normal(loc=0.005, scale=0.001, size=500)
    kwargs = {"block": 21, "draws": 50, "seed": 3, "horizon": 126, "mdd_limit": -0.5, "sessions_per_year": 252}
    up = bootstrap_profile(rising, **kwargs)
    assert up.p_cagr_ge_abs_mdd > 0.9
    assert up.p_cagr_le_zero < 0.1
    down = bootstrap_profile(-rising, **kwargs)
    assert down.p_cagr_ge_abs_mdd < 0.1
    assert down.p_cagr_le_zero > 0.9


def test_underwater_quantiles_reflect_long_drawdown() -> None:
    from src.research.stats import bootstrap_profile, point_metrics

    x = np.concatenate([np.full(50, 0.01), np.full(100, -0.005), np.full(50, 0.02)])
    assert point_metrics(x, sessions_per_year=252).underwater_sessions == 125
    profile = bootstrap_profile(x, block=10, draws=40, seed=5, horizon=200, mdd_limit=-0.5, sessions_per_year=252)
    assert profile.underwater_median_sessions >= 50
    assert profile.underwater_p95_sessions >= 100
    assert profile.underwater_p95_sessions >= profile.underwater_median_sessions


def test_dsr_from_dispersion_monotonic_and_guards() -> None:
    from src.research.stats import deflated_sharpe_ratio_from_dispersion

    rng = np.random.default_rng(41)
    candidate = rng.normal(loc=0.001, scale=0.01, size=750)
    values = [
        deflated_sharpe_ratio_from_dispersion(candidate, sharpe_std_per_period=0.3, n_trials=n)
        for n in (1, 2, 50, 500)
    ]
    assert all(0.0 < value < 1.0 for value in values)
    assert values[0] > values[1] > values[2] > values[3]
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio_from_dispersion(candidate, sharpe_std_per_period=0.3, n_trials=0)
    with pytest.raises(ValueError, match="dispersion"):
        deflated_sharpe_ratio_from_dispersion(candidate, sharpe_std_per_period=0.0, n_trials=50)
    with pytest.raises(ValueError, match="dispersion"):
        deflated_sharpe_ratio_from_dispersion(candidate, sharpe_std_per_period=float("nan"), n_trials=50)
    with pytest.raises(ValueError, match="positive variance"):
        deflated_sharpe_ratio_from_dispersion(np.full(10, 0.5), sharpe_std_per_period=0.3, n_trials=50)
    with pytest.raises(ValueError, match="at least 2 sessions"):
        deflated_sharpe_ratio_from_dispersion(np.array([0.1]), sharpe_std_per_period=0.3, n_trials=50)
