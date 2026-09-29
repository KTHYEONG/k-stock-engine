"""Research multiplicity statistics invariants."""
from __future__ import annotations

import math
from datetime import date

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


def test_pbo_of_dominant_column_is_zero() -> None:
    from src.research.stats import cscv_pbo

    rng = np.random.default_rng(17)
    noise = rng.normal(scale=0.01, size=(800, 4))
    dominant = rng.normal(loc=0.005, scale=0.01, size=800)
    matrix = np.column_stack([dominant, noise])
    assert cscv_pbo(matrix, blocks=4).pbo == 0.0


def test_pbo_of_pure_noise_near_one_half() -> None:
    from src.research.stats import cscv_pbo

    rng = np.random.default_rng(19)
    matrix = rng.normal(size=(800, 20))
    assert 0.3 <= cscv_pbo(matrix, blocks=4).pbo <= 0.7


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


def test_yearly_active_sums() -> None:
    from src.research.stats import yearly_active_log

    sessions = [date(2022, 12, 30), date(2023, 1, 2), date(2023, 6, 1)]
    active = np.array([0.01, 0.02, -0.005])
    assert yearly_active_log(sessions, active) == {2022: pytest.approx(0.01), 2023: pytest.approx(0.015)}


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


def test_information_ratio_and_t_stat() -> None:
    from src.research.stats import active_t_stat, information_ratio

    rng = np.random.default_rng(29)
    x = rng.normal(loc=0.001, scale=0.01, size=100)
    assert information_ratio(x, sessions_per_year=252) == pytest.approx(
        float(np.mean(x) / np.std(x, ddof=1) * math.sqrt(252))
    )
    assert active_t_stat(x) == pytest.approx(float(np.mean(x) / np.std(x, ddof=1) * math.sqrt(100)))
    flat = np.full(10, 0.5)
    assert information_ratio(flat, sessions_per_year=252) == 0.0
    assert active_t_stat(flat) == 0.0
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        information_ratio(np.array([0.1]), sessions_per_year=252)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        active_t_stat(np.array([0.1]))


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


def test_sharpe_of_branches() -> None:
    from src.research.stats import _sharpe_of

    assert _sharpe_of(np.array([0.1])) == 0.0
    assert _sharpe_of(np.full(5, 0.01)) == 0.0
    assert _sharpe_of(np.array([0.01, 0.02, 0.03])) > 0.0


def test_cscv_validation() -> None:
    from src.research.stats import cscv_pbo

    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        cscv_pbo(np.ones(10), blocks=2)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        cscv_pbo(np.ones((10, 2)), blocks=3)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        cscv_pbo(np.ones((10, 1)), blocks=2)
    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        cscv_pbo(np.ones((2, 3)), blocks=4)


def test_yearly_length_mismatch() -> None:
    from src.research.stats import yearly_active_log

    with pytest.raises(ValueError, match=r"invalid|empty|positive|length|1-D|2-D|column|horizon|draws|block|session|trial|variance|must|non|at least"):
        yearly_active_log([date(2023, 1, 2)], np.array([0.01, 0.02]))
