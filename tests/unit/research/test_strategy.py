"""Declarative strategy spec and target-construction invariants."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.data.research_protocol import BenchmarkPolicy
from src.research.cube import ResearchCube
from src.research.strategy import (
    StrategySpec,
    benchmark_targets,
    load_strategy_spec,
    rebalance_rows,
    target_weights,
    universe_mask,
)

INSTS = [f"KRX:{i:06d}" for i in range(10)]


def _sessions(n: int = 8, start: date = date(2020, 1, 6)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _cube(
    sessions: list[date],
    n: int = 10,
    *,
    adtv: float = 1e9,
    close: float = 5000.0,
    vol: float = 0.02,
) -> ResearchCube:
    ids = [f"KRX:{i:06d}" for i in range(n)]
    shape = (len(sessions), n)
    ones_b = np.ones(shape, dtype=bool)
    zeros_b = np.zeros(shape, dtype=bool)
    arrays: dict[str, Any] = {
        "present": ones_b,
        "eligible": ones_b,
        "entry_blocked": zeros_b,
        "open_at_upper": zeros_b,
        "open_at_lower": zeros_b,
        "traded": ones_b,
        "volume": np.full(shape, 1000.0),
        "open": np.full(shape, close),
        "close": np.full(shape, close),
        "base_price": np.full(shape, close),
        "market_cap": np.full(shape, 1e11),
        "adtv20": np.full(shape, adtv),
        "ret_vol60": np.full(shape, vol),
        "tick_at_open": np.ones(shape),
        "sell_tax_rate": np.zeros(shape),
        "r_on": np.zeros(shape),
        "r_id": np.zeros(shape),
    }
    return ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=list(sessions),
        instrument_ids=ids,
        arrays=arrays,
        exit_at=np.full(n, -1, dtype=np.int64),
        exit_halted=np.zeros(n, dtype=np.bool_),
    )


def _spec(**overrides: Any) -> StrategySpec:
    base: dict[str, Any] = {
        "family": "probe",
        "universe": {"min_adtv20_krw": 0, "min_price_krw": 0},
        "junk": None,
        "score": {"size": 1.0},
        "n": 3,
        "keep_rank_multiple": 1.0,
        "rebalance": "D",
    }
    base.update(overrides)
    return StrategySpec.model_validate(base)


def _features(sessions: list[date], n: int, values: dict[str, list[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, row in values.items():
        arr = np.full((len(sessions), n), np.nan)
        arr[0, :] = np.asarray(row, dtype=np.float64)
        out[name] = arr
    return out


def test_spec_hash_is_order_independent(tmp_path: Path) -> None:
    """Permuted TOML keys give equal hashes."""
    first = tmp_path / "a.toml"
    second = tmp_path / "b.toml"
    first.write_text(
        'family = "pead"\nn = 20\nkeep_rank_multiple = 2.0\nrebalance = "M"\n'
        '[universe]\nmin_adtv20_krw = 500000000\nmin_price_krw = 1000\n'
        '[score]\nsue_op = 1.0\nsue_ni = 1.0\n',
        encoding="utf-8",
    )
    second.write_text(
        'rebalance = "M"\nkeep_rank_multiple = 2.0\nn = 20\nfamily = "pead"\n'
        '[score]\nsue_ni = 1.0\nsue_op = 1.0\n'
        '[universe]\nmin_price_krw = 1000\nmin_adtv20_krw = 500000000\n',
        encoding="utf-8",
    )
    assert load_strategy_spec(first).spec_hash == load_strategy_spec(second).spec_hash


def test_unknown_feature_rejected() -> None:
    """Unknown score features fail validation."""
    with pytest.raises(ValueError, match="unknown feature"):
        _spec(score={"nope": 1.0})


def test_junk_filter_excludes_low_percentile_names() -> None:
    """Low lowturn names are excluded even when their score is highest."""
    sessions = _sessions(2)
    cube = _cube(sessions)
    lowturn = np.full((2, 10), 0.5)
    lowturn[0, 0] = -10.0
    lowturn[0, 1] = -9.0
    size = np.full((2, 10), 0.0)
    size[0, 0] = 100.0
    size[0, 1] = 90.0
    features = {"lowturn": lowturn, "size": size}
    spec = _spec(
        junk={"features": ["lowturn"], "min_rank": 0.2},
        score={"size": 1.0},
        n=3,
        rebalance="D",
    )
    weights = target_weights(spec, cube, features, lo=0, hi=1)
    chosen = set(np.flatnonzero(weights[0] > 0).tolist())
    assert 0 not in chosen
    assert 1 not in chosen


def test_hysteresis_keeps_incumbents() -> None:
    """Incumbents within the keep band stay; far fallers are replaced."""
    sessions = _sessions(3)
    cube = _cube(sessions, n=50)
    second = np.full((3, 50), 0.0)
    second[0, :] = np.arange(50, dtype=np.float64)[::-1]
    second[1, :] = np.arange(50, dtype=np.float64)[::-1]
    second[1, 0] = 24.5
    second[2, :] = np.arange(50, dtype=np.float64)[::-1]
    second[2, 0] = -1000.0
    features = {"size": second}
    spec = _spec(score={"size": 1.0}, n=20, keep_rank_multiple=2.0, rebalance="D")
    weights = target_weights(spec, cube, features, lo=0, hi=3)
    assert 0 in set(np.flatnonzero(weights[1] > 0).tolist())
    assert 0 not in set(np.flatnonzero(weights[2] > 0).tolist())


def test_negative_weight_flips_direction() -> None:
    """Negative weights select the largest capitalizations."""
    sessions = _sessions(2)
    cube = _cube(sessions)
    size = np.full((2, 10), 0.0)
    size[0, :] = np.arange(10, dtype=np.float64)
    features = {"size": size}
    spec = _spec(score={"size": -1.0}, n=2, rebalance="D")
    weights = target_weights(spec, cube, features, lo=0, hi=1)
    chosen = set(np.flatnonzero(weights[0] > 0).tolist())
    assert chosen == {0, 1}


def test_monthly_rows_are_month_end_decisions() -> None:
    """Monthly rows are the last sessions of Jan and Feb."""
    sessions = [date(2020, 1, 30), date(2020, 1, 31), date(2020, 2, 28), date(2020, 2, 29), date(2020, 3, 2)]
    rows = rebalance_rows(sessions, lo=0, hi=5, freq="M")
    assert rows == (1, 3)


def test_empty_universe_yields_zero_weights() -> None:
    """No qualifying names give all-zero weights."""
    sessions = _sessions(2)
    cube = _cube(sessions, adtv=0.0)
    size = np.full((2, 10), 1.0)
    spec = _spec(
        universe={"min_adtv20_krw": 1_000_000_000, "min_price_krw": 0},
        score={"size": 1.0},
        n=3,
        rebalance="D",
    )
    weights = target_weights(spec, cube, {"size": size}, lo=0, hi=1)
    assert bool(np.all(weights[0] == 0.0))


def test_benchmarks_sum_to_one() -> None:
    """Each benchmark row sums to one over positive weights."""
    sessions = [date(2020, 1, 30), date(2020, 1, 31), date(2020, 2, 28), date(2020, 2, 29)]
    cube = _cube(sessions, n=6)
    policy = BenchmarkPolicy(universe_min_adtv20_krw=1, universe_min_price_krw=1, rebalance="M")
    out = benchmark_targets(cube, policy, lo=0, hi=4)
    assert set(out) == {"U_EW", "CW"}
    assert len(next(iter(out.values()))) > 0
    for group in out.values():
        for weights in group.values():
            assert float(np.sum(weights)) == pytest.approx(1.0)
            assert bool(np.all(weights >= 0.0))


def test_universe_mask_applies_floors() -> None:
    """Universe floors filter low-liquidity and blocked names."""
    sessions = _sessions(2)
    cube = _cube(sessions, n=3)
    rule = {"min_adtv20_krw": 100, "min_price_krw": 100}
    spec_rule = StrategySpec.model_validate(
        {
            "family": "f",
            "universe": rule,
            "junk": None,
            "score": {"size": 1.0},
            "n": 1,
            "keep_rank_multiple": 1.0,
            "rebalance": "D",
        }
    )
    mask = universe_mask(cube, spec_rule.universe)
    assert mask.shape == (2, 3)
    assert bool(np.all(mask))


def test_spec_validation_rejects_bad_inputs() -> None:
    """All spec validators fail closed on bad values."""
    with pytest.raises(ValueError, match="non-negative int"):
        StrategySpec.model_validate(
            {"family": "f", "universe": {"min_adtv20_krw": -1, "min_price_krw": 0},
             "score": {"size": 1.0}, "n": 1, "keep_rank_multiple": 1.0, "rebalance": "D"}
        )
    with pytest.raises(ValueError, match="min_rank"):
        StrategySpec.model_validate(
            {"family": "f", "universe": {"min_adtv20_krw": 0, "min_price_krw": 0},
             "junk": {"features": ["size"], "min_rank": 1.0},
             "score": {"size": 1.0}, "n": 1, "keep_rank_multiple": 1.0, "rebalance": "D"}
        )
    with pytest.raises(ValueError, match="unknown feature"):
        StrategySpec.model_validate(
            {"family": "f", "universe": {"min_adtv20_krw": 0, "min_price_krw": 0},
             "junk": {"features": ["nope"], "min_rank": 0.2},
             "score": {"size": 1.0}, "n": 1, "keep_rank_multiple": 1.0, "rebalance": "D"}
        )
    with pytest.raises(ValueError, match="non-empty"):
        StrategySpec.model_validate(
            {"family": "  ", "universe": {"min_adtv20_krw": 0, "min_price_krw": 0},
             "score": {"size": 1.0}, "n": 1, "keep_rank_multiple": 1.0, "rebalance": "D"}
        )
    with pytest.raises(ValueError, match="n must"):
        _spec(n=0)
    with pytest.raises(ValueError, match="keep_rank_multiple"):
        _spec(keep_rank_multiple=0.5)
    with pytest.raises(ValueError, match="non-empty mapping"):
        _spec(score={})
    with pytest.raises(ValueError, match="finite non-zero"):
        _spec(score={"size": 0.0})
    with pytest.raises(ValueError, match="finite non-zero"):
        _spec(score={"size": float("nan")})
    with pytest.raises(ValueError, match="must satisfy"):
        _spec(junk={"features": ["size"], "min_rank": 2.0})
    with pytest.raises(ValueError, match="must be >= 1"):
        _spec(keep_rank_multiple=0.5)
    assert _spec().canonical_json()
    assert _spec().spec_hash
    assert set(_spec(junk={"features": ["lowturn"], "min_rank": 0.2}).feature_names()) == {"lowturn", "size"}


def test_rebalance_validation_and_frequencies() -> None:
    """Bad windows fail; W/2W/Q rows follow calendar rules."""
    sessions = _sessions(4)
    with pytest.raises(ValueError, match="ints"):
        rebalance_rows(sessions, lo=True, hi=2, freq="D")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="within"):
        rebalance_rows(sessions, lo=0, hi=99, freq="D")
    with pytest.raises(ValueError, match="unknown rebalance"):
        rebalance_rows(sessions, lo=0, hi=2, freq="X")
    week = [date(2020, 1, 3), date(2020, 1, 4), date(2020, 1, 6), date(2020, 1, 7)]
    assert rebalance_rows(week, lo=0, hi=4, freq="W") == (1,)
    assert rebalance_rows(week, lo=0, hi=4, freq="2W") in ((), (1,))
    quarter = [date(2020, 3, 30), date(2020, 3, 31), date(2020, 4, 1), date(2020, 4, 2)]
    assert rebalance_rows(quarter, lo=0, hi=4, freq="Q") == (1,)
    assert rebalance_rows(sessions, lo=0, hi=4, freq="D") == (0, 1, 2, 3)


def test_target_weights_edge_branches() -> None:
    """Sparse ranks, missing components and bad inputs are handled."""
    sessions = _sessions(2)
    cube = _cube(sessions, n=4)
    size = np.full((2, 4), np.nan)
    size[0, :] = [1.0, 2.0, 3.0, 4.0]
    spec = _spec(score={"size": 1.0}, n=2, rebalance="D")
    weights = target_weights(spec, cube, {"size": size}, lo=0, hi=1)
    assert bool(np.all(weights[0] == 0.0))
    size2 = np.full((2, 4), 1.0)
    other = np.full((2, 4), np.nan)
    spec2 = _spec(score={"size": 1.0, "lowturn": 1.0}, n=2, rebalance="D")
    weights2 = target_weights(spec2, cube, {"size": size2, "lowturn": other}, lo=0, hi=1)
    assert bool(np.all(weights2[0] == 0.0))
    with pytest.raises(ValueError, match="within"):
        target_weights(spec, cube, {"size": size2}, lo=0, hi=99)
    with pytest.raises(ValueError, match="wrong shape"):
        target_weights(spec, cube, {"size": np.zeros((2, 5))}, lo=0, hi=1)
    with pytest.raises(KeyError):
        target_weights(spec, cube, {}, lo=0, hi=1)


def test_benchmark_without_traded_array() -> None:
    """CW falls back to present/volume/open when traded is absent."""
    sessions = [date(2020, 1, 30), date(2020, 1, 31), date(2020, 2, 28), date(2020, 2, 29)]
    cube = _cube(sessions, n=4)
    arrays = {key: np.asarray(value).copy() for key, value in cube.arrays.items()}
    del arrays["traded"]
    bare = ResearchCube.from_arrays(
        cube_id=cube.cube_id, sessions=list(sessions), instrument_ids=list(cube.instrument_ids),
        arrays=arrays, exit_at=np.asarray(cube.exit_at), exit_halted=np.asarray(cube.exit_halted),
    )
    policy = BenchmarkPolicy(universe_min_adtv20_krw=1, universe_min_price_krw=1, rebalance="M")
    out = benchmark_targets(bare, policy, lo=0, hi=4)
    assert set(out) == {"U_EW", "CW"}
    with pytest.raises(ValueError, match="within"):
        benchmark_targets(bare, policy, lo=0, hi=99)
