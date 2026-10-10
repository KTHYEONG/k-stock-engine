"""Operational daily decision invariants: stateful sleeves, sealed decisions, parity."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from src.core.pit import PITDataError
from src.research.book import build_sleeve_targets, combine_sleeve_targets, sleeve_capital_krw
from src.research.cube import ResearchCube
from src.research.model import FrozenScorer, ScorerConfig, ScoreMatrix, score_rows
from src.research.operate import (
    SealedDecision,
    advance as _advance,
    bootstrap_state,
    load_state,
    overlay_signals,
    read_chain,
    save_state,
    seal,
    state_digest,
)
from src.research.panel import FeaturePanel, feature_names_for
from src.research.policy import TrendCashPolicy, UniverseRule, decision_rows, select_row

KRX = ZoneInfo("Asia/Seoul")

N = 4
CAPITAL = 10_000_000
BUFFER = 0.0


def _series(cube: ResearchCube) -> dict[str, np.ndarray]:
    return {
        "cash_returns": np.zeros(len(cube.sessions), dtype=np.float64),
        "hedge_levels": np.ones(len(cube.sessions), dtype=np.float64),
        "hedge_inverse": np.ones(len(cube.sessions), dtype=np.float64),
        "trend_levels": np.ones(len(cube.sessions), dtype=np.float64),
    }


def advance(
    state: Any, spec: Any, cube: ResearchCube, panel: FeaturePanel, scorer: FrozenScorer,
    *, capital_krw: int, cash_buffer: float,
) -> Any:
    return _advance(state, spec, cube, panel, scorer, capital_krw=capital_krw,
                    cash_buffer=cash_buffer, series=_series(cube))


class _StubBooster:
    def predict(self, data: np.ndarray) -> np.ndarray:
        mat = np.asarray(data, dtype=np.float64)
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            return np.zeros(mat.shape[0], dtype=np.float64)
        return np.ascontiguousarray(mat[:, 0], dtype=np.float64)


def _sessions(n: int = 60, start: date = date(2021, 1, 4)) -> list[date]:
    from src.core.krx_calendar import xkrx_calendar_through

    return [instant.date() for instant in xkrx_calendar_through(start).sessions if instant.date() >= start][:n]


def _spec() -> SimpleNamespace:
    policy = TrendCashPolicy(
        family="ml_trend_cash",
        n=2,
        keep_rank_multiple=3.0,
        rebalance_every_sessions=2,
        universe=UniverseRule(min_adtv20_krw=0, min_price_krw=0),
        trend_min_dev_ma20=None,
        trend_min_ret21=None,
        min_units_per_slot=0,
    )
    book = SimpleNamespace(sleeves=2, stock_capital_fraction=0.75, rebalance_band=0.0)
    from src.research.book import BookSpec as _Book

    return SimpleNamespace(
        policy=policy,
        book=_Book(sleeves=2, stock_capital_fraction=0.75),
        spec_hash="spec-test-hash",
        trend_overlay=None,
        regime_hedge=None,
    )


def _scorer_config() -> ScorerConfig:
    return ScorerConfig(horizons=(5,), min_cross_section=2, min_train_rows=1)


def _scorer(year: int = 2021) -> FrozenScorer:
    config = _scorer_config()
    return FrozenScorer(config=config, year=year, boosters={(int(config.seed), 5): _StubBooster()})  # type: ignore[dict-item]


def _cube(sessions: list[date] | None = None, n: int = N, seed: int = 7) -> ResearchCube:
    sess = sessions if sessions is not None else _sessions()
    shape = (len(sess), n)
    rng = np.random.default_rng(seed)
    close = 5000.0 + rng.random(shape) * 500.0 + np.arange(n)[None, :] * 100.0
    arrays = {
        "present": np.ones(shape, dtype=bool),
        "eligible": np.ones(shape, dtype=bool),
        "entry_blocked": np.zeros(shape, dtype=bool),
        "volume": np.full(shape, 1_000_000.0),
        "adtv20": np.full(shape, 1_000_000_000.0),
        "close": np.ascontiguousarray(close, dtype=np.float64),
        "ret_vol60": np.full(shape, 0.2),
    }
    return ResearchCube.from_arrays(
        cube_id="research_cube_test",
        sessions=sess,
        instrument_ids=[f"KRX:{i + 1:06d}" for i in range(n)],
        arrays=arrays,
        exit_at=np.full(n, -1, dtype=np.int64),
        exit_halted=np.zeros(n, dtype=np.bool_),
    )


def _panel(cube: ResearchCube, seed: int = 11) -> FeaturePanel:
    rng = np.random.default_rng(seed)
    shape = (len(cube.sessions), len(cube.instrument_ids))
    feats = {
        name: np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32)
        for name in feature_names_for("full62")
    }
    return FeaturePanel(features=feats, labels={}, last_row=len(cube.sessions) - 1)


def _scores(cube: ResearchCube, panel: FeaturePanel, scorer: FrozenScorer) -> np.ndarray:
    from src.research.policy import universe_mask as _mask

    uni = np.ascontiguousarray(np.asarray(_mask(cube, _spec().policy.universe), dtype=bool))
    mats = [np.asarray(panel.features[name]) for name in feature_names_for("full62")]
    rows = list(range(len(cube.sessions)))
    return np.asarray(score_rows(mats, uni, rows, scorer), dtype=np.float64)


def _research_reference(
    spec: SimpleNamespace,
    cube: ResearchCube,
    panel: FeaturePanel,
    scores: np.ndarray,
    *,
    lo: int,
    hi: int,
) -> tuple[tuple[dict[int, np.ndarray], ...], dict[int, np.ndarray]]:
    from src.research.policy import universe_mask as _mask

    uni = np.ascontiguousarray(np.asarray(_mask(cube, spec.policy.universe), dtype=bool))[: hi + 1]
    matrix = ScoreMatrix(
        scores=np.ascontiguousarray(scores[: hi + 1], dtype=np.float32),
        test_years=(),
        config_hash="test",
        last_row=hi,
    )
    view = SimpleNamespace(arrays={"close": np.asarray(cube.arrays["close"], dtype=np.float64)[: hi + 1]})
    sleeves = build_sleeve_targets(
        spec.policy,
        view,  # type: ignore[arg-type]
        panel,
        matrix,
        uni,
        sessions=list(cube.sessions),
        sleeves=int(spec.book.sleeves),
        lo=lo,
        hi=hi + 1,
        sleeve_capital_krw=sleeve_capital_krw(CAPITAL, spec.book),
        cash_buffer=BUFFER,
    )
    combined = combine_sleeve_targets(sleeves, lo=max(lo, 1), hi=hi + 1)
    return sleeves, combined


def _prefix(cube: ResearchCube, end: int) -> ResearchCube:
    arrays = {name: np.ascontiguousarray(arr[: end + 1]) for name, arr in cube.arrays.items()}
    exit_at = np.array(cube.exit_at, dtype=np.int64, copy=True)
    halted = np.array(cube.exit_halted, dtype=bool, copy=True)
    exit_at[exit_at >= end + 1] = -1
    halted[exit_at < 0] = False
    return ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions[: end + 1]),
        instrument_ids=list(cube.instrument_ids),
        arrays=arrays,
        exit_at=exit_at,
        exit_halted=halted,
    )


def _incumbents_for(
    spec: SimpleNamespace, scores: np.ndarray, cube: ResearchCube, panel: FeaturePanel, *, lo: int, hi: int
) -> list[frozenset[int]]:
    from src.research.policy import universe_mask as _mask

    uni = np.asarray(_mask(cube, spec.policy.universe), dtype=bool)
    close = np.asarray(cube.arrays["close"], dtype=np.float64)
    dev = np.asarray(panel.features["dev_ma20"], dtype=np.float64)
    ret = np.asarray(panel.features["ret_21"], dtype=np.float64)
    slot = float(sleeve_capital_krw(CAPITAL, spec.book)) * (1.0 - BUFFER) / float(spec.policy.n)
    every = int(spec.policy.rebalance_every_sessions)
    out: list[frozenset[int]] = []
    for phase in range(int(spec.book.sleeves)):
        carried: frozenset[int] = frozenset()
        for row in decision_rows(list(cube.sessions), lo=lo, hi=hi + 1, every=every, phase=phase):
            _, carried = select_row(
                spec.policy,
                score_row=scores[row],
                universe_row=uni[row],
                close_row=close[row],
                dev_row=dev[row],
                ret_row=ret[row],
                incumbents=carried,
                slot_krw=slot,
            )
        out.append(carried)
    return out


def test_advance_equals_research_loop_step_by_step() -> None:
    spec = _spec()
    cube = _cube()
    panel = _panel(cube)
    scorer = _scorer()
    scores = _scores(cube, panel, scorer)
    lo, r = 10, 30
    sleeves, _ = _research_reference(spec, cube, panel, scores, lo=lo, hi=r)
    state = bootstrap_state(spec, cube, sleeves, _incumbents_for(spec, scores, cube, panel, lo=lo, hi=r), lo=lo, last_row=r)
    current = state
    for target in range(r + 1, r + 11):
        sub = _prefix(cube, target)
        sub_panel = _panel(sub)
        sub_scores = _scores(sub, sub_panel, scorer)
        _, ref = _research_reference(spec, sub, sub_panel, sub_scores, lo=lo, hi=target)
        nxt, decisions = advance(current, spec, sub, sub_panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
        assert len(decisions) == 1
        day, weights = decisions[0]
        assert day == sub.sessions[target]
        expect = {sub.instrument_ids[c]: float(v) for c, v in enumerate(np.asarray(ref[target])) if float(v) != 0.0}
        assert dict(weights) == pytest.approx(expect)
        current = nxt
    _, ref_full = _research_reference(spec, cube, panel, scores, lo=lo, hi=r + 10)
    final_ref = bootstrap_state(
        spec, cube, *_research_reference(spec, cube, panel, scores, lo=lo, hi=r + 10)[:1],
        _incumbents_for(spec, scores, cube, panel, lo=lo, hi=r + 10), lo=lo, last_row=r + 10,
    )
    assert state_digest(current) == state_digest(final_ref)
    assert current == final_ref


def test_catch_up_matches_one_at_a_time() -> None:
    spec = _spec()
    cube = _cube()
    panel = _panel(cube)
    scorer = _scorer()
    scores = _scores(cube, panel, scorer)
    lo, r = 10, 30
    sleeves, _ = _research_reference(spec, cube, panel, scores, lo=lo, hi=r)
    state = bootstrap_state(spec, cube, sleeves, _incumbents_for(spec, scores, cube, panel, lo=lo, hi=r), lo=lo, last_row=r)
    batched_state, batched = advance(state, spec, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    assert [day for day, _ in batched[:3]] == list(cube.sessions[r + 1 : r + 4])
    current = state
    single: list[tuple] = []
    for target in range(r + 1, r + 4):
        sub = _prefix(cube, target)
        nxt, decisions = advance(current, spec, sub, _panel(sub), scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
        single.extend(decisions)
        current = nxt
    assert [day for day, _ in single] == [day for day, _ in batched[:3]]
    for (_, a), (_, b) in zip(single, batched[:3], strict=True):
        assert dict(a) == pytest.approx(dict(b))
    sub = _prefix(cube, r + 3)
    expected, _ = advance(state, spec, sub, _panel(sub), scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    assert current == expected


def test_gap_fails_closed() -> None:
    spec = _spec()
    cube = _cube()
    panel = _panel(cube)
    scorer = _scorer()
    scores = _scores(cube, panel, scorer)
    lo, r = 10, 30
    sleeves, _ = _research_reference(spec, cube, panel, scores, lo=lo, hi=r)
    state = bootstrap_state(spec, cube, sleeves, _incumbents_for(spec, scores, cube, panel, lo=lo, hi=r), lo=lo, last_row=r)
    before = state_digest(state)
    tail = _prefix(cube, len(cube.sessions) - 1)
    arrays = {name: np.ascontiguousarray(arr[r + 2 :]) for name, arr in tail.arrays.items()}
    gapped = ResearchCube.from_arrays(
        cube_id=tail.cube_id,
        sessions=list(tail.sessions[r + 2 :]),
        instrument_ids=list(tail.instrument_ids),
        arrays=arrays,
        exit_at=np.array(tail.exit_at, copy=True),
        exit_halted=np.array(tail.exit_halted, copy=True),
    )
    with pytest.raises(PITDataError):
        advance(state, spec, gapped, _panel(gapped), scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    assert state_digest(state) == before


def test_wrong_year_scorer_refused() -> None:
    spec = _spec()
    sessions = _sessions(n=40, start=date(2021, 12, 1))
    cube = _cube(sessions=sessions)
    panel = _panel(cube)
    scorer = _scorer(year=2021)
    scores = _scores(cube, panel, scorer)
    lo, r = 4, 10
    sleeves, _ = _research_reference(spec, cube, panel, scores, lo=lo, hi=r)
    state = bootstrap_state(spec, cube, sleeves, _incumbents_for(spec, scores, cube, panel, lo=lo, hi=r), lo=lo, last_row=r)
    with pytest.raises(ValueError, match="year"):
        advance(state, spec, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)


def test_delisted_incumbent_dropped() -> None:
    spec = _spec()
    cube = _cube()
    panel = _panel(cube)
    scorer = _scorer()
    scores = _scores(cube, panel, scorer)
    lo, r = 10, 30
    sleeves, _ = _research_reference(spec, cube, panel, scores, lo=lo, hi=r)
    incumb = _incumbents_for(spec, scores, cube, panel, lo=lo, hi=r)
    state = bootstrap_state(spec, cube, sleeves, incumb, lo=lo, last_row=r)
    keep = list(cube.instrument_ids[:2])
    arrays = {name: np.ascontiguousarray(arr[:, :2]) for name, arr in cube.arrays.items()}
    slim = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions),
        instrument_ids=keep,
        arrays=arrays,
        exit_at=np.array(cube.exit_at[:2], copy=True),
        exit_halted=np.array(cube.exit_halted[:2], copy=True),
    )
    nxt, decisions = advance(state, spec, slim, _panel(slim), scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    assert decisions
    gone = set(cube.instrument_ids[2:])
    for sleeve in nxt.sleeves:
        assert not (set(sleeve.incumbents) & gone)
        assert not (set(sleeve.weights) & gone)


def _sealed(day: date, prev: str = "", digest: str = "d" * 64) -> SealedDecision:
    return SealedDecision(
        data_session=day,
        execution_session=day + timedelta(days=1),
        sealed_at=datetime(2021, 6, 1, 9, 0, tzinfo=KRX),
        spec_hash="spec",
        models_id="models",
        inputs={"cube": "c1"},
        targets={"KRX:000001": 0.25},
        reference={"KRX:000001": 5000.0},
        overlay={"regime_fraction": 0.0, "trend_signal": 1.0, "trend_vol_scale": 1.0},
        state_digest=digest,
        prev_hash=prev,
        hash="",
    )


def test_seal_chain_tamper_evident_and_idempotent(tmp_path: Path) -> None:
    first = seal(tmp_path, _sealed(date(2021, 6, 1)))
    second = seal(tmp_path, _sealed(date(2021, 6, 2)))
    assert second.prev_hash == first.hash
    assert len(read_chain(tmp_path)) == 2
    again = seal(tmp_path, _sealed(date(2021, 6, 1)))
    assert again.hash == first.hash
    assert len(read_chain(tmp_path)) == 2
    with pytest.raises(ValueError, match="conflicting"):
        seal(tmp_path, _sealed(date(2021, 6, 1), digest="e" * 64))
    assert len(read_chain(tmp_path)) == 2
    target = tmp_path / "2021-06-01.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["targets"] = {"KRX:000001": 0.5}
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"modified"):
        read_chain(tmp_path)


def test_future_data_cannot_change_sealed_decision() -> None:
    spec = _spec()
    cube = _cube()
    panel = _panel(cube)
    scorer = _scorer()
    scores = _scores(cube, panel, scorer)
    lo, r = 10, 30
    sleeves, _ = _research_reference(spec, cube, panel, scores, lo=lo, hi=r)
    state = bootstrap_state(spec, cube, sleeves, _incumbents_for(spec, scores, cube, panel, lo=lo, hi=r), lo=lo, last_row=r)
    _, first_run = advance(state, spec, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    want = dict(first_run[0][1])
    rng = np.random.default_rng(3)
    arrays = {}
    for name, arr in cube.arrays.items():
        work = np.array(arr, copy=True)
        if work.dtype != bool:
            work[r + 2 :] = rng.standard_normal(work[r + 2 :].shape)
        arrays[name] = np.ascontiguousarray(work)
    mutated = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions),
        instrument_ids=list(cube.instrument_ids),
        arrays=arrays,
        exit_at=np.array(cube.exit_at, copy=True),
        exit_halted=np.array(cube.exit_halted, copy=True),
    )
    _, second_run = advance(state, spec, mutated, _panel(mutated), scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    assert dict(second_run[0][1]) == pytest.approx(want)


def test_state_roundtrip(tmp_path: Path) -> None:
    spec = _spec()
    cube = _cube()
    panel = _panel(cube)
    scorer = _scorer()
    scores = _scores(cube, panel, scorer)
    lo, r = 10, 30
    sleeves, _ = _research_reference(spec, cube, panel, scores, lo=lo, hi=r)
    state = bootstrap_state(spec, cube, sleeves, _incumbents_for(spec, scores, cube, panel, lo=lo, hi=r), lo=lo, last_row=r)
    path = tmp_path / "state.json"
    save_state(path, state)
    assert load_state(path) == state


def test_overlay_ranges() -> None:
    spec = _spec()
    levels = np.linspace(100.0, 130.0, 60)
    out = overlay_signals(spec, trend_levels=None, hedge_levels=np.ascontiguousarray(levels))
    assert out["trend_signal"] == 0.0
    assert out["trend_vol_scale"] == 1.0
    assert out["regime_fraction"] == 0.0


def _bootstrap(lo: int = 10, r: int = 30, **over: object) -> tuple[object, object, object, object, object]:
    spec = _spec()
    cube = _cube()
    panel = _panel(cube)
    scorer = _scorer()
    scores = _scores(cube, panel, scorer)
    sleeves, _ = _research_reference(spec, cube, panel, scores, lo=lo, hi=r)
    state = bootstrap_state(spec, cube, sleeves, _incumbents_for(spec, scores, cube, panel, lo=lo, hi=r), lo=lo, last_row=r)
    return spec, cube, panel, scorer, state


def test_bootstrap_guards() -> None:
    spec, cube, panel, scorer, state = _bootstrap()
    scores = _scores(cube, panel, scorer)
    sleeves, _ = _research_reference(spec, cube, panel, scores, lo=10, hi=30)
    incumb = _incumbents_for(spec, scores, cube, panel, lo=10, hi=30)
    with pytest.raises(ValueError, match="sleeves"):
        bootstrap_state(spec, cube, sleeves[:1], incumb, lo=10, last_row=30)
    with pytest.raises(ValueError, match="sleeves"):
        bootstrap_state(spec, cube, sleeves, incumb[:1], lo=10, last_row=30)
    with pytest.raises(ValueError, match="lo"):
        bootstrap_state(spec, cube, sleeves, incumb, lo=0, last_row=30)
    with pytest.raises(ValueError, match="last_row"):
        bootstrap_state(spec, cube, sleeves, incumb, lo=10, last_row=8)


def test_bootstrap_sleeve_without_decisions() -> None:
    spec = _spec()
    cube = _cube()
    state = bootstrap_state(spec, cube, ({}, {}), (frozenset(), frozenset()), lo=10, last_row=30)
    assert all(sleeve.decided_session is None for sleeve in state.sleeves)
    assert all(sleeve.weights == {} for sleeve in state.sleeves)
    assert state.anchor_session == cube.sessions[9]
    assert state.sessions_since_anchor == 21


def test_advance_guards() -> None:
    from dataclasses import replace

    spec, cube, panel, scorer, state = _bootstrap()
    other = SimpleNamespace(policy=spec.policy, book=spec.book, spec_hash="other",
                            trend_overlay=None, regime_hedge=None)
    with pytest.raises(ValueError, match="spec hash"):
        advance(other, spec, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    with pytest.raises(ValueError, match="capital_krw"):
        advance(state, spec, cube, panel, scorer, capital_krw=0, cash_buffer=BUFFER)
    with pytest.raises(ValueError, match="cash_buffer"):
        advance(state, spec, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=1.0)
    bad_every = SimpleNamespace(policy=SimpleNamespace(rebalance_every_sessions=0), book=spec.book,
                                spec_hash=state.spec_hash, trend_overlay=None, regime_hedge=None)
    with pytest.raises(ValueError, match="rebalance"):
        advance(state, bad_every, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    slim_spec = SimpleNamespace(policy=spec.policy,
                                book=SimpleNamespace(sleeves=1, stock_capital_fraction=0.75, rebalance_band=0.0),
                                spec_hash=state.spec_hash, trend_overlay=None, regime_hedge=None)
    with pytest.raises(ValueError, match="sleeves"):
        advance(state, slim_spec, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    shuffled = replace(state, sleeves=(state.sleeves[1], state.sleeves[0]))
    with pytest.raises(ValueError, match="phase"):
        advance(shuffled, spec, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    short_panel = _panel(_prefix(cube, 40))
    with pytest.raises(ValueError, match="last_row"):
        advance(state, spec, cube, short_panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    narrow_feats = {name: mat[:40] for name, mat in panel.features.items()}
    narrow = FeaturePanel(features=narrow_feats, labels={}, last_row=len(cube.sessions) - 1)
    with pytest.raises(ValueError, match="aligned"):
        advance(state, spec, cube, narrow, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)


def test_advance_at_end_is_a_noop() -> None:
    spec, cube, panel, scorer, _ = _bootstrap()
    end = len(cube.sessions) - 1
    scores = _scores(cube, panel, scorer)
    sleeves, _ = _research_reference(spec, cube, panel, scores, lo=10, hi=end)
    state = bootstrap_state(spec, cube, sleeves, _incumbents_for(spec, scores, cube, panel, lo=10, hi=end),
                            lo=10, last_row=end)
    nxt, decisions = advance(state, spec, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)
    assert decisions == ()
    assert nxt == state


def _trend_spec(**over: object) -> object:
    from src.research.trend_overlay import TrendOverlaySpec

    kwargs: dict[str, object] = {
        "signal": "ma", "ma_sessions": 5,
        "long_fraction": 0.5, "short_fraction": 0.5, "rebalance_every_sessions": 5,
        "contract_multiplier_krw": 10000, "initial_margin_rate": 0.2, "margin_buffer_rate": 0.1,
        "margin_topup_trigger_fraction": 0.75, "futures_cost_rate": 0.0003, "futures_tax_rate": 0.1,
        "futures_annual_deduction_krw": 0,
    }
    kwargs.update(over)
    return TrendOverlaySpec(**kwargs)  # type: ignore[arg-type]


def _regime_spec(**over: object) -> object:
    from src.research.regime_hedge import RegimeHedgeSpec

    kwargs: dict[str, object] = {
        "tsmom_horizons": (5, 10), "target_vol": 0.15, "max_fraction": 0.5, "vol_window_sessions": 10,
        "rebalance_every_sessions": 5, "contract_multiplier_krw": 10000,
        "initial_margin_rate": 0.2, "margin_buffer_rate": 0.1,
        "margin_topup_trigger_fraction": 0.75, "futures_cost_rate": 0.0003, "futures_tax_rate": 0.1,
        "futures_annual_deduction_krw": 0,
    }
    kwargs.update(over)
    return RegimeHedgeSpec(**kwargs)  # type: ignore[arg-type]


def test_overlay_with_trend_and_regime() -> None:
    up = np.linspace(100.0, 160.0, 60)
    down = np.linspace(160.0, 100.0, 60)
    spec = _spec()
    spec.trend_overlay = _trend_spec()
    out = overlay_signals(spec, trend_levels=np.ascontiguousarray(up), hedge_levels=np.ascontiguousarray(up))
    assert out["trend_signal"] == 1.0
    assert out["trend_vol_scale"] == pytest.approx(1.0)
    assert out["regime_fraction"] == 0.0
    spec.regime_hedge = _regime_spec()
    out = overlay_signals(spec, trend_levels=np.ascontiguousarray(up), hedge_levels=np.ascontiguousarray(down))
    assert 0.0 < out["regime_fraction"] <= 0.5
    tsmom = _spec()
    tsmom.trend_overlay = _trend_spec(signal="tsmom", tsmom_horizons=(5, 10), ma_sessions=None)
    out = overlay_signals(tsmom, trend_levels=np.ascontiguousarray(up), hedge_levels=np.ascontiguousarray(up))
    assert out["trend_signal"] == pytest.approx(1.0)
    capped = _spec()
    capped.trend_overlay = _trend_spec(vol_cap=0.01, vol_window_sessions=10)
    out = overlay_signals(capped, trend_levels=np.ascontiguousarray(up), hedge_levels=np.ascontiguousarray(up))
    assert 0.0 < out["trend_vol_scale"] <= 1.0
    short = _spec()
    short.trend_overlay = _trend_spec()
    out = overlay_signals(short, trend_levels=np.ascontiguousarray(up[:3]), hedge_levels=np.ascontiguousarray(up[:3]))
    assert out["trend_signal"] == 0.0
    assert out["trend_vol_scale"] == 1.0
    with pytest.raises(PITDataError, match="trend"):
        overlay_signals(short, trend_levels=None, hedge_levels=np.ascontiguousarray(up))


def test_chain_error_branches(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    seal(missing, _sealed(date(2021, 6, 1)))
    (missing / "2021-06-01.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        read_chain(missing)
    garbage = tmp_path / "garbage"
    garbage.mkdir()
    (garbage / "chain.jsonl").write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        read_chain(garbage)
    short = tmp_path / "short"
    short.mkdir()
    (short / "chain.jsonl").write_text('{"data_session": "2021-06-01"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        read_chain(short)
    tampered = tmp_path / "tampered"
    first = seal(tampered, _sealed(date(2021, 6, 1)))
    lines = (tampered / "chain.jsonl").read_text(encoding="utf-8").splitlines()
    broken = dict(json.loads(lines[-1]))
    broken["prev_hash"] = "f" * 64
    (tampered / "chain.jsonl").write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken"):
        read_chain(tampered)
    assert first.hash != ""


def test_chain_path_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seal(tmp_path, _sealed(date(2021, 6, 1)))
    real = Path.read_text

    def _boom(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "chain.jsonl":
            raise OSError("denied")
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", _boom)
    with pytest.raises(ValueError, match="unreadable"):
        read_chain(tmp_path)


def test_seal_unreadable_existing_file(tmp_path: Path) -> None:
    (tmp_path / "2021-06-01.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        seal(tmp_path, _sealed(date(2021, 6, 1)))


def test_load_state_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unreadable"):
        load_state(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "other", "sleeves": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_state(bad)
    broken = tmp_path / "broken.json"
    broken.write_text('{"schema": "operate-v1", "sleeves": [{}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_state(broken)


def test_overlay_vol_unknown_and_regime_flat() -> None:
    up = np.linspace(100.0, 160.0, 60)
    capped = _spec()
    capped.trend_overlay = _trend_spec(vol_cap=0.5, vol_window_sessions=10)
    out = overlay_signals(
        capped,
        trend_levels=np.ascontiguousarray(up[:5]),
        hedge_levels=np.ascontiguousarray(up[:5]),
    )
    assert out["trend_vol_scale"] == 1.0
    spec = _spec()
    spec.trend_overlay = _trend_spec()
    spec.regime_hedge = _regime_spec()
    out = overlay_signals(spec, trend_levels=np.ascontiguousarray(up), hedge_levels=np.ascontiguousarray(up))
    assert out["regime_fraction"] == 0.0
    out = overlay_signals(
        spec, trend_levels=np.ascontiguousarray(up[:5]), hedge_levels=np.ascontiguousarray(up[:5])
    )
    assert out["regime_fraction"] == 0.0
    thin = _spec()
    thin.trend_overlay = _trend_spec()
    thin.regime_hedge = _regime_spec(tsmom_horizons=(1, 2))
    out = overlay_signals(
        thin,
        trend_levels=np.ascontiguousarray(up[:5]),
        hedge_levels=np.ascontiguousarray(np.linspace(160.0, 100.0, 3)),
    )
    assert out["regime_fraction"] == 0.0


def test_chain_blank_lines_and_bad_hash(tmp_path: Path) -> None:
    first = seal(tmp_path, _sealed(date(2021, 6, 1)))
    with (tmp_path / "chain.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert len(read_chain(tmp_path)) == 1
    lines = (tmp_path / "chain.jsonl").read_text(encoding="utf-8").splitlines()
    corrupted = dict(json.loads(lines[0]))
    corrupted["targets"] = {"KRX:000001": 0.99}
    (tmp_path / "chain.jsonl").write_text(json.dumps(corrupted) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken"):
        read_chain(tmp_path)
    assert first.hash != ""


def test_advance_rejects_internal_calendar_gap() -> None:
    spec, cube, panel, scorer, state = _bootstrap()
    keep = np.arange(len(cube.sessions)) != 32
    gapped = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=[day for idx, day in enumerate(cube.sessions) if keep[idx]],
        instrument_ids=cube.instrument_ids,
        arrays={name: arr[keep] for name, arr in cube.arrays.items()},
        exit_at=cube.exit_at,
        exit_halted=cube.exit_halted,
    )
    with pytest.raises(PITDataError, match="skips"):
        advance(state, spec, gapped, _panel(gapped), scorer, capital_krw=CAPITAL, cash_buffer=BUFFER)


def test_five_sleeve_mean_is_bitwise_research_mean() -> None:
    from src.research.book import BookSpec

    spec = _spec()
    spec.book = BookSpec(sleeves=5, stock_capital_fraction=0.75)
    spec.policy = spec.policy.model_copy(update={"rebalance_every_sessions": 5, "n": 3})
    cube = _prefix(_cube(), 31)
    panel = _panel(cube)
    weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])
    targets = [{30: weights} for _ in range(5)]
    state = bootstrap_state(spec, cube, targets, [frozenset({0, 1, 2})] * 5, lo=10, last_row=30)
    final, decisions = advance(state, spec, cube, panel, _scorer(), capital_krw=CAPITAL, cash_buffer=BUFFER)
    expected = combine_sleeve_targets(
        [{31: np.array([sleeve.weights.get(inst, 0.0) for inst in cube.instrument_ids])}
         for sleeve in final.sleeves], lo=32, hi=32,
    )[31]
    actual = np.array([decisions[0][1].get(inst, 0.0) for inst in cube.instrument_ids])
    np.testing.assert_array_equal(actual, expected)


def test_seal_recovers_file_written_before_chain(tmp_path: Path) -> None:
    decision = _sealed(date(2021, 6, 1))
    first = seal(tmp_path, decision)
    (tmp_path / "chain.jsonl").write_text("", encoding="utf-8")
    assert seal(tmp_path, decision) == first
    assert read_chain(tmp_path) == (first,)


def test_seal_refuses_orphan_with_wrong_link(tmp_path: Path) -> None:
    from dataclasses import replace
    from src.research.operate import _decision_hash, _decision_payload

    decision = _sealed(date(2021, 6, 1))
    first = seal(tmp_path, decision)
    (tmp_path / "chain.jsonl").write_text("", encoding="utf-8")
    orphan = replace(first, prev_hash="f" * 64)
    payload = _decision_payload(orphan)
    payload["hash"] = _decision_hash(orphan.prev_hash, payload)
    (tmp_path / "2021-06-01.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="orphan"):
        seal(tmp_path, decision)


def test_chain_rejects_reordered_sessions(tmp_path: Path) -> None:
    from dataclasses import replace
    from src.research.operate import _decision_hash, _decision_payload

    first = seal(tmp_path, _sealed(date(2021, 6, 2)))
    earlier = _sealed(date(2021, 6, 1))
    with pytest.raises(ValueError, match="increasing"):
        seal(tmp_path, earlier)
    earlier = replace(earlier, prev_hash=first.hash)
    payload = _decision_payload(earlier)
    payload["hash"] = _decision_hash(earlier.prev_hash, payload)
    with (tmp_path / "chain.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="increasing"):
        read_chain(tmp_path)


def test_seal_rejects_wrong_execution_session(tmp_path: Path) -> None:
    from dataclasses import replace

    decision = replace(_sealed(date(2021, 6, 1)), execution_session=date(2021, 6, 3))
    with pytest.raises(ValueError, match="next XKRX"):
        seal(tmp_path, decision)
    assert not (tmp_path / "chain.jsonl").exists()


def test_state_rejects_corrupt_counts_and_mutable_weights(tmp_path: Path) -> None:
    from dataclasses import replace
    from src.research.operate import SleeveState

    spec, cube, panel, scorer, state = _bootstrap()
    with pytest.raises(ValueError, match="non-negative"):
        replace(state, sessions_since_anchor=-1)
    with pytest.raises(PITDataError, match="count"):
        advance(replace(state, sessions_since_anchor=1), spec, cube, panel, scorer,
                capital_krw=CAPITAL, cash_buffer=BUFFER)
    with pytest.raises(ValueError, match="sorted"):
        SleeveState(0, ("KRX:000002", "KRX:000001"), {}, None)
    with pytest.raises(ValueError, match="finite"):
        SleeveState(0, (), {"KRX:000001": float("nan")}, None)
    weights = {"KRX:000001": 0.5}
    sleeve = SleeveState(0, (), weights, None)
    weights["KRX:000001"] = 0.9
    assert sleeve.weights["KRX:000001"] == 0.5
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_state(path)


def test_regime_zero_volatility_keeps_capped_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.research.trend_overlay as trend_module

    spec = _spec()
    spec.regime_hedge = _regime_spec()
    monkeypatch.setattr(trend_module, "realised_vol", lambda levels, idx, window: 0.0)
    out = overlay_signals(spec, trend_levels=None, hedge_levels=np.linspace(160.0, 100.0, 60))
    assert out["regime_fraction"] == 0.5


@pytest.mark.parametrize("name", ["cash_returns", "hedge_levels", "hedge_inverse", "trend_levels"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_advance_validates_every_pending_series_before_scoring(
    name: str, invalid: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.research.operate as operate_module

    spec, cube, panel, scorer, state = _bootstrap()
    spec.trend_overlay = _trend_spec()
    series = _series(cube)
    series[name][-1] = invalid
    before = state_digest(state)

    def forbidden_score(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid catch-up series reached scoring")

    monkeypatch.setattr(operate_module, "score_rows", forbidden_score)
    with pytest.raises(PITDataError, match=name):
        _advance(state, spec, cube, panel, scorer, capital_krw=CAPITAL,
                 cash_buffer=BUFFER, series=series)
    assert state_digest(state) == before


@pytest.mark.parametrize("name", ["cash_returns", "hedge_levels", "hedge_inverse", "trend_levels"])
def test_advance_rejects_missing_and_misaligned_series(name: str) -> None:
    spec, cube, panel, scorer, state = _bootstrap()
    spec.trend_overlay = _trend_spec()
    series = _series(cube)
    del series[name]
    with pytest.raises(PITDataError, match=name):
        _advance(state, spec, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER, series=series)
    series[name] = np.ones(len(cube.sessions) - 1)
    with pytest.raises(PITDataError, match=name):
        _advance(state, spec, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER, series=series)


@pytest.mark.parametrize("name", ["hedge_levels", "hedge_inverse", "trend_levels"])
def test_advance_rejects_nonpositive_levels(name: str) -> None:
    spec, cube, panel, scorer, state = _bootstrap()
    spec.trend_overlay = _trend_spec()
    series = _series(cube)
    series[name][-1] = 0.0
    with pytest.raises(PITDataError, match=name):
        _advance(state, spec, cube, panel, scorer, capital_krw=CAPITAL, cash_buffer=BUFFER, series=series)
