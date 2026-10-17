"""Investor-flow session-grid causality and perturbation invariance."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from src.core.time import KRX_TZ
from src.data.investor_flow_silver import InvestorFlowSilverPolicy, flow_availability_frame
from src.research.cube import ResearchCube, assemble_flows
from src.research.panel import build_panel

N_SESS = 30
INSTS = ("KRX:000001", "KRX:000002")
_POLICY = InvestorFlowSilverPolicy()


def _sessions() -> list[date]:
    start = date(2020, 1, 2)
    return [start + timedelta(days=i) for i in range(N_SESS)]


def _stamp(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 8, 0, tzinfo=KRX_TZ)


def _stamps(sessions: list[date]) -> dict[date, datetime]:
    frame = flow_availability_frame(sessions, lag=_POLICY.available_session_lag)
    return {
        session: _stamp(available)
        for session, available in zip(frame["session"].to_list(), frame["available_session"].to_list(), strict=True)
        if available is not None
    }


def _close(n_s: int = N_SESS, n_n: int = 1) -> np.ndarray:
    return np.full((n_s, n_n), 5000.0)


def _single_cell(sessions: list[date], source: int, stamp: datetime) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session": [sessions[source]],
            "instrument_id": [INSTS[0]],
            "foreign_net_shares": [100],
            "institution_net_shares": [200],
            "individual_net_shares": [-50],
            "available_at": [stamp],
        },
        schema={
            "session": pl.Date,
            "instrument_id": pl.String,
            "foreign_net_shares": pl.Int64,
            "institution_net_shares": pl.Int64,
            "individual_net_shares": pl.Int64,
            "available_at": pl.Datetime("us", "Asia/Seoul"),
        },
    )


def _finite_rows(arrays: dict[str, np.ndarray]) -> list[int]:
    rows = set()
    for arr in arrays.values():
        rows.update(int(i) for i in np.where(np.isfinite(arr[:, 0]))[0].tolist())
    return sorted(rows)


def test_flow_source_session_strictly_before_row() -> None:
    sessions = _sessions()
    stamps = _stamps(sessions)
    empty = _single_cell(sessions, 0, stamps[sessions[0]]).clear()
    for k in range(N_SESS):
        if sessions[k] in stamps:
            frame, expected = _single_cell(sessions, k, stamps[sessions[k]]), [k + 1]
        else:
            frame, expected = empty, []
        arrays = assemble_flows(frame, _close(), sessions=sessions, instrument_ids=[INSTS[0]])
        assert _finite_rows(arrays) == expected


def test_stamp_after_decision_clock_moves_one_row() -> None:
    sessions = _sessions()
    late = datetime(2020, 1, 2 + 6, 18, 0, 0, 1, tzinfo=KRX_TZ)
    arrays = assemble_flows(_single_cell(sessions, 5, late), _close(), sessions=sessions, instrument_ids=[INSTS[0]])
    assert _finite_rows(arrays) == [7]


def _full_frame(sessions: list[date]) -> pl.DataFrame:
    stamps = _stamps(sessions)
    rows = [
        {
            "session": session,
            "ticker": ticker,
            "foreign_net_shares": 100,
            "institution_net_shares": 200,
            "individual_net_shares": -50,
            "available_at": stamps[session],
        }
        for session in sessions
        if session in stamps
        for ticker in ("000001", "000002")
    ]
    return pl.DataFrame(
        rows,
        schema={
            "session": pl.Date,
            "ticker": pl.String,
            "foreign_net_shares": pl.Int64,
            "institution_net_shares": pl.Int64,
            "individual_net_shares": pl.Int64,
            "available_at": pl.Datetime("us", "Asia/Seoul"),
        },
    )


def _cube_with_flows(sessions: list[date], flows: dict[str, np.ndarray]) -> ResearchCube:
    from tests.unit.research.test_panel import _synthetic_cube

    base = _synthetic_cube(n_s=len(sessions), n_n=2)
    arrays = dict(base.arrays)
    arrays.update(flows)
    return ResearchCube.from_arrays(
        cube_id=base.cube_id,
        sessions=list(sessions),
        instrument_ids=list(base.instrument_ids),
        arrays=arrays,
        exit_at=np.asarray(base.exit_at),
        exit_halted=np.asarray(base.exit_halted),
    )


def _equal_with_nan(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return bool(
        a.shape == b.shape
        and np.array_equal(np.isnan(a), np.isnan(b))
        and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
    )


def _perturb_after(frame: pl.DataFrame, sessions: list[date], cut: int, *, seed: int, delete: bool) -> pl.DataFrame:
    from src.research.cube import _decision_at

    horizon = _decision_at(sessions[cut])
    rng = np.random.default_rng(seed)
    rows = frame.to_dicts()
    future = [row for row in rows if row["available_at"] > horizon]
    if delete:
        rows = [row for row in rows if row["available_at"] <= horizon]
    else:
        for row in future:
            row["foreign_net_shares"] = int(rng.integers(-10_000, 10_000))
            row["institution_net_shares"] = int(rng.integers(-10_000, 10_000))
            row["individual_net_shares"] = int(rng.integers(-10_000, 10_000))
    return pl.DataFrame(rows, schema=frame.schema)


def test_flow_perturbation_invariance() -> None:
    """Future-stamped flow must not move sealed rows; scores compose via test_model.py::test_year_scores_ignore_the_future."""
    sessions = _sessions()
    cut = 20
    close = np.full((N_SESS, 2), 5000.0)
    base_frame = _full_frame(sessions)
    base_arrays = assemble_flows(base_frame, close, sessions=sessions, instrument_ids=list(INSTS))
    base_cube = _cube_with_flows(sessions, base_arrays)
    base_panel = build_panel(base_cube, last_row=cut)
    for seed, delete in ((11, False), (22, True)):
        other_arrays = assemble_flows(
            _perturb_after(base_frame, sessions, cut, seed=seed, delete=delete),
            close,
            sessions=sessions,
            instrument_ids=list(INSTS),
        )
        for name in base_arrays:
            assert _equal_with_nan(np.asarray(base_arrays[name][: cut + 1]), np.asarray(other_arrays[name][: cut + 1]))
        other_panel = build_panel(_cube_with_flows(sessions, other_arrays), last_row=cut)
        for name in ("fl_for5", "fl_for20", "fl_for60", "fl_ins5", "fl_ins20", "fl_ins60", "fl_ind5", "fl_ind20", "fl_ind60"):
            assert _equal_with_nan(
                np.asarray(base_panel.features[name][: cut + 1]), np.asarray(other_panel.features[name][: cut + 1])
            )


def test_flow_perturbation_not_vacuous() -> None:
    sessions = _sessions()
    cut = 20
    close = np.full((N_SESS, 2), 5000.0)
    base_frame = _full_frame(sessions)
    base_arrays = assemble_flows(base_frame, close, sessions=sessions, instrument_ids=list(INSTS))
    rows = base_frame.to_dicts()
    for row in rows:
        if row["session"] == sessions[cut - 1] and row["ticker"] == "000001":
            row["foreign_net_shares"] = 9_999_999
    other_arrays = assemble_flows(pl.DataFrame(rows, schema=base_frame.schema), close, sessions=sessions, instrument_ids=list(INSTS))
    changed = any(
        not _equal_with_nan(np.asarray(base_arrays[name][: cut + 1]), np.asarray(other_arrays[name][: cut + 1]))
        for name in base_arrays
    )
    assert changed
