"""Dense, read-only session x instrument research arrays with structural PIT causality."""

from __future__ import annotations

import bisect
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from src.backtest.market import load_market_arrays
from src.core.market_rules import KrxMarket, KrxMarketRules
from src.core.pit import PITDataError
from src.core.time import KRX_TZ
from src.data.datasets import load_manifest

__all__ = [
    "REVISION",
    "CubeInputs",
    "ResearchCube",
    "assemble_dividends",
    "assemble_flows",
    "assemble_fundamentals",
    "assemble_releases",
    "build_research_cube",
    "load_research_cube",
    "research_cube_id",
]

REVISION = "research-cube-v2"

#: Cache layout version. Format 1 was a single ``<cube_id>.npz``; format 2 is a directory of ``.npy`` arrays
#: that can be memory-mapped instead of read into anonymous memory.
_CACHE_FORMAT = 2
_CHECKSUM_CHUNK_BYTES = 1 << 23

_LOG = logging.getLogger(__name__)

_FISCAL_RE = re.compile(r"^(\d{4})Q([1-4])$")
_FLOW_FACTS: tuple[str, ...] = (
    "sales",
    "gross_profit",
    "operating_profit",
    "net_income",
    "operating_cash_flow",
    "capex",
)
_STOCK_FACTS: tuple[str, ...] = ("assets", "equity", "debt", "cash")
_RELEASE_METRICS: tuple[str, ...] = ("sales", "operating_profit", "net_income")
_FACT_TO_COLUMN: dict[str, str] = {
    "sales": "sales",
    "gross_profit": "gross_profit",
    "operating_profit": "operating_profit",
    "net_income": "net_income",
    "operating_cash_flow": "operating_cash_flow",
    "capex": "capex",
    "assets": "assets",
    "equity": "equity",
    "debt": "debt",
    "cash": "cash",
}


@dataclass(frozen=True, slots=True)
class CubeInputs:
    """Verified upstream datasets and run constants that fully determine a cube."""

    market_panel: Path
    dividend_events: Path
    financial_facts: Path
    investor_flow: Path
    earnings_releases: Path
    market_rules: KrxMarketRules
    dividend_withholding_rate: Decimal


@dataclass(frozen=True, slots=True)
class ResearchCube:
    """Dense, read-only session x instrument research arrays.

    Row ``t`` of every array holds only information observable at session ``t`` 18:00 KST,
    except the execution arrays (``open``, ``r_on``, ``r_id``, ``tick_at_open``, lock flags),
    which describe session ``t`` itself and are consumed only when executing decisions made at
    ``t-1``.

    Attributes:
        cube_id: ``research_cube_<hash16>`` over policy version, input dataset ids, rules
            fingerprint and withholding rate.
        sessions: Ascending panel sessions (length S).
        instrument_ids: Ascending instrument ids (length N), identical to the market panel.
        arrays: Named S x N arrays (see invariants); float64 unless stated, NaN = unknown.
        exit_at: int64 (N,) session index at which a delisted instrument is settled
            (last session + 1), -1 when it never exits inside the panel.
        exit_halted: bool (N,) True when the final session had no trading volume.
    """

    cube_id: str
    sessions: tuple[date, ...]
    instrument_ids: tuple[str, ...]
    arrays: Mapping[str, NDArray[Any]]
    exit_at: NDArray[np.int64]
    exit_halted: NDArray[np.bool_]

    @classmethod
    def from_arrays(
        cls,
        *,
        cube_id: str,
        sessions: Sequence[date],
        instrument_ids: Sequence[str],
        arrays: Mapping[str, NDArray[Any]],
        exit_at: NDArray[np.int64],
        exit_halted: NDArray[np.bool_],
    ) -> ResearchCube:
        sess = tuple(sessions)
        insts = tuple(instrument_ids)
        shape = (len(sess), len(insts))
        frozen: dict[str, NDArray[Any]] = {}
        for name, arr in arrays.items():
            arr = np.ascontiguousarray(arr)
            if arr.shape != shape:
                raise PITDataError(f"research cube array has wrong shape: {name}")
            arr.flags.writeable = False
            frozen[name] = arr
        exit_at_c = np.ascontiguousarray(exit_at, dtype=np.int64)
        halted_c = np.ascontiguousarray(exit_halted, dtype=np.bool_)
        if exit_at_c.shape != (len(insts),) or halted_c.shape != (len(insts),):
            raise PITDataError("research cube exit arrays have wrong shape")
        exit_at_c.flags.writeable = False
        halted_c.flags.writeable = False
        return cls(
            cube_id=cube_id,
            sessions=sess,
            instrument_ids=insts,
            arrays=frozen,
            exit_at=exit_at_c,
            exit_halted=halted_c,
        )


def _rules_fingerprint(rules: KrxMarketRules) -> str:
    return hashlib.sha256(repr(rules).encode("utf-8")).hexdigest()[:16]


def research_cube_id(inputs: CubeInputs) -> str:
    """Return ``research_cube_<hash16>`` for the given inputs."""
    payload = "|".join(
        [
            REVISION,
            Path(inputs.market_panel).name,
            Path(inputs.dividend_events).name,
            Path(inputs.financial_facts).name,
            Path(inputs.investor_flow).name,
            Path(inputs.earnings_releases).name,
            _rules_fingerprint(inputs.market_rules),
            str(inputs.dividend_withholding_rate),
        ]
    )
    return f"research_cube_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _decision_at(session_day: date) -> datetime:
    return datetime(session_day.year, session_day.month, session_day.day, 18, 0, tzinfo=KRX_TZ)


def _decision_times(sessions: Sequence[date]) -> list[datetime]:
    return [_decision_at(day) for day in sessions]


def _avail_index(available_at: Any, sessions: Sequence[date]) -> int | None:
    return _avail_index_in(available_at, _decision_times(sessions))


def _avail_index_in(available_at: Any, decision_times: Sequence[datetime]) -> int | None:
    """First session index whose 18:00 KST decision time is at or after ``available_at``."""
    try:
        if available_at is None:
            return None
        if isinstance(available_at, date) and not isinstance(available_at, datetime):
            at = datetime(available_at.year, available_at.month, available_at.day, tzinfo=KRX_TZ)
        else:
            at = available_at
            if not isinstance(at, datetime):
                return None
            at = at.replace(tzinfo=KRX_TZ) if at.tzinfo is None else at.astimezone(KRX_TZ)
    except (ValueError, TypeError, OverflowError):  # pragma: no cover
        return None
    idx = bisect.bisect_left(decision_times, at)
    return idx if idx < len(decision_times) else None


def _parse_qk(period: Any) -> tuple[int, int, int] | None:
    if not isinstance(period, str):
        return None
    match = _FISCAL_RE.fullmatch(period.strip())
    if match is None:
        return None
    year = int(match.group(1))
    quarter = int(match.group(2))
    return (year * 4 + quarter - 1, year, quarter)


def _session_qk(day: date) -> int:
    return day.year * 4 + (day.month - 1) // 3


def _instrument_of(row: dict[str, Any]) -> str | None:
    for key in ("instrument_id", "company_id"):
        value = row.get(key)
        if isinstance(value, str) and value.startswith("KRX:"):
            return value
    ticker = row.get("ticker")
    if isinstance(ticker, str) and re.fullmatch(r"\d{6}", ticker.strip()):
        return f"KRX:{ticker.strip()}"
    company = row.get("company_id")
    if isinstance(company, str) and re.fullmatch(r"\d{6}", company.strip()):
        return f"KRX:{company.strip()}"
    instrument = row.get("instrument_id")
    if isinstance(instrument, str) and re.fullmatch(r"\d{6}", instrument.strip()):
        return f"KRX:{instrument.strip()}"
    return None


def _coverage(shape: tuple[int, int], arr: NDArray[np.float64] | NDArray[np.bool_]) -> float:
    if isinstance(arr.dtype, np.dtype) and arr.dtype == np.bool_:
        return float(np.mean(arr)) if arr.size else 0.0
    finite = np.isfinite(np.asarray(arr, dtype=np.float64))
    return float(np.mean(finite)) if finite.size else 0.0


def _log_phase(name: str, arrays: Mapping[str, NDArray[Any]], shape: tuple[int, int]) -> None:
    parts: list[str] = []
    for key in sorted(arrays):
        arr = arrays[key]
        if arr.shape == shape:
            parts.append(f"{key}={_coverage(shape, arr):.3f}")
            if len(parts) >= 4:
                break
    _LOG.info("[DATA] stage=%s shape=%dx%d %s", name, shape[0], shape[1], " ".join(parts))


def _quarter_fact_map(
    facts: pl.DataFrame, sessions: Sequence[date], instrument_ids: Sequence[str]
) -> tuple[
    dict[tuple[int, int], dict[str, tuple[float, int]]],
    dict[tuple[int, int], int],
]:
    """Collect preferred-basis three-month/stock values per (instrument, quarter)."""
    index_of = {inst: idx for idx, inst in enumerate(instrument_ids)}
    grouped: dict[tuple[int, int, str], list[tuple[int, int, float, bool]]] = {}
    order = 0
    times = _decision_times(sessions)
    try:
        rows = facts.to_dicts()
    except (ValueError, pl.exceptions.PolarsError) as exc:  # pragma: no cover
        raise PITDataError(f"invalid financial facts frame: {exc}") from exc
    for row in rows:
        order += 1
        inst = _instrument_of(row)
        if inst is None or inst not in index_of:
            continue
        parsed = _parse_qk(row.get("fiscal_period"))
        if parsed is None:
            continue
        qk, _, _ = parsed
        fact = str(row.get("fact") or "").strip()
        if fact not in _FACT_TO_COLUMN:
            continue
        value = row.get("value")
        if value is None:
            continue
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if amount != amount or amount in (float("inf"), float("-inf")):
            continue
        avail = _avail_index_in(row.get("available_at"), times)
        if avail is None:
            continue
        consolidated = row.get("consolidated")
        is_consolidated = bool(consolidated) if consolidated is not None else True
        grouped.setdefault((index_of[inst], qk, fact), []).append((avail, order, amount, is_consolidated))
    quarter_fact: dict[tuple[int, int], dict[str, tuple[float, int]]] = {}
    for (nidx, qk, fact), entries in grouped.items():
        has_consolidated = any(item[3] for item in entries)
        candidates = [item for item in entries if item[3]] if has_consolidated else entries
        best = max(candidates, key=lambda item: (item[0], item[1]))
        quarter_fact.setdefault((nidx, qk), {})[fact] = (best[2], best[0])
    quarter_avail: dict[tuple[int, int], int] = {}
    for key, values in quarter_fact.items():
        quarter_avail[key] = max(avail for _, avail in values.values())
    return quarter_fact, quarter_avail


def assemble_fundamentals(
    facts: pl.DataFrame, *, sessions: Sequence[date], instrument_ids: Sequence[str]
) -> dict[str, NDArray[np.float64]]:
    """Build as-of fundamental arrays with Q4 derivation and largest-quarter-wins display."""
    sess = list(sessions)
    insts = list(instrument_ids)
    n_s, n_n = len(sess), len(insts)
    shape = (n_s, n_n)
    quarter_fact, quarter_avail = _quarter_fact_map(facts, sess, insts)
    sess_qk = np.asarray([_session_qk(day) for day in sess], dtype=np.int64)

    def _three_month(nidx: int, qk: int, fact: str) -> tuple[float, int | None]:
        parsed_year = qk // 4
        quarter = qk - parsed_year * 4 + 1
        if quarter == 4:
            annual = quarter_fact.get((nidx, qk), {}).get(fact)
            parts: list[tuple[float, int]] = []
            for back in (3, 2, 1):
                entry = quarter_fact.get((nidx, qk - back), {}).get(fact)
                if entry is None:
                    return (float("nan"), None)
                parts.append(entry)
            if annual is None:
                return (float("nan"), None)
            derived = annual[0] - sum(item[0] for item in parts)
            avail = max([annual[1]] + [item[1] for item in parts])
            return (derived, avail)
        entry = quarter_fact.get((nidx, qk), {}).get(fact)
        if entry is None:
            return (float("nan"), None)
        return entry

    out: dict[str, NDArray[np.float64]] = {}
    stock_cols = [f"f_{name}" for name in ("assets", "equity", "debt", "cash")]
    for name in [*stock_cols, "f_assets_ly"]:
        out[name] = np.full(shape, np.nan, dtype=np.float64)
    for flow in _FLOW_FACTS:
        for suffix in ("_q", "_q_ly", "_ttm", "_ttm_ly"):
            out[f"f_{flow}{suffix}"] = np.full(shape, np.nan, dtype=np.float64)
    out["f_qk"] = np.full(shape, np.nan, dtype=np.float64)
    out["f_age_q"] = np.full(shape, np.nan, dtype=np.float64)
    out["f_avail_t"] = np.full(shape, np.nan, dtype=np.float64)

    qk_min: int | None = None
    qk_max: int | None = None
    if quarter_avail:
        qk_min = min(qk for _, qk in quarter_avail)
        qk_max = max(qk for _, qk in quarter_avail)
    if qk_min is None or qk_max is None or n_s == 0 or n_n == 0:
        _log_phase("fundamentals", out, shape)
        return out
    span = qk_max - qk_min + 1

    owned_by_inst: dict[int, list[int]] = {}
    for n_key, qk_key in quarter_avail:
        owned_by_inst.setdefault(n_key, []).append(qk_key)
    t_idx = np.arange(n_s, dtype=np.int64)
    never = np.iinfo(np.int64).max

    for nidx, owned_list in owned_by_inst.items():
        owned = sorted(owned_list)
        # 분기 위치별 값과 그 값이 처음 관측 가능해진 세션. 표시 시점 t에서 가용 세션 > t인 값은 숨긴다
        # (늦게 공시된 과거 분기·Q4 환산에 쓰인 늦은 분기가 앞선 시점에 새지 않도록).
        stock_val = {name: np.full(span, np.nan) for name in _STOCK_FACTS}
        stock_avail = {name: np.full(span, never, dtype=np.int64) for name in _STOCK_FACTS}
        flow_val = {flow: np.full(span, np.nan) for flow in _FLOW_FACTS}
        flow_avail = {flow: np.full(span, never, dtype=np.int64) for flow in _FLOW_FACTS}
        table_avail = np.full(span, -1, dtype=np.int64)
        for qk in owned:
            pos = qk - qk_min
            facts_here = quarter_fact.get((nidx, qk), {})
            for name in _STOCK_FACTS:
                entry = facts_here.get(name)
                if entry is not None:
                    stock_val[name][pos] = entry[0]
                    stock_avail[name][pos] = entry[1]
            for flow in _FLOW_FACTS:
                amount, avail = _three_month(nidx, qk, flow)
                if avail is not None and amount == amount:
                    flow_val[flow][pos] = amount
                    flow_avail[flow][pos] = avail
            table_avail[pos] = quarter_avail[(nidx, qk)]
        ev_avail = np.asarray([quarter_avail[(nidx, qk)] for qk in owned], dtype=np.int64)
        ev_qk = np.asarray(owned, dtype=np.int64)
        order = np.argsort(ev_avail, kind="stable")
        ev_avail = ev_avail[order]
        ev_qk = ev_qk[order]
        cum_best = np.maximum.accumulate(ev_qk)
        positions = np.searchsorted(ev_avail, t_idx, side="right") - 1
        best = np.where(positions >= 0, cum_best[np.clip(positions, 0, len(cum_best) - 1)], -1)
        best_pos = best - qk_min
        has = best >= 0
        cur = np.clip(best_pos, 0, span - 1)
        prev_pos = best_pos - 4
        in_range = has & (prev_pos >= 0)
        prev = np.clip(prev_pos, 0, span - 1)
        out["f_qk"][:, nidx] = np.where(has, best.astype(np.float64), np.nan)
        out["f_avail_t"][:, nidx] = np.where(has, table_avail[cur].astype(np.float64), np.nan)
        out["f_age_q"][:, nidx] = np.where(has, (sess_qk - best).astype(np.float64), np.nan)

        def _visible(values: NDArray[np.float64], avails: NDArray[np.int64], at: NDArray[np.int64],
                     ok: NDArray[np.bool_]) -> NDArray[np.float64]:
            return np.where(ok & (avails[at] <= t_idx), values[at], np.nan)

        for name in _STOCK_FACTS:
            out[f"f_{name}"][:, nidx] = _visible(stock_val[name], stock_avail[name], cur, has)
        out["f_assets_ly"][:, nidx] = _visible(stock_val["assets"], stock_avail["assets"], prev, in_range)
        for flow in _FLOW_FACTS:
            values, avails = flow_val[flow], flow_avail[flow]
            out[f"f_{flow}_q"][:, nidx] = _visible(values, avails, cur, has)
            out[f"f_{flow}_q_ly"][:, nidx] = _visible(values, avails, prev, in_range)
            # TTM: 연속 4개 분기 3개월치 합, 가용 세션은 네 값 중 가장 늦은 것
            ttm_val = np.full(span, np.nan)
            ttm_avail = np.full(span, never, dtype=np.int64)
            if span >= 4:
                windows = np.lib.stride_tricks.sliding_window_view(values, 4)
                win_avail = np.lib.stride_tricks.sliding_window_view(avails, 4)
                ttm_val[3:] = windows.sum(axis=1)
                ttm_avail[3:] = win_avail.max(axis=1)
            out[f"f_{flow}_ttm"][:, nidx] = _visible(ttm_val, ttm_avail, cur, has)
            out[f"f_{flow}_ttm_ly"][:, nidx] = _visible(ttm_val, ttm_avail, prev, in_range)
    _log_phase("fundamentals", out, shape)
    return out


def _release_tables(
    releases: pl.DataFrame,
    quarter_fact: dict[tuple[int, int], dict[str, tuple[float, int]]],
    sessions: Sequence[date],
    instrument_ids: Sequence[str],
) -> tuple[
    dict[tuple[int, int], list[tuple[int, int, dict[str, tuple[float | None, float | None]]]]],
    dict[tuple[int, int], int],
]:
    index_of = {inst: idx for idx, inst in enumerate(instrument_ids)}
    try:
        rows = releases.to_dicts()
    except (ValueError, pl.exceptions.PolarsError) as exc:  # pragma: no cover
        raise PITDataError(f"invalid earnings releases frame: {exc}") from exc
    per_key: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    order = 0
    times = _decision_times(sessions)
    for row in rows:
        order += 1
        inst = _instrument_of(row)
        if inst is None or inst not in index_of:
            continue
        parsed = _parse_qk(row.get("fiscal_period"))
        if parsed is None:
            continue
        qk, _, _ = parsed
        avail = _avail_index_in(row.get("available_at"), times)
        if avail is None:
            continue
        basis = str(row.get("basis") or "consolidated").strip().lower()
        if basis not in ("consolidated", "separate"):
            basis = "consolidated"
        row["_order"] = order
        row["_avail"] = avail
        per_key.setdefault((index_of[inst], qk, basis), []).append(row)
    versions: dict[tuple[int, int], list[tuple[int, int, dict[str, tuple[float | None, float | None]]]]] = {}
    first_avail: dict[tuple[int, int], int] = {}
    by_quarter: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = {}
    for (nidx, qk, basis), group in per_key.items():
        by_quarter.setdefault((nidx, qk), {})[basis] = group
    for (nidx, qk), bases in by_quarter.items():
        chosen = bases["consolidated"] if "consolidated" in bases else next(iter(bases.values()))
        ordered = sorted(chosen, key=lambda row: (int(row["_avail"]), int(row["_order"])))
        first_avail[(nidx, qk)] = int(ordered[0]["_avail"])
        for row in ordered:
            avail = int(row["_avail"])
            values: dict[str, tuple[float | None, float | None]] = {}
            kind = str(row.get("release_kind") or row.get("kind") or "").strip().lower()
            span = str(row.get("span") or "").strip().lower()
            metric = str(row.get("metric") or "").strip()
            is_prelim = kind == "preliminary" or (not kind and span == "quarter")
            is_profit = kind == "profit_change" or (not kind and span == "annual")
            if metric in _RELEASE_METRICS and (is_prelim or span == "quarter"):
                current = row.get("value_krw")
                prior = row.get("prior_year_value_krw")
                try:
                    cur = None if current is None else float(current)
                except (TypeError, ValueError):
                    cur = None
                try:
                    prv = None if prior is None else float(prior)
                except (TypeError, ValueError):
                    prv = None
                if cur is not None and cur != cur:
                    cur = None
                if prv is not None and prv != prv:
                    prv = None
                values[metric] = (cur, prv)
            elif metric in _RELEASE_METRICS and (is_profit or span == "annual"):
                annual_cur = row.get("value_krw")
                annual_prv = row.get("prior_year_value_krw")
                try:
                    annual_cur_f = None if annual_cur is None else float(annual_cur)
                except (TypeError, ValueError):
                    annual_cur_f = None
                try:
                    annual_prv_f = None if annual_prv is None else float(annual_prv)
                except (TypeError, ValueError):
                    annual_prv_f = None
                fact_name = metric
                cur_q4: float | None = None
                if annual_cur_f is not None and annual_cur_f == annual_cur_f:
                    parts: list[float] = []
                    ok = True
                    for back in (3, 2, 1):
                        entry = quarter_fact.get((nidx, qk - back), {}).get(fact_name)
                        if entry is None or entry[1] > avail:
                            ok = False
                            break
                        parts.append(entry[0])
                    if ok:
                        cur_q4 = annual_cur_f - sum(parts)
                prv_q4: float | None = None
                if annual_prv_f is not None and annual_prv_f == annual_prv_f:
                    parts_ly: list[float] = []
                    ok_ly = True
                    for back in (7, 6, 5):
                        entry = quarter_fact.get((nidx, qk - back), {}).get(fact_name)
                        if entry is None or entry[1] > avail:
                            ok_ly = False
                            break
                        parts_ly.append(entry[0])
                    if ok_ly:
                        prv_q4 = annual_prv_f - sum(parts_ly)
                values[metric] = (cur_q4, prv_q4)
            else:
                continue
            versions.setdefault((nidx, qk), []).append((avail, int(row["_order"]), values))
    return versions, first_avail


def assemble_releases(
    releases: pl.DataFrame,
    fundamentals: Mapping[str, NDArray[np.float64]],
    facts: pl.DataFrame,
    *,
    sessions: Sequence[date],
    instrument_ids: Sequence[str],
) -> dict[str, NDArray[np.float64]]:
    """Build early-release and merged best-available quarterly arrays."""
    sess = list(sessions)
    insts = list(instrument_ids)
    n_s, n_n = len(sess), len(insts)
    shape = (n_s, n_n)
    out: dict[str, NDArray[np.float64]] = {}
    for metric in _RELEASE_METRICS:
        out[f"e_{metric}_q"] = np.full(shape, np.nan, dtype=np.float64)
        out[f"e_{metric}_q_ly"] = np.full(shape, np.nan, dtype=np.float64)
    out["e_qk"] = np.full(shape, np.nan, dtype=np.float64)
    out["e_avail_t"] = np.full(shape, np.nan, dtype=np.float64)
    for metric in _RELEASE_METRICS:
        out[f"earn_{metric}_q"] = np.full(shape, np.nan, dtype=np.float64)
        out[f"earn_{metric}_q_ly"] = np.full(shape, np.nan, dtype=np.float64)
    out["earn_qk"] = np.full(shape, np.nan, dtype=np.float64)
    out["earn_avail_t"] = np.full(shape, np.nan, dtype=np.float64)
    quarter_fact, _ = _quarter_fact_map(facts, sess, insts)
    versions, _first_avail = _release_tables(releases, quarter_fact, sess, insts)
    _ = fundamentals
    by_inst: dict[int, list[tuple[int, int, int]]] = {}
    for (nidx, qk), ver_list in versions.items():
        for avail, order, _ in ver_list:
            by_inst.setdefault(nidx, []).append((avail, qk, order))
    latest_at: dict[tuple[int, int, int], dict[str, tuple[float | None, float | None]]] = {}
    for (nidx, qk), ver_list in versions.items():
        ordered = sorted(ver_list, key=lambda item: (item[0], item[1]))
        running: dict[str, tuple[float | None, float | None]] = {}
        for avail, _order, values in ordered:
            running = {**running, **values}
            latest_at[(nidx, qk, avail)] = dict(running)
    for nidx in range(n_n):
        events = sorted(by_inst.get(nidx, []))
        if not events:
            continue
        live: dict[int, dict[str, tuple[float | None, float | None]]] = {}
        seen_qk: dict[int, int] = {}
        cursor = 0
        for t in range(n_s):
            while cursor < len(events) and events[cursor][0] <= t:
                avail, qk, _ = events[cursor]
                key = (nidx, qk, avail)
                if key in latest_at:
                    live[qk] = latest_at[key]
                if qk not in seen_qk:
                    seen_qk[qk] = avail
                cursor += 1
            if not live:
                continue
            best_qk = max(live)
            out["e_qk"][t, nidx] = float(best_qk)
            out["e_avail_t"][t, nidx] = float(seen_qk[best_qk])
            for metric in _RELEASE_METRICS:
                pair = live[best_qk].get(metric)
                if pair is None:
                    continue
                cur, prv = pair
                if cur is not None:
                    out[f"e_{metric}_q"][t, nidx] = cur
                if prv is not None:
                    out[f"e_{metric}_q_ly"][t, nidx] = prv
    f_qk = np.asarray(fundamentals.get("f_qk", np.full(shape, np.nan)), dtype=np.float64)
    f_avail = np.asarray(fundamentals.get("f_avail_t", np.full(shape, np.nan)), dtype=np.float64)
    e_qk = out["e_qk"]
    e_avail = out["e_avail_t"]
    use_filing = np.isfinite(f_qk) & (~np.isfinite(e_qk) | (f_qk >= e_qk))
    use_release = np.isfinite(e_qk) & ~use_filing
    has_any = use_filing | use_release
    out["earn_qk"] = np.where(use_filing, f_qk, np.where(use_release, e_qk, np.nan))
    f_early = np.where(use_filing, f_avail, np.nan)
    e_early = np.where(use_release, e_avail, np.nan)
    tie = use_filing & np.isfinite(e_qk) & (f_qk == e_qk)
    f_safe = np.where(np.isfinite(f_early), f_early, np.inf)
    e_safe = np.where(np.isfinite(e_early), e_early, np.inf)
    best_early = np.minimum(f_safe, e_safe)
    earn_avail = np.where(has_any, np.where(np.isfinite(best_early), best_early, np.nan), np.nan)
    f_all = np.where(np.isfinite(f_avail), f_avail, np.inf)
    e_all = np.where(np.isfinite(e_avail), e_avail, np.inf)
    best_all = np.minimum(f_all, e_all)
    earn_avail = np.where(tie, np.where(np.isfinite(best_all), best_all, np.nan), earn_avail)
    out["earn_avail_t"] = earn_avail
    filing_map = {"sales": "f_sales_q", "operating_profit": "f_operating_profit_q", "net_income": "f_net_income_q"}
    filing_ly = {
        "sales": "f_sales_q_ly",
        "operating_profit": "f_operating_profit_q_ly",
        "net_income": "f_net_income_q_ly",
    }
    for metric in _RELEASE_METRICS:
        f_col = np.asarray(fundamentals.get(filing_map[metric], np.full(shape, np.nan)), dtype=np.float64)
        f_ly_col = np.asarray(fundamentals.get(filing_ly[metric], np.full(shape, np.nan)), dtype=np.float64)
        e_col = out[f"e_{metric}_q"]
        e_ly_col = out[f"e_{metric}_q_ly"]
        out[f"earn_{metric}_q"] = np.where(use_filing, f_col, np.where(use_release, e_col, np.nan))
        out[f"earn_{metric}_q_ly"] = np.where(use_filing, f_ly_col, np.where(use_release, e_ly_col, np.nan))
    _log_phase("releases", out, shape)
    return out


def _instrument_key_expr(columns: Sequence[str]) -> pl.Expr:
    """Vectorized ``_instrument_of``: a ``KRX:`` id first, else a six-digit ticker/company code."""
    candidates: list[pl.Expr] = []
    for key in ("instrument_id", "company_id"):
        if key in columns:
            text = pl.col(key).cast(pl.String, strict=False)
            candidates.append(pl.when(text.str.starts_with("KRX:")).then(text))
    for key in ("ticker", "company_id", "instrument_id"):
        if key in columns:
            text = pl.col(key).cast(pl.String, strict=False).str.strip_chars()
            candidates.append(pl.when(text.str.contains(r"^\d{6}$")).then(pl.lit("KRX:") + text))
    return pl.coalesce(candidates)


def _numeric_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    dtype = frame.schema[column]
    if dtype.is_numeric():
        return pl.col(column).cast(pl.Float64)
    return pl.col(column).cast(pl.String, strict=False).cast(pl.Float64, strict=False)


def assemble_flows(
    flows: pl.DataFrame, close: NDArray[np.float64], *, sessions: Sequence[date], instrument_ids: Sequence[str]
) -> dict[str, NDArray[np.float64]]:
    """Accumulate investor net-share flows into KRW at their availability session.

    Vectorized over the full flow history (millions of rows): building per-row Python objects
    would exceed the cube's memory budget.
    """
    sess = list(sessions)
    insts = list(instrument_ids)
    n_s, n_n = len(sess), len(insts)
    shape = (n_s, n_n)
    sources = (
        ("flow_for_krw", "foreign_net_shares"),
        ("flow_ins_krw", "institution_net_shares"),
        ("flow_ind_krw", "individual_net_shares"),
    )
    out = {name: np.full(shape, np.nan, dtype=np.float64) for name, _ in sources}
    required = {"session", "available_at"} | {column for _, column in sources}
    if not required <= set(flows.columns) or not {"instrument_id", "ticker", "company_id"} & set(flows.columns):
        raise PITDataError(f"investor flow frame lacks required columns: {sorted(required)} plus an instrument key")
    at_dtype = flows.schema["available_at"]
    if not isinstance(at_dtype, pl.Datetime) or at_dtype.time_zone is None:
        raise PITDataError("investor flow available_at must be a timezone-aware datetime")
    close_arr = np.asarray(close, dtype=np.float64)
    if close_arr.shape != shape:
        raise PITDataError(f"investor flow close array shape {close_arr.shape} != {shape}")
    index_frame = pl.DataFrame({"_inst": insts, "_nidx": np.arange(n_n, dtype=np.int64)})
    session_frame = pl.DataFrame({"_session": sess, "_sidx": np.arange(n_s, dtype=np.int64)})
    frame = (
        flows.select(
            _instrument_key_expr(flows.columns).alias("_inst"),
            (
                pl.col("session")
                if flows.schema["session"] == pl.Date
                else pl.col("session").cast(pl.String, strict=False).str.slice(0, 10).str.to_date(strict=False)
            ).alias("_session"),
            pl.col("available_at").dt.convert_time_zone("UTC").dt.epoch("us").alias("_at"),
            *[_numeric_expr(flows, column).alias(column) for _, column in sources],
            *[pl.col(column).is_not_null().alias(f"_{column}_given") for _, column in sources],
        )
        .join(index_frame, on="_inst", how="inner", maintain_order="left")
        .join(session_frame, on="_session", how="inner", maintain_order="left")
    )
    before = frame.height
    frame = frame.unique(subset=["_sidx", "_nidx"], keep="first", maintain_order=True)
    dropped = before - frame.height
    # 값이 주어졌는데 숫자로 읽히지 않는 행은 행 전체를 버린다(일부 투자자만 반영하지 않음).
    malformed = pl.lit(False)
    for _, column in sources:
        malformed = malformed | (pl.col(f"_{column}_given") & pl.col(column).is_null())
    frame = frame.filter(~malformed & pl.col("_at").is_not_null())
    if dropped:
        _LOG.debug("investor flow duplicate rows dropped: %d", dropped)
    decision_us = np.asarray(
        [int(_decision_at(day).timestamp() * 1_000_000) for day in sess], dtype=np.int64
    )
    avail = np.searchsorted(decision_us, frame["_at"].to_numpy(), side="left")
    sidx = frame["_sidx"].to_numpy()
    nidx = frame["_nidx"].to_numpy()
    price = close_arr[sidx, nidx]
    keep = (avail < n_s) & np.isfinite(price) & (price > 0)
    avail, sidx, nidx, price = avail[keep], sidx[keep], nidx[keep], price[keep]
    for name, column in sources:
        values = frame[column].to_numpy()[keep].astype(np.float64)
        valid = np.isfinite(values)
        totals = np.zeros(shape, dtype=np.float64)
        counts = np.zeros(shape, dtype=np.int64)
        np.add.at(totals, (avail[valid], nidx[valid]), values[valid] * price[valid])
        np.add.at(counts, (avail[valid], nidx[valid]), 1)
        mask = counts > 0
        out[name][mask] = totals[mask]
    _log_phase("flows", out, shape)
    return out


def assemble_dividends(
    dividends: pl.DataFrame,
    base_price: NDArray[np.float64],
    close: NDArray[np.float64],
    *,
    sessions: Sequence[date],
    instrument_ids: Sequence[str],
    withholding_rate: Decimal,
) -> dict[str, NDArray[np.float64]]:
    """Build ex-session dividend returns and trailing dividend-per-share inputs."""
    sess = list(sessions)
    insts = list(instrument_ids)
    n_s, n_n = len(sess), len(insts)
    shape = (n_s, n_n)
    factor = 1.0 - float(withholding_rate)
    base = np.asarray(base_price, dtype=np.float64)
    if base.shape != shape:
        raise PITDataError("dividend base_price has wrong shape")
    div_ret = np.zeros(shape, dtype=np.float64)
    dps_ttm = np.zeros(shape, dtype=np.float64)
    try:
        rows = dividends.to_dicts()
    except (ValueError, pl.exceptions.PolarsError) as exc:  # pragma: no cover
        raise PITDataError(f"invalid dividend events frame: {exc}") from exc
    index_of = {inst: idx for idx, inst in enumerate(insts)}
    session_of = {day: idx for idx, day in enumerate(sess)}
    times = _decision_times(sess)
    events: list[tuple[int, int, float, int]] = []
    for row in rows:
        inst = _instrument_of(row)
        if inst is None or inst not in index_of:
            continue
        ex_val = row.get("ex_session")
        if not isinstance(ex_val, date):
            try:
                ex_val = date.fromisoformat(str(ex_val)[:10])
            except (ValueError, TypeError):
                continue
        if ex_val not in session_of:
            continue
        dps = row.get("dps_krw")
        if dps is None:
            continue
        try:
            dps_f = float(dps)
        except (TypeError, ValueError):
            continue
        if dps_f != dps_f or dps_f < 0:
            continue
        avail = _avail_index_in(row.get("available_at"), times)
        if avail is None:
            continue
        events.append((session_of[ex_val], index_of[inst], dps_f, avail))
    for ex_idx, nidx, dps_f, _ in events:
        denom = base[ex_idx, nidx]
        if np.isfinite(denom) and denom > 0:
            div_ret[ex_idx, nidx] += dps_f * factor / denom
    sess_ord = np.asarray([day.toordinal() for day in sess], dtype=np.int64)
    for ex_idx, nidx, dps_f, avail in events:
        # 공시 가용 세션부터 배당락 후 365일까지 누적 (세션 날짜는 오름차순)
        stop = int(np.searchsorted(sess_ord, sess[ex_idx].toordinal() + 365, side="right"))
        if stop > avail:
            dps_ttm[avail:stop, nidx] += dps_f
    _log_phase("dividends", {"dps_ttm": dps_ttm, "div_ret": div_ret}, shape)
    _ = close
    return {"dps_ttm": dps_ttm, "div_ret": div_ret}


def _read_frame(dataset_dir: Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    from src.data.datasets import read_dataset

    try:
        return read_dataset(Path(dataset_dir), columns=columns).collect()
    except PITDataError:
        raise
    except Exception as exc:  # pragma: no cover
        raise PITDataError(f"unreadable dataset: {dataset_dir}") from exc


def _require_kind(dataset_dir: Path, expected: str) -> None:
    try:
        manifest = load_manifest(Path(dataset_dir))
    except PITDataError as exc:
        raise PITDataError(f"invalid dataset manifest: {dataset_dir}") from exc
    if manifest.kind != expected:
        raise PITDataError(f"dataset kind mismatch at {dataset_dir}: {manifest.kind} != {expected}")


def _check_references(
    frame: pl.DataFrame, panel_ids: set[str], label: str, id_keys: Sequence[str] = ("instrument_id",)
) -> None:
    try:
        rows = frame.select([key for key in id_keys if key in frame.columns]).to_dicts()
    except (ValueError, pl.exceptions.PolarsError):  # pragma: no cover
        return
    for row in rows:
        inst = _instrument_of(row)
        if inst is not None and inst not in panel_ids:
            raise PITDataError(f"{label} references unknown instrument: {inst}")


def _fact_instruments_outside_panel(facts: pl.DataFrame, panel_ids: set[str]) -> set[str]:
    """Instruments named by financial facts that have no panel row (never listed in the universe)."""
    if "company_id" not in facts.columns:
        return set()
    columns = [key for key in ("company_id", "ticker") if key in facts.columns]
    rows = facts.select(columns).unique().to_dicts()
    found = {_instrument_of(row) for row in rows}
    return {inst for inst in found if inst is not None and inst not in panel_ids}


def _panel_float_columns(
    panel_dir: Path, names: Sequence[str], sessions: Sequence[date], instrument_ids: Sequence[str]
) -> dict[str, NDArray[np.float64]]:
    """Dense arrays of panel columns that ``load_market_arrays`` does not carry."""
    from src.data.datasets import dataset_partition_paths

    paths = [str(path) for path in dataset_partition_paths(panel_dir) if path.suffix == ".parquet"
             and path.name != "instrument_exits.parquet"]
    frame = pl.scan_parquet(paths).select(["session", "instrument_id", *names]).collect()
    session_index = {day: idx for idx, day in enumerate(sessions)}
    instrument_index = {inst: idx for idx, inst in enumerate(instrument_ids)}
    rows = frame["session"].replace_strict(session_index, return_dtype=pl.Int64).to_numpy()
    cols = frame["instrument_id"].replace_strict(instrument_index, return_dtype=pl.Int64).to_numpy()
    out: dict[str, NDArray[np.float64]] = {}
    for name in names:
        dense = np.full((len(sessions), len(instrument_ids)), np.nan, dtype=np.float64)
        dense[rows, cols] = frame[name].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        out[name] = dense
    return out


def _build_market_arrays(
    inputs: CubeInputs, sessions: tuple[date, ...], instrument_ids: tuple[str, ...]
) -> dict[str, NDArray[Any]]:
    arrays = load_market_arrays(panel_dir=Path(inputs.market_panel), cache_root=Path(tempfile.gettempdir()))
    panel_sessions = list(arrays.sessions)
    panel_insts = list(arrays.instrument_ids)
    if tuple(panel_sessions) != tuple(sessions) or tuple(panel_insts) != tuple(instrument_ids):
        raise PITDataError("market panel sessions or instruments changed during cube build")  # pragma: no cover
    n_s, n_n = len(sessions), len(instrument_ids)
    shape = (n_s, n_n)
    get_int = arrays.int_fields.get
    get_float = arrays.float_fields.get
    get_bool = arrays.bool_fields.get

    def _f(name: str, default: float = float("nan")) -> NDArray[np.float64]:
        if name in arrays.int_fields:
            return np.asarray(get_int(name), dtype=np.float64)
        if name in arrays.float_fields:
            return np.asarray(get_float(name), dtype=np.float64)
        return np.full(shape, default, dtype=np.float64)

    def _b(name: str) -> NDArray[np.bool_]:
        if name in arrays.bool_fields:
            return np.asarray(get_bool(name), dtype=bool)
        return np.zeros(shape, dtype=bool)

    open_px = _f("open")
    high = _f("high")
    low = _f("low")
    close = _f("close")
    base = _f("base_price")
    volume = _f("volume")
    panel_floats = _panel_float_columns(
        Path(inputs.market_panel), ("trading_value", "market_cap", "listed_shares"), sessions, instrument_ids
    )
    trading_value = panel_floats["trading_value"]
    market_cap = panel_floats["market_cap"]
    listed_shares = panel_floats["listed_shares"]
    adtv20 = _f("adtv20")
    ret_vol60 = _f("ret_vol60")
    sell_tax = _f("sell_tax_rate")
    present = _b("present")
    eligible = _b("eligible")
    entry_blocked = _b("entry_blocked")
    open_at_upper = _b("open_at_upper")
    open_at_lower = _b("open_at_lower")
    market = np.asarray(arrays.market, dtype=np.int8)
    tick_at_open = np.zeros(shape, dtype=np.float64)
    for t, day in enumerate(sessions):
        try:
            regime = inputs.market_rules.tick_regime_at(day)
        except PITDataError:  # pragma: no cover - session precedes the rules' coverage
            continue
        quotable = (open_px[t] > 0) & np.isfinite(open_px[t])
        for code, market_key in ((1, KrxMarket.KOSPI), (2, KrxMarket.KOSDAQ)):
            mask = quotable & (market[t] == code)
            if not mask.any():
                continue
            bands = regime.bands[market_key]
            lowers = np.asarray([band.lower_price_inclusive for band in bands], dtype=np.float64)
            ticks = np.asarray([band.tick for band in bands], dtype=np.float64)
            position = np.searchsorted(lowers, np.floor(open_px[t, mask]), side="right") - 1
            tick_at_open[t, mask] = ticks[np.clip(position, 0, len(bands) - 1)]
    traded = present & (volume > 0) & (open_px > 0)
    ret_cc = np.zeros(shape, dtype=np.float64)
    valid_px = present & (close > 0) & (base > 0) & np.isfinite(close) & np.isfinite(base)
    ret_cc[valid_px] = close[valid_px] / base[valid_px] - 1.0
    out: dict[str, NDArray[Any]] = {
        "open": open_px,
        "high": high,
        "low": low,
        "close": close,
        "base_price": base,
        "volume": volume,
        "trading_value": trading_value,
        "market_cap": market_cap,
        "listed_shares": listed_shares,
        "adtv20": adtv20,
        "ret_vol60": ret_vol60,
        "sell_tax_rate": sell_tax,
        "present": present,
        "eligible": eligible,
        "entry_blocked": entry_blocked,
        "open_at_upper": open_at_upper,
        "open_at_lower": open_at_lower,
        "market": market,
        "tick_at_open": tick_at_open,
        "traded": traded,
        "ret_cc": ret_cc,
    }
    _log_phase("market", out, shape)
    return out


def _exit_arrays(panel_dir: Path, sessions: Sequence[date], instrument_ids: Sequence[str]) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    sess_index = {day: idx for idx, day in enumerate(sessions)}
    inst_index = {inst: idx for idx, inst in enumerate(instrument_ids)}
    exit_at = np.full(len(instrument_ids), -1, dtype=np.int64)
    halted = np.zeros(len(instrument_ids), dtype=np.bool_)
    candidates = [Path(panel_dir) / "instrument_exits.parquet", Path(panel_dir) / "exits.parquet"]
    frame: pl.DataFrame | None = None
    for path in candidates:
        if path.is_file():
            try:
                frame = pl.read_parquet(path)
                break
            except (OSError, ValueError, pl.exceptions.PolarsError) as exc:  # pragma: no cover
                raise PITDataError(f"unreadable instrument exits: {path}") from exc
    if frame is None or frame.height == 0:
        return exit_at, halted
    try:
        rows = frame.to_dicts()
    except (ValueError, pl.exceptions.PolarsError) as exc:  # pragma: no cover
        raise PITDataError(f"invalid instrument exits: {exc}") from exc
    panel_last = sessions[-1] if sessions else None
    for row in rows:
        inst = _instrument_of(row)
        if inst is None or inst not in inst_index:
            continue
        last = row.get("last_session")
        if not isinstance(last, date):
            try:
                last = date.fromisoformat(str(last)[:10])
            except (ValueError, TypeError):
                continue
        if panel_last is not None and last == panel_last:
            continue
        if last not in sess_index:
            continue
        nidx = inst_index[inst]
        exit_at[nidx] = int(sess_index[last] + 1)
        volume = row.get("last_volume")
        try:
            halted[nidx] = volume is not None and int(volume) == 0
        except (TypeError, ValueError):
            halted[nidx] = False
    return exit_at, halted


def build_research_cube(inputs: CubeInputs) -> ResearchCube:
    """Load verified datasets and assemble the cube.

    Raises:
        PITDataError: any input manifest or partition hash fails verification, an input dataset
            kind mismatches its role, fact keys (company, fiscal period, fact) are duplicated, or an
            event references an instrument absent from the panel.
    """
    for role, expected in (
        (inputs.market_panel, "market_panel"),
        (inputs.dividend_events, "dividend_events"),
        (inputs.financial_facts, "financial_facts"),
        (inputs.investor_flow, "investor_flow"),
        (inputs.earnings_releases, "earnings_releases"),
    ):
        _require_kind(Path(role), expected)
    arrays = load_market_arrays(panel_dir=Path(inputs.market_panel), cache_root=Path(tempfile.gettempdir()))
    sessions = tuple(arrays.sessions)
    instrument_ids = tuple(arrays.instrument_ids)
    panel_ids = set(instrument_ids)
    facts = _read_frame(Path(inputs.financial_facts))
    dividends = _read_frame(Path(inputs.dividend_events))
    flows = _read_frame(Path(inputs.investor_flow))
    releases = _read_frame(Path(inputs.earnings_releases))
    if "company_id" in facts.columns and "fiscal_period" in facts.columns and "fact" in facts.columns:
        try:
            keys = facts.select(["company_id", "fiscal_period", "fact"]).to_dicts()
            seen_keys: set[tuple[str, str, str]] = set()
            for row in keys:
                key = (str(row.get("company_id")), str(row.get("fiscal_period")), str(row.get("fact")))
                if key in seen_keys:
                    raise PITDataError(f"duplicated financial fact key: {key}")
                seen_keys.add(key)
        except (ValueError, pl.exceptions.PolarsError) as exc:  # pragma: no cover
            raise PITDataError(f"invalid financial facts keys: {exc}") from exc
    for frame, label in (
        (dividends, "dividend events"),
        (flows, "investor flow"),
        (releases, "earnings releases"),
    ):
        _check_references(frame, panel_ids, label)
    outside = _fact_instruments_outside_panel(facts, panel_ids)
    if outside:
        # 시세가 없는 비유니버스 종목의 공시라 매매 대상이 될 수 없다. 조용히 버리지 않고 규모를 남긴다.
        _LOG.info("[DATA] stage=facts_outside_panel instruments=%d skipped", len(outside))
    market_part = _build_market_arrays(inputs, sessions, instrument_ids)
    shape = (len(sessions), len(instrument_ids))
    base = np.asarray(market_part["base_price"], dtype=np.float64)
    close = np.asarray(market_part["close"], dtype=np.float64)
    fundamentals = assemble_fundamentals(facts, sessions=sessions, instrument_ids=instrument_ids)
    releases_out = assemble_releases(
        releases, fundamentals, facts, sessions=sessions, instrument_ids=instrument_ids
    )
    flows_out = assemble_flows(flows, close, sessions=sessions, instrument_ids=instrument_ids)
    dividends_out = assemble_dividends(
        dividends,
        base,
        close,
        sessions=sessions,
        instrument_ids=instrument_ids,
        withholding_rate=inputs.dividend_withholding_rate,
    )
    div_ret = np.asarray(dividends_out["div_ret"], dtype=np.float64)
    traded = np.asarray(market_part["traded"], dtype=bool)
    ret_cc = np.asarray(market_part["ret_cc"], dtype=np.float64)
    open_px = np.asarray(market_part["open"], dtype=np.float64)
    close_px = np.asarray(market_part["close"], dtype=np.float64)
    r_on = np.zeros(shape, dtype=np.float64)
    r_id = np.zeros(shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        open_ret = np.where(traded & (base > 0), open_px / base - 1.0, ret_cc)
    r_on = open_ret + div_ret
    r_on[~np.isfinite(r_on)] = 0.0
    traded_id = traded & (open_px > 0) & np.isfinite(open_px) & np.isfinite(close_px)
    r_id[traded_id] = close_px[traded_id] / open_px[traded_id] - 1.0
    growth = 1.0 + ret_cc
    growth[~np.isfinite(growth)] = 1.0
    adj_px = np.cumprod(growth, axis=0)
    growth_tr = 1.0 + ret_cc + div_ret
    growth_tr[~np.isfinite(growth_tr)] = 1.0
    adj_tr = np.cumprod(growth_tr, axis=0)
    if shape[0] > 0:
        adj_px[0, :] = 1.0
        adj_tr[0, :] = 1.0
        if shape[0] > 1:
            adj_px[1:, :] = np.cumprod(growth[1:, :], axis=0)
            adj_tr[1:, :] = np.cumprod(growth_tr[1:, :], axis=0)
    full: dict[str, NDArray[Any]] = {
        **market_part,
        **fundamentals,
        **releases_out,
        **flows_out,
        "dps_ttm": np.asarray(dividends_out["dps_ttm"], dtype=np.float64),
        "div_ret": div_ret,
        "r_on": r_on,
        "r_id": r_id,
        "adj_px": adj_px,
        "adj_tr": adj_tr,
    }
    exit_at, exit_halted = _exit_arrays(Path(inputs.market_panel), sessions, instrument_ids)
    cube_id = research_cube_id(inputs)
    return ResearchCube.from_arrays(
        cube_id=cube_id,
        sessions=sessions,
        instrument_ids=instrument_ids,
        arrays=full,
        exit_at=exit_at,
        exit_halted=exit_halted,
    )


def _update_digest(digest: Any, arr: NDArray[Any]) -> None:
    """Feed one array's C-order bytes into ``digest`` in fixed-size chunks.

    The cube is several GB and is memory-mapped on a cache hit, so hashing it must never call ``tobytes()`` on
    a whole array: that would allocate a second full-size anonymous copy just to verify the cache.
    """
    flat = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)
    for start in range(0, flat.size, _CHECKSUM_CHUNK_BYTES):
        digest.update(flat[start : start + _CHECKSUM_CHUNK_BYTES].tobytes())


def _array_checksum(arrays: Mapping[str, NDArray[Any]], exit_at: NDArray[np.int64], halted: NDArray[np.bool_]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        _update_digest(digest, arrays[name])
    _update_digest(digest, exit_at)
    _update_digest(digest, halted)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _CubeCache:
    """One verified cache payload: the arrays plus the identity and digest they were stored under.

    ``arrays`` is either memory-mapped (format 2) or anonymous (the legacy ``.npz``), so callers must not
    assume either; what is guaranteed is that the bytes are the ones the checksum covers.
    """

    cube_id: str
    sessions: tuple[date, ...]
    instrument_ids: tuple[str, ...]
    arrays: Mapping[str, NDArray[Any]]
    exit_at: NDArray[np.int64]
    exit_halted: NDArray[np.bool_]
    checksum: str


def _read_cache_dir(directory: Path, cube_id: str) -> _CubeCache | None:
    """Verify and memory-map a format-2 cache directory, or return ``None`` when it is unusable."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or int(manifest["format"]) != _CACHE_FORMAT:
            return None
        if str(manifest["cube_id"]) != cube_id:
            return None
        names = [str(item) for item in manifest["array_names"]]
        sessions = tuple(date.fromordinal(int(item)) for item in manifest["sessions"])
        instrument_ids = tuple(str(item) for item in manifest["instruments"])
        checksum = str(manifest["checksum"])
        arrays: dict[str, NDArray[Any]] = {
            name: np.load(directory / f"arr_{name}.npy", mmap_mode="r") for name in names
        }
        exit_at = np.load(directory / "exit_at.npy")
        halted = np.load(directory / "exit_halted.npy")
    except (OSError, ValueError, TypeError, KeyError, EOFError):
        return None
    shape = (len(sessions), len(instrument_ids))
    if exit_at.dtype != np.int64 or halted.dtype != np.bool_:
        return None
    if exit_at.shape != (len(instrument_ids),) or halted.shape != (len(instrument_ids),):
        return None
    if any(arr.shape != shape or not arr.flags.c_contiguous for arr in arrays.values()):
        return None
    if _array_checksum(arrays, exit_at, halted) != checksum:
        return None
    return _CubeCache(cube_id, sessions, instrument_ids, arrays, exit_at, halted, checksum)


def _read_cache_npz(path: Path, cube_id: str) -> _CubeCache | None:
    """Read a format-1 ``.npz`` cache, or return ``None`` when it is corrupt or belongs to another cube."""
    try:
        with np.load(str(path), allow_pickle=True) as store:
            if str(store["cube_id"]) != cube_id:
                return None
            checksum = str(store["checksum"])
            sessions = tuple(date.fromordinal(int(item)) for item in np.asarray(store["sessions"]).tolist())
            instrument_ids = tuple(str(item) for item in store["instruments"].tolist())
            names = [str(item) for item in store["array_names"].tolist()]
            arrays: dict[str, NDArray[Any]] = {name: np.asarray(store[f"arr_{name}"]) for name in names}
            exit_at = np.asarray(store["exit_at"], dtype=np.int64)
            halted = np.asarray(store["exit_halted"], dtype=np.bool_)
    except Exception:  # noqa: BLE001 - an unreadable legacy cache is simply rebuilt
        return None
    if _array_checksum(arrays, exit_at, halted) != checksum:
        return None
    return _CubeCache(cube_id, sessions, instrument_ids, arrays, exit_at, halted, checksum)


def _write_cache_dir(payload: _CubeCache, directory: Path) -> None:
    """Write one ``.npy`` per array plus ``manifest.json`` into a fresh ``directory``."""
    directory.mkdir(parents=True)
    for name, arr in payload.arrays.items():
        np.save(directory / f"arr_{name}.npy", np.ascontiguousarray(arr), allow_pickle=False)
    np.save(directory / "exit_at.npy", np.asarray(payload.exit_at, dtype=np.int64), allow_pickle=False)
    np.save(directory / "exit_halted.npy", np.asarray(payload.exit_halted, dtype=np.bool_), allow_pickle=False)
    manifest = {
        "array_names": sorted(payload.arrays),
        "checksum": payload.checksum,
        "cube_id": payload.cube_id,
        "format": _CACHE_FORMAT,
        "instruments": list(payload.instrument_ids),
        "sessions": [day.toordinal() for day in payload.sessions],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _publish_cache(payload: _CubeCache, directory: Path) -> bool:
    """Write ``payload`` into a staged sibling directory and move it into place atomically.

    Staging under a pid-tagged name means a crashed or losing writer never leaves a half-written directory at
    the real path: readers only ever see a fully written, verified one.
    """
    temp = directory.parent / f".{directory.name}.{os.getpid()}.tmp"
    try:
        shutil.rmtree(temp, ignore_errors=True)
        _write_cache_dir(payload, temp)
        shutil.rmtree(directory, ignore_errors=True)
        os.replace(temp, directory)
    except OSError:
        _LOG.warning("[DATA] cube cache could not be written: %s", directory)
        return False
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return True


def _payload_of(cube: ResearchCube) -> _CubeCache:
    """The cube as a cache payload, sealed with the digest that identifies its bytes."""
    return _CubeCache(
        cube_id=cube.cube_id,
        sessions=cube.sessions,
        instrument_ids=cube.instrument_ids,
        arrays=dict(cube.arrays),
        exit_at=cube.exit_at,
        exit_halted=cube.exit_halted,
        checksum=_array_checksum(cube.arrays, cube.exit_at, cube.exit_halted),
    )


def _cube_from_cache(payload: _CubeCache) -> ResearchCube:
    """Freeze a verified payload into a cube, leaving every array as it was read.

    Deliberately not ``from_arrays``: that iterates the whole mapping to re-validate it and would flatten each
    ``np.memmap`` back to a plain view, losing the read-only file-backed handle a copy-on-write consumer needs.
    """
    exit_at = np.ascontiguousarray(payload.exit_at, dtype=np.int64)
    halted = np.ascontiguousarray(payload.exit_halted, dtype=np.bool_)
    exit_at.flags.writeable = False
    halted.flags.writeable = False
    return ResearchCube(
        cube_id=payload.cube_id,
        sessions=payload.sessions,
        instrument_ids=payload.instrument_ids,
        arrays=dict(payload.arrays),
        exit_at=exit_at,
        exit_halted=halted,
    )


def load_research_cube(inputs: CubeInputs, *, cache_root: Path) -> ResearchCube:
    """Return the cube from a checksum-verified, memory-mapped cache, building and caching it on a miss.

    Cache layout (format 2) is a directory ``<cache_root>/<cube_id>/`` holding ``manifest.json`` (``format``,
    ``cube_id``, ``checksum``, ``sessions`` as ordinals, ``instruments``, ``array_names``) plus one ``.npy`` per
    array (``arr_<name>.npy``, ``exit_at.npy``, ``exit_halted.npy``). Arrays are returned as read-only memory
    maps rather than anonymous copies: the cube is several GB and almost entirely read-only, so file-backed
    pages stay reclaimable under memory pressure, while a consumer that needs a modified view (the causal
    perturbation) can copy on write only the pages it changes.

    The checksum is the same digest as format 1 (array names in sorted order, then raw bytes, then the exit
    arrays) and is computed over the mapped buffers without a second in-memory copy. A format-1 cache whose
    checksum verifies is converted to format 2 once and its ``.npz`` is removed only after the new directory
    verifies; a corrupt or mismatching cache of either format is removed and rebuilt.

    Raises: whatever ``build_research_cube`` raises on a miss; never returns an unverified cube.
    """
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    cube_id = research_cube_id(inputs)
    directory = root / cube_id
    legacy = root / f"{cube_id}.npz"

    cached = _read_cache_dir(directory, cube_id)
    if cached is not None:
        return _cube_from_cache(cached)
    if legacy.is_file():
        payload = _read_cache_npz(legacy, cube_id)
        promoted: _CubeCache | None = None
        if payload is not None and _publish_cache(payload, directory):
            # Re-read the published directory: the converted load must be memory-mapped like any other hit,
            # and the ``.npz`` only disappears once the new layout has verified.
            promoted = _read_cache_dir(directory, cube_id)
        if promoted is not None or payload is None:
            # A verified legacy file is only dropped once its replacement verifies; a corrupt one is dropped.
            with contextlib.suppress(OSError):
                legacy.unlink()
        if promoted is not None:
            return _cube_from_cache(promoted)
    shutil.rmtree(directory, ignore_errors=True)
    staged = root / f".{cube_id}.{os.getpid()}.tmp"
    try:
        cube = build_research_cube(inputs)
        published = _publish_cache(_payload_of(cube), directory)
    finally:
        # A crash between staging and renaming, or a build that never reaches the publish, must not leave a
        # directory the next run would have to unpick.
        shutil.rmtree(staged, ignore_errors=True)
    # Serve the miss from the published maps too: returning the anonymous build would leave the first run
    # holding the whole cube in private memory and force full-size copies in the perturbation.
    mapped = _read_cache_dir(directory, cube_id) if published else None
    return _cube_from_cache(mapped) if mapped is not None else cube
