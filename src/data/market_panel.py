"""Decision-safe session-by-instrument Gold market panel."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.core.market_rules import KrxMarket, KrxMarketRules
from src.core.pit import PITDataError
from src.core.time import KRX_TZ
from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    dataset_partition_paths,
    dataset_reference,
    load_manifest,
    publish_dataset,
    read_dataset,
)

REVISION = "krx-market-panel-v4"

_LOG = logging.getLogger(__name__)

_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "instrument_id": pl.String,
    "ticker": pl.String,
    "market": pl.String,
    "open": pl.Int64,
    "high": pl.Int64,
    "low": pl.Int64,
    "close": pl.Int64,
    "change": pl.Int64,
    "base_price": pl.Int64,
    "volume": pl.Int64,
    "trading_value": pl.Int64,
    "market_cap": pl.Int64,
    "listed_shares": pl.Int64,
    "price_state": pl.String,
    "eligible": pl.Boolean,
    "exclusion_reason": pl.String,
    "gap_before": pl.Boolean,
    "ret_price": pl.Float64,
    "share_factor": pl.Float64,
    "tick_size": pl.Int64,
    "sell_tax_rate": pl.Float64,
    "upper_limit": pl.Int64,
    "lower_limit": pl.Int64,
    "limits_applicable": pl.Boolean,
    "open_at_upper": pl.Boolean,
    "open_at_lower": pl.Boolean,
    "close_at_upper": pl.Boolean,
    "close_at_lower": pl.Boolean,
    "entry_blocked": pl.Boolean,
    "entry_block_reason": pl.String,
    "adtv20": pl.Float64,
    "adtv60": pl.Float64,
    "ret_vol60": pl.Float64,
    "available_at": pl.Datetime("us", "Asia/Seoul"),
    "source_hash": pl.String,
    "policy_version": pl.String,
}

_EXITS_SCHEMA: dict[str, Any] = {
    "instrument_id": pl.String,
    "last_session": pl.Date,
    "last_close": pl.Int64,
    "last_volume": pl.Int64,
    "exit_kind": pl.String,
}


def _panel_policy_toml(path: Path | None = None) -> tuple[dict[str, Any], Path]:
    resolved = (
        path
        if path is not None
        else Path(__file__).resolve().parents[2] / "config" / "data" / "market_panel.toml"
    )
    try:
        with open(resolved, "rb") as handle:
            return tomllib.load(handle), resolved
    except OSError as exc:
        raise PITDataError(f"market panel policy is missing: {resolved}") from exc
    except ValueError as exc:
        raise PITDataError(f"market panel policy is invalid TOML: {resolved}") from exc


def _default_block_administrative(path: Path | None = None) -> bool:
    """Return the configured default for blocking entries on administrative designations."""
    raw, resolved = _panel_policy_toml(path)
    value = raw.get("block_administrative")
    if not isinstance(value, bool):
        raise PITDataError(f"market panel policy needs a boolean block_administrative: {resolved}")
    return value


def _default_delisting_block_max_sessions(path: Path | None = None) -> int:
    """Return the configured cap on how long a delisting or liquidation notice blocks entries."""
    raw, resolved = _panel_policy_toml(path)
    value = raw.get("delisting_block_max_sessions")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PITDataError(f"market panel policy needs a positive integer delisting_block_max_sessions: {resolved}")
    return value


def _default_limitless_move_audit_threshold(path: Path | None = None) -> float:
    """Return the configured fraction above which a no-limit session must be explained."""
    raw, resolved = _panel_policy_toml(path)
    value = raw.get("limitless_move_audit_threshold")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) < 1:
        raise PITDataError(
            f"market panel policy needs a limitless_move_audit_threshold in (0, 1): {resolved}"
        )
    return float(value)


def _default_max_unexplained_limitless_moves(path: Path | None = None) -> int:
    """Return the configured tolerance for unexplained no-limit sessions."""
    raw, resolved = _panel_policy_toml(path)
    value = raw.get("max_unexplained_limitless_moves")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PITDataError(
            f"market panel policy needs a non-negative integer max_unexplained_limitless_moves: {resolved}"
        )
    return value


@dataclass(frozen=True, slots=True)
class MarketPanelPolicy:
    """Rolling-window contract for decision-time liquidity and risk columns.

    Attributes:
        adtv_short_sessions: Trailing observations for ``adtv20``.
        adtv_long_sessions: Trailing observations for ``adtv60``.
        return_vol_sessions: Trailing non-null price returns for ``ret_vol60``.
        block_administrative: Block new entries while an administrative
            (management-issue) designation is active. Delisting and
            liquidation-trading blocks always apply.
        delisting_block_max_sessions: A delisting or liquidation notice blocks entries for at most
            this many sessions after the latest such notice. A name still trading beyond that
            was reprieved (injunction, appeal, improvement period), and the title of the reversal is
            not always machine-recognizable; the cap uses only past information.
    """

    adtv_short_sessions: int = 20
    adtv_long_sessions: int = 60
    return_vol_sessions: int = 60
    block_administrative: bool = field(default_factory=_default_block_administrative)
    delisting_block_max_sessions: int = field(default_factory=_default_delisting_block_max_sessions)
    limitless_move_audit_threshold: float = field(default_factory=_default_limitless_move_audit_threshold)
    max_unexplained_limitless_moves: int = field(default_factory=_default_max_unexplained_limitless_moves)


@dataclass(frozen=True, slots=True)
class MarketPanelResult:
    dataset_path: Path
    dataset_id: str
    rows: int
    years: tuple[int, ...]
    corporate_action_rows: int
    gap_rows: int
    limit_inapplicable_rows: int
    open_at_upper_rows: int
    open_at_lower_rows: int
    exits_traded: int
    exits_halted: int
    entry_blocked_rows: int = 0
    unexplained_limitless_moves: int = 0


def _load_input_manifest(
    dataset_dir: Path, *, expected_kind: str
) -> tuple[str, list[date], dict[date, Path]]:
    """Load one verified input and align its physical partitions by session."""

    directory = Path(dataset_dir)
    try:
        paths = dataset_partition_paths(directory, allow_legacy=False)
    except PITDataError as exc:
        raise PITDataError(f"invalid market-panel input manifest: {directory}") from exc
    if not paths:
        raise PITDataError(f"invalid market-panel input manifest: {directory}")
    try:
        manifest = load_manifest(directory)
    except PITDataError:
        manifest = None
    if manifest is not None and manifest.kind != expected_kind:
        raise PITDataError(f"market-panel input kind mismatch: {directory}")
    by_session: dict[date, Path] = {}
    previous_session: date | None = None
    for path in paths:
        try:
            frame = pl.read_parquet(path, columns=["session"])
            values = frame.get_column("session").unique().to_list()
        except pl.exceptions.ColumnNotFoundError:
            match = re.search(r"(?:^|/)session=([^/]+)(?:/|$)", path.as_posix())
            if match is None:
                raise PITDataError(f"market-panel input partition has no session: {path}") from None
            try:
                values = [date.fromisoformat(match.group(1))]
            except ValueError as exc:
                raise PITDataError(f"market-panel input partition has an invalid session: {path}") from exc
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            raise PITDataError(f"market-panel input partition is unreadable: {path}") from exc
        if len(values) != 1:
            raise PITDataError(f"market-panel input partition has no unique session: {path}")
        value = values[0]
        if not isinstance(value, date):
            value = date.fromisoformat(str(value)[:10])
        if previous_session is not None and value <= previous_session:
            raise PITDataError(f"market-panel input sessions are not strictly ordered: {directory}")
        previous_session = value
        by_session[value] = path
    sessions = sorted(by_session)
    return directory.name, sessions, by_session


def _bucket_of(instrument_id: str, buckets: int) -> int:
    digest = hashlib.sha256(instrument_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


def _tick_expr(rules: KrxMarketRules, price_col: str) -> pl.Expr:
    expr: pl.Expr = pl.lit(None, dtype=pl.Int64)
    for regime in rules.tick_regimes:
        for market in (KrxMarket.KOSPI, KrxMarket.KOSDAQ):
            bands = regime.bands[market]
            band_expr: pl.Expr = pl.lit(bands[0].tick, dtype=pl.Int64)
            for band in bands[1:]:
                band_expr = (
                    pl.when(pl.col(price_col) >= band.lower_price_inclusive)
                    .then(pl.lit(band.tick, dtype=pl.Int64))
                    .otherwise(band_expr)
                )
            cond = (pl.col("session") >= regime.effective_from) & (pl.col("market") == market.value)
            expr = pl.when(cond).then(band_expr).otherwise(expr)
    return expr


def _tax_expr(rules: KrxMarketRules) -> pl.Expr:
    expr: pl.Expr = pl.lit(None, dtype=pl.Float64)
    for regime in rules.sell_tax_regimes:
        for market in (KrxMarket.KOSPI, KrxMarket.KOSDAQ):
            cond = (pl.col("session") >= regime.effective_from) & (pl.col("market") == market.value)
            expr = pl.when(cond).then(pl.lit(float(regime.rates[market]))).otherwise(expr)
    return expr


def _width_expr(rules: KrxMarketRules) -> pl.Expr:
    expr: pl.Expr = pl.lit(None, dtype=pl.Int64)
    for regime in rules.price_limit_regimes:
        num, den = regime.ratio.as_integer_ratio()
        width = (
            (pl.col("base_price") * num) // (den * pl.col("base_tick")) * pl.col("base_tick")
        ).cast(pl.Int64)
        cond = pl.col("session") >= regime.effective_from
        expr = pl.when(cond).then(width).otherwise(expr)
    return expr


_BLOCK_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "instrument_id": pl.String,
    "entry_blocked": pl.Boolean,
    "entry_block_reason": pl.String,
}

def _entry_block_frame(
    market_actions_path: Path,
    sessions: list[date],
    *,
    block_administrative: bool,
    delisting_block_max_sessions: int,
) -> pl.DataFrame:
    """Return blocked ``(session, instrument)`` cells from point-in-time market actions.

    An action applies from its ``available_at`` session onward, so ``entry_blocked``
    at session ``t`` depends only on actions available at the decision time for ``t``.
    A delisting cancellation clears a pending delisting block; a delisting or liquidation block also
    lapses ``delisting_block_max_sessions`` sessions after the latest such notice; administrative
    blocks apply only when the run policy enables them.

    A delisting or liquidation action that states its window (``effective_end``) blocks entries
    from its availability session through ``effective_end`` and is exempt from the
    ``delisting_block_max_sessions`` lapse, because the exchange itself fixed the end.
    """
    frame = read_dataset(Path(market_actions_path)).collect()
    missing = {"instrument_id", "kind", "rcept_no", "available_at", "cancellation",
               "effective_start", "effective_end"} - set(frame.columns)
    if missing:
        raise PITDataError(f"market actions dataset is missing columns {sorted(missing)}")
    if frame.height == 0:
        return pl.DataFrame([], schema=_BLOCK_SCHEMA)
    by_instrument: dict[str, list[tuple[date, str, str, str, bool, date | None, date | None]]] = {}
    for row in frame.iter_rows(named=True):
        available_at = row["available_at"]
        if not isinstance(available_at, datetime):
            raise PITDataError(f"market action has an invalid available_at: {row['rcept_no']!r}")
        if available_at.tzinfo is None:
            available_at = available_at.replace(tzinfo=KRX_TZ)
        effective = available_at.astimezone(KRX_TZ).date()
        instrument_id = str(row["instrument_id"])
        kind = str(row["kind"])
        if kind not in {
            "delisting_decided",
            "liquidation_trading",
            "administrative_designated",
            "administrative_released",
            "trading_halted",
            "trading_resumed",
        }:
            raise PITDataError(f"market action has an unknown kind {kind!r}")
        start_raw = row.get("effective_start")
        end_raw = row.get("effective_end")
        start = start_raw if isinstance(start_raw, date) else None
        end = end_raw if isinstance(end_raw, date) else None
        by_instrument.setdefault(instrument_id, []).append(
            (effective, str(available_at.isoformat()), str(row["rcept_no"]), kind, bool(row["cancellation"]),
             start, end)
        )
    for actions in by_instrument.values():
        actions.sort()
    blocked: list[dict[str, Any]] = []
    for instrument_id, actions in by_instrument.items():
        pending_delisting = False
        liquidation = False
        administrative = False
        last_notice_idx = -1
        bounded: list[tuple[int, date | None, date]] = []
        position = 0
        for session_idx, session in enumerate(sessions):
            while position < len(actions) and actions[position][0] <= session:
                _, _, _, kind, cancellation, start, end = actions[position]
                if cancellation:
                    pending_delisting = False
                    liquidation = False
                    bounded.clear()
                elif kind == "delisting_decided":
                    if end is not None:
                        bounded.append((session_idx, start, end))
                    else:
                        pending_delisting = True
                        last_notice_idx = session_idx
                elif kind == "liquidation_trading":
                    if end is not None:
                        bounded.append((session_idx, start, end))
                    else:
                        liquidation = True
                        last_notice_idx = session_idx
                elif kind == "administrative_designated":
                    administrative = True
                elif kind == "administrative_released":
                    administrative = False
                position += 1
            if (pending_delisting or liquidation) and session_idx - last_notice_idx > delisting_block_max_sessions:
                pending_delisting = False
                liquidation = False
            covering = [
                (available_idx, start, end)
                for available_idx, start, end in bounded
                if available_idx <= session_idx and session <= end
            ]
            reason: str | None = None
            if covering:
                if any(start is not None and session >= start for _, start, _ in covering):
                    reason = "liquidation_trading"
                else:
                    reason = "delisting_decided"
            elif pending_delisting:
                reason = "delisting_decided"
            elif liquidation:
                reason = "liquidation_trading"
            elif administrative and block_administrative:
                reason = "administrative_designated"
            if reason is not None:
                blocked.append(
                    {
                        "session": session,
                        "instrument_id": instrument_id,
                        "entry_blocked": True,
                        "entry_block_reason": reason,
                    }
                )
    if not blocked:
        return pl.DataFrame([], schema=_BLOCK_SCHEMA)
    return pl.DataFrame(blocked, schema=_BLOCK_SCHEMA).sort(["session", "instrument_id"])


def _build_bucket_frame(
    *,
    daily: pl.DataFrame,
    universe: pl.DataFrame,
    calendar_frame: pl.DataFrame,
    rules: KrxMarketRules,
    policy: MarketPanelPolicy,
    blocks: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if daily.height == 0:
        return pl.DataFrame([], schema=_SCHEMA)
    if daily.select(["session", "instrument_id"]).is_duplicated().any():
        dup = daily.filter(daily.select(["session", "instrument_id"]).is_duplicated())
        key = (dup["session"].to_list()[0], dup["instrument_id"].to_list()[0])
        raise PITDataError(f"duplicate market panel key: {key}")
    df = daily.join(calendar_frame, on="session", how="left").sort(["instrument_id", "session"])
    df = df.with_columns(
        prev_pos=pl.col("pos").shift(1).over("instrument_id"),
        prev_close=pl.col("close").shift(1).over("instrument_id"),
        invalid=pl.col("price_state") == "invalid",
        tradable=pl.col("price_state") == "tradable",
    ).with_columns(
        first=pl.col("prev_pos").is_null(),
        gap_before=pl.col("prev_pos").is_not_null() & (pl.col("prev_pos") != pl.col("pos") - 1),
    ).with_columns(
        ret_price=pl.when(~pl.col("first") & ~pl.col("invalid")).then(
            pl.col("change").cast(pl.Float64) / pl.col("base_price").cast(pl.Float64)
        ),
        share_factor=pl.when(~pl.col("first") & ~pl.col("gap_before") & ~pl.col("invalid")).then(
            pl.col("prev_close").cast(pl.Float64) / pl.col("base_price").cast(pl.Float64)
        ),
        base_tick=_tick_expr(rules, "base_price"),
    ).with_columns(
        width=_width_expr(rules),
    ).with_columns(
        raw_upper=pl.col("base_price") + pl.col("width"),
        raw_lower=pl.col("base_price") - pl.col("width"),
    ).with_columns(
        up_tick=_tick_expr(rules, "raw_upper"),
        lo_tick=_tick_expr(rules, "raw_lower"),
    ).with_columns(
        upper_limit=pl.when(~pl.col("first")).then(pl.col("raw_upper") // pl.col("up_tick") * pl.col("up_tick")),
        lower_limit=pl.when(~pl.col("first")).then(
            pl.max_horizontal(
                -(-pl.col("raw_lower") // pl.col("lo_tick")) * pl.col("lo_tick"), pl.col("lo_tick")
            )
        ),
    ).with_columns(
        limits_applicable=~(
            pl.col("first")
            | pl.col("invalid")
            | (
                pl.col("tradable")
                & ((pl.col("high") > pl.col("upper_limit")) | (pl.col("low") < pl.col("lower_limit")))
            )
        ).fill_null(True),
    ).with_columns(
        lock=pl.col("limits_applicable") & pl.col("tradable"),
    ).with_columns(
        open_at_upper=pl.col("lock") & (pl.col("open") >= pl.col("upper_limit")),
        open_at_lower=pl.col("lock") & (pl.col("open") <= pl.col("lower_limit")),
        close_at_upper=pl.col("lock") & (pl.col("close") >= pl.col("upper_limit")),
        close_at_lower=pl.col("lock") & (pl.col("close") <= pl.col("lower_limit")),
        adtv20=pl.col("trading_value")
        .cast(pl.Float64)
        .rolling_mean(policy.adtv_short_sessions, min_samples=policy.adtv_short_sessions)
        .over("instrument_id"),
        adtv60=pl.col("trading_value")
        .cast(pl.Float64)
        .rolling_mean(policy.adtv_long_sessions, min_samples=policy.adtv_long_sessions)
        .over("instrument_id"),
        tick_size=_tick_expr(rules, "close"),
        sell_tax_rate=_tax_expr(rules),
    )
    rets = (
        df.filter(pl.col("ret_price").is_not_null())
        .select("instrument_id", "session", "ret_price")
        .with_columns(
            rv=pl.col("ret_price")
            .rolling_std(policy.return_vol_sessions, min_samples=policy.return_vol_sessions, ddof=1)
            .over("instrument_id")
        )
        .select("instrument_id", "session", "rv")
    )
    df = (
        df.join(rets, on=["instrument_id", "session"], how="left")
        .with_columns(ret_vol60=pl.col("rv").forward_fill().over("instrument_id"))
        .join(universe, on=["session", "instrument_id"], how="left")
        .with_columns(
            eligible=pl.col("eligible").fill_null(False),
            exclusion_reason=pl.col("exclusion_reason").fill_null("absent_from_universe"),
            policy_version=pl.lit(REVISION),
        )
    )
    if blocks is not None and blocks.height:
        df = df.join(blocks, on=["session", "instrument_id"], how="left")
    df = df.with_columns(
        entry_blocked=pl.col("entry_blocked").fill_null(False) if "entry_blocked" in df.columns else pl.lit(False),
        entry_block_reason=pl.col("entry_block_reason").fill_null("")
        if "entry_block_reason" in df.columns
        else pl.lit(""),
    )
    if df.filter(pl.col("tick_size").is_null() | pl.col("sell_tax_rate").is_null()).height:
        bad = df.filter(pl.col("tick_size").is_null() | pl.col("sell_tax_rate").is_null())
        raise PITDataError(f"market panel session outside rule coverage: {bad['session'].to_list()[0]}")
    return df.select(list(_SCHEMA))


def _rules_fingerprint(rules: KrxMarketRules) -> str:
    payload = {
        "version": rules.version,
        "tick_regimes": [
            {
                "effective_from": regime.effective_from.isoformat(),
                "bands": {
                    market.value: [
                        {"lower": band.lower_price_inclusive, "tick": band.tick}
                        for band in regime.bands[market]
                    ]
                    for market in (KrxMarket.KOSPI, KrxMarket.KOSDAQ)
                },
            }
            for regime in rules.tick_regimes
        ],
        "sell_tax_regimes": [
            {
                "effective_from": regime.effective_from.isoformat(),
                "rates": {market.value: str(regime.rates[market]) for market in (KrxMarket.KOSPI, KrxMarket.KOSDAQ)},
            }
            for regime in rules.sell_tax_regimes
        ],
        "price_limit_regimes": [
            {
                "effective_from": regime.effective_from.isoformat(),
                "ratio": str(regime.ratio),
            }
            for regime in rules.price_limit_regimes
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def materialize_market_panel(
    *,
    daily_market_path: Path,
    universe_path: Path,
    rules: KrxMarketRules,
    gold_root: Path,
    policy: MarketPanelPolicy = MarketPanelPolicy(),  # noqa: B008
    instrument_buckets: int = 16,
    market_actions_path: Path | None = None,
) -> MarketPanelResult:
    """Build and publish the decision-safe session-by-instrument market panel."""

    if (
        isinstance(policy.adtv_short_sessions, bool)
        or isinstance(policy.adtv_long_sessions, bool)
        or isinstance(policy.return_vol_sessions, bool)
        or policy.adtv_short_sessions < 1
        or policy.adtv_long_sessions < 1
        or policy.return_vol_sessions < 1
    ):
        raise PITDataError("market panel windows must be positive integers")
    if isinstance(instrument_buckets, bool) or not isinstance(instrument_buckets, int) or instrument_buckets < 1:
        raise PITDataError("market panel instrument buckets must be a positive integer")
    threshold = policy.limitless_move_audit_threshold
    max_allowed = policy.max_unexplained_limitless_moves
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < float(threshold) < 1:
        raise PITDataError("market panel limitless_move_audit_threshold must be in (0, 1)")
    if isinstance(max_allowed, bool) or not isinstance(max_allowed, int) or max_allowed < 0:
        raise PITDataError("market panel max_unexplained_limitless_moves must be a non-negative integer")
    threshold_value = float(threshold)
    daily_id, daily_sessions, daily_by_session = _load_input_manifest(
        Path(daily_market_path), expected_kind="daily_market"
    )
    universe_id, universe_sessions_, universe_by_session = _load_input_manifest(
        Path(universe_path), expected_kind="ordinary_universe"
    )
    if daily_sessions != universe_sessions_:
        raise PITDataError("market panel inputs cover different session sets")
    calendar = daily_sessions
    if (
        not calendar
        or calendar[0] < rules.tick_regimes[0].effective_from
        or calendar[0] < rules.sell_tax_regimes[0].effective_from
        or calendar[0] < rules.price_limit_regimes[0].effective_from
    ):
        raise PITDataError(f"market panel session outside rule coverage: {calendar[0] if calendar else 'empty'}")

    identity = DatasetIdentity(
        kind="market_panel",
        layer=DatasetLayer.GOLD,
        policy_version=REVISION,
        inputs={
            "daily_market": dataset_reference(daily_id, kind="daily_market"),
            "universe": dataset_reference(universe_id, kind="ordinary_universe"),
            **(
                {"market_actions": dataset_reference(Path(market_actions_path).name, kind="market_actions")}
                if market_actions_path is not None
                else {}
            ),
        },
        params={
            "adtv_short_sessions": policy.adtv_short_sessions,
            "adtv_long_sessions": policy.adtv_long_sessions,
            "return_vol_sessions": policy.return_vol_sessions,
            "block_administrative": policy.block_administrative,
            "delisting_block_max_sessions": policy.delisting_block_max_sessions,
            "rules_version": rules.version,
            "rules_fingerprint": _rules_fingerprint(rules),
        },
    )
    calendar_frame = pl.DataFrame(
        {"session": calendar, "pos": list(range(len(calendar)))},
        schema={"session": pl.Date, "pos": pl.Int64},
    )
    blocks_all: pl.DataFrame | None = None
    if market_actions_path is not None:
        blocks_all = _entry_block_frame(
            Path(market_actions_path),
            calendar,
            block_administrative=policy.block_administrative,
            delisting_block_max_sessions=policy.delisting_block_max_sessions,
        )
    daily_files = [str(daily_by_session[session]) for session in calendar]
    partitions: dict[str, pl.DataFrame] = {}
    partition_details: list[dict[str, object]] = []
    corporate_action_rows = 0
    gap_rows = 0
    limit_inapplicable_rows = 0
    open_at_upper_rows = 0
    open_at_lower_rows = 0
    entry_blocked_rows = 0
    total_rows = 0
    last_obs: dict[str, tuple[int, int, int]] = {}
    audit_candidates: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="market-panel-") as temporary:
        shard_dir = Path(temporary)
        instrument_ids = (
            pl.scan_parquet(daily_files).select("instrument_id").unique().collect()["instrument_id"].to_list()
        )
        bucket_members: list[frozenset[str]] = [
            frozenset(
                str(iid) for iid in instrument_ids if _bucket_of(str(iid), instrument_buckets) == bucket
            )
            for bucket in range(instrument_buckets)
        ]
        for bucket in range(instrument_buckets):
            members = bucket_members[bucket]
            if members:
                daily_bucket = (
                    pl.scan_parquet(daily_files)
                    .filter(pl.col("instrument_id").is_in(members))
                    .collect()
                )
                universe_frames: list[pl.DataFrame] = []
                for session in calendar:
                    universe_frame = pl.read_parquet(universe_by_session[session])
                    if "session" not in universe_frame.columns:
                        universe_frame = universe_frame.with_columns(
                            pl.lit(session, dtype=pl.Date).alias("session")
                        )
                    universe_frames.append(
                        universe_frame.select("session", "instrument_id", "eligible", "exclusion_reason")
                    )
                universe_bucket = pl.concat(universe_frames, how="vertical_relaxed").filter(
                    pl.col("instrument_id").is_in(members)
                )
            else:
                daily_bucket = pl.scan_parquet(daily_files).filter(pl.lit(False)).collect()
                universe_bucket = pl.DataFrame(
                    [],
                    schema={
                        "session": pl.Date,
                        "instrument_id": pl.String,
                        "eligible": pl.Boolean,
                        "exclusion_reason": pl.String,
                    },
                )
            frame = _build_bucket_frame(
                daily=daily_bucket,
                universe=universe_bucket,
                calendar_frame=calendar_frame,
                rules=rules,
                policy=policy,
                blocks=blocks_all.filter(pl.col("instrument_id").is_in(members))
                if blocks_all is not None and members
                else None,
            )
            corporate_action_rows += frame.filter(
                pl.col("share_factor").is_not_null() & (pl.col("share_factor") != 1.0)
            ).height
            gap_rows += frame.filter(pl.col("gap_before")).height
            limit_inapplicable_rows += frame.filter(~pl.col("limits_applicable")).height
            open_at_upper_rows += frame.filter(pl.col("open_at_upper")).height
            open_at_lower_rows += frame.filter(pl.col("open_at_lower")).height
            entry_blocked_rows += frame.filter(pl.col("entry_blocked")).height
            total_rows += frame.height
            if frame.height and market_actions_path is not None:
                firsts = frame.group_by("instrument_id").agg(
                    pl.col("session").min().alias("first_session")
                )
                joined = frame.join(firsts, on="instrument_id", how="left")
                candidates = (
                    joined.filter(
                        (pl.col("session") != pl.col("first_session"))
                        & pl.col("eligible")
                        & (pl.col("volume") > 0)
                        & (pl.col("base_price") > 0)
                        & (~pl.col("limits_applicable"))
                        & (
                            (
                                pl.col("close").cast(pl.Float64)
                                / pl.col("base_price").cast(pl.Float64)
                                - 1.0
                            ).abs()
                            > threshold_value
                        )
                        & (~pl.col("entry_blocked"))
                    )
                    .select("instrument_id", "session", "close", "base_price")
                    .to_dicts()
                )
                for candidate in candidates:
                    base = int(candidate["base_price"])
                    audit_candidates.append(
                        {
                            "instrument_id": str(candidate["instrument_id"]),
                            "session": candidate["session"],
                            "move": float(int(candidate["close"]) / base - 1.0),
                        }
                    )
            if frame.height:
                tails = frame.sort(["instrument_id", "session"]).group_by("instrument_id").agg(
                    pl.col("session").last().alias("session"),
                    pl.col("close").last().alias("close"),
                    pl.col("volume").last().alias("volume"),
                )
                pos_of = {session: index for index, session in enumerate(calendar)}
                for row in tails.to_dicts():
                    last_obs[str(row["instrument_id"])] = (
                        pos_of[row["session"]],
                        int(row["close"]),
                        int(row["volume"]),
                    )
            shard_path = shard_dir / f"bucket={bucket:04d}.parquet"
            frame.write_parquet(shard_path)
            _LOG.info("[DATA] stage=market_panel bucket=%d/%d rows=%d", bucket + 1, instrument_buckets, frame.height)

        years = sorted({session.year for session in calendar})
        for year in years:
            year_frame = (
                pl.scan_parquet(str(shard_dir / "bucket=*.parquet"))
                .filter(pl.col("session").dt.year() == year)
                .collect()
                .sort(["session", "instrument_id"])
                .select(list(_SCHEMA))
            )
            relative_path = f"year={year}/part.parquet"
            partitions[relative_path] = year_frame
            partition_details.append({"year": year, "rows": year_frame.height})
            _LOG.info("[DATA] stage=market_panel year=%d rows=%d", year, year_frame.height)

    exits = [
        {
            "instrument_id": instrument_id,
            "last_session": calendar[position],
            "last_close": close,
            "last_volume": volume,
            "exit_kind": "halted_exit" if volume == 0 else "traded_exit",
        }
        for instrument_id, (position, close, volume) in sorted(last_obs.items())
        if position < len(calendar) - 1
    ]
    exits_frame = (
        pl.DataFrame(exits, schema=_EXITS_SCHEMA).sort("instrument_id")
        if exits
        else pl.DataFrame([], schema=_EXITS_SCHEMA)
    )
    partitions["instrument_exits.parquet"] = exits_frame
    exits_traded = int(exits_frame.filter(pl.col("exit_kind") == "traded_exit").height)
    exits_halted = int(exits_frame.filter(pl.col("exit_kind") == "halted_exit").height)
    if market_actions_path is None:
        audit_detail: object = "not_audited"
        unexplained = 0
    else:
        ordered = sorted(audit_candidates, key=lambda item: (str(item["session"]), str(item["instrument_id"])))
        unexplained = len(ordered)
        samples = [
            {"instrument_id": item["instrument_id"], "session": item["session"], "move": item["move"]}
            for item in ordered[:20]
        ]
        audit_detail = {
            "threshold": threshold_value,
            "max_allowed": max_allowed,
            "unexplained": unexplained,
            "samples": samples,
        }
        if unexplained > max_allowed:
            preview = "; ".join(
                f"{item['instrument_id']}@{item['session']}:{float(item['move']):.4f}"
                for item in ordered[:5]
            )
            raise PITDataError(
                f"unexplained limitless moves {unexplained} exceed {max_allowed}: {preview}"
            )
    details = {
        "years": years,
        "daily_market_dataset_id": daily_id,
        "universe_dataset_id": universe_id,
        "corporate_action_rows": corporate_action_rows,
        "gap_rows": gap_rows,
        "limit_inapplicable_rows": limit_inapplicable_rows,
        "open_at_upper_rows": open_at_upper_rows,
        "open_at_lower_rows": open_at_lower_rows,
        "entry_blocked_rows": entry_blocked_rows,
        "market_actions_dataset_id": Path(market_actions_path).name if market_actions_path is not None else None,
        "exits_traded": exits_traded,
        "exits_halted": exits_halted,
        "exits": {
            "path": "instrument_exits.parquet",
            "rows": exits_frame.height,
            "decision_safe": False,
        },
        "dividends": "not_integrated",
        "exits_decision_safe": False,
        "limitless_move_audit": audit_detail,
        "partitions": partition_details,
    }
    published = publish_dataset(
        layer_root=Path(gold_root),
        identity=identity,
        partitions=partitions,
        details=details,
    )
    return MarketPanelResult(
        dataset_path=published.path,
        dataset_id=published.dataset_id,
        rows=total_rows,
        years=tuple(years),
        corporate_action_rows=corporate_action_rows,
        gap_rows=gap_rows,
        limit_inapplicable_rows=limit_inapplicable_rows,
        open_at_upper_rows=open_at_upper_rows,
        open_at_lower_rows=open_at_lower_rows,
        exits_traded=exits_traded,
        exits_halted=exits_halted,
        entry_blocked_rows=entry_blocked_rows,
        unexplained_limitless_moves=unexplained,
    )
