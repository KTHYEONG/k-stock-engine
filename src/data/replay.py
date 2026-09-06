"""Bounded point-in-time replay with streaming Gold publication."""
from __future__ import annotations

import pickle
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl

from src.core.datasets import DatasetCertification
from src.core.time import SessionCalendar
from src.data.schemas import PITDataError, SilverTable
from src.features.contracts import QvefFeaturePolicy, QvefFeatureRow
from src.strategy.scoring import ChampionScoreRow
from src.strategy.universe import UniverseDecision, UniversePolicy

_FLOW_LOOKBACK = 20

_FLOAT64_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trading_value",
        "market_cap",
        "shares_outstanding",
        "foreign_buy_value",
        "foreign_sell_value",
        "foreign_net_value",
        "institution_net_value",
        "retail_net_value",
        "value",
        "factor",
        "cash_amount",
    }
)


def _require_aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PITDataError(f"{name} must be timezone-aware")


def _as_float64(frame: pl.DataFrame) -> pl.DataFrame:
    present = [c for c in _FLOAT64_COLUMNS if c in frame.columns]
    if not present:
        return frame
    return frame.with_columns(pl.col(c).cast(pl.Float64, strict=False) for c in present)


def _literal_in_column_tz(dtype: object, decision_time: datetime) -> datetime:
    tz = getattr(dtype, "time_zone", None)
    if tz is None or decision_time.tzinfo is None:
        return decision_time
    from zoneinfo import ZoneInfo

    return decision_time.astimezone(ZoneInfo(str(tz)))


def _pit_available(frame: pl.DataFrame, decision_time: datetime) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "available_at" not in frame.columns:
        return frame.clear()
    literal = _literal_in_column_tz(frame["available_at"].dtype, decision_time)
    try:
        kept = frame.filter(pl.col("available_at") <= literal)
    except Exception as exc:
        raise PITDataError("invalid certified Silver table: available_at") from exc
    if kept.is_empty():
        return kept
    return kept


def _session_window(
    calendar: SessionCalendar, session: datetime, *, window: int, include_decision: bool
) -> tuple[datetime, ...]:
    idx = calendar.sessions.index(session)
    if include_decision:
        start = idx - window + 1
        if start < 0:
            # Late window start: return exactly what exists; downstream marks ineligible.
            start = 0
        return tuple(calendar.sessions[start : idx + 1])
    if idx < window:
        return ()
    return tuple(calendar.sessions[idx - window : idx])


@dataclass(frozen=True, slots=True)
class PITReplaySession:
    session: datetime
    decision_time: datetime
    security_master: pl.DataFrame
    daily_market: pl.DataFrame
    investor_flow: pl.DataFrame
    financial_facts: pl.DataFrame
    corporate_actions: pl.DataFrame


class PITReplayReader:
    def __init__(
        self,
        *,
        calendar: SessionCalendar,
        silver_root: Path | None = None,
        decision_time: datetime | None = None,
        security_master: pl.DataFrame | None = None,
        daily_market: pl.DataFrame | None = None,
        investor_flow: pl.DataFrame | None = None,
        financial_facts: pl.DataFrame | None = None,
        corporate_actions: pl.DataFrame | None = None,
        dataset_files: Mapping[SilverTable, tuple[Path, ...]] | None = None,
    ) -> None:
        self._calendar = calendar
        self._silver_root = Path(silver_root) if silver_root is not None else None
        self._bound_time = decision_time
        self._master = security_master
        self._daily = daily_market
        self._flow = investor_flow
        self._facts = financial_facts
        self._actions = corporate_actions
        self._dataset_files = dict(dataset_files or {})

    @classmethod
    def from_silver_root(
        cls, *, silver_root: Path, decision_time: datetime, calendar: SessionCalendar
    ) -> PITReplayReader:
        _require_aware(decision_time, "decision_time")
        root = Path(silver_root)
        if not root.exists():
            raise PITDataError("missing certified Silver table")
        from src.data.silver import latest_silver_dataset_path

        files: dict[SilverTable, tuple[Path, ...]] = {}
        for table in (
            SilverTable.DAILY_MARKET,
            SilverTable.SECURITY_MASTER,
            SilverTable.INVESTOR_FLOW,
            SilverTable.FINANCIAL_FACTS,
            SilverTable.CORPORATE_ACTIONS,
        ):
            dataset = latest_silver_dataset_path(root=root, table=table, decision_time=decision_time)
            paths = tuple(sorted(dataset.rglob("*.parquet")))
            files[table] = paths
        return cls(calendar=calendar, silver_root=root, decision_time=decision_time, dataset_files=files)

    @classmethod
    def from_frames_for_test(
        cls,
        *,
        calendar: SessionCalendar,
        security_master: pl.DataFrame,
        daily_market: pl.DataFrame,
        investor_flow: pl.DataFrame,
        financial_facts: pl.DataFrame,
        corporate_actions: pl.DataFrame,
    ) -> PITReplayReader:
        return cls(
            calendar=calendar,
            security_master=security_master,
            daily_market=daily_market,
            investor_flow=investor_flow,
            financial_facts=financial_facts,
            corporate_actions=corporate_actions,
        )

    def session_input(
        self,
        *,
        session: datetime,
        decision_time: datetime,
        universe_policy: UniversePolicy,
        qvef_policy: QvefFeaturePolicy,
    ) -> PITReplaySession:
        _ = qvef_policy
        _require_aware(session, "session")
        _require_aware(decision_time, "decision_time")
        if session > decision_time:
            raise PITDataError("session must not be after decision_time")
        if session not in self._calendar.sessions:
            raise PITDataError("calendar does not contain session")
        window = int(universe_policy.liquidity_window_sessions)
        daily_window = _session_window(self._calendar, session, window=window, include_decision=True)
        flow_window = _session_window(
            self._calendar, session, window=_FLOW_LOOKBACK, include_decision=False
        )
        if self._silver_root is not None:
            return self._session_input_lazy(
                session=session,
                decision_time=decision_time,
                daily_window=daily_window,
                flow_window=flow_window,
            )
        return self._session_input_frames(
            session=session, decision_time=decision_time, daily_window=daily_window, flow_window=flow_window
        )

    def _session_input_frames(
        self,
        *,
        session: datetime,
        decision_time: datetime,
        daily_window: tuple[datetime, ...],
        flow_window: tuple[datetime, ...],
    ) -> PITReplaySession:
        master_src = self._master if self._master is not None else pl.DataFrame()
        daily_src = self._daily if self._daily is not None else pl.DataFrame()
        flow_src = self._flow if self._flow is not None else pl.DataFrame()
        facts_src = self._facts if self._facts is not None else pl.DataFrame()
        actions_src = self._actions if self._actions is not None else pl.DataFrame()

        daily_pit = _pit_available(daily_src, decision_time)
        if not daily_pit.is_empty() and "session" in daily_pit.columns and daily_window:
            allowed = set(daily_window)
            daily_slice = daily_pit.filter(pl.col("session").is_in(list(allowed)))
        elif not daily_pit.is_empty():
            daily_slice = daily_pit.clear()
        else:
            daily_slice = daily_pit

        flow_pit = _pit_available(flow_src, decision_time)
        if not flow_pit.is_empty() and "session" in flow_pit.columns and flow_window:
            allowed_f = set(flow_window)
            flow_slice = flow_pit.filter(pl.col("session").is_in(list(allowed_f)))
        elif not flow_pit.is_empty():
            flow_slice = flow_pit.clear()
        else:
            flow_slice = flow_pit

        master_pit = _pit_available(master_src, decision_time)
        if not master_pit.is_empty() and "valid_from" in master_pit.columns:
            kept_rows: list[dict[str, object]] = []
            for row in master_pit.to_dicts():
                vf = row.get("valid_from")
                vt = row.get("valid_to")
                if vf is None:
                    continue
                try:
                    if vf <= session and (vt is None or session <= vt):
                        kept_rows.append(row)
                except TypeError:
                    continue
            master_slice = pl.DataFrame(kept_rows) if kept_rows else master_pit.clear()
        else:
            master_slice = master_pit

        facts_pit = _pit_available(facts_src, decision_time)
        # Fiscal-period filtering is forbidden; availability governs corrections.
        eligible: set[str] | None = None
        if not master_slice.is_empty() and "company_id" in master_slice.columns:
            eligible = {str(v) for v in master_slice["company_id"].to_list() if v is not None}
        if not facts_pit.is_empty() and eligible is not None and "company_id" in facts_pit.columns:
            facts_slice = facts_pit.filter(pl.col("company_id").is_in(sorted(eligible)))
        else:
            facts_slice = facts_pit

        actions_slice = _pit_available(actions_src, decision_time)

        return PITReplaySession(
            session=session,
            decision_time=decision_time,
            security_master=_as_float64(master_slice),
            daily_market=_as_float64(daily_slice),
            investor_flow=_as_float64(flow_slice),
            financial_facts=_as_float64(facts_slice),
            corporate_actions=_as_float64(actions_slice),
        )

    def _session_input_lazy(
        self,
        *,
        session: datetime,
        decision_time: datetime,
        daily_window: tuple[datetime, ...],
        flow_window: tuple[datetime, ...],
    ) -> PITReplaySession:
        assert self._silver_root is not None

        def _scan(
            table: SilverTable,
            *,
            upper: datetime | None = None,
            lower: datetime | None = None,
        ) -> pl.LazyFrame:
            files = list(self._dataset_files.get(table, ()))
            if upper is not None or lower is not None:
                selected: list[Path] = []
                for path in files:
                    match = re.search(r"year=(\d{4})/month=(\d{2})", str(path))
                    if match is None:
                        selected.append(path)
                        continue
                    bound = upper if upper is not None else lower
                    assert bound is not None
                    month = datetime(
                        int(match.group(1)), int(match.group(2)), 1, tzinfo=bound.tzinfo
                    )
                    if upper is not None and month > upper.replace(day=1, hour=0, minute=0, second=0, microsecond=0):
                        continue
                    if lower is not None:
                        lower_month = lower.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                        if month < lower_month:
                            continue
                    selected.append(path)
                files = selected
            if not files:
                raise PITDataError(f"invalid certified Silver table: {table.value}")
            return pl.scan_parquet([str(p) for p in files])

        def _collect_filtered(
            lazy: pl.LazyFrame,
            table: SilverTable,
            *,
            sessions: tuple[datetime, ...] = (),
            company_ids: tuple[str, ...] | None = None,
        ) -> pl.DataFrame:
            try:
                schema = lazy.collect_schema()
                cols = list(schema.names())
                plan = lazy
                if "available_at" in cols:
                    literal = _literal_in_column_tz(schema.get("available_at"), decision_time)
                    plan = plan.filter(pl.col("available_at") <= literal)
                if sessions and "session" in cols:
                    # Calendar keys are trading dates at midnight; source rows
                    # may carry the same date at the market-open timestamp.
                    plan = plan.filter(
                        pl.col("session").cast(pl.Date).is_in([item.date() for item in sessions])
                    )
                if company_ids is not None and "company_id" in cols:
                    plan = plan.filter(pl.col("company_id").is_in(list(company_ids)))
                frame = plan.collect()
            except Exception as exc:
                raise PITDataError(f"invalid certified Silver table: {table.value}") from exc
            if "available_at" in frame.columns:
                frame = frame.filter(pl.col("available_at").is_not_null())
            return frame

        daily_frame = _collect_filtered(
            _scan(
                SilverTable.DAILY_MARKET,
                lower=daily_window[0] if daily_window else None,
                upper=daily_window[-1] if daily_window else session,
            ),
            SilverTable.DAILY_MARKET,
            sessions=daily_window,
        )

        flow_frame = _collect_filtered(
            _scan(
                SilverTable.INVESTOR_FLOW,
                lower=flow_window[0] if flow_window else None,
                upper=flow_window[-1] if flow_window else session,
            ),
            SilverTable.INVESTOR_FLOW,
            sessions=flow_window,
        )
        if not flow_frame.is_empty() and "session" in flow_frame.columns and flow_window:
            flow_frame = flow_frame.filter(pl.col("session").is_in(list(set(flow_window))))
        elif not flow_frame.is_empty():
            flow_frame = flow_frame.clear()

        master_lazy = _scan(SilverTable.SECURITY_MASTER, upper=session)
        master_schema = master_lazy.collect_schema()
        if "valid_from" in master_schema.names():
            master_lazy = master_lazy.filter(pl.col("valid_from").cast(pl.Date) <= session.date())
            if "valid_to" in master_schema.names():
                master_lazy = master_lazy.filter(
                    pl.col("valid_to").is_null()
                    | (pl.col("valid_to").cast(pl.Date) >= session.date())
                )
        master_frame = _collect_filtered(master_lazy, SilverTable.SECURITY_MASTER)
        if not master_frame.is_empty() and "valid_from" in master_frame.columns:
            rows: list[dict[str, object]] = []
            for row in master_frame.to_dicts():
                vf = row.get("valid_from")
                vt = row.get("valid_to")
                vf_date = vf.date() if isinstance(vf, datetime) else vf
                vt_date = vt.date() if isinstance(vt, datetime) else vt
                if vf_date is not None and vf_date <= session.date() and (vt_date is None or session.date() <= vt_date):
                    rows.append(row)
            master_frame = pl.DataFrame(rows) if rows else master_frame.clear()

        eligible_ids = tuple(
            sorted({str(v) for v in master_frame["company_id"].to_list() if v is not None})
        ) if "company_id" in master_frame.columns else ()
        facts_frame = _collect_filtered(
            _scan(SilverTable.FINANCIAL_FACTS, upper=decision_time),
            SilverTable.FINANCIAL_FACTS,
            company_ids=eligible_ids,
        )

        actions_frame = _collect_filtered(
            _scan(SilverTable.CORPORATE_ACTIONS, upper=session), SilverTable.CORPORATE_ACTIONS
        )
        return PITReplaySession(
            session=session,
            decision_time=decision_time,
            security_master=_as_float64(master_frame),
            daily_market=_as_float64(daily_frame),
            investor_flow=_as_float64(flow_frame),
            financial_facts=_as_float64(facts_frame),
            corporate_actions=_as_float64(actions_frame),
        )


class StreamingGoldWriter:
    def __init__(
        self,
        *,
        root: Path,
        dataset_id: str,
        decision_time: datetime,
        certification: DatasetCertification,
        source_hashes: Mapping[str, str],
        expected_sessions: tuple[datetime, ...],
        require_scores: bool = True,
    ) -> None:
        _require_aware(decision_time, "decision_time")
        if not dataset_id or not dataset_id.strip():
            raise PITDataError("dataset_id must be non-empty")
        if not expected_sessions:
            raise PITDataError("expected_sessions must be non-empty")
        for sess in expected_sessions:
            _require_aware(sess, "expected session")
        if sorted(expected_sessions) != list(expected_sessions):
            raise PITDataError("expected_sessions must be chronological")
        self._root = Path(root)
        self._dataset_id = dataset_id
        self._decision_time = decision_time
        self._certification = certification
        self._source_hashes = dict(source_hashes)
        self._expected = tuple(expected_sessions)
        self._require_scores = bool(require_scores)
        self._universe_batches: list[Path] = []
        self._feature_batches: list[Path] = []
        self._score_batches: list[Path] = []
        self._universe_sessions: set[datetime] = set()
        self._eligible_sessions: set[datetime] = set()
        self._feature_sessions: set[datetime] = set()
        self._score_sessions: set[datetime] = set()

    def append_universe(self, decisions: tuple[UniverseDecision, ...]) -> None:
        for item in decisions:
            if not isinstance(item.instrument_id, str) or not item.instrument_id.strip():
                raise PITDataError("invalid universe key")
            _require_aware(item.decision_session, "decision_session")
        self._universe_sessions.update(item.decision_session for item in decisions)
        self._eligible_sessions.update(item.decision_session for item in decisions if item.eligible)
        self._universe_batches.append(self._write_batch("universe", decisions))

    def append_features(self, rows: tuple[QvefFeatureRow, ...]) -> None:
        for item in rows:
            _require_aware(item.decision_session, "decision_session")
        self._feature_sessions.update(item.decision_session for item in rows)
        self._feature_batches.append(self._write_batch("features", rows))

    def append_scores(self, rows: tuple[ChampionScoreRow, ...]) -> None:
        for item in rows:
            _require_aware(item.decision_session, "decision_session")
        self._score_sessions.update(item.decision_session for item in rows)
        self._score_batches.append(self._write_batch("scores", rows))

    def _write_batch(self, kind: str, rows: object) -> Path:
        staging = self._root / f".staging-{self._dataset_id}" / kind
        staging.mkdir(parents=True, exist_ok=True)
        marker = staging / f"batch-{len(list(staging.glob('batch-*.pkl'))):05d}.pkl"
        if not isinstance(rows, tuple):  # pragma: no cover - typed append APIs only
            raise PITDataError("invalid Gold staging batch")
        with marker.open("wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return marker

    @staticmethod
    def _load_batches(paths: list[Path]) -> tuple[object, ...]:
        rows: list[object] = []
        for path in paths:
            with path.open("rb") as handle:
                batch = pickle.load(handle)  # noqa: S301 - private local staging only
            if not isinstance(batch, tuple):  # pragma: no cover - private staging corruption
                raise PITDataError("invalid Gold staging batch")
            rows.extend(batch)
        return tuple(rows)

    def close(self) -> Mapping[str, Path]:
        expected = set(self._expected)
        universe_sessions = self._universe_sessions
        if universe_sessions != expected:
            raise ValueError(f"incomplete requested sessions: {sorted(expected - universe_sessions)!r} missing")
        eligible = self._eligible_sessions
        if eligible and self._require_scores:
            feature_sessions = self._feature_sessions
            score_sessions = self._score_sessions
            if not eligible <= feature_sessions or not eligible <= score_sessions:
                raise ValueError("incomplete requested sessions: eligible sessions lack features/scores")
        # Early publication is forbidden; final dirs are created only below.
        universe_dir = self._root / "universe" / self._dataset_id
        qvef_dir = self._root / "qvef" / self._dataset_id
        scores_dir = self._root / "champion_scores" / self._dataset_id
        if universe_dir.exists() or qvef_dir.exists() or scores_dir.exists():
            raise ValueError("incomplete requested sessions: existing Gold mixes incompatible inputs")
        from src.features.contracts import QvefFeaturePolicy
        from src.features.materialize import materialize_qvef_features
        from src.strategy.scoring import ChampionScorePolicy, materialize_champion_scores
        from src.strategy.universe import UniversePolicy, materialize_historical_universe

        u_policy = UniversePolicy()
        f_policy = QvefFeaturePolicy()
        s_policy = ChampionScorePolicy()
        calendar_hash = str(self._source_hashes.get("calendar", "c"))
        master_hash = str(self._source_hashes.get("security_master", "m"))
        quality_hash = str(self._source_hashes.get("quality_report", self._dataset_id))
        out: dict[str, Path] = {}
        universe_rows = cast(tuple[UniverseDecision, ...], self._load_batches(self._universe_batches))
        universe_path = materialize_historical_universe(
            tuple(sorted(universe_rows, key=lambda d: (d.decision_session, d.instrument_id))),
            root=self._root / "universe",
            dataset_id=self._dataset_id,
            decision_time=self._decision_time,
            policy=u_policy,
            provider_version="official-pit-v1",
            calendar_hash=calendar_hash,
            master_hash=master_hash,
            quality_report_hash=quality_hash,
            certification=self._certification,
        )
        out["universe"] = universe_path
        features = cast(tuple[QvefFeatureRow, ...], self._load_batches(self._feature_batches))
        scores = cast(tuple[ChampionScoreRow, ...], self._load_batches(self._score_batches))
        if features:
            out["qvef"] = materialize_qvef_features(
                tuple(sorted(features, key=lambda r: (r.decision_session, r.instrument_id))),
                root=self._root / "qvef",
                dataset_id=self._dataset_id,
                decision_time=self._decision_time,
                policy=f_policy,
                provider_version="official-pit-v1",
                calendar_hash=calendar_hash,
                master_hash=master_hash,
                quality_report_hash=quality_hash,
                certification=self._certification,
            )
        if scores:
            out["champion_scores"] = materialize_champion_scores(
                tuple(sorted(scores, key=lambda r: (r.decision_session, r.instrument_id))),
                root=self._root / "champion_scores",
                dataset_id=self._dataset_id,
                decision_time=self._decision_time,
                policy=s_policy,
                provider_version="official-pit-v1",
                calendar_hash=calendar_hash,
                master_hash=master_hash,
                quality_report_hash=quality_hash,
                certification=self._certification,
            )
        staging_root = self._root / f".staging-{self._dataset_id}"
        if staging_root.exists():
            import shutil

            shutil.rmtree(staging_root, ignore_errors=True)
        return out
