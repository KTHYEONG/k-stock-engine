"""Fixed Champion v1 scoring and immutable score dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import polars as pl

from src.core.datasets import HIVE_PARTITION_LAYOUT, DatasetCertification, make_manifest
from src.core.instruments import AssetKind
from src.features.contracts import QvefFeatureRow
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash


@dataclass(frozen=True, slots=True)
class ChampionScorePolicy:
    version: str = "champion-v1-scoring-v1"
    required_feature_policy_version: str = "champion-v1-qvef-v1"
    min_required_factors: int = 4

    def __post_init__(self) -> None:
        if not self.version or not self.version.strip():
            raise ValueError("policy version must be non-empty")
        if not self.required_feature_policy_version or not self.required_feature_policy_version.strip():
            raise ValueError("required_feature_policy_version must be non-empty")
        if self.min_required_factors < 1 or self.min_required_factors > 4:
            raise ValueError("min_required_factors must be between 1 and 4")


class ChampionScoreReason(StrEnum):
    MISSING_QUALITY = "missing_quality"
    MISSING_VALUE = "missing_value"
    MISSING_EARNINGS = "missing_earnings"
    MISSING_FOREIGN_FLOW = "missing_foreign_flow"


@dataclass(frozen=True, slots=True)
class ChampionScoreRow:
    decision_session: datetime
    instrument_id: str
    eligible: bool
    champion_score: float | None
    rank: int | None
    exclusion_reasons: tuple[ChampionScoreReason, ...]
    feature_policy_version: str
    score_policy_version: str


@dataclass(slots=True)
class _Interim:
    row: QvefFeatureRow
    eligible: bool
    champion_score: float | None
    reasons: tuple[ChampionScoreReason, ...]
    rank: int | None = None


def score_champion_rows(
    rows: tuple[QvefFeatureRow, ...],
    *,
    decision_time: datetime,
    policy: ChampionScorePolicy = ChampionScorePolicy(),  # noqa: B008
) -> tuple[ChampionScoreRow, ...]:
    if not rows:
        raise ValueError("rows must be non-empty")
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if not policy.version or not policy.version.strip():
        raise ValueError("policy version must be non-empty")
    if not policy.required_feature_policy_version or not policy.required_feature_policy_version.strip():
        raise ValueError("policy required_feature_policy_version must be non-empty")

    # Validate each row preconditions and collect decision_sessions
    seen_ids: set[str] = set()
    decision_sessions: set[datetime] = set()
    for row in rows:
        # instrument_id blank or duplicate
        if not row.instrument_id or not row.instrument_id.strip():
            raise ValueError("instrument_id must be non-empty")
        if row.instrument_id in seen_ids:
            raise ValueError(f"duplicate instrument_id {row.instrument_id!r}")
        seen_ids.add(row.instrument_id)

        # decision_session naive and future
        if row.decision_session.tzinfo is None:
            raise ValueError("decision_session must be timezone-aware")
        if row.decision_session > decision_time:
            raise ValueError("decision_session must not be after decision_time")
        decision_sessions.add(row.decision_session)

        # source_available_at validation
        saa = row.source_available_at
        if saa is None or not isinstance(saa, tuple) or len(saa) == 0:
            raise ValueError("source_available_at must be a non-empty tuple")
        source_names: set[str] = set()
        for saa_entry in saa:
            if not isinstance(saa_entry, tuple) or len(saa_entry) != 2:
                raise ValueError("source_available_at entry must be a (source, available_at) tuple")
            source, avail = saa_entry
            if not isinstance(source, str) or not source.strip():
                raise ValueError("source_available_at source name must be non-blank")
            if source in source_names:
                raise ValueError(f"duplicate source_available_at source {source!r}")
            source_names.add(source)
            if not isinstance(avail, datetime):
                raise ValueError("source_available_at timestamp must be a datetime")
            if avail.tzinfo is None:
                raise ValueError("source_available_at timestamp must be timezone-aware")
            if avail > decision_time:
                raise ValueError("source_available_at timestamp must not be after decision_time (available)")

        # policy_version mismatch
        if row.policy_version != policy.required_feature_policy_version:
            raise ValueError(f"policy version mismatch: {row.policy_version!r} != {policy.required_feature_policy_version!r} (policy)")

        # factor finite check
        for field in ("quality_score", "value_score", "earnings_score", "foreign_flow_score"):
            val = getattr(row, field)
            if val is not None and not math.isfinite(float(val)):
                raise ValueError(f"non-finite value for {field}: {val!r}")

    if len(decision_sessions) != 1:
        raise ValueError("mixed decision_session values")

    # Compute scores and reasons
    # First pass: determine eligibility and raw scores
    interim: list[_Interim] = []
    for row in rows:
        reasons: list[ChampionScoreReason] = []
        factors: list[float] = []
        if row.quality_score is None:
            reasons.append(ChampionScoreReason.MISSING_QUALITY)
        else:
            factors.append(float(row.quality_score))
        if row.value_score is None:
            reasons.append(ChampionScoreReason.MISSING_VALUE)
        else:
            factors.append(float(row.value_score))
        if row.earnings_score is None:
            reasons.append(ChampionScoreReason.MISSING_EARNINGS)
        else:
            factors.append(float(row.earnings_score))
        if row.foreign_flow_score is None:
            reasons.append(ChampionScoreReason.MISSING_FOREIGN_FLOW)
        else:
            factors.append(float(row.foreign_flow_score))
        reasons_sorted = tuple(sorted(reasons, key=lambda r: r.value))
        if len(factors) >= policy.min_required_factors:
            eligible = True
            champion_score = sum(factors) / float(len(factors))
            effective_reasons: tuple[ChampionScoreReason, ...] = ()
        else:
            eligible = False
            champion_score = None
            effective_reasons = reasons_sorted
        interim.append(
            _Interim(
                row=row,
                eligible=eligible,
                champion_score=champion_score,
                reasons=effective_reasons,
            )
        )

    # Rank eligible rows
    eligible_entries = [e for e in interim if e.eligible]
    eligible_entries_sorted: list[_Interim] = sorted(
        eligible_entries,
        key=lambda e: (-float(e.champion_score) if e.champion_score is not None else 0.0, e.row.instrument_id),
    )
    for idx, scored in enumerate(eligible_entries_sorted, start=1):
        scored.rank = idx

    # Map instrument_id to rank for fast lookup
    rank_by_id: dict[str, int] = {
        e.row.instrument_id: e.rank
        for e in eligible_entries_sorted
        if e.rank is not None
    }

    # Build final ChampionScoreRow objects
    result: list[ChampionScoreRow] = []
    for scored_entry in interim:
        r = scored_entry.row
        eligible = scored_entry.eligible
        cs = scored_entry.champion_score
        scored_reasons = scored_entry.reasons
        if eligible:
            rank_val = rank_by_id[r.instrument_id]
            assert not scored_reasons
        else:
            rank_val = None
        result.append(
            ChampionScoreRow(
                decision_session=r.decision_session,
                instrument_id=r.instrument_id,
                eligible=eligible,
                champion_score=cs,
                rank=rank_val,
                exclusion_reasons=scored_reasons,
                feature_policy_version=r.policy_version,
                score_policy_version=policy.version,
            )
        )

    # Return sorted by instrument_id ascending, preserving rank
    result_sorted = tuple(sorted(result, key=lambda x: x.instrument_id))
    return result_sorted


def materialize_champion_scores(
    scores: tuple[ChampionScoreRow, ...],
    *,
    root: Path,
    dataset_id: str,
    decision_time: datetime,
    policy: ChampionScorePolicy,
    provider_version: str,
    calendar_hash: str,
    master_hash: str,
    quality_report_hash: str,
    certification: DatasetCertification,
) -> Path:
    if not scores:
        raise ValueError("scores must be non-empty")
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if not policy.version or not policy.version.strip():
        raise ValueError("policy version must be non-empty")
    if not policy.required_feature_policy_version or not policy.required_feature_policy_version.strip():
        raise ValueError("policy required_feature_policy_version must be non-empty")
    if not provider_version or not provider_version.strip():
        raise ValueError("provider_version must be non-empty")
    if not calendar_hash or not calendar_hash.strip():
        raise ValueError("calendar_hash must be non-empty")
    if not master_hash or not master_hash.strip():
        raise ValueError("master_hash must be non-empty")
    if not quality_report_hash or not quality_report_hash.strip():
        raise ValueError("quality_report_hash must be non-empty")
    if certification not in (DatasetCertification.RESEARCH, DatasetCertification.PRODUCTION):
        raise ValueError("certification must be RESEARCH or PRODUCTION")
    if not dataset_id or not dataset_id.strip():
        raise ValueError("dataset_id must be non-empty")

    root = Path(root)
    dataset_dir = root / dataset_id
    if dataset_dir.exists():
        raise ValueError(f"dataset already exists: {dataset_id}")

    seen_keys: set[tuple[datetime, str]] = set()
    for s in scores:
        key = (s.decision_session, s.instrument_id)
        if key in seen_keys:
            raise ValueError(f"duplicate key {key!r}")
        seen_keys.add(key)

        if s.decision_session.tzinfo is None:
            raise ValueError("decision_session must be timezone-aware")
        if s.decision_session > decision_time:
            raise ValueError("decision_session must not be after decision_time")

        if not s.score_policy_version or not s.score_policy_version.strip():
            raise ValueError("score_policy_version must be non-empty")
        if not s.feature_policy_version or not s.feature_policy_version.strip():
            raise ValueError("feature_policy_version must be non-empty")
        if s.score_policy_version != policy.version:
            raise ValueError(f"policy version mismatch: {s.score_policy_version!r} != {policy.version!r} (policy)")
        if s.feature_policy_version != policy.required_feature_policy_version:
            raise ValueError(f"policy version mismatch: {s.feature_policy_version!r} != {policy.required_feature_policy_version!r} (policy)")

        if s.eligible and s.exclusion_reasons:
            raise ValueError(f"eligible row must have no exclusion_reasons: {s.instrument_id}")
        if not s.eligible and not s.exclusion_reasons:
            raise ValueError(f"ineligible row must have exclusion_reasons: {s.instrument_id}")

        if s.eligible:
            if s.champion_score is None:
                raise ValueError(f"eligible row missing champion_score: {s.instrument_id}")
            if not math.isfinite(float(s.champion_score)):
                raise ValueError(f"non-finite champion_score for {s.instrument_id!r}")
            if s.rank is None:
                raise ValueError(f"eligible row missing rank: {s.instrument_id}")
            if not isinstance(s.rank, int) or s.rank <= 0:
                raise ValueError(f"rank must be positive integer: {s.instrument_id}")
        else:
            if s.champion_score is not None:
                raise ValueError(f"ineligible row must have no champion_score: {s.instrument_id}")
            if s.rank is not None:
                raise ValueError(f"ineligible row must have no rank: {s.instrument_id}")

    from collections import defaultdict

    eligible_by_session: dict[datetime, list[ChampionScoreRow]] = defaultdict(list)
    for s in scores:
        if s.eligible:
            eligible_by_session[s.decision_session].append(s)

    for sess_scores in eligible_by_session.values():
        scored_values: list[tuple[ChampionScoreRow, float, int]] = [
            (s, float(s.champion_score), int(s.rank))  # type: ignore[arg-type]
            for s in sess_scores
        ]
        expected_ranks = list(range(1, len(sess_scores) + 1))
        actual_ranks = sorted(rank for _, _, rank in scored_values)
        if actual_ranks != expected_ranks:
            raise ValueError("eligible ranks must be unique and consecutive starting at 1")

    ordered_columns = [
        "decision_session",
        "instrument_id",
        "eligible",
        "champion_score",
        "rank",
        "exclusion_reasons",
        "feature_policy_version",
        "score_policy_version",
        "generated_at",
    ]

    # Build records sorted by instrument_id for determinism
    sorted_scores = sorted(scores, key=lambda x: (x.instrument_id, x.decision_session))
    records: list[dict[str, object]] = []
    for s in sorted_scores:
        excl_str = ",".join(r.value for r in s.exclusion_reasons)
        records.append(
            {
                "decision_session": s.decision_session,
                "instrument_id": s.instrument_id,
                "eligible": bool(s.eligible),
                "champion_score": s.champion_score,
                "rank": s.rank,
                "exclusion_reasons": excl_str,
                "feature_policy_version": s.feature_policy_version,
                "score_policy_version": s.score_policy_version,
                "generated_at": decision_time,
            }
        )

    frame = pl.DataFrame(
        records,
        schema_overrides={"champion_score": pl.Float64, "rank": pl.Int64},
    ).select(ordered_columns)

    sessions = [s.decision_session for s in sorted_scores]
    time_start = min(sessions)
    time_end = max(sessions)

    content_hash = canonical_content_hash(frame, ordered_columns)

    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set="stock_champion_scores_v1",
        label_definition="none",
        label_horizon_sessions=1,
        time_start=time_start,
        time_end=time_end,
        provider_version=provider_version,
        universe_policy_version=policy.version,
        row_count=frame.height,
        generated_time=decision_time,
        certification=certification,
        calendar_hash=calendar_hash,
        master_hash=master_hash,
        quality_report_hash=quality_report_hash,
        schema_version="v2",
        content_hash=content_hash,
        storage_layout=HIVE_PARTITION_LAYOUT,
    )

    store = ParquetDatasetStore(root)
    path = store.write_partitioned(
        frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set="stock_champion_scores_v1",
        decision_time=decision_time,
    )
    return path
