"""Frictionless Gold reference benchmark indices built from the market panel."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import polars as pl

from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    dataset_partition_paths,
    dataset_reference,
    load_manifest,
    publish_dataset,
)
from src.data.schemas import PITDataError

DEFINITIONS_VERSION = "reference-benchmarks-v1"

_COLUMNS: list[str] = [
    "session",
    "instrument_id",
    "eligible",
    "price_state",
    "adtv20",
    "market_cap",
    "ret_price",
]

_OUTPUT_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "benchmark_id": pl.String,
    "ret": pl.Float64,
    "index_level": pl.Float64,
    "constituents": pl.Int32,
    "dropped_exit_weight": pl.Float64,
}


class Weighting(StrEnum):
    EQUAL = "equal"
    CAP = "cap"


@dataclass(frozen=True, slots=True)
class BenchmarkDefinition:
    benchmark_id: str
    weighting: Weighting
    min_adtv20_krw: float | None


@dataclass(frozen=True, slots=True)
class ReferenceBenchmarkResult:
    dataset_path: Path
    dataset_id: str
    sessions: int
    benchmarks: tuple[str, ...]
    dropped_exit_weight_max: float


def load_benchmark_definitions(path: Path) -> tuple[str, tuple[BenchmarkDefinition, ...]]:
    """Parse the versioned benchmark definition file.

    Raises:
        PITDataError: duplicate id, unknown weighting, or non-positive threshold.
    """
    try:
        document = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except OSError:
        raise
    except ValueError as exc:
        raise PITDataError(f"reference benchmark definitions are unreadable: {path}") from exc
    if not isinstance(document, dict) or document.get("version") != DEFINITIONS_VERSION:
        raise PITDataError(f"reference benchmark definitions require version {DEFINITIONS_VERSION!r}: {path}")
    raw_definitions = document.get("benchmarks")
    if not isinstance(raw_definitions, list):
        raise PITDataError(f"reference benchmark definitions require a benchmarks list: {path}")
    definitions: list[BenchmarkDefinition] = []
    seen: set[str] = set()
    for raw in raw_definitions:
        if not isinstance(raw, dict) or set(raw) - {"benchmark_id", "weighting", "min_adtv20_krw"}:
            raise PITDataError(f"reference benchmark definition has invalid keys: {raw!r}")
        benchmark_id = raw.get("benchmark_id")
        if not isinstance(benchmark_id, str) or not benchmark_id:
            raise PITDataError(f"reference benchmark definition requires a benchmark_id: {raw!r}")
        if benchmark_id in seen:
            raise PITDataError(f"duplicate reference benchmark id: {benchmark_id!r}")
        seen.add(benchmark_id)
        try:
            weighting = Weighting(str(raw.get("weighting")))
        except ValueError as exc:
            raise PITDataError(f"unknown weighting for {benchmark_id!r}: {raw.get('weighting')!r}") from exc
        raw_threshold = raw.get("min_adtv20_krw")
        if raw_threshold is None:
            threshold: float | None = None
        elif isinstance(raw_threshold, bool) or not isinstance(raw_threshold, (int, float)) or not raw_threshold > 0:
            raise PITDataError(f"non-positive threshold for {benchmark_id!r}: {raw_threshold!r}")
        else:
            threshold = float(raw_threshold)
        definitions.append(
            BenchmarkDefinition(benchmark_id=benchmark_id, weighting=weighting, min_adtv20_krw=threshold)
        )
    return (DEFINITIONS_VERSION, tuple(definitions))


def _load_panel_frame(market_panel_path: Path) -> tuple[str, pl.DataFrame, list[Any]]:
    try:
        paths = dataset_partition_paths(market_panel_path, allow_legacy=False)
        try:
            manifest = load_manifest(market_panel_path)
        except PITDataError:
            manifest = None
        if manifest is not None and manifest.kind != "market_panel":
            raise PITDataError(f"invalid reference-benchmark input kind: {market_panel_path}")
        frames: list[pl.DataFrame] = []
        for path in paths:
            try:
                candidate = pl.read_parquet(path)
            except (OSError, ValueError, pl.exceptions.PolarsError):
                continue
            if set(_COLUMNS).issubset(candidate.columns):
                frames.append(candidate.select(_COLUMNS))
    except PITDataError as exc:
        raise PITDataError(f"invalid reference-benchmark input manifest: {market_panel_path}") from exc
    if not frames:
        raise PITDataError(f"invalid reference-benchmark input manifest: {market_panel_path}")
    frame = pl.concat(frames, how="vertical_relaxed").sort(["instrument_id", "session"])
    sessions = frame.select("session").unique().sort("session")["session"].to_list()
    return market_panel_path.name, frame, sessions


def _benchmark_frame(
    *,
    base: pl.DataFrame,
    calendar: pl.DataFrame,
    sessions: list[Any],
    definition: BenchmarkDefinition,
) -> pl.DataFrame:
    weight = (
        pl.lit(1.0)
        if definition.weighting is Weighting.EQUAL
        else pl.col("market_cap").cast(pl.Float64)
    )
    candidates = (
        base.with_columns(
            w=weight,
            next_ret=pl.col("ret_price").shift(-1).over("instrument_id"),
            next_pos=pl.col("pos").shift(-1).over("instrument_id"),
            target_pos=pl.col("pos") + 1,
        )
        .join(
            calendar.select(pl.col("session").alias("t"), pl.col("pos").alias("target_pos")),
            on="target_pos",
            how="left",
        )
        .filter(
            pl.col("eligible")
            & (pl.col("price_state") == "tradable")
            & (
                pl.lit(True)
                if definition.min_adtv20_krw is None
                else pl.col("adtv20").is_not_null() & (pl.col("adtv20") >= definition.min_adtv20_krw)
            )
            & pl.col("t").is_not_null()
        )
        .select(
            "t",
            "w",
            "next_ret",
            kept=(pl.col("next_pos") == pl.col("target_pos")) & pl.col("next_ret").is_not_null(),
        )
    )
    grouped = (
        candidates.group_by("t", maintain_order=True)
        .agg(
            n_total=pl.len(),
            total_w=pl.col("w").sum(),
            kept_n=pl.col("kept").sum(),
            kept_w=pl.when(pl.col("kept")).then(pl.col("w")).otherwise(0.0).sum(),
            ret_num=pl.when(pl.col("kept")).then(pl.col("w") * pl.col("next_ret")).otherwise(0.0).sum(),
        )
        .sort("t")
    )
    if grouped.height == 0:
        raise PITDataError(f"benchmark {definition.benchmark_id} never forms a constituent set")
    # 유동성 필터 지수는 롤링 창이 찰 때까지 구성할 수 없다: 첫 구성 가능 세션의 전일이 기준일(1.0)이다.
    inception_index = sessions.index(grouped["t"][0]) - 1
    inception = sessions[inception_index]
    expected = sessions[inception_index + 1 :]
    formed = set(grouped["t"].to_list())
    missing = [session for session in expected if session not in formed]
    if missing:
        raise PITDataError(f"empty constituent set for {[str(session) for session in missing]}")
    if grouped.filter(pl.col("kept_n") == 0).height:
        raise PITDataError("no kept constituents for a session after the first")
    body = grouped.with_columns(
        session=pl.col("t"),
        benchmark_id=pl.lit(definition.benchmark_id),
        ret=pl.col("ret_num") / pl.col("kept_w"),
        constituents=pl.col("n_total").cast(pl.Int32),
        dropped_exit_weight=(pl.col("total_w") - pl.col("kept_w")) / pl.col("total_w"),
    ).select("session", "benchmark_id", "ret", "constituents", "dropped_exit_weight")
    first = pl.DataFrame(
        {
            "session": [inception],
            "benchmark_id": [definition.benchmark_id],
            "ret": [None],
            "constituents": [0],
            "dropped_exit_weight": [0.0],
        },
        schema={
            "session": pl.Date,
            "benchmark_id": pl.String,
            "ret": pl.Float64,
            "constituents": pl.Int32,
            "dropped_exit_weight": pl.Float64,
        },
    )
    return (
        pl.concat([first, body], how="vertical_relaxed")
        .sort("session")
        .with_columns(index_level=(pl.lit(1.0) + pl.col("ret").fill_null(0.0)).cum_prod())
        .select(list(_OUTPUT_SCHEMA))
    )


def materialize_reference_benchmarks(
    *,
    market_panel_path: Path,
    definitions: tuple[BenchmarkDefinition, ...],
    definitions_version: str,
    gold_root: Path,
) -> ReferenceBenchmarkResult:
    """Build and publish frictionless reference benchmark indices."""

    if not definitions:
        raise PITDataError("reference benchmarks require at least one definition")
    panel_id, frame, sessions = _load_panel_frame(Path(market_panel_path))
    calendar = frame.select("session").unique().sort("session").with_row_index("pos")
    base = frame.join(calendar, on="session", how="left").sort(["instrument_id", "session"])
    combined = pl.concat(
        [
            _benchmark_frame(
                base=base,
                calendar=calendar,
                sessions=sessions,
                definition=definition,
            )
            for definition in definitions
        ],
        how="vertical_relaxed",
    ).sort(["benchmark_id", "session"])
    definition_payload = [
        {
            "benchmark_id": definition.benchmark_id,
            "weighting": definition.weighting.value,
            "min_adtv20_krw": definition.min_adtv20_krw,
        }
        for definition in sorted(definitions, key=lambda item: item.benchmark_id)
    ]
    identity = DatasetIdentity(
        kind="reference_benchmarks",
        layer=DatasetLayer.GOLD,
        policy_version=definitions_version,
        inputs={"market_panel": dataset_reference(panel_id, kind="market_panel")},
        params={
            "definitions_version": definitions_version,
            "definitions": json.dumps(definition_payload, sort_keys=True, separators=(",", ":")),
        },
    )
    dropped_exit_weight_max = (
        float(cast("float", combined["dropped_exit_weight"].max()))
        if combined.height
        else 0.0
    )
    published = publish_dataset(
        layer_root=Path(gold_root),
        identity=identity,
        partitions={"benchmarks.parquet": combined},
        details={
            "definitions_version": definitions_version,
            "definitions": definition_payload,
            "sessions": len(sessions),
            "benchmarks": [definition.benchmark_id for definition in definitions],
            "dropped_exit_weight_max": dropped_exit_weight_max,
            "dividends": "not_integrated",
        },
    )
    return ReferenceBenchmarkResult(
        dataset_path=published.path,
        dataset_id=published.dataset_id,
        sessions=len(sessions),
        benchmarks=tuple(definition.benchmark_id for definition in definitions),
        dropped_exit_weight_max=dropped_exit_weight_max,
    )
