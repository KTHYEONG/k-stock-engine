"""Dense session-by-instrument market arrays built from a Gold market panel."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from src.core.pit import PITDataError
from src.data.datasets import dataset_partition_paths

INT_FIELD_NAMES: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "base_price",
    "volume",
    "tick_size",
    "upper_limit",
    "lower_limit",
)
FLOAT_FIELD_NAMES: tuple[str, ...] = ("sell_tax_rate", "adtv20", "ret_vol60", "share_factor")
PANEL_BOOL_FIELD_NAMES: tuple[str, ...] = ("eligible", "open_at_upper", "open_at_lower")

_MARKET_CODES: dict[str, int] = {"KOSPI": 1, "KOSDAQ": 2}


@dataclass(frozen=True, slots=True)
class MarketArrays:
    """Dense session x instrument view of one Gold market-panel dataset.

    The engine loop is path dependent (integer shares, cash, events), so it
    walks sessions sequentially; dense arrays make every per-session step a
    vectorized slice instead of a frame filter. Prices are integer KRW so the
    ledger can stay exact.

    Attributes:
        dataset_id: Source ``market_panel_<hash16>`` id.
        sessions: Ascending trading dates (length S).
        instrument_ids: Ascending instrument ids (length N).
        int_fields: ``open, high, low, close, base_price, volume, tick_size,
            upper_limit, lower_limit`` as int64 SxN, ``0`` where absent.
        float_fields: ``sell_tax_rate, adtv20, ret_vol60, share_factor`` as
            float64 SxN, NaN where absent.
        bool_fields: ``present, eligible, open_at_upper, open_at_lower`` SxN.
        market: int8 SxN market code (1=KOSPI, 2=KOSDAQ, 0=absent).
    """

    dataset_id: str
    sessions: tuple[date, ...]
    instrument_ids: tuple[str, ...]
    int_fields: dict[str, NDArray[np.int64]]
    float_fields: dict[str, NDArray[np.float64]]
    bool_fields: dict[str, NDArray[np.bool_]]
    market: NDArray[np.int8]


def _read_manifest(panel_dir: Path) -> tuple[str, list[dict[str, Any]], bytes]:
    try:
        raw = (panel_dir / "manifest.json").read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        parts = manifest["partitions"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PITDataError(f"invalid market-panel manifest: {panel_dir}") from exc
    dataset_id = manifest.get("dataset_id")
    if dataset_id != panel_dir.name or not isinstance(parts, list) or not parts:
        raise PITDataError(f"invalid market-panel manifest: {panel_dir}")
    for part in parts:
        if (
            not isinstance(part, dict)
            or not isinstance(part.get("path"), str)
            or not isinstance(part.get("parquet_sha256", part.get("sha256")), str)
        ):
            raise PITDataError(f"invalid market-panel partition entry: {panel_dir}")
    return (cast("str", dataset_id), cast("list[dict[str, Any]]", parts), raw)


def _verified_files(panel_dir: Path, parts: list[dict[str, Any]]) -> list[str]:
    """Verify the dataset and return only the dense-panel partitions."""

    try:
        verified = dataset_partition_paths(panel_dir)
    except PITDataError as exc:
        raise PITDataError(f"market-panel partition verification failed: {panel_dir}") from exc
    files = [str(path) for path in verified if path.name not in {"instrument_exits.parquet", "exits.parquet"}]
    if not files:
        raise PITDataError(f"market-panel has no dense partitions: {panel_dir}")
    return files


def _scan_columns(files: list[str], columns: list[str]) -> pl.DataFrame:
    try:
        return pl.scan_parquet(files).select(columns).collect()
    except Exception as exc:
        raise PITDataError(f"market panel is missing columns {columns!r}") from exc


def _locate(
    frame: pl.DataFrame,
    sessions_ord: NDArray[np.int64],
    instruments: NDArray[np.str_],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    day_list = cast("list[date]", frame["session"].to_list())
    rows_ord = np.asarray([day.toordinal() for day in day_list], dtype=np.int64)
    rows_inst = np.asarray(cast("list[str]", frame["instrument_id"].to_list()))
    return (np.searchsorted(sessions_ord, rows_ord), np.searchsorted(instruments, rows_inst))


def _freeze(arrays: MarketArrays) -> None:
    for field in (*arrays.int_fields.values(), *arrays.float_fields.values(), *arrays.bool_fields.values()):
        field.flags.writeable = False
    arrays.market.flags.writeable = False


def _manifest_source_digest(raw_manifest: bytes, parts: list[dict[str, Any]]) -> str:
    """Bind a cache to the verified manifest and declared partition digests."""

    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw_manifest + b"\x00" + canonical).hexdigest()


def _array_checksum(source_digest: str, arrays: MarketArrays) -> str:
    """Return a content checksum for the materialized numeric arrays."""

    digest = hashlib.sha256(source_digest.encode("utf-8"))
    for name in INT_FIELD_NAMES:
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(arrays.int_fields[name]).tobytes())
    for name in FLOAT_FIELD_NAMES:
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(arrays.float_fields[name]).tobytes())
    for name in ("present", *PANEL_BOOL_FIELD_NAMES):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(arrays.bool_fields[name]).tobytes())
    digest.update(np.ascontiguousarray(arrays.market).tobytes())
    return digest.hexdigest()


def _load_cache(cache_path: Path, *, dataset_id: str, source_digest: str) -> MarketArrays | None:
    try:
        with np.load(str(cache_path), allow_pickle=False) as store:
            if str(store["dataset_id"]) != dataset_id or str(store["source_digest"]) != source_digest:
                return None
            cache_checksum = str(store["checksum"]) if "checksum" in store.files else ""
            sessions_ord = np.asarray(store["sessions"], dtype=np.int64)
            instruments = [str(item) for item in store["instruments"].tolist()]
            int_fields = {name: np.asarray(store[f"int_{name}"], dtype=np.int64) for name in INT_FIELD_NAMES}
            float_fields = {
                name: np.asarray(store[f"float_{name}"], dtype=np.float64) for name in FLOAT_FIELD_NAMES
            }
            bool_fields = {
                name: np.asarray(store[f"bool_{name}"], dtype=bool)
                for name in ("present", *PANEL_BOOL_FIELD_NAMES)
            }
            market = np.asarray(store["market"], dtype=np.int8)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    arrays = MarketArrays(
        dataset_id=dataset_id,
        sessions=tuple(date.fromordinal(int(item)) for item in sessions_ord.tolist()),
        instrument_ids=tuple(instruments),
        int_fields=int_fields,
        float_fields=float_fields,
        bool_fields=bool_fields,
        market=market,
    )
    expected_shape = (len(arrays.sessions), len(arrays.instrument_ids))
    if any(field.shape != expected_shape for field in (*arrays.int_fields.values(), *arrays.float_fields.values(), *arrays.bool_fields.values())):
        return None
    if arrays.market.shape != expected_shape:
        return None
    if cache_checksum != _array_checksum(source_digest, arrays):
        return None
    _freeze(arrays)
    return arrays


def _store_cache(
    cache_root: Path,
    cache_path: Path,
    *,
    dataset_id: str,
    source_digest: str,
    arrays: MarketArrays,
    sessions_ord: NDArray[np.int64],
    instruments: NDArray[np.str_],
) -> None:
    payload: dict[str, Any] = {
        "dataset_id": np.asarray(dataset_id),
        "source_digest": np.asarray(source_digest),
        "checksum": np.asarray(_array_checksum(source_digest, arrays)),
        "sessions": sessions_ord,
        "instruments": instruments,
        "market": arrays.market,
    }
    for name in INT_FIELD_NAMES:
        payload[f"int_{name}"] = arrays.int_fields[name]
    for name in FLOAT_FIELD_NAMES:
        payload[f"float_{name}"] = arrays.float_fields[name]
    for name in ("present", *PANEL_BOOL_FIELD_NAMES):
        payload[f"bool_{name}"] = arrays.bool_fields[name]
    tmp_path = cache_root / f".{dataset_id}.{os.getpid()}.tmp.npz"
    try:
        np.savez(str(tmp_path), **payload)
        os.replace(tmp_path, cache_path)
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink()


def load_market_arrays(*, panel_dir: Path, cache_root: Path) -> MarketArrays:
    """Load a market panel as dense arrays, reusing a verified on-disk cache.

    Args:
        panel_dir: ``gold/<scope>/market_panel_<hash16>`` directory with ``manifest.json``.
        cache_root: Directory for ``<dataset_id>.npz``-style caches keyed by dataset id.

    Returns:
        Read-only arrays for the whole panel.

    Raises:
        PITDataError: manifest ``dataset_id`` differs from the directory name, a
            partition sha256 mismatches its manifest, duplicate (session,
            instrument) rows exist, or a cache's recorded dataset id differs.
    """
    panel_dir = Path(panel_dir)
    cache_root = Path(cache_root)
    dataset_id, parts, raw_manifest = _read_manifest(panel_dir)
    # Verify the source before consulting the derived cache.  Otherwise a
    # stale cache can make a tampered panel appear healthy merely because its
    # manifest bytes are unchanged.
    files = _verified_files(panel_dir, parts)
    source_digest = _manifest_source_digest(raw_manifest, parts)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{dataset_id}.npz"
    if cache_path.is_file():
        cached = _load_cache(cache_path, dataset_id=dataset_id, source_digest=source_digest)
        if cached is not None:
            return cached
    keys = _scan_columns(files, ["session", "instrument_id"])
    if keys.select(["session", "instrument_id"]).is_duplicated().any():
        raise PITDataError(f"duplicate market panel key in {panel_dir}")
    sessions = tuple(sorted(cast("list[date]", keys["session"].unique().to_list())))
    instrument_ids = tuple(sorted(cast("list[str]", keys["instrument_id"].unique().to_list())))
    sessions_ord = np.asarray([day.toordinal() for day in sessions], dtype=np.int64)
    instruments = np.asarray(list(instrument_ids))
    size = (len(sessions), len(instrument_ids))
    int_fields = {name: np.zeros(size, dtype=np.int64) for name in INT_FIELD_NAMES}
    float_fields = {name: np.full(size, np.nan, dtype=np.float64) for name in FLOAT_FIELD_NAMES}
    bool_fields = {
        name: np.zeros(size, dtype=bool) for name in ("present", *PANEL_BOOL_FIELD_NAMES)
    }
    market = np.zeros(size, dtype=np.int8)
    present_idx = _locate(keys, sessions_ord, instruments)
    bool_fields["present"][present_idx] = True
    for name in INT_FIELD_NAMES:
        frame = _scan_columns(files, ["session", "instrument_id", name])
        idx = _locate(frame, sessions_ord, instruments)
        int_fields[name][idx] = frame[name].cast(pl.Int64).fill_null(0).to_numpy().astype(np.int64)
    for name in FLOAT_FIELD_NAMES:
        frame = _scan_columns(files, ["session", "instrument_id", name])
        idx = _locate(frame, sessions_ord, instruments)
        float_fields[name][idx] = (
            frame[name].cast(pl.Float64).fill_null(float("nan")).to_numpy().astype(np.float64)
        )
    for name in PANEL_BOOL_FIELD_NAMES:
        frame = _scan_columns(files, ["session", "instrument_id", name])
        idx = _locate(frame, sessions_ord, instruments)
        bool_fields[name][idx] = frame[name].cast(pl.Boolean).fill_null(False).to_numpy().astype(bool)
    market_frame = _scan_columns(files, ["session", "instrument_id", "market"])
    market_idx = _locate(market_frame, sessions_ord, instruments)
    market_codes = np.asarray(
        [_MARKET_CODES.get(item, 0) if isinstance(item, str) else 0 for item in market_frame["market"].to_list()],
        dtype=np.int8,
    )
    market[market_idx] = market_codes
    arrays = MarketArrays(
        dataset_id=dataset_id,
        sessions=sessions,
        instrument_ids=instrument_ids,
        int_fields=int_fields,
        float_fields=float_fields,
        bool_fields=bool_fields,
        market=market,
    )
    _freeze(arrays)
    _store_cache(
        cache_root,
        cache_path,
        dataset_id=dataset_id,
        source_digest=source_digest,
        arrays=arrays,
        sessions_ord=sessions_ord,
        instruments=instruments,
    )
    return arrays
