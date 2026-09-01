"""Hash-bound executable hedge overlay loader."""
from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

from src.core.costs import CostSchedule
from src.core.instruments import AssetKind, Instrument
from src.storage.parquet_datasets import ParquetDatasetStore

_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def load_executable_overlay_data(
    *,
    root: Path,
    dataset_id: str,
    instrument_id: str,
    beta: float,
    decision_time: datetime,
    expected_content_hash: str,
    base_cost_schedule: CostSchedule,
    stress_cost_schedule: CostSchedule,
) -> object:
    """Load one immutable inverse-ETF dataset with hash and type gates.

    Validates explicit inputs before any repository access: finite negative
    beta, 64-hex content hash, ETF asset kind, manifest hash equality,
    single instrument, monotonic timezone-aware sessions, and finite positive
    OHLC.
    """
    from src.stocks.ml.contracts import ExecutableOverlayData

    # R2 explicit beta validation
    try:
        beta_f = float(beta)
    except Exception as exc:
        raise ValueError("hedge-beta-invalid") from exc
    if not math.isfinite(beta_f) or beta_f >= 0.0:
        raise ValueError("hedge-beta-invalid")
    # R2 explicit hash validation before IO where possible
    if not isinstance(expected_content_hash, str) or not _HEX64_RE.match(expected_content_hash):
        raise ValueError("hedge-content-hash-mismatch")
    if not instrument_id or not isinstance(instrument_id, str):
        raise ValueError("hedge-instrument-id-invalid")
    if ":" not in instrument_id:
        raise ValueError("hedge-instrument-id-invalid")
    if not isinstance(decision_time, datetime):
        raise ValueError("hedge-decision-time-invalid")
    if decision_time.tzinfo is None:
        raise ValueError("hedge-decision-time-invalid")
    # Load manifest and gate asset kind / hash
    store = ParquetDatasetStore(Path(root))
    try:
        manifest = store.read_manifest(dataset_id)
    except FileNotFoundError as exc:
        raise ValueError("hedge-dataset-missing") from exc
    if manifest.asset_kind != AssetKind.ETF:
        raise ValueError("hedge-asset-kind-mismatch")
    if manifest.content_hash != expected_content_hash:
        raise ValueError("hedge-content-hash-mismatch")
    # Read dataset - reuse store read with manifest's feature_set
    frame = store.read(
        dataset_id,
        expected_asset_kind=AssetKind.ETF,
        expected_feature_set=manifest.feature_set,
        decision_time=decision_time,
    )
    if frame.is_empty():
        raise ValueError("hedge-price-invalid")
    # Require expected columns
    required = {"instrument_id", "session", "open", "high", "low", "close"}
    missing = sorted(required.difference(set(frame.columns)))
    if missing:
        raise ValueError(f"hedge-price-invalid:{','.join(missing)}")
    # Exactly one requested instrument
    unique_ids = frame["instrument_id"].unique().to_list()
    if len(unique_ids) != 1 or unique_ids[0] != instrument_id:
        # also check if requested instrument missing
        if instrument_id not in unique_ids:
            raise ValueError("hedge-instrument-missing")
        raise ValueError("hedge-instrument-mismatch")
    # Filter to requested instrument (already single)
    # Validate sessions: timezone-aware, unique, monotonic
    sessions = frame["session"].to_list()
    for s in sessions:
        if not isinstance(s, datetime) or s.tzinfo is None:
            raise ValueError("hedge-price-invalid")
    if len(set(sessions)) != len(sessions):
        raise ValueError("hedge-price-invalid")
    if sessions != sorted(sessions):
        raise ValueError("hedge-price-invalid")
    # Finite positive OHLC
    for col in ("open", "high", "low", "close"):
        vals = frame[col].to_list()
        for v in vals:
            try:
                fv = float(v)
            except Exception as exc:
                raise ValueError("hedge-price-invalid") from exc
            if not math.isfinite(fv) or fv <= 0.0:
                raise ValueError("hedge-price-invalid")
    # Volume optional but if present check finite?
    # Construct instrument object - beta never inferred
    exchange, symbol = instrument_id.split(":", 1)
    instrument = Instrument(instrument_id, AssetKind.ETF, exchange, symbol, "KRW")
    return ExecutableOverlayData(
        instrument=instrument,
        frame=frame,
        evidence_hash=expected_content_hash,
        base_cost_schedule=base_cost_schedule,
        stress_cost_schedule=stress_cost_schedule,
        beta=beta_f,
    )
