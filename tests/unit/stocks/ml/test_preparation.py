"""Prepared training matrix and label alignment contract tests."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from types import SimpleNamespace

from src.stocks.ml.preparation import (
    prepare_folds,
    prepare_horizon_labels,
    prepare_matrix_from_frame,
    prepare_training_matrix,
)
from src.stocks.research.folds import Fold


def _panel(n_sessions: int = 8, n_instruments: int = 3) -> pl.DataFrame:
    rows = []
    for session_index in range(n_sessions):
        session = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=session_index)
        rows.extend(
            {
                "instrument_id": f"KRX:{ticker + 1:05d}",
                "session": session,
                "feature__a": float(session_index) + ticker,
                "feature__b": float(ticker) * 0.5,
                "net_alpha_target": 0.01 * (session_index + ticker),
                "label_available_time": session + timedelta(days=5),
                "open": 100.0,
            }
            for ticker in range(n_instruments)
        )
    return pl.DataFrame(rows)


class _SchemaView:
    learner_columns = ("feature__a", "feature__b")


class _DataView:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.feature_frame = frame


def test_prepared_matrix_is_contiguous_float32_and_excludes_forbidden_columns() -> None:
    panel = _panel()
    matrix = prepare_matrix_from_frame(panel, ("feature__a", "feature__b"))
    assert matrix.X.dtype == np.float32
    assert matrix.X.flags["C_CONTIGUOUS"]
    assert matrix.X.shape == (panel.height, 2)
    assert matrix.num_rows == len(matrix.instrument_code)
    assert matrix.num_rows == len(matrix.session_code)
    for forbidden in ("net_alpha_target", "label_available_time", "open"):
        assert forbidden not in matrix.feature_columns


def test_prepared_sessions_are_chronological_codes() -> None:
    panel = _panel()
    matrix = prepare_matrix_from_frame(panel, ("feature__a",))
    first_row_session = matrix.session_timestamps_ns[matrix.session_code[0]]
    assert first_row_session == matrix.session_timestamps_ns.min()
    assert matrix.num_sessions == panel["session"].n_unique()


def test_duplicate_keys_fail_closed() -> None:
    panel = pl.concat([_panel(), _panel().head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        prepare_matrix_from_frame(panel, ("feature__a",))


def test_prepare_training_matrix_delegates_to_frame_path() -> None:
    panel = _panel()
    via_view = prepare_training_matrix(_DataView(panel), _SchemaView(), ())
    direct = prepare_matrix_from_frame(panel, ("feature__a", "feature__b"))
    assert np.array_equal(via_view.X, direct.X)
    assert np.array_equal(via_view.sorted_keys, direct.sorted_keys)


def test_label_alignment_is_one_to_one_and_sorted() -> None:
    from datetime import datetime

    labels = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001", "KRX:00002"],
            "session": [
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
            "net_alpha_target": [0.1, 0.2],
            "label_available_time": [datetime(2024, 1, 7, tzinfo=UTC)] * 2,
            "risk_residual": [0.12, 0.22],
            "reference_cost": [0.02, 0.02],
        }
    )
    data = type("D", (), {"labels_by_horizon": {10: labels}})()
    matrix = prepare_matrix_from_frame(_panel(), ("feature__a",))
    horizon = prepare_horizon_labels(matrix, data, 10)
    assert np.all(np.diff(horizon.row_index) > 0)
    assert horizon.row_index.size == 2
    assert np.allclose(horizon.realized, [0.10, 0.20])


def test_prepare_folds_freezes_integer_arrays() -> None:
    fold = Fold(
        train_mask=[2, 0, 1],
        validation_mask=[3, 4],
        train_label_end=5,
        validation_decision_start=6,
        segment_id=0,
    )
    prepared = prepare_folds([fold])[0]
    assert np.array_equal(prepared.train_rows, np.asarray([0, 1, 2]))
    assert np.array_equal(prepared.validation_rows, np.asarray([3, 4]))


def _legacy_labels_without_gross() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["KRX:00001", "KRX:00002"],
            "session": [datetime(2024, 1, 2, tzinfo=UTC)] * 2,
            "net_alpha_target": [0.1, 0.2],
            "label_available_time": [datetime(2024, 1, 7, tzinfo=UTC)] * 2,
            "risk_residual": [0.12, 0.22],
            "reference_cost": [0.02, 0.02],
        }
    )


def test_prepare_horizon_labels_route_preserves_gross_and_projects_target() -> None:
    # Given
    matrix = prepare_matrix_from_frame(_panel(), ("feature__a",))
    labels = pl.DataFrame({"instrument_id": ["KRX:00001", "KRX:00002"], "session": [datetime(2024, 1, 2, tzinfo=UTC)] * 2, "net_alpha_target": [9.0, 9.0], "label_available_time": [datetime(2024, 1, 7, tzinfo=UTC)] * 2, "gross_return": [0.10, 0.20], "risk_residual": [-0.4, -0.3], "reference_cost": [0.02, 0.02]})
    data = type("D", (), {"labels_by_horizon": {10: labels}})()
    # When
    result = prepare_horizon_labels(matrix, data, 10, route_objective=SimpleNamespace(kind="unhedged_absolute"))
    # Then
    assert np.allclose(result.target, [0.08, 0.18])
    assert np.allclose(result.realized, [0.08, 0.18])
    assert np.allclose(result.gross_return, [0.10, 0.20])
    assert np.all(np.diff(result.row_index) > 0)


def test_prepare_horizon_labels_unhedged_missing_gross_fails_closed() -> None:
    # Given
    matrix = prepare_matrix_from_frame(_panel(), ("feature__a",))
    data = type("D", (), {"labels_by_horizon": {10: _legacy_labels_without_gross()}})()
    # When / Then
    with pytest.raises(ValueError, match="gross"):
        prepare_horizon_labels(matrix, data, 10, route_objective=SimpleNamespace(kind="unhedged_absolute"))
    assert prepare_horizon_labels(matrix, data, 10).row_index.size > 0
