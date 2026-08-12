"""Unit: build_research_v2 CLI parser and materializer dispatch."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.stocks.cli.build_research_v2 as cli


def _minimal_args() -> list[str]:
    return [
        "--source-snapshot-id",
        "source",
        "--feature-dataset-id",
        "features",
        "--label-dataset-id",
        "labels",
        "--snapshot-id",
        "snap",
        "--train-start",
        "2024-01-01",
        "--train-end",
        "2024-01-02",
        "--validation-start",
        "2024-01-03",
        "--validation-end",
        "2024-01-04",
        "--test-start",
        "2024-01-05",
        "--test-end",
        "2024-01-06",
    ]


def _fake_result() -> SimpleNamespace:
    return SimpleNamespace(
        feature_dataset_id="features",
        label_dataset_id="labels",
        snapshot_id="snap",
        feature_content_hash="f-hash",
        label_content_hash="l-hash",
        feature_row_count=10,
        label_row_count=10,
        min_coverage=0.75,
        certification=SimpleNamespace(value="provisional"),
    )


def test_label_horizon_mode_defaults_to_five_day() -> None:
    parsed = cli.build_parser().parse_args(_minimal_args())
    assert parsed.label_horizon_mode == "five_day"


def test_label_horizon_mode_multi_horizon_is_explicit_choice() -> None:
    parsed = cli.build_parser().parse_args(
        [*_minimal_args(), "--label-horizon-mode", "multi_horizon"]
    )
    assert parsed.label_horizon_mode == "multi_horizon"


def test_label_horizon_mode_rejects_unknown_choice() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [*_minimal_args(), "--label-horizon-mode", "ten_day"]
        )


def test_main_dispatches_five_day_to_v2_materializer(monkeypatch, capsys) -> None:
    called: list[str] = []
    fake = _fake_result()
    monkeypatch.setattr(
        cli,
        "materialize_stock_alpha_v2_snapshot",
        lambda request: (called.append("v2") or fake),
    )
    monkeypatch.setattr(
        cli,
        "materialize_stock_alpha_v3_snapshot",
        lambda request: (called.append("v3") or fake),
    )
    exit_code = cli.main(_minimal_args())
    assert exit_code == 0
    assert called == ["v2"]
    assert "label_horizon_mode=five_day" in capsys.readouterr().out


def test_main_dispatches_multi_horizon_to_v3_materializer(
    monkeypatch, capsys
) -> None:
    called: list[str] = []
    fake = _fake_result()
    monkeypatch.setattr(
        cli,
        "materialize_stock_alpha_v2_snapshot",
        lambda request: (called.append("v2") or fake),
    )
    monkeypatch.setattr(
        cli,
        "materialize_stock_alpha_v3_snapshot",
        lambda request: (called.append("v3") or fake),
    )
    exit_code = cli.main([*_minimal_args(), "--label-horizon-mode", "multi_horizon"])
    assert exit_code == 0
    assert called == ["v3"]
    assert "label_horizon_mode=multi_horizon" in capsys.readouterr().out
