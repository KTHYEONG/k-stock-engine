"""Catalog CLI parser and behavior tests for validate-readiness."""
from __future__ import annotations

import pytest

from src.stocks.cli import catalog
from tests.fixtures.stocks.helpers import feature_readiness_dataset


def test_catalog_validate_readiness_requires_dataset_dir_and_feature() -> None:
    with pytest.raises(SystemExit):
        catalog.main(["validate-readiness"])
    with pytest.raises(SystemExit):
        catalog.main(["validate-readiness", "--dataset-dir", "features"])


def test_catalog_validate_readiness_fails_on_unusable_feature(tmp_path, capsys) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    code = catalog.main(
        [
            "--catalog-root",
            str(tmp_path / "catalog"),
            "validate-readiness",
            "--dataset-dir",
            str(dataset_dir),
            "--feature",
            "inactive",
        ]
    )
    assert code == 1
    assert "readiness failed" in capsys.readouterr().out


def test_catalog_validate_readiness_reports_usable_selection(tmp_path, capsys) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    code = catalog.main(
        [
            "--catalog-root",
            str(tmp_path / "catalog"),
            "validate-readiness",
            "--dataset-dir",
            str(dataset_dir),
            "--feature",
            "overnight_ret",
            "--feature",
            "ret_21_60d",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert "feature__inactive" in out
