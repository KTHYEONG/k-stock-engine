"""Frozen ticker bridge reads from loose and packed Bronze evidence."""
from pathlib import Path

import pytest

from src.core.pit import PITDataError
from src.data.jobs.universe import read_corp_code_bridge
from src.data.payload_store import PACK_DIR_NAME, pack_loose
from src.data.receipt_catalog import ReceiptCatalog
from tests.fixtures import seed_corp_code_bridge


def test_bridge_mapping_and_source_hash_survive_packing(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    raw = b'[{"corp_code":"00126380","ticker":"005930"}]'
    digest = seed_corp_code_bridge(bronze, raw)
    catalog = ReceiptCatalog(bronze / "catalog")
    expected = ({"00126380": "005930"}, digest)
    assert read_corp_code_bridge(catalog) == expected
    pack_loose(bronze, kinds=["dart_corp_codes"], offsite_paths=None, remove=True)
    assert read_corp_code_bridge(catalog) == expected
    next((bronze / PACK_DIR_NAME / "dart_corp_codes").glob("pack-*.bin")).unlink()
    with pytest.raises(PITDataError, match="unreadable"):
        read_corp_code_bridge(catalog)
