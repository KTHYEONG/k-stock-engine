from __future__ import annotations

import src.core.pit as core_pit
import src.data.schemas as data_schemas


def test_live_pit_facade_exports_are_shared() -> None:
    assert data_schemas.BronzeReceipt is core_pit.BronzeReceipt
    assert data_schemas.EvidenceKind is core_pit.EvidenceKind
    assert data_schemas.PITDataError is core_pit.PITDataError
    assert set(data_schemas.__all__) == {"BronzeReceipt", "EvidenceKind", "PITDataError"}
