def test_data_schemas_is_core_pit_compatibility_facade() -> None:
    import src.core.pit as core_pit
    import src.data.schemas as data_schemas

    assert data_schemas.PITDataError is core_pit.PITDataError
    assert data_schemas.EvidenceKind is core_pit.EvidenceKind
    assert data_schemas.SilverTable is core_pit.SilverTable
    assert data_schemas.BronzeReceipt is core_pit.BronzeReceipt
    assert data_schemas.PITSnapshotRequest is core_pit.PITSnapshotRequest
    assert data_schemas.CertificationReport is core_pit.CertificationReport
    assert set(data_schemas.__all__) == {
        "BronzeReceipt",
        "CertificationReport",
        "EvidenceKind",
        "PITDataError",
        "PITSnapshotRequest",
        "SilverTable",
    }
