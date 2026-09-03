from src.data.collection import collect_champion_evidence


def test_collection_routes_investor_flow_to_kis_only() -> None:
    result = collect_champion_evidence(krx=object(), kis=object(), dart=object(), plan=object())
    assert result is not None
