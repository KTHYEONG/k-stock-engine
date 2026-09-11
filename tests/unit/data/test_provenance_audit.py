import json

import pytest


def test_audit_keeps_empty_responses_unverified_and_retries_provider_errors(tmp_path) -> None:
    from src.data.provenance_audit import audit_production_provenance

    bronze = tmp_path / 'bronze'
    for name, payload in {'a': {'symbol': '000001', 'provider': 'ls', 'status': 'source_unavailable'}, 'b': {'symbol': '000002', 'provider': 'ls', 'status': 'source_unavailable'}, 'c': {'symbol': '000002', 'provider': 'kiwoom', 'status': 'source_unavailable'}, 'd': {'symbol': '000003', 'provider': 'ls', 'status': 'source_unavailable'}, 'e': {'symbol': '000003', 'provider': 'kiwoom', 'status': 'provider_error'}}.items():
        path = bronze / 'investor_flow' / name
        path.mkdir(parents=True)
        (path / 'payload.json').write_text(json.dumps(payload))
    report = audit_production_provenance(bronze_root=bronze, silver_root=tmp_path / 'silver', artifact_root=tmp_path / 'artifacts')
    states = {row.symbol: row.state for row in report.investor_flow}
    assert states == {'000001': 'unverified_empty_response', '000002': 'unverified_empty_response', '000003': 'retry_required'}
    ignored = bronze / 'investor_flow' / 'ignored'
    ignored.mkdir()
    (ignored / 'payload.json').write_text(json.dumps({'symbol': '000004'}))
    fixture = tmp_path / 'silver' / 'daily_market' / 'fixture-id'
    fixture.mkdir(parents=True)
    (fixture / 'dataset_manifest.json').write_text(json.dumps({'provider_version': 'fixture'}))
    report = audit_production_provenance(bronze_root=bronze, silver_root=tmp_path / 'silver', artifact_root=tmp_path / 'artifacts')
    assert report.fixture_tables == ('daily_market',)
    invalid = bronze / 'investor_flow' / 'invalid'
    invalid.mkdir()
    (invalid / 'payload.json').write_text(json.dumps({'provider': 'ls', 'status': 'source_unavailable'}))
    with pytest.raises(ValueError, match='invalid investor-flow negative receipt'):
        audit_production_provenance(bronze_root=bronze, silver_root=tmp_path / 'silver', artifact_root=tmp_path / 'artifacts')
