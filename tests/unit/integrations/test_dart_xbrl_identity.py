import pytest

from src.domain.stock_data import PITDataError
from src.integrations.dart.xbrl import DartXbrlCollector


def test_dart_rejects_missing_filing_identity() -> None:
    collector = DartXbrlCollector(api_key='test-key', request_json=lambda endpoint, params: {})
    with pytest.raises(PITDataError, match='filing identity'):
        tuple(collector.fetch_xbrl_facts(({'filing_id': 'F1'},)))
