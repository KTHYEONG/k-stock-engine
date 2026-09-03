import pytest

from src.data.collection_plan import CollectionReadinessReport
from src.domain.stock_data import PITDataError


def test_uncertain_status_blocks_certification() -> None:
    report = CollectionReadinessReport.incomplete(corporate_status_reason='unvalidated provider provenance')
    with pytest.raises(PITDataError, match='unvalidated'):
        report.require_certifiable()
