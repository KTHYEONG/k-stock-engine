import pytest

from src.data.collection_plan import CollectionReadinessReport
from src.domain.stock_data import PITDataError


def test_uncertain_status_blocks_certification() -> None:
    report = CollectionReadinessReport.incomplete(corporate_status_reason='unvalidated provider provenance')
    with pytest.raises(PITDataError, match='unvalidated'):
        report.require_certifiable()


def test_readiness_allows_local_gaps_only_with_policy_sized_daily_cohort(tmp_path) -> None:
    from datetime import date
    from src.data.collection_plan import EvidenceCoverage, audit_historical_readiness, build_historical_collection_plan
    from src.data.schemas import EvidenceKind

    session = date(2016, 1, 4)
    plan = build_historical_collection_plan(sessions=(session,), universe=({'symbol': '005930', 'is_common_stock': True},), start=session, end=session, artifact_root=tmp_path / 'plans')
    gap = EvidenceCoverage(kind=EvidenceKind.INVESTOR_FLOW, instrument_id='KRX:000001', session=session, state='source_unavailable', receipt_hash='a' * 64, reason='provider_has_no_history')
    passing = audit_historical_readiness(plan=plan, coverage=(gap,), usable_feature_count_by_session={session: 10}, minimum_cohort=10)
    failing = audit_historical_readiness(plan=plan, coverage=(gap,), usable_feature_count_by_session={session: 9}, minimum_cohort=10)

    assert passing.certifiable is True
    assert passing.coverage_gaps
    assert failing.certifiable is False
    assert any('usable cohort' in reason for reason in failing.unresolved_reasons)
    complete = EvidenceCoverage(kind=EvidenceKind.INVESTOR_FLOW, instrument_id='KRX:000001', session=session, state='complete', receipt_hash='b' * 64, reason='ok')
    assert audit_historical_readiness(plan=plan, coverage=(complete,), usable_feature_count_by_session={session: 10}, minimum_cohort=10).certifiable is True


def test_readiness_rejects_invalid_minimum_cohort(tmp_path) -> None:
    from datetime import date
    from src.data.collection_plan import audit_historical_readiness, build_historical_collection_plan

    day = date(2016, 1, 4)
    plan = build_historical_collection_plan(
        sessions=(day,), universe=({'symbol': '005930', 'is_common_stock': True},),
        start=day, end=day, artifact_root=tmp_path / 'plans',
    )
    with pytest.raises(PITDataError, match='minimum_cohort'):
        audit_historical_readiness(plan=plan, coverage=(), usable_feature_count_by_session={day: 1}, minimum_cohort=0)
