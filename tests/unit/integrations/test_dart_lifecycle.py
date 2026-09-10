

from datetime import date, datetime

from src.core.time import KRX_TZ, SessionCalendar
from src.data.lifecycle import LifecycleCandidate, LifecycleResolutionKind
from src.integrations.dart.lifecycle import DartLifecycleCollector

class FakeDart:
    def load_corp_code_records(self):
        return ()

def test_dart_lifecycle_collector_persists_unresolved_mapping_gap():
    session = datetime(2016, 1, 21, 9, tzinfo=KRX_TZ)
    candidate = LifecycleCandidate('KRX:003945', '003945', datetime(2016, 1, 20, 9, tzinfo=KRX_TZ), session, ('master-hash',))
    evidence = DartLifecycleCollector(dart=FakeDart(), calendar=SessionCalendar((candidate.last_tradable_session, session)), coverage_start=date(2015, 1, 1)).collect(candidate)
    assert evidence.resolution_kind is LifecycleResolutionKind.UNRESOLVED
    assert evidence.evidence_reason == 'missing_corp_code'
    assert evidence.cash_settlement_per_share is None


def test_dart_lifecycle_collector_resolves_cleanup_cash_settlement():
    from datetime import date, datetime
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.lifecycle import LifecycleCandidate
    from src.integrations.dart.lifecycle import DartLifecycleCollector
    from src.data.lifecycle import LifecycleResolutionKind
    sessions = (datetime(2016, 5, 3, 9, tzinfo=KRX_TZ), datetime(2016, 5, 10, 9, tzinfo=KRX_TZ), datetime(2016, 5, 18, 9, tzinfo=KRX_TZ), datetime(2016, 5, 19, 9, tzinfo=KRX_TZ))
    candidate = LifecycleCandidate("KRX:008020", "008020", sessions[2], sessions[3], ("master-hash",))
    archive = "정리매매 기간은 2016.05.10.부터 2016.05.18.까지이며 2016.05.19.자로 상장폐지. 주당 10,200원.".encode()
    class FakeDart:
        def load_corp_code_records(self):
            from src.integrations.dart.client import DartCorpCodeRecord
            return (DartCorpCodeRecord(ticker="008020", corp_code="00101336", corp_name="경남에너지"),)
        def list_disclosures(self, start, end, *, corp_code=None):
            assert corp_code == "00101336"
            return [{"rcept_no": "20160502800560", "rcept_dt": "20160502", "corp_code": corp_code, "corp_name": "경남에너지", "report_nm": "기타경영사항(자율공시)(상장폐지결정 및 정리매매)", "rm": "유"}]
        def fetch_document_archive(self, rcept_no):
            assert rcept_no == "20160502800560"
            return archive
    evidence = DartLifecycleCollector(dart=FakeDart(), calendar=SessionCalendar(sessions), coverage_start=date(2015, 1, 1)).collect(candidate)
    assert evidence.resolution_kind is LifecycleResolutionKind.CASH_SETTLEMENT
    assert evidence.cash_settlement_per_share == 10200.0
    assert evidence.document_receipt_no == "20160502800560"


def test_dart_lifecycle_collector_skips_permanently_unavailable_archive():
    from src.integrations.dart.client import DartCorpCodeRecord, DartTerminalError
    from src.data.lifecycle import LifecycleCandidate
    session = datetime(2016, 1, 21, 9, tzinfo=KRX_TZ)
    candidate = LifecycleCandidate("KRX:003450", "003450", datetime(2016, 1, 20, 9, tzinfo=KRX_TZ), session, ("h",))
    class FakeDart:
        def load_corp_code_records(self):
            return (DartCorpCodeRecord(ticker="003450", corp_code="c", corp_name="x"),)
        def list_disclosures(self, start, end, *, corp_code=None):
            return [{"rcept_no": "20160121000001", "rcept_dt": "20160121", "report_nm": "상장폐지결정"}]
        def fetch_document_archive(self, rcept_no):
            raise DartTerminalError("014")
    evidence = DartLifecycleCollector(dart=FakeDart(), calendar=SessionCalendar((candidate.last_tradable_session, session)), coverage_start=date(2015, 1, 1)).collect(candidate)
    assert evidence.evidence_reason == "document_unavailable"


def test_dart_lifecycle_collector_keeps_verified_unpriced_fallback():
    from src.integrations.dart.client import DartCorpCodeRecord
    sessions = (datetime(2016, 5, 3, 9, tzinfo=KRX_TZ), datetime(2016, 5, 10, 9, tzinfo=KRX_TZ), datetime(2016, 5, 18, 9, tzinfo=KRX_TZ), datetime(2016, 5, 19, 9, tzinfo=KRX_TZ))
    candidate = LifecycleCandidate("KRX:008020", "008020", sessions[2], sessions[3], ("h",))
    class FakeDart:
        def load_corp_code_records(self):
            return (DartCorpCodeRecord(ticker="008020", corp_code="c", corp_name="x"),)
        def list_disclosures(self, start, end, *, corp_code=None):
            return [{"rcept_no": "20160502000001", "rcept_dt": "20160502", "report_nm": "상장폐지결정"}]
        def fetch_document_archive(self, rcept_no):
            return "정리매매기간(2016.05.10~2016.05.18) 후 상장폐지일: 2016.05.19".encode()
    evidence = DartLifecycleCollector(dart=FakeDart(), calendar=SessionCalendar(sessions), coverage_start=date(2015, 1, 1)).collect(candidate)
    assert evidence.evidence_status == "verified"
    assert evidence.resolution_kind is LifecycleResolutionKind.UNSETTLED_DELISTING


def test_dart_lifecycle_collector_returns_auditable_validation_failure():
    from src.integrations.dart.client import DartCorpCodeRecord
    sessions = (datetime(2016, 5, 3, 9, tzinfo=KRX_TZ), datetime(2016, 5, 10, 9, tzinfo=KRX_TZ), datetime(2016, 5, 18, 9, tzinfo=KRX_TZ), datetime(2016, 5, 19, 9, tzinfo=KRX_TZ))
    candidate = LifecycleCandidate("KRX:008020", "008020", sessions[2], sessions[3], ("h",))
    class FakeDart:
        def load_corp_code_records(self):
            return (DartCorpCodeRecord(ticker="008020", corp_code="c", corp_name="x"),)
        def list_disclosures(self, start, end, *, corp_code=None):
            return [{"rcept_no": "20160502000002", "rcept_dt": "20160502", "report_nm": "상장폐지결정"}]
        def fetch_document_archive(self, rcept_no):
            return "정리매매기간(2016.05.10~2016.05.17) 후 상장폐지일: 2016.05.19".encode()
    evidence = DartLifecycleCollector(dart=FakeDart(), calendar=SessionCalendar(sessions), coverage_start=date(2015, 1, 1)).collect(candidate)
    assert evidence.evidence_status == "unresolved"
    assert evidence.evidence_reason.startswith("document_validation:")
