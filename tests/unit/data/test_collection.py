

from datetime import datetime

from src.core.time import KRX_TZ
from src.data.collection import collect_dart_lifecycle_evidence
from src.data.lifecycle import LifecycleCandidate, LifecycleEvidence, LifecycleResolutionKind

class FakeCollector:
    def collect(self, candidate):
        return LifecycleEvidence(candidate, LifecycleResolutionKind.UNRESOLVED, 'unresolved', 'missing_terms', None, None, None, None, None, None, 'opendart', None, None, None)

def test_collect_dart_lifecycle_evidence_writes_one_auditable_bronze_envelope_per_candidate(tmp_path):
    from src.data.bronze import BronzeStore
    session = datetime(2016, 1, 21, 9, tzinfo=KRX_TZ)
    candidate = LifecycleCandidate('KRX:003945', '003945', datetime(2016, 1, 20, 9, tzinfo=KRX_TZ), session, ('master-hash',))
    receipts = collect_dart_lifecycle_evidence(candidates=(candidate,), collector=FakeCollector(), bronze=BronzeStore(tmp_path), retrieved_at=session)
    assert len(receipts) == 1
    assert receipts[0].kind.value == 'lifecycle_events'
    assert receipts[0].content_hash
