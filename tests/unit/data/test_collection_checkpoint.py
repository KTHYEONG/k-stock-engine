from src.data.collection_plan import CollectionCheckpointStore


def test_resume_recollects_chunk_with_changed_receipt(tmp_path) -> None:
    store = CollectionCheckpointStore(tmp_path)
    store.mark_complete(plan_id='plan-a', chunk_id='chunk-a', receipt_digest='old')
    assert store.is_pending(plan_id='plan-a', chunk_id='chunk-a', receipt_digest='new')
