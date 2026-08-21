from __future__ import annotations

from src.stocks.ml.fitting import OofCache


def test_oof_cache_closes_without_error(tmp_path) -> None:
    cache = OofCache(tmp_path)
    assert cache.root.exists()
    cache.close()
