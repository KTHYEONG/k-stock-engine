from __future__ import annotations

import inspect

from src.stocks.ml.promotion import publish_training_outcome


def test_promotion_entrypoint_is_callable() -> None:
    assert inspect.isfunction(publish_training_outcome)
