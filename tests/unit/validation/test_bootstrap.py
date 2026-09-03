from __future__ import annotations


def test_moving_block_bootstrap_is_seeded_and_preserves_contiguous_blocks() -> None:
    from src.validation.bootstrap import (
        BootstrapConfig,
        BootstrapMethod,
        moving_block_bootstrap_indices,
    )

    config = BootstrapConfig(
        BootstrapMethod.MOVING_BLOCK,
        resamples=3,
        block_length_sessions=20,
        seed=17,
    )

    first = moving_block_bootstrap_indices(40, config)
    second = moving_block_bootstrap_indices(40, config)

    assert first == second
    assert all(len(path) == 40 for path in first)
    for path in first:
        for start in (0, 20):
            block = path[start : start + 20]
            assert all(block[index] == (block[0] + index) % 40 for index in range(20))


def test_bootstrap_promotion_run_requires_five_thousand_resamples() -> None:
    import pytest

    from src.validation.bootstrap import BootstrapConfig, BootstrapMethod

    with pytest.raises(ValueError, match="5000"):
        BootstrapConfig(
            BootstrapMethod.STATIONARY,
            resamples=4_999,
            block_length_sessions=20,
            seed=7,
            promotion_run=True,
        )
