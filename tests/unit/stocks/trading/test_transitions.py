from __future__ import annotations

from src.stocks.trading.transitions import TransitionEvidence


def test_transition_evidence_defaults_to_zero() -> None:
    evidence = TransitionEvidence()

    assert evidence.retained_count == 0
    assert evidence.turnover_bps == 0.0
