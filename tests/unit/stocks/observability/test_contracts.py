from __future__ import annotations

from src.stocks.observability.contracts import (
    DiagnosticCategory,
    DiagnosticEvent,
    DiagnosticStage,
    DiagnosticStatus,
)


def test_event_contract_has_required_categories_and_statuses() -> None:
    event = DiagnosticEvent(
        run_id="run",
        sequence=0,
        category=DiagnosticCategory.DATA,
        component="test",
        stage=DiagnosticStage.INPUT,
        event="input",
        status=DiagnosticStatus.PASS,
    )

    assert event.category is DiagnosticCategory.DATA
    assert event.status is DiagnosticStatus.PASS
