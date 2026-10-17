"""Host slot gate invariants."""
from __future__ import annotations


def test_host_pacer_satisfies_slot_gate(tmp_path) -> None:
    from types import SimpleNamespace

    from src.integrations.kis.client import _LimiterPacer
    from src.integrations.kis.admission import HostPacedLimiter
    from src.integrations.transport import HostPacer, HttpTransport, RetryPolicy

    state = tmp_path / "slot.state"
    pacer = HostPacer(state, min_interval_seconds=0.01)
    limiter = HostPacedLimiter(
        tmp_path / "limiter.state", max_lead_seconds=None, clock=lambda: 0.0, sleep=lambda _s: None
    )
    adapted = _LimiterPacer(limiter)

    def _transport(host_pacer):
        return HttpTransport(
            provider="TEST",
            base_url="https://example.invalid",
            min_interval_seconds=0.0,
            retry=RetryPolicy(max_attempts=2),
            quota=None,
            timeout_seconds=5.0,
            session=SimpleNamespace(
                get=lambda *_a, **_k: SimpleNamespace(status_code=200, headers={}, json=lambda: {})
            ),
            sleep=lambda _s: None,
            host_pacer=host_pacer,
        )

    for gate in (pacer, adapted):
        transport = _transport(gate)
        assert transport.get("ep", {}).status_code == 200
