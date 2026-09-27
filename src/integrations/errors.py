"""Shared provider-transport error tree.

Transport failures (talking to an external provider) live here. Data-content
violations stay in ``src.core.pit.PITDataError`` and a transport failure never
raises it.
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Base for failures talking to an external data provider."""

    def __init__(self, message: str, *, provider: str = "", endpoint: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.endpoint = endpoint


class ProviderRetryableError(ProviderError):
    """Transient transport failure, 408/429/5xx, or provider try-later code."""


class ProviderTerminalError(ProviderError):
    """Permanent failure: 4xx, auth, malformed request, business rejection."""


class ProviderQuotaExhaustedError(ProviderError):
    """Daily quota or provider quota code; stop the job."""


class DartApiError(ProviderError):
    """DART transport failure; catch ``ProviderError`` in new code."""
