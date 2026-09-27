"""KRX transport-only client."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, ClassVar, Final

import requests

from src.config.providers import KrxPolicy
from src.config.secrets import read_secret
from src.core.pit import PITDataError
from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
from src.integrations.quota import LedgerQuotaGate, ProviderQuotaStateStore
from src.integrations.transport import HttpTransport, RetryPolicy

__all__ = [
    "KrxApiClient",
    "KrxHolidayError",
    "KrxMarket",
    "build_scoped_krx_client",
]

_PROVIDER = "KRX"
_BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"
_TIMEOUT_SECONDS: Final = 30.0
_MAX_HTTP_ATTEMPTS: Final = 3
_DEFAULT_MIN_INTERVAL_SECONDS: Final = 1.5

_CLOSE_ALIASES: Final = ("TDD_CLSPRC", "CLSPRC", "close", "CLOSE")
_MCAP_ALIASES: Final = ("MKTCAP", "market_cap", "marcap")
_SHRS_ALIASES: Final = ("LIST_SHRS", "shares_outstanding", "list_shrs")
_HOLIDAY_TOKENS: Final = ("휴장", "holiday", "non-trading", "non trading")


class KrxHolidayError(PITDataError):
    """KRX explicitly reports no trading for an exchange-calendar session."""


class KrxMarket(str, Enum):  # noqa: UP042
    KOSPI = "KOSPI"
    KOSDAQ = "KOSDAQ"
    ALL = "ALL"


def _validate_daily_market_records(records: list[dict[str, Any]], *, session: date) -> None:
    """Require KRX valuation fields before a daily page may enter Bronze."""
    missing: set[str] = set()
    for record in records:
        if not any(record.get(name) not in (None, "") for name in _CLOSE_ALIASES):
            missing.add("TDD_CLSPRC")
        if not any(record.get(name) not in (None, "") for name in _MCAP_ALIASES):
            missing.add("MKTCAP")
        if not any(record.get(name) not in (None, "") for name in _SHRS_ALIASES):
            missing.add("LIST_SHRS")
    if missing:
        names = " and ".join(sorted(missing))
        raise PITDataError(f"KRX daily market missing {names} for {session}; certification blocked")


def _validate_master_records(records: list[dict[str, Any]], *, session: date) -> None:
    """Require ticker and ISIN identity on every master row."""
    for record in records:
        if not str(record.get("ISU_SRT_CD") or "").strip() or not str(record.get("ISU_CD") or "").strip():
            raise PITDataError(
                f"KRX security master record lacks ticker or ISIN for {session}; certification blocked"
            )


def _holiday_message(payload: Mapping[str, Any]) -> str | None:
    """Return the provider's holiday notice when it answers without rows, else ``None``."""
    for key, value in payload.items():
        if key == "OutBlock_1":
            continue
        if isinstance(value, Mapping):
            nested = _holiday_message(value)
            if nested is not None:
                return nested
        elif isinstance(value, str) and value.strip():
            lowered = value.casefold()
            if any(token in lowered for token in _HOLIDAY_TOKENS):
                return value.strip()
    return None


class KrxApiClient:
    """KRX OpenAPI client returning validated record lists.

    Transport-only: pacing, retries, and quotas belong to ``HttpTransport``
    and the quota ledger. Bronze writes belong to the data layer.
    """

    BASE_URL = _BASE_URL
    ENDPOINTS: ClassVar[dict[str, str]] = {  # noqa: RUF012
        "KOSPI_INFO": "sto/stk_isu_base_info",
        "KOSDAQ_INFO": "sto/ksq_isu_base_info",
        "KOSPI_TRADE": "sto/stk_bydd_trd",
        "KOSDAQ_TRADE": "sto/ksq_bydd_trd",
    }

    def __init__(
        self,
        api_key: str,
        *,
        transport: HttpTransport | None = None,
        quota_store: ProviderQuotaStateStore | None = None,
        daily_limit: int | None = None,
        min_interval_seconds: float | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        cleaned = api_key.strip().strip("\"'") if isinstance(api_key, str) else ""
        if not cleaned:
            raise ValueError("KRX api_key is required")
        self._api_key = cleaned
        self._now = now or (lambda: datetime.now(UTC))
        if transport is not None:
            self._transport = transport
            return
        gate: LedgerQuotaGate | None = None
        if quota_store is not None:
            gate = LedgerQuotaGate(quota_store, provider=_PROVIDER, daily_limit=daily_limit)
            gate.bind_now(self._now)
        pace = _DEFAULT_MIN_INTERVAL_SECONDS if min_interval_seconds is None else float(min_interval_seconds)
        self._transport = HttpTransport(
            provider=_PROVIDER,
            base_url=self.BASE_URL,
            min_interval_seconds=pace,
            retry=RetryPolicy(max_attempts=_MAX_HTTP_ATTEMPTS),
            quota=gate,
            timeout_seconds=_TIMEOUT_SECONDS,
            session=requests.Session(),
            sleep=lambda seconds: time.sleep(seconds),
            monotonic=lambda: time.monotonic(),
        )

    def _records(self, endpoint: str, as_of: date) -> list[dict[str, Any]]:
        response = self._transport.get(
            endpoint,
            {"basDd": as_of.strftime("%Y%m%d")},
            headers={"AUTH_KEY": self._api_key},
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderRetryableError(
                f"KRX returned invalid JSON for {endpoint}", provider=_PROVIDER, endpoint=endpoint
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderTerminalError(
                f"KRX response must be an object for {endpoint}", provider=_PROVIDER, endpoint=endpoint
            )
        records = payload.get("OutBlock_1", [])
        if not isinstance(records, list):
            raise ProviderTerminalError(
                f"KRX records must be a list for {endpoint}", provider=_PROVIDER, endpoint=endpoint
            )
        if not records:
            holiday = _holiday_message(payload)
            if holiday is not None:
                raise KrxHolidayError(
                    f"KRX reports no trading for {as_of.isoformat()} ({holiday}); review required"
                )
            return []
        return [record for record in records if isinstance(record, dict)]

    @staticmethod
    def _coerce_market(market: KrxMarket | str) -> KrxMarket:
        if isinstance(market, KrxMarket):
            return market
        try:
            return KrxMarket(market)
        except ValueError as exc:
            raise ValueError(f"unknown KRX market {market!r}") from exc

    def fetch_daily_records(self, as_of: date, market: KrxMarket | str = KrxMarket.ALL) -> list[dict[str, Any]]:
        """Return validated daily trade records for one session across the requested markets."""
        if not isinstance(as_of, date):
            raise ValueError("as_of must be a date")
        resolved = self._coerce_market(market)
        records: list[dict[str, Any]] = []
        if resolved in (KrxMarket.KOSPI, KrxMarket.ALL):
            records.extend(self._records(self.ENDPOINTS["KOSPI_TRADE"], as_of))
        if resolved in (KrxMarket.KOSDAQ, KrxMarket.ALL):
            records.extend(self._records(self.ENDPOINTS["KOSDAQ_TRADE"], as_of))
        if records:
            _validate_daily_market_records(records, session=as_of)
        return records

    def fetch_master_records(self, as_of: date, market: KrxMarket | str = KrxMarket.ALL) -> list[dict[str, Any]]:
        """Return validated master records for one session across the requested markets."""
        if not isinstance(as_of, date):
            raise ValueError("as_of must be a date")
        resolved = self._coerce_market(market)
        if resolved == KrxMarket.KOSPI:
            records = self._records(self.ENDPOINTS["KOSPI_INFO"], as_of)
        elif resolved == KrxMarket.KOSDAQ:
            records = self._records(self.ENDPOINTS["KOSDAQ_INFO"], as_of)
        else:
            records = [
                *self._records(self.ENDPOINTS["KOSPI_INFO"], as_of),
                *self._records(self.ENDPOINTS["KOSDAQ_INFO"], as_of),
            ]
        if records:
            _validate_master_records(records, session=as_of)
        return records

    def health_check(self) -> None:
        """Confirm collection credentials were verified without spending quota."""


def build_scoped_krx_client(
    *,
    policy: KrxPolicy,
    quota_store: ProviderQuotaStateStore,
    now: Callable[[], datetime] | None = None,
) -> KrxApiClient:
    """Build the KRX collector for one declared key from its provider policy.

    Raises:
        ValueError: the key's environment variable is unset.
    """
    api_key = read_secret(policy.api_key_env).strip().strip("\"'")
    if not api_key:
        raise ValueError(f"{policy.api_key_env} is not set")
    return KrxApiClient(
        api_key,
        quota_store=quota_store,
        daily_limit=policy.daily_limit,
        min_interval_seconds=policy.min_interval_seconds,
        now=now,
    )
