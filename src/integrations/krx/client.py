"""KRX transport-only client."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
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
        "ETF_TRADE": "etp/etf_bydd_trd",
        "KOSDAQ_INDEX": "idx/kosdaq_dd_trd",
        "KOSPI_INDEX": "idx/kospi_dd_trd",
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

    def fetch_hedge_records(
        self, as_of: date, *, etf_tickers: Sequence[str], index_name: str, index_class: str = "KOSDAQ"
    ) -> list[dict[str, Any]]:
        """Return the validated hedge-series records of one session: the requested ETF rows (tagged
        ``"_endpoint": "etf"``) and the index row with ``IDX_CLSS == index_class`` and ``IDX_NM == index_name``
        (tagged ``"_endpoint": "index"``).

        ``index_class`` selects the KRX index page: ``"KOSDAQ"`` reads ``idx/kosdaq_dd_trd``, ``"KOSPI"`` reads
        ``idx/kospi_dd_trd``. Why a class and not a free endpoint: the class is also the row filter, so a page and a
        filter can never disagree.

        Empty list only when both pages are empty without a holiday message.

        Raises:
            ValueError: ``index_class`` is not ``"KOSDAQ"`` or ``"KOSPI"``, or as before for the other arguments.
            KrxHolidayError, ProviderTerminalError: as before.
        """
        if not isinstance(as_of, date):
            raise ValueError("as_of must be a date")
        if index_class not in ("KOSDAQ", "KOSPI"):
            raise ValueError(f'index_class must be "KOSDAQ" or "KOSPI", got {index_class!r}')
        if not isinstance(index_name, str) or not index_name.strip():
            raise ValueError("index_name must be a non-empty string")
        wanted = tuple(str(ticker).strip() for ticker in etf_tickers)
        if any(not ticker for ticker in wanted):
            raise ValueError("etf_tickers must contain non-empty strings")
        index_endpoint = (
            self.ENDPOINTS["KOSDAQ_INDEX"] if index_class == "KOSDAQ" else self.ENDPOINTS["KOSPI_INDEX"]
        )
        etf_page = self._records(self.ENDPOINTS["ETF_TRADE"], as_of)
        index_page = self._records(index_endpoint, as_of)
        if not etf_page and not index_page:
            return []
        if not etf_page or not index_page:
            raise ProviderTerminalError(
                f"KRX hedge-series page is incomplete for {as_of.isoformat()}",
                provider=_PROVIDER,
                endpoint=self.ENDPOINTS["ETF_TRADE"] if not etf_page else index_endpoint,
            )
        index_rows = [
            record
            for record in index_page
            if str(record.get("IDX_CLSS") or "").strip() == index_class
            and str(record.get("IDX_NM") or "") == index_name
        ]
        if len(index_rows) != 1:
            raise ProviderTerminalError(
                f"KRX hedge-series index row is missing or duplicated for {as_of.isoformat()}",
                provider=_PROVIDER,
                endpoint=index_endpoint,
            )
        out: list[dict[str, Any]] = []
        for ticker in wanted:
            matches = [record for record in etf_page if str(record.get("ISU_CD") or "").strip() == ticker]
            if len(matches) > 1:
                raise ProviderTerminalError(
                    f"KRX hedge-series ETF row is duplicated for {as_of.isoformat()}: {ticker}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                )
            for record in matches:
                tagged = dict(record)
                tagged["_endpoint"] = "etf"
                out.append(tagged)
        tagged_index = dict(index_rows[0])
        tagged_index["_endpoint"] = "index"
        out.append(tagged_index)
        for record in out:
            bas_dd = str(record.get("BAS_DD") or "").strip()
            if len(bas_dd) != 8 or not bas_dd.isdigit():
                raise ProviderTerminalError(
                    f"KRX hedge-series page date conflicts for {as_of.isoformat()}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                )
            try:
                page_day = date(int(bas_dd[:4]), int(bas_dd[4:6]), int(bas_dd[6:8]))
            except ValueError as exc:
                raise ProviderTerminalError(
                    f"KRX hedge-series page date conflicts for {as_of.isoformat()}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                ) from exc
            if page_day != as_of:
                raise ProviderTerminalError(
                    f"KRX hedge-series page date conflicts for {as_of.isoformat()}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                )
            if record.get("_endpoint") == "index":
                try:
                    level = float(str(record.get("CLSPRC_IDX")).replace(",", "").strip())
                except (TypeError, ValueError):
                    raise ProviderTerminalError(
                        f"KRX hedge-series index level is not positive for {as_of.isoformat()}",
                        provider=_PROVIDER,
                        endpoint=index_endpoint,
                    ) from None
                if not level > 0:
                    raise ProviderTerminalError(
                        f"KRX hedge-series index level is not positive for {as_of.isoformat()}",
                        provider=_PROVIDER,
                        endpoint=index_endpoint,
                    )
            else:
                try:
                    close = float(str(record.get("TDD_CLSPRC")).replace(",", "").strip())
                except (TypeError, ValueError):
                    raise ProviderTerminalError(
                        f"KRX hedge-series ETF close is not positive for {as_of.isoformat()}",
                        provider=_PROVIDER,
                        endpoint=self.ENDPOINTS["ETF_TRADE"],
                    ) from None
                if not close > 0:
                    raise ProviderTerminalError(
                        f"KRX hedge-series ETF close is not positive for {as_of.isoformat()}",
                        provider=_PROVIDER,
                        endpoint=self.ENDPOINTS["ETF_TRADE"],
                    )
        return out

    def health_check(self) -> None:
        """Confirm collection credentials were verified without spending quota."""

    def fetch_etf_rows(self, as_of: date, *, tickers: Sequence[str]) -> list[dict[str, Any]]:
        """Return the ETF-page rows of ``tickers`` for one session, each tagged ``"_endpoint": "etf"``.

        Empty list only when the ETF page is empty without a holiday message (a ticker absent from a
        non-empty page is simply not returned; the Silver builder decides whether that is an error).

        Raises:
            ValueError: ``as_of`` not a date or a blank ticker.
            KrxHolidayError: the shared page reader reports a holiday.
            ProviderTerminalError: a requested ticker appears twice, ``BAS_DD`` differs from ``as_of``, or
                ``TDD_CLSPRC`` of a returned row is not a positive integer.
        """
        if not isinstance(as_of, date):
            raise ValueError("as_of must be a date")
        wanted = tuple(str(ticker).strip() for ticker in tickers)
        if any(not ticker for ticker in wanted):
            raise ValueError("tickers must contain non-empty strings")
        etf_page = self._records(self.ENDPOINTS["ETF_TRADE"], as_of)
        if not etf_page:
            return []
        out: list[dict[str, Any]] = []
        for ticker in wanted:
            matches = [record for record in etf_page if str(record.get("ISU_CD") or "").strip() == ticker]
            if len(matches) > 1:
                raise ProviderTerminalError(
                    f"KRX cash-series ETF row is duplicated for {as_of.isoformat()}: {ticker}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                )
            for record in matches:
                tagged = dict(record)
                tagged["_endpoint"] = "etf"
                out.append(tagged)
        for record in out:
            bas_dd = str(record.get("BAS_DD") or "").strip()
            if len(bas_dd) != 8 or not bas_dd.isdigit():
                raise ProviderTerminalError(
                    f"KRX cash-series page date conflicts for {as_of.isoformat()}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                )
            try:
                page_day = date(int(bas_dd[:4]), int(bas_dd[4:6]), int(bas_dd[6:8]))
            except ValueError as exc:
                raise ProviderTerminalError(
                    f"KRX cash-series page date conflicts for {as_of.isoformat()}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                ) from exc
            if page_day != as_of:
                raise ProviderTerminalError(
                    f"KRX cash-series page date conflicts for {as_of.isoformat()}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                )
            raw_close = record.get("TDD_CLSPRC")
            if raw_close is None or (isinstance(raw_close, str) and not raw_close.strip()):
                raise ProviderTerminalError(
                    f"KRX cash-series ETF close is not positive for {as_of.isoformat()}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                )
            if isinstance(raw_close, bool):
                raise ProviderTerminalError(
                    f"KRX cash-series ETF close is not positive for {as_of.isoformat()}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                )
            try:
                if isinstance(raw_close, int):
                    close = raw_close
                elif isinstance(raw_close, float):
                    if not raw_close.is_integer():
                        raise ValueError("non-integral close")
                    close = int(raw_close)
                else:
                    text = str(raw_close).replace(",", "").strip()
                    if not text.isdigit():
                        # Allow decimal-integral forms like "5000.0" while rejecting fractions.
                        from decimal import Decimal, InvalidOperation

                        try:
                            parsed = Decimal(text)
                        except InvalidOperation:
                            raise ValueError("invalid close") from None
                        if parsed != parsed.to_integral_value() or parsed <= 0:
                            raise ValueError("non-positive close")
                        close = int(parsed)
                    else:
                        close = int(text)
            except (TypeError, ValueError):
                raise ProviderTerminalError(
                    f"KRX cash-series ETF close is not positive for {as_of.isoformat()}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                ) from None
            if close <= 0:
                raise ProviderTerminalError(
                    f"KRX cash-series ETF close is not positive for {as_of.isoformat()}",
                    provider=_PROVIDER,
                    endpoint=self.ENDPOINTS["ETF_TRADE"],
                )
        return out


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
