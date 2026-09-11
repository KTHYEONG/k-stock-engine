"""LS Securities OpenAPI client."""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import requests

from src.core.pit import PITDataError

LS_MIN_REQUEST_INTERVAL_SECONDS: Final[float] = 1.05
_LS_RETRY_BACKOFF_SECONDS: Final[float] = 1.2
_LS_MAX_ATTEMPTS: Final[int] = 3


@dataclass(frozen=True, slots=True)
class LsCredentials:
    app_key: str
    app_secret: str

    @classmethod
    def from_env(cls) -> LsCredentials:
        app_key = os.getenv("LS_APP_KEY", "").strip()
        app_secret = os.getenv("LS_APP_SECRET", "").strip()
        if not app_key or not app_secret:
            raise PITDataError("missing required LS OpenAPI credentials (LS_APP_KEY, LS_APP_SECRET)")
        return cls(app_key=app_key, app_secret=app_secret)


class LsClient:
    """LS Securities OpenAPI client for investor flow (t1702) and charts."""

    _slot_lock = threading.Lock()
    _last_slot: float | None = None

    def __init__(
        self,
        credentials: LsCredentials | None = None,
        *,
        base_url: str = "https://openapi.ls-sec.co.kr:8080",
        session: requests.Session | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self._token: str | None = None
        self._session = session if session is not None else requests.Session()
        self._monotonic = monotonic
        self._sleeper = sleeper

    def ensure_token(self) -> str:
        if self._token:
            return self._token
        creds = self.credentials or LsCredentials.from_env()
        resp = self._session.post(
            f"{self.base_url}/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "appkey": creds.app_key,
                "appsecretkey": creds.app_secret,
                "scope": "oob",
            },
            timeout=10,
        )
        data = resp.json()
        token = str(data.get("access_token", ""))
        if not token:
            raise PITDataError(f"LS token issuance failed: {data}")
        self._token = token
        return token

    def _wait_for_request_slot(self) -> None:
        with LsClient._slot_lock:
            now = self._monotonic()
            last = LsClient._last_slot
            if last is None:
                LsClient._last_slot = now
                return
            wait = LS_MIN_REQUEST_INTERVAL_SECONDS - (now - last)
            if wait > 0:
                self._sleeper(wait)
                now = last + LS_MIN_REQUEST_INTERVAL_SECONDS
            LsClient._last_slot = now

    def _post_investor_trend(self, *, headers: Mapping[str, str], body: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        url = f"{self.base_url}/stock/frgr-itt"
        last_error: str = ""
        for attempt in range(1, _LS_MAX_ATTEMPTS + 1):
            self._wait_for_request_slot()
            resp = self._session.post(url, headers=dict(headers), json=dict(body), timeout=15)
            try:
                resp.raise_for_status()
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is None:
                    status = getattr(resp, "status_code", None)
                if status == 429 and attempt < _LS_MAX_ATTEMPTS:
                    self._sleeper(_LS_RETRY_BACKOFF_SECONDS)
                    continue
                raise PITDataError(f"LS t1702 HTTP error: {exc}") from exc
            try:
                data = resp.json()
            except ValueError as exc:
                raise PITDataError(f"LS t1702 malformed JSON: {exc}") from exc
            if not isinstance(data, Mapping):
                raise PITDataError("LS t1702 non-mapping payload; certification blocked")
            rsp_cd = str(data.get("rsp_cd", "")).strip()
            if rsp_cd == "IGW00201":
                last_error = rsp_cd
                if attempt < _LS_MAX_ATTEMPTS:
                    self._sleeper(_LS_RETRY_BACKOFF_SECONDS)
                    continue
                raise PITDataError(f"LS t1702 throttled ({last_error}); certification blocked")
            if rsp_cd != "00000":
                raise PITDataError(f"LS t1702 non-success rsp_cd {rsp_cd!r}; certification blocked")
            rows = data.get("t1702OutBlock1")
            if not isinstance(rows, list):
                raise PITDataError("LS t1702 t1702OutBlock1 missing or not a list; certification blocked")
            return tuple(dict(r) for r in rows if isinstance(r, dict))
        raise PITDataError(f"LS t1702 throttled ({last_error}); certification blocked")  # pragma: no cover

    def inquire_investor_trend(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        unit: str = "amount",
    ) -> tuple[dict[str, Any], ...]:
        token = self.ensure_token()
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "tr_cd": "t1702",
            "tr_cont": "N",
            "tr_cont_key": "",
        }
        volvalgb = "1" if unit == "amount" else "0"
        body: dict[str, Any] = {
            "t1702InBlock": {
                "shcode": symbol.strip(),
                "fromdt": start_date.strftime("%Y%m%d"),
                "todt": end_date.strftime("%Y%m%d"),
                "volvalgb": volvalgb,
                "msmdgb": "0",
                "gubun": "0",
                "exchgubun": "K",
            },
            "tr_cd": "t1702",
        }
        return self._post_investor_trend(headers=headers, body=body)
