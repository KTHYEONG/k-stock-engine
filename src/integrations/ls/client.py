"""LS Securities OpenAPI client."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

from src.core.pit import PITDataError


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

    def __init__(
        self,
        credentials: LsCredentials | None = None,
        *,
        base_url: str = "https://openapi.ls-sec.co.kr:8080",
    ) -> None:
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self._token: str | None = None
        self._session = requests.Session()

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
        resp = self._session.post(
            f"{self.base_url}/stock/frgr-itt",
            headers=headers,
            json=body,
            timeout=15,
        )
        data = resp.json()
        rows = data.get("t1702OutBlock1")
        if not isinstance(rows, list):
            return ()
        return tuple(dict(r) for r in rows if isinstance(r, dict))
