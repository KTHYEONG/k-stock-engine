"""Kiwoom REST API client."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

from src.data.schemas import PITDataError


@dataclass(frozen=True, slots=True)
class KiwoomCredentials:
    app_key: str
    secret_key: str

    @classmethod
    def from_env(cls) -> KiwoomCredentials:
        app_key = (os.getenv("KIWOM_APP_KEY") or os.getenv("KIWOOM_APP_KEY") or "").strip()
        secret_key = (os.getenv("KIWOM_SECRET_KEY") or os.getenv("KIWOOM_SECRET_KEY") or "").strip()
        if not app_key or not secret_key:
            raise PITDataError("missing required Kiwoom credentials (KIWOM_APP_KEY, KIWOM_SECRET_KEY)")
        return cls(app_key=app_key, secret_key=secret_key)


class KiwoomClient:
    """Kiwoom REST API client for investor flow (ka10059) and charts."""

    def __init__(
        self,
        credentials: KiwoomCredentials | None = None,
        *,
        base_url: str = "https://api.kiwoom.com",
    ) -> None:
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self._token: str | None = None
        self._session = requests.Session()

    def ensure_token(self) -> str:
        if self._token:
            return self._token
        creds = self.credentials or KiwoomCredentials.from_env()
        resp = self._session.post(
            f"{self.base_url}/oauth2/token",
            json={
                "grant_type": "client_credentials",
                "appkey": creds.app_key,
                "secretkey": creds.secret_key,
            },
            timeout=10,
        )
        data = resp.json()
        token = str(data.get("token", ""))
        if not token:
            raise PITDataError(f"Kiwoom token issuance failed: {data}")
        self._token = token
        return token

    def inquire_investor_trend(
        self,
        symbol: str,
        target_date: date,
        unit: str = "amount",
    ) -> tuple[dict[str, Any], ...]:
        token = self.ensure_token()
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "api-id": "ka10059",
            "cont-yn": "N",
            "next-key": "",
        }
        amt_qty_tp = "2" if unit == "amount" else "1"
        body = {
            "stk_cd": symbol.strip(),
            "dt": target_date.strftime("%Y%m%d"),
            "amt_qty_tp": amt_qty_tp,
            "trde_tp": "0",
            "unit_tp": "1",
        }
        resp = self._session.post(
            f"{self.base_url}/api/dostk/stkinfo",
            headers=headers,
            json=body,
            timeout=15,
        )
        data = resp.json()
        rows = data.get("stk_invsr_orgn")
        if not isinstance(rows, list):
            return ()
        return tuple(dict(r) for r in rows if isinstance(r, dict))
