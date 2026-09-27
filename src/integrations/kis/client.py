"""KIS transport-only client for active integrations."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from src.config.providers import KisPolicy
from src.config.runtime import load_runtime_config
from src.config.secrets import read_secret
from src.core.market_rules import KrxMarket, KrxMarketRules
from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
from src.integrations.transport import HttpTransport, RetryPolicy, TokenCache


@dataclass
class KisCredentials:
    app_key: str
    app_secret: str
    account_no: str
    account_product_code: str
    env: str = "real"
    base_url: str | None = None

    @classmethod
    def from_env(cls, policy: KisPolicy | None = None, *, env: str | None = None) -> KisCredentials:
        """Read KIS credentials using the env names declared in ``policy``."""
        from src.config.providers import load_provider_policy

        resolved = policy if policy is not None else load_provider_policy(load_runtime_config()).kis
        app_key = read_secret(resolved.app_key_env).strip()
        app_secret = read_secret(resolved.app_secret_env).strip()
        account_raw = read_secret(resolved.account_no_env).strip()
        prdt = read_secret(resolved.account_product_code_env, default="").strip()
        env_val = env if env is not None else read_secret(resolved.env_env, default="real")
        target_env = env_val.strip().lower() if env_val else "real"

        if "-" in account_raw and not prdt:
            acc, parsed_prdt = account_raw.split("-", 1)
            account_raw = acc
            prdt = parsed_prdt

        if not account_raw or not prdt:
            raise ValueError("KIS account number and product code are required.")

        return cls(
            app_key=app_key,
            app_secret=app_secret,
            account_no=account_raw,
            account_product_code=prdt,
            env=target_env,
        )


def _default_kis_policy() -> KisPolicy | None:
    try:
        from src.config.providers import load_provider_policy

        return load_provider_policy(load_runtime_config()).kis
    except Exception:  # noqa: BLE001
        # reason: adapter boundary — client must stay constructible in tests without runtime config.
        return None


class KisClient:
    """Korea Investment OpenAPI client (domestic stock)."""

    REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"
    DEMO_BASE_URL = "https://openapivts.koreainvestment.com:29443"

    def __init__(
        self,
        credentials: KisCredentials,
        timeout: int = 15,
        retry: int = 2,
        *,
        token_cache_dir: Path | None = None,
        policy: KisPolicy | None = None,
        min_interval_seconds: float | None = None,
    ) -> None:
        self.creds = credentials
        self.timeout = timeout
        self.retry = retry
        self.session = requests.Session()
        self.base_url = credentials.base_url or (
            self.DEMO_BASE_URL if credentials.env.startswith("demo") or credentials.env.startswith("v") else self.REAL_BASE_URL
        )
        self._access_token: str | None = None
        self._token_expire_at: datetime | None = None
        resolved_policy = policy if policy is not None else _default_kis_policy()
        if min_interval_seconds is not None:
            pace = float(min_interval_seconds)
        elif resolved_policy is not None:
            pace = float(resolved_policy.min_interval_seconds)
        else:
            pace = 0.0
        max_attempts = int(resolved_policy.max_attempts) if resolved_policy is not None else int(retry) + 1
        self._retryable_codes: tuple[str, ...] = (
            tuple(resolved_policy.retryable_msg_codes) if resolved_policy is not None else ()
        )
        cache_dir = token_cache_dir if token_cache_dir is not None else load_runtime_config().logs_root
        self._token_cache_path = Path(cache_dir) / f"kis_token_{self.creds.env}.json"
        self.token_cache = TokenCache(cache_dir, provider="kis", env=self.creds.env)
        self._transport = HttpTransport(
            provider="KIS",
            base_url=self.base_url,
            min_interval_seconds=pace,
            retry=RetryPolicy(max_attempts=max(1, max_attempts)),
            quota=None,
            timeout_seconds=float(timeout),
            session=self.session,
            sleep=lambda seconds: time.sleep(seconds),
            monotonic=lambda: time.monotonic(),
        )

    def _sync_session(self) -> None:
        self._transport._session = self.session

    def _load_cached_token(self) -> None:
        try:
            if not self._token_cache_path.exists():
                return
            if self._token_cache_path.stat().st_mode & 0o077:
                return
            import json

            payload = json.loads(self._token_cache_path.read_text(encoding="utf-8"))
            token = payload.get("access_token")
            expire_at = payload.get("expire_at")
            if not token or not expire_at:
                return
            dt_exp = datetime.fromisoformat(expire_at)
            if dt_exp <= datetime.now() + timedelta(minutes=2):
                return
            self._access_token = token
            self._token_expire_at = dt_exp
        except (OSError, ValueError):
            # reason: adapter boundary — a corrupt token cache must read as a miss, never crash collection.
            return

    def _save_cached_token(self, token: str, expire_at: datetime) -> None:
        import contextlib
        import json
        import os
        import threading

        self._token_cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"access_token": token, "expire_at": expire_at.isoformat()}
        tmp_path = self._token_cache_path.parent / f".{self._token_cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(data, ensure_ascii=False))
            os.replace(tmp_path, self._token_cache_path)
            self._token_cache_path.chmod(0o600)
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    def _request_new_token(self) -> str:
        url = f"{self.base_url}/oauth2/tokenP"
        body = {"grant_type": "client_credentials", "appkey": self.creds.app_key, "appsecret": self.creds.app_secret}
        resp = self.session.post(url, json=body, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        raw_token = payload.get("access_token")
        if not raw_token:
            raise ProviderTerminalError(f"Failed to issue token: {payload}", provider="KIS", endpoint="oauth2/tokenP")
        token = str(raw_token)
        expires_in = int(payload.get("expires_in", 86400))
        expire_at = datetime.now() + timedelta(seconds=max(expires_in - 120, 60))
        self._access_token = token
        self._token_expire_at = expire_at
        self._save_cached_token(token, expire_at)
        return str(token)

    def ensure_token(self) -> str:
        if self._access_token and self._token_expire_at and self._token_expire_at > datetime.now() + timedelta(minutes=1):
            return self._access_token
        self._load_cached_token()
        if self._access_token and self._token_expire_at and self._token_expire_at > datetime.now() + timedelta(minutes=1):
            return self._access_token
        return self._request_new_token()

    def _headers(
        self,
        tr_id: str | None = None,
        tr_cont: str = "",
        custtype: str = "P",
        include_auth: bool = True,
        hashkey: str | None = None,
    ) -> dict[str, str]:
        headers = {
            "content-type": "application/json; charset=utf-8",
            "appkey": self.creds.app_key,
            "appsecret": self.creds.app_secret,
            "custtype": custtype,
        }
        if tr_id:
            headers["tr_id"] = tr_id
        if tr_cont:
            headers["tr_cont"] = tr_cont
        if include_auth:
            headers["authorization"] = f"Bearer {self.ensure_token()}"
        if hashkey:
            headers["hashkey"] = hashkey
        return headers

    def _classify_business(self, payload: dict[str, Any], *, endpoint: str) -> None:
        rt_cd = payload.get("rt_cd")
        if rt_cd in (None, "0"):
            return
        msg = str(payload.get("msg1", "Unknown KIS API error"))
        code = str(payload.get("msg_cd", ""))
        for retryable in self._retryable_codes:
            if (code and (code == retryable or code.startswith(retryable))) or (msg and msg.startswith(retryable)):
                raise ProviderRetryableError(f"KIS API error ({rt_cd}): {msg}", provider="KIS", endpoint=endpoint)
        raise ProviderTerminalError(f"KIS API error ({rt_cd}): {msg}", provider="KIS", endpoint=endpoint)

    def _call(
        self,
        method: str,
        path: str,
        tr_id: str | None = None,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        include_auth: bool = True,
        use_hashkey: bool = False,
    ) -> dict[str, Any]:
        request_body = body or {}
        hashkey = None
        if use_hashkey and request_body:
            hashkey = self.get_hashkey(request_body)
        headers = self._headers(tr_id=tr_id, include_auth=include_auth, hashkey=hashkey)
        parsed: list[dict[str, Any]] = []
        endpoint = path

        def _classify(response: requests.Response) -> None:
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderRetryableError(
                    f"KIS transient invalid JSON for {endpoint}", provider="KIS", endpoint=endpoint
                ) from exc
            if not isinstance(payload, dict):
                raise ProviderTerminalError(f"KIS response must be an object: {payload}", provider="KIS", endpoint=endpoint)
            self._classify_business(payload, endpoint=endpoint)
            parsed.append(payload)

        self._sync_session()
        str_params = {str(k): str(v) for k, v in (params or {}).items()}
        if method.upper() == "GET":
            self._transport.get(endpoint, str_params, headers=headers, classify=_classify)
        else:
            self._transport.post(endpoint, {str(k): v for k, v in request_body.items()}, headers=headers, classify=_classify)
        return parsed[0]

    def get_hashkey(self, data: dict[str, Any]) -> str:
        payload = self._call(method="POST", path="/uapi/hashkey", body=data, include_auth=False)
        hashkey = payload.get("HASH")
        if not hashkey:
            raise ProviderTerminalError(f"Failed to get hashkey: {payload}", provider="KIS", endpoint="/uapi/hashkey")
        return str(hashkey)

    def inquire_price(self, symbol: str) -> dict[str, Any]:
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        payload = self._call(
            method="GET",
            path="/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
        )
        out = payload.get("output", {})
        if not isinstance(out, dict):
            raise ProviderTerminalError(f"KIS inquire_price malformed output: {payload}", provider="KIS", endpoint="inquire-price")
        return out

    def search_stock_info(self, symbol: str) -> dict[str, Any]:
        """Return KIS's stock-basic-information output (``CTPF1002R``) for one ticker."""
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        payload = self._call(
            method="GET",
            path="/uapi/domestic-stock/v1/quotations/search-stock-info",
            tr_id="CTPF1002R",
            params={"PRDT_TYPE_CD": "300", "PDNO": symbol},
        )
        out = payload.get("output", {})
        if not isinstance(out, dict):
            raise ProviderTerminalError(
                f"KIS search-stock-info malformed output: {payload}", provider="KIS", endpoint="search-stock-info"
            )
        return out

    def inquire_investor_trade_by_stock_daily(self, symbol: str, anchor: date) -> tuple[dict[str, Any], ...]:
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        if not isinstance(anchor, date):
            raise ValueError("anchor must be a date")
        payload = self._call(
            method="GET",
            path="/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily",
            tr_id="FHPTJ04160001",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol.strip(),
                "FID_INPUT_DATE_1": anchor.strftime("%Y%m%d"),
                "FID_ORG_ADJ_PRC": "",
                "FID_ETC_CLS_CODE": "",
            },
        )
        raw = payload.get("output2", [])
        if not isinstance(raw, list):
            raise ProviderTerminalError("KIS investor flow output2 must be a list", provider="KIS", endpoint="investor-flow")
        return tuple(item for item in raw if isinstance(item, dict))

    def inquire_balance(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tr_id = "VTTC8434R" if self.creds.env.startswith("demo") or self.creds.env.startswith("v") else "TTTC8434R"
        payload = self._call(
            method="GET",
            path="/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id=tr_id,
            params={
                "CANO": self.creds.account_no,
                "ACNT_PRDT_CD": self.creds.account_product_code,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
        )
        o1 = payload.get("output1", [])
        o2 = payload.get("output2", [{}])
        if not isinstance(o1, list):
            raise ProviderTerminalError(f"KIS inquire_balance output1 malformed: {payload}", provider="KIS", endpoint="inquire-balance")
        if not isinstance(o2, list):
            raise ProviderTerminalError(f"KIS inquire_balance output2 malformed: {payload}", provider="KIS", endpoint="inquire-balance")
        return o1 or [], o2[0] if o2 else {}

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float | None = None,
        order_type: str = "market",
        exchange_code: str = "KRX",
        *,
        market: KrxMarket,
        session: date,
        rules: KrxMarketRules,
    ) -> dict[str, Any]:
        if qty <= 0:
            raise ValueError("qty must be > 0")
        side_lower = side.lower()
        if side_lower not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        if order_type.lower() == "market":
            ord_dvsn = "01"
            ord_unpr = "0"
        elif order_type.lower() == "limit":
            if price is None or price <= 0:
                raise ValueError("limit order requires positive price")
            rounded = round(price)
            tick = rules.tick_size(session=session, market=market, price=rounded)
            ord_dvsn = "00"
            ord_unpr = str((rounded // tick) * tick)
        else:
            raise ValueError("order_type must be 'market' or 'limit'")
        if self.creds.env.startswith("demo") or self.creds.env.startswith("v"):
            tr_id = "VTTC0012U" if side_lower == "buy" else "VTTC0011U"
        else:
            tr_id = "TTTC0012U" if side_lower == "buy" else "TTTC0011U"
        body = {
            "CANO": self.creds.account_no,
            "ACNT_PRDT_CD": self.creds.account_product_code,
            "PDNO": symbol,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": ord_unpr,
            "EXCG_ID_DVSN_CD": exchange_code,
        }
        return self._call(method="POST", path="/uapi/domestic-stock/v1/trading/order-cash", tr_id=tr_id, body=body, use_hashkey=True)

    @staticmethod
    def parse_positions(output1: list[dict[str, Any]]) -> dict[str, int]:
        positions: dict[str, int] = {}
        for row in output1:
            if not isinstance(row, dict):
                continue
            ticker = (row.get("pdno") or row.get("mksc_shrn_iscd") or "").strip()
            if not ticker:
                continue
            qty_raw = row.get("hldg_qty") or row.get("hold_qty") or "0"
            try:
                qty = int(float(str(qty_raw)))
            except (ValueError, TypeError):
                qty = 0
            if qty > 0:
                positions[ticker] = qty
        return positions

    @staticmethod
    def parse_sellable_quantities(output1: list[dict[str, Any]]) -> dict[str, int]:
        sellable: dict[str, int] = {}
        for row in output1:
            if not isinstance(row, dict):
                continue
            ticker = (row.get("pdno") or row.get("mksc_shrn_iscd") or "").strip()
            if not ticker:
                continue
            qty_raw = row.get("ord_psbl_qty") or "0"
            try:
                qty = int(float(str(qty_raw)))
            except (ValueError, TypeError):
                qty = 0
            if qty > 0:
                sellable[ticker] = qty
        return sellable

    @staticmethod
    def extract_cash(output2: dict[str, Any]) -> float:
        for key in ["dnca_tot_amt", "nxdy_excc_amt", "prvs_rcdl_excc_amt"]:
            if key in output2:
                try:
                    val = float(output2[key])
                    if val >= 0:
                        return val
                except (ValueError, TypeError):
                    continue
        return 0.0

    @staticmethod
    def extract_total_equity(output2: dict[str, Any]) -> float:
        for key in ["tot_evlu_amt", "tot_evlu_pfls_amt", "nass_amt", "total_eval_amount"]:
            if key in output2:
                try:
                    return float(output2[key])
                except (ValueError, TypeError):
                    # reason: adapter boundary — vendor numeric fields vary; unparseable values read as absent.
                    continue
        return 0.0

    def inquire_psbl_order(self, symbol: str, price: float = 0.0, ord_dvsn: str = "01") -> dict[str, Any]:
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        is_demo = self.creds.env.startswith("demo") or self.creds.env.startswith("v")
        tr_id = "VTTC8908R" if is_demo else "TTTC8908R"
        unpr_str = str(int(price)) if price > 0 else ""
        payload = self._call(
            method="GET",
            path="/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            tr_id=tr_id,
            params={
                "CANO": self.creds.account_no,
                "ACNT_PRDT_CD": self.creds.account_product_code,
                "PDNO": symbol,
                "ORD_UNPR": unpr_str,
                "ORD_DVSN": ord_dvsn,
                "CMA_EVLU_AMT_ICLD_YN": "N",
                "OVRS_ICLD_YN": "N",
            },
        )
        out = payload.get("output", {})
        if not isinstance(out, dict):
            raise ProviderTerminalError(f"KIS inquire_psbl_order malformed output: {payload}", provider="KIS", endpoint="inquire-psbl-order")
        return out

    @staticmethod
    def extract_order_number(order_response: dict[str, Any]) -> str:
        if not isinstance(order_response, dict):
            raise ValueError("order_response must be an object")
        output = order_response.get("output", {})
        if not isinstance(output, dict):
            return ""
        return str(output.get("ODNO") or output.get("odno") or "")

    @staticmethod
    def extract_current_price(price_output: dict[str, Any]) -> float:
        if not isinstance(price_output, dict):
            raise ValueError("price_output must be an object")
        for key in ["stck_prpr", "cur_prc", "last"]:
            if key in price_output:
                try:
                    return float(price_output[key])
                except (ValueError, TypeError):
                    # reason: adapter boundary — vendor numeric fields vary; unparseable values read as absent.
                    continue
        return 0.0

    @staticmethod
    def extract_limit_prices(price_output: dict[str, Any]) -> tuple[float, float]:
        if not isinstance(price_output, dict):
            raise ValueError("price_output must be an object")
        try:
            mxpr = float(price_output.get("stck_mxpr") or 0.0)
            llam = float(price_output.get("stck_llam") or 0.0)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid limit price payload: {price_output}") from exc
        return mxpr, llam
