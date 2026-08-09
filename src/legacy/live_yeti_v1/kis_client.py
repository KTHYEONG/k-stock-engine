import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from config.base import BASE_DIR


@dataclass
class KisCredentials:
    app_key: str
    app_secret: str
    account_no: str
    account_product_code: str
    env: str = "real"
    base_url: Optional[str] = None

    @classmethod
    def from_env(cls, env: Optional[str] = None) -> "KisCredentials":
        app_key = os.getenv("KIS_APP_KEY", "").strip()
        app_secret = os.getenv("KIS_APP_SECRET", "").strip()
        account_raw = os.getenv("KIS_ACCOUNT_NO", "").strip()
        prdt = os.getenv("KIS_ACCOUNT_PRODUCT_CODE", "").strip()
        target_env = (env or os.getenv("KIS_ENV", "real")).strip().lower()
        base_url = os.getenv("KIS_BASE_URL", "").strip() or None

        if "-" in account_raw and not prdt:
            acc, parsed_prdt = account_raw.split("-", 1)
            account_raw = acc
            prdt = parsed_prdt

        if not app_key or not app_secret:
            raise ValueError("KIS_APP_KEY/KIS_APP_SECRET is required.")
        if not account_raw or not prdt:
            raise ValueError("KIS_ACCOUNT_NO and KIS_ACCOUNT_PRODUCT_CODE are required.")

        return cls(
            app_key=app_key,
            app_secret=app_secret,
            account_no=account_raw,
            account_product_code=prdt,
            env=target_env,
            base_url=base_url,
        )


class KisClient:
    """Korea Investment OpenAPI client (domestic stock)."""

    REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"
    DEMO_BASE_URL = "https://openapivts.koreainvestment.com:29443"

    def __init__(
        self,
        credentials: KisCredentials,
        timeout: int = 15,
        retry: int = 2,
    ):
        self.creds = credentials
        self.timeout = timeout
        self.retry = retry
        self.session = requests.Session()
        self.base_url = credentials.base_url or (
            self.DEMO_BASE_URL if credentials.env.startswith("demo") or credentials.env.startswith("v") else self.REAL_BASE_URL
        )
        self._access_token: Optional[str] = None
        self._token_expire_at: Optional[datetime] = None
        self._token_cache_path = Path(BASE_DIR) / "logs" / f"kis_token_{self.creds.env}.json"

    def _load_cached_token(self) -> None:
        if not self._token_cache_path.exists():
            return
        try:
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
        except Exception:
            return

    def _save_cached_token(self, token: str, expire_at: datetime) -> None:
        self._token_cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "access_token": token,
            "expire_at": expire_at.isoformat(),
        }
        self._token_cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _request_new_token(self) -> str:
        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.creds.app_key,
            "appsecret": self.creds.app_secret,
        }
        resp = self.session.post(url, json=body, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise RuntimeError(f"Failed to issue token: {payload}")

        expires_in = int(payload.get("expires_in", 86400))
        expire_at = datetime.now() + timedelta(seconds=max(expires_in - 120, 60))
        self._access_token = token
        self._token_expire_at = expire_at
        self._save_cached_token(token, expire_at)
        return token

    def ensure_token(self) -> str:
        if self._access_token and self._token_expire_at and self._token_expire_at > datetime.now() + timedelta(minutes=1):
            return self._access_token
        self._load_cached_token()
        if self._access_token and self._token_expire_at and self._token_expire_at > datetime.now() + timedelta(minutes=1):
            return self._access_token
        return self._request_new_token()

    def _headers(
        self,
        tr_id: Optional[str] = None,
        tr_cont: str = "",
        custtype: str = "P",
        include_auth: bool = True,
        hashkey: Optional[str] = None,
    ) -> Dict[str, str]:
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

    def _call(
        self,
        method: str,
        path: str,
        tr_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        include_auth: bool = True,
        use_hashkey: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        request_body = body or {}
        hashkey = None

        if use_hashkey and request_body:
            hashkey = self.get_hashkey(request_body)

        last_error: Optional[Exception] = None
        for attempt in range(self.retry + 1):
            try:
                resp = self.session.request(
                    method=method.upper(),
                    url=url,
                    headers=self._headers(tr_id=tr_id, include_auth=include_auth, hashkey=hashkey),
                    params=params,
                    json=request_body if request_body else None,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
                rt_cd = payload.get("rt_cd")
                if rt_cd not in (None, "0"):
                    msg = payload.get("msg1", "Unknown KIS API error")
                    raise RuntimeError(f"KIS API error ({rt_cd}): {msg}")
                return payload
            except Exception as exc:  # noqa: PERF203
                last_error = exc
                if attempt < self.retry:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                raise

        raise RuntimeError(f"KIS API call failed: {last_error}")

    def get_hashkey(self, data: Dict[str, Any]) -> str:
        payload = self._call(
            method="POST",
            path="/uapi/hashkey",
            body=data,
            include_auth=False,
        )
        hashkey = payload.get("HASH")
        if not hashkey:
            raise RuntimeError(f"Failed to get hashkey: {payload}")
        return hashkey

    def inquire_price(self, symbol: str) -> Dict[str, Any]:
        payload = self._call(
            method="GET",
            path="/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
            },
        )
        return payload.get("output", {})

    def inquire_balance(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
        return payload.get("output1", []) or [], payload.get("output2", [{}])[0] if payload.get("output2") else {}

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: Optional[float] = None,
        order_type: str = "market",
        exchange_code: str = "KRX",
    ) -> Dict[str, Any]:
        if qty <= 0:
            raise ValueError("qty must be > 0")
        side_lower = side.lower()
        if side_lower not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        if order_type.lower() == "market":
            ord_dvsn = "01"
            ord_unpr = "0"
        elif order_type.lower() == "limit":
            if price is None or price <= 0:
                raise ValueError("limit order requires positive price")
            ord_dvsn = "00"
            ord_unpr = str(KisClient.round_to_krx_tick_size(price))
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
        return self._call(
            method="POST",
            path="/uapi/domestic-stock/v1/trading/order-cash",
            tr_id=tr_id,
            body=body,
            use_hashkey=True,
        )

    @staticmethod
    def parse_positions(output1: List[Dict[str, Any]]) -> Dict[str, int]:
        """보유수량(hldg_qty) 기준으로 포지션 파싱.

        Notes:
            hldg_qty: 실제 보유수량 (체결 기준)
            ord_psbl_qty: 당일 주문 가능 수량 (매도 가능)
        """
        positions: Dict[str, int] = {}
        for row in output1:
            ticker = (row.get("pdno") or row.get("mksc_shrn_iscd") or "").strip()
            if not ticker:
                continue
            # hldg_qty: 실제 보유수량 우선
            qty_raw = row.get("hldg_qty") or row.get("hold_qty") or "0"
            try:
                qty = int(float(qty_raw))
            except (ValueError, TypeError):
                qty = 0
            if qty > 0:
                positions[ticker] = qty
        return positions

    @staticmethod
    def parse_sellable_quantities(output1: List[Dict[str, Any]]) -> Dict[str, int]:
        """주문가능수량(ord_psbl_qty) 기준으로 매도 가능 수량 파싱.

        Notes:
            ord_psbl_qty: 당일 매도 가능 수량. 당일 매수한 주식은 D+2 결제이므로 0일 수 있음.
        """
        sellable: Dict[str, int] = {}
        for row in output1:
            ticker = (row.get("pdno") or row.get("mksc_shrn_iscd") or "").strip()
            if not ticker:
                continue
            qty_raw = row.get("ord_psbl_qty") or "0"
            try:
                qty = int(float(qty_raw))
            except (ValueError, TypeError):
                qty = 0
            if qty > 0:
                sellable[ticker] = qty
        return sellable

    @staticmethod
    def extract_cash(output2: Dict[str, Any]) -> float:
        """주문가능현금(dnca_tot_amt) 기준으로 예수금 파싱.

        Notes:
            dnca_tot_amt: 예수금 총금액 (output2 필드, 주식잔고조회)
            nxdy_excc_amt: 익일정산금액 (D+1 기준)
        """
        for key in [
            "dnca_tot_amt",      # 예수금 총금액 (API 문서 기준 primary)
            "nxdy_excc_amt",     # 익일정산금액
            "prvs_rcdl_excc_amt",  # 전일이후 정산금액 (가장 보수적)
        ]:
            if key in output2:
                try:
                    val = float(output2[key])
                    if val >= 0:
                        return val
                except (ValueError, TypeError):
                    continue
        return 0.0

    @staticmethod
    def extract_total_equity(output2: Dict[str, Any]) -> float:
        for key in [
            "tot_evlu_amt",
            "tot_evlu_pfls_amt",
            "nass_amt",
            "total_eval_amount",
        ]:
            if key in output2:
                try:
                    return float(output2[key])
                except Exception:
                    continue
        return 0.0

    def inquire_psbl_order(
        self,
        symbol: str,
        price: float = 0.0,
        ord_dvsn: str = "01",
    ) -> Dict[str, Any]:
        """매수가능조회 (TTTC8908R / VTTC8908R).

        Args:
            symbol: 종목코드 (6자리)
            price: 조회 단가 (시장가 조회 시 0 또는 공란)
            ord_dvsn: 주문구분 (기본: 01=시장가, 증거금율 반영)

        Returns:
            output dict with fields:
                nrcvb_buy_amt: 미수없는매수금액 (미수 미사용 기준 매수가능금액)
                nrcvb_buy_qty: 미수없는매수수량
                ord_psbl_cash: 주문가능현금 (예수금 기준)
                max_buy_amt: 최대매수금액 (미수 포함)
        """
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
        return payload.get("output", {})

    @staticmethod
    def extract_order_number(order_response: Dict[str, Any]) -> str:
        output = order_response.get("output", {})
        return str(output.get("ODNO") or output.get("odno") or "")

    @staticmethod
    def extract_current_price(price_output: Dict[str, Any]) -> float:
        for key in ["stck_prpr", "cur_prc", "last"]:
            if key in price_output:
                try:
                    return float(price_output[key])
                except Exception:
                    continue
        return 0.0

    @staticmethod
    def extract_limit_prices(price_output: Dict[str, Any]) -> Tuple[float, float]:
        mxpr = float(price_output.get("stck_mxpr") or 0.0)
        llam = float(price_output.get("stck_llam") or 0.0)
        return mxpr, llam

    @staticmethod
    def round_to_krx_tick_size(price: float) -> int:
        p = int(round(price))
        if p < 2000:
            tick = 1
        elif p < 5000:
            tick = 5
        elif p < 20000:
            tick = 10
        elif p < 50000:
            tick = 50
        elif p < 200000:
            tick = 100
        elif p < 500000:
            tick = 500
        else:
            tick = 1000
        return (p // tick) * tick
