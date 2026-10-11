"""KIS mock-domain broker adapter; live credentials are refused."""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import date, datetime

from src.core.instruments import AssetKind, Instrument
from src.core.market_rules import KrxMarket, KrxMarketRules
from src.execution.domain.orders import OrderRequest, OrderSide, OrderState, OrderStateRecord
from src.execution.domain.portfolio import PortfolioSnapshot, Position

_LOG = logging.getLogger("execution.kis_broker")


class LiveTradingForbiddenError(RuntimeError):
    """The adapter was constructed with non-mock credentials."""


def _is_mock_env(env: str) -> bool:
    lowered = env or ""
    return lowered.startswith("demo") or lowered.startswith("v")


def _masked_account(account_no: str) -> str:
    stripped = str(account_no or "")
    if len(stripped) <= 2:
        return "**"
    return "*" * (len(stripped) - 2) + stripped[-2:]


class KisMockBroker:
    """Order adapter bound to the KIS mock domain."""

    def __init__(
        self,
        client: object,
        *,
        rules: KrxMarketRules,
        markets: Mapping[str, KrxMarket],
        session: date,
    ) -> None:
        from src.integrations.kis.client import KisClient

        if not isinstance(client, KisClient):
            raise TypeError("client must be a KisClient")
        env = str(getattr(client.creds, "env", "") or "")
        if not _is_mock_env(env):
            raise LiveTradingForbiddenError("non-mock credentials are forbidden")
        if client.base_url != KisClient.DEMO_BASE_URL or client._transport._base_url != KisClient.DEMO_BASE_URL:
            raise LiveTradingForbiddenError("non-mock endpoint is forbidden")
        self._client = client
        self._rules = rules
        self._markets = dict(markets)
        self._session = session

    @property
    def account(self) -> str:
        return str(self._client.creds.account_no)

    @staticmethod
    def _symbol_for(instrument_id: str) -> str:
        token = str(instrument_id).strip()
        if not token:
            raise ValueError("instrument_id must be non-empty")
        return token.split(":")[-1].strip()

    def submit(self, request: OrderRequest) -> OrderStateRecord:
        quantity = request.quantity
        if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
            raise ValueError("quantity must be a positive integer")
        if not math.isfinite(quantity) or float(quantity) <= 0 or float(quantity) != float(int(float(quantity))):
            raise ValueError("quantity must be a positive integer")
        qty = int(float(quantity))
        if request.instrument_id not in self._markets:
            raise ValueError(f"unknown market for {request.instrument_id!r}")
        market = self._markets[request.instrument_id]
        symbol = self._symbol_for(request.instrument_id)
        side = "buy" if request.side is OrderSide.BUY else "sell"
        try:
            response = self._client.place_order(
                symbol,
                side,
                qty,
                order_type="market",
                market=market,
                session=self._session,
                rules=self._rules,
            )
        except Exception as exc:
            from src.integrations.errors import ProviderTerminalError

            if isinstance(exc, ProviderTerminalError) and str(exc).startswith("KIS API error ("):
                message = str(exc)
                _LOG.warning(
                    "mock order rejected account=%s instrument=%s",
                    _masked_account(self.account),
                    request.instrument_id,
                )
                return OrderStateRecord(
                    order_id=request.order_id,
                    intent_id=request.intent_id,
                    instrument_id=request.instrument_id,
                    asset_kind=request.asset_kind,
                    side=request.side,
                    state=OrderState.REJECTED,
                    submitted_quantity=float(qty),
                    reason=message[:500],
                )
            raise
        if not isinstance(response, dict) or "rt_cd" not in response:
            raise ValueError("malformed order response")
        rt_cd = str(response["rt_cd"])
        if rt_cd != "0":
            message = str(response.get("msg1", f"rt_cd={rt_cd}")) if isinstance(response, dict) else f"rt_cd={rt_cd}"
            _LOG.warning(
                "mock order rejected account=%s instrument=%s",
                _masked_account(self.account),
                request.instrument_id,
            )
            return OrderStateRecord(
                order_id=request.order_id,
                intent_id=request.intent_id,
                instrument_id=request.instrument_id,
                asset_kind=request.asset_kind,
                side=request.side,
                state=OrderState.REJECTED,
                submitted_quantity=float(qty),
                reason=message[:500],
            )
        order_no = ""
        try:
            order_no = str(self._client.extract_order_number(response) or "")
        except ValueError:
            order_no = ""
        if not order_no:
            raise ValueError("accepted order response has no broker order number")
        _LOG.info(
            "mock order submitted account=%s instrument=%s qty=%d",
            _masked_account(self.account),
            request.instrument_id,
            qty,
        )
        return OrderStateRecord(
            order_id=request.order_id,
            intent_id=request.intent_id,
            instrument_id=request.instrument_id,
            asset_kind=request.asset_kind,
            side=request.side,
            state=OrderState.SUBMITTED,
            submitted_quantity=float(qty),
            reason=order_no[:200],
        )

    def reconcile(self, account: str) -> list[OrderStateRecord]:
        raise NotImplementedError("daily order inquiry TR is unconfirmed for the mock domain")

    def snapshot(self, *, as_of: datetime) -> PortfolioSnapshot:
        output1, output2 = self._client.inquire_balance()
        quantities = self._client.parse_positions(list(output1))
        cash = float(self._client.extract_cash(dict(output2)))
        positions: list[Position] = []
        for ticker in sorted(quantities):
            qty = int(quantities[ticker])
            if qty <= 0:
                continue
            instrument_id = ticker if ":" in ticker else f"KRX:{ticker}"
            symbol = ticker.split(":")[-1]
            positions.append(
                Position(
                    instrument=Instrument(instrument_id, AssetKind.STOCK, "KRX", symbol, "KRW"),
                    quantity=float(qty),
                    average_cost=0.0,
                )
            )
        return PortfolioSnapshot(
            account_snapshot_id=f"kis-mock:{as_of.isoformat()}",
            as_of=as_of,
            settled_cash=cash,
            unsettled_cash=0.0,
            positions=tuple(positions),
        )
