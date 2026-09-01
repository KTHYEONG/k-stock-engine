"""Typed cost-evidence loading and shared fill-cost resolution.

The cost evidence artifact is the single hash-bound source for the BanKIS
lifetime-preferential commission assumption, statutory sell taxes, KRX tick
sizes, and the liquidity slippage model. ``resolve_fill_cost`` is the one
common per-fill cost path used by both the backtest engine and the simulator so
the two execution paths resolve identical per-fill costs for identical inputs.

The loader fails closed: malformed JSON, unknown schema versions, unsorted or
duplicated effective dates, a gap before the requested coverage, negative
rates, unsupported market codes, an incomplete tick band tiling, or a missing
source URI raise ``ValueError``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from math import inf
from pathlib import Path
from typing import Any

from src.core.costs import (
    CostPoint,
    CostSchedule,
    FillCostBreakdown,
    LiquiditySlippageModel,
    TickSizeRule,
    TickSizeSchedule,
)
from legacy.stocks.data.contracts import CoverageRange

KRX_MARKETS = ("KOSPI", "KOSDAQ")
SUPPORTED_SCHEMA_VERSIONS = (1,)
DEFAULT_SETTLEMENT_DAYS = 2


def krx_market_for_code(instrument_id: str) -> str:
    """Resolve the KRX listing market from a canonical instrument id.

    KRX 6-digit codes starting with ``0`` trade on KOSPI and codes starting
    with ``1`` trade on KOSDAQ.
    """
    code = instrument_id.rsplit(":", 1)[-1]
    if code.startswith("0"):
        return "KOSPI"
    if code.startswith("1"):
        return "KOSDAQ"
    raise ValueError(f"cannot resolve KRX market from instrument id {instrument_id!r}")


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One statutory or policy source bound by URI, retrieval time, and hash."""

    uri: str
    retrieved_at: datetime
    content_hash: str


@dataclass(frozen=True, slots=True)
class CommissionRule:
    """Effective-dated per-side commission assumption."""

    effective_from: datetime
    buy_rate: float
    sell_rate: float


@dataclass(frozen=True, slots=True)
class SellTaxRule:
    """Effective-dated statutory sell tax for one KRX market.

    ``sell_tax_rate`` is the exact sum of the securities transaction tax and
    the rural special tax components, so buy/sell tax separation is explicit.
    """

    effective_from: datetime
    market: str
    securities_transaction_tax_rate: float
    rural_special_tax_rate: float
    source_uri: str
    source_hash: str

    @property
    def sell_tax_rate(self) -> float:
        return self.securities_transaction_tax_rate + self.rural_special_tax_rate


@dataclass(frozen=True, slots=True)
class LiquidityModelSpec:
    """Explicit liquidity-model calibration carried by the artifact."""

    model_id: str
    impact_coefficient: float
    stress_multiplier: float


@dataclass(frozen=True, slots=True)
class CostEvidence:
    """Typed, immutable cost evidence artifact.

    ``content_hash`` is the SHA-256 of the artifact file bytes at load time,
    binding every fill to the exact evidence version. ``path`` records where
    the artifact was read from for provenance.
    """

    schema_version: int
    coverage: CoverageRange
    assumption_id: str
    sources: tuple[SourceRecord, ...]
    commission: tuple[CommissionRule, ...]
    sell_taxes: tuple[SellTaxRule, ...]
    tick_size_rules: tuple[TickSizeRule, ...]
    liquidity_model: LiquidityModelSpec
    settlement_days: int = DEFAULT_SETTLEMENT_DAYS
    content_hash: str = ""
    path: Path | None = None

    @property
    def tick_schedule(self) -> TickSizeSchedule:
        return TickSizeSchedule(self.tick_size_rules)

    @property
    def base_liquidity_model(self) -> LiquiditySlippageModel:
        return LiquiditySlippageModel(
            impact_coefficient=self.liquidity_model.impact_coefficient,
            tick_schedule=self.tick_schedule,
            stress_multiplier=1.0,
            model_id=self.liquidity_model.model_id,
        )

    @property
    def stress_liquidity_model(self) -> LiquiditySlippageModel:
        return LiquiditySlippageModel(
            impact_coefficient=self.liquidity_model.impact_coefficient,
            tick_schedule=self.tick_schedule,
            stress_multiplier=self.liquidity_model.stress_multiplier,
            model_id=self.liquidity_model.model_id,
        )

    def base_schedule(self, market: str = "KOSPI") -> CostSchedule:
        """Effective-dated simplified ``CostSchedule`` from the evidence.

        The commission is the per-side buy rate and the tax is the statutory
        sell tax for ``market``; slippage is left to the provenance-bound
        liquidity model, so the schedule contributes commission and tax only.
        """
        points = tuple(
            CostPoint(
                effective_from=rule.effective_from,
                commission_rate=rule.buy_rate,
                tax_rate=self.sell_tax_for(market, rule.effective_from).sell_tax_rate,
                slippage_bps=0.0,
                settlement_days=self.settlement_days,
            )
            for rule in self.commission
        )
        return CostSchedule(name=f"{self.assumption_id}-base", points=points)

    def stress_schedule(self, market: str = "KOSPI") -> CostSchedule:
        """Effective-dated simplified stress ``CostSchedule`` from the evidence."""
        points = tuple(
            CostPoint(
                effective_from=rule.effective_from,
                commission_rate=rule.sell_rate,
                tax_rate=self.sell_tax_for(market, rule.effective_from).sell_tax_rate,
                slippage_bps=0.0,
                settlement_days=self.settlement_days,
            )
            for rule in self.commission
        )
        return CostSchedule(name=f"{self.assumption_id}-stress", points=points)

    def commission_for(self, effective_time: datetime) -> CommissionRule:
        """Resolve the commission rule covering ``effective_time``."""
        when = _as_utc(effective_time)
        chosen = [rule for rule in self.commission if _as_utc(rule.effective_from) <= when]
        if not chosen:
            raise ValueError(f"no commission coverage at {when.isoformat()}")
        return max(chosen, key=lambda rule: _as_utc(rule.effective_from))

    def sell_tax_for(self, market: str, effective_time: datetime) -> SellTaxRule:
        """Resolve the sell tax rule for ``market`` at ``effective_time``."""
        if market not in KRX_MARKETS:
            raise ValueError(f"unsupported market {market!r}")
        when = _as_utc(effective_time)
        chosen = [
            rule
            for rule in self.sell_taxes
            if rule.market == market and _as_utc(rule.effective_from) <= when
        ]
        if not chosen:
            raise ValueError(
                f"no sell-tax coverage for {market} at {when.isoformat()}"
            )
        return max(chosen, key=lambda rule: _as_utc(rule.effective_from))


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"cost evidence artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid cost evidence artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"cost evidence artifact must be a JSON object: {path}")
    return payload


def _required(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"cost evidence artifact missing {key!r}")
    return value


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _effective_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 date or datetime")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO-8601 date or datetime") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _non_negative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    rate = float(value)
    if rate < 0:
        raise ValueError(f"{field} must be non-negative")
    return rate


def _positive_upper(value: Any, field: str) -> float:
    if value is None:
        return inf
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number or null")
    upper = float(value)
    if upper <= 0:
        raise ValueError(f"{field} must be positive")
    return upper


def _assert_sorted_effective(
    rules: list[tuple[datetime, tuple[str, ...]]], label: str
) -> None:
    previous: datetime | None = None
    for when, _ident in rules:
        if previous is not None and when < previous:
            raise ValueError(f"{label} effective dates must be sorted ascending")
        if previous is not None and when == previous:
            raise ValueError(f"{label} effective dates must not duplicate")
        previous = when


def load_cost_evidence(path: Path, required_range: CoverageRange) -> CostEvidence:
    """Load and fail-closed validate a cost evidence artifact.

    Validation covers schema version, coverage containment of
    ``required_range``, sorted non-duplicate effective dates, exact tax
    component sums, supported markets, gapless tick band tiling, and the
    presence of a source URI/hash on every sell-tax row.
    """
    payload = _read_object(path)
    schema_version = _required(payload, "schema_version")
    if not isinstance(schema_version, int) or schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported cost evidence schema_version {schema_version!r}"
        )

    raw_coverage = _required(payload, "coverage")
    if not isinstance(raw_coverage, dict):
        raise ValueError("cost evidence coverage must be an object")
    coverage = CoverageRange.from_json(raw_coverage)
    if not coverage.contains(required_range):
        raise ValueError(
            f"cost evidence coverage {coverage.start}..{coverage.end} does not "
            f"contain required range {required_range.start}..{required_range.end}"
        )

    sources: list[SourceRecord] = []
    raw_sources = _required(payload, "sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("cost evidence sources must be a non-empty list")
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise ValueError("cost evidence source must be an object")
        uri = str(_required(raw, "uri"))
        if not uri.startswith(("https://", "http://")):
            raise ValueError(f"cost evidence source uri must be absolute: {uri!r}")
        content_hash = str(_required(raw, "content_hash"))
        if not content_hash:
            raise ValueError("cost evidence source content_hash must not be empty")
        sources.append(
            SourceRecord(
                uri=uri,
                retrieved_at=_effective_datetime(
                    _required(raw, "retrieved_at"), "retrieved_at"
                ),
                content_hash=content_hash,
            )
        )

    commission: list[CommissionRule] = []
    raw_commission = _required(payload, "commission")
    if not isinstance(raw_commission, list) or not raw_commission:
        raise ValueError("cost evidence commission must be a non-empty list")
    for raw in raw_commission:
        if not isinstance(raw, dict):
            raise ValueError("cost evidence commission entry must be an object")
        commission.append(
            CommissionRule(
                effective_from=_effective_datetime(
                    _required(raw, "effective_from"), "effective_from"
                ),
                buy_rate=_non_negative(_required(raw, "buy_rate"), "buy_rate"),
                sell_rate=_non_negative(_required(raw, "sell_rate"), "sell_rate"),
            )
        )
    _assert_sorted_effective(
        [(_as_utc(rule.effective_from), ()) for rule in commission], "commission"
    )
    if _as_utc(commission[0].effective_from) > datetime.combine(
        required_range.start, datetime.min.time(), tzinfo=UTC
    ):
        raise ValueError(
            "cost evidence commission coverage starts after the required range"
        )

    sources_by_uri = {source.uri: source for source in sources}
    sell_taxes: list[SellTaxRule] = []
    raw_sell_taxes = _required(payload, "sell_taxes")
    if not isinstance(raw_sell_taxes, list) or not raw_sell_taxes:
        raise ValueError("cost evidence sell_taxes must be a non-empty list")
    for raw in raw_sell_taxes:
        if not isinstance(raw, dict):
            raise ValueError("cost evidence sell_tax entry must be an object")
        market = str(_required(raw, "market"))
        if market not in KRX_MARKETS:
            raise ValueError(f"unsupported sell_tax market {market!r}")
        securities_rate = _non_negative(
            _required(raw, "securities_transaction_tax_rate"),
            "securities_transaction_tax_rate",
        )
        rural_rate = _non_negative(
            _required(raw, "rural_special_tax_rate"), "rural_special_tax_rate"
        )
        raw_sell_tax_rate = _non_negative(_required(raw, "sell_tax_rate"), "sell_tax_rate")
        if raw_sell_tax_rate != securities_rate + rural_rate:
            raise ValueError(
                f"sell_tax_rate {raw_sell_tax_rate} must equal the sum of its "
                f"components {securities_rate + rural_rate}"
            )
        source_uri = str(_required(raw, "source_uri"))
        if source_uri not in sources_by_uri:
            raise ValueError(f"sell_tax entry references unknown source {source_uri!r}")
        source_hash = str(_required(raw, "source_hash"))
        if source_hash != sources_by_uri[source_uri].content_hash:
            raise ValueError(
                f"sell_tax entry source_hash does not match source {source_uri!r}"
            )
        sell_taxes.append(
            SellTaxRule(
                effective_from=_effective_datetime(
                    _required(raw, "effective_from"), "effective_from"
                ),
                market=market,
                securities_transaction_tax_rate=securities_rate,
                rural_special_tax_rate=rural_rate,
                source_uri=source_uri,
                source_hash=source_hash,
            )
        )
    for market in KRX_MARKETS:
        market_rules = [rule for rule in sell_taxes if rule.market == market]
        _assert_sorted_effective(
            [(_as_utc(rule.effective_from), ()) for rule in market_rules],
            f"sell_taxes[{market}]",
        )
        if not market_rules:
            raise ValueError(f"cost evidence has no sell_tax coverage for {market}")
        if _as_utc(market_rules[0].effective_from) > datetime.combine(
            required_range.start, datetime.min.time(), tzinfo=UTC
        ):
            raise ValueError(
                f"sell_tax coverage for {market} starts after the required range"
            )

    tick_rules: list[TickSizeRule] = []
    raw_tick_rules = _required(payload, "tick_size_rules")
    if not isinstance(raw_tick_rules, list) or not raw_tick_rules:
        raise ValueError("cost evidence tick_size_rules must be a non-empty list")
    for raw in raw_tick_rules:
        if not isinstance(raw, dict):
            raise ValueError("cost evidence tick rule must be an object")
        tick_rules.append(
            TickSizeRule(
                rule_id=str(_required(raw, "rule_id")),
                effective_from=_effective_datetime(
                    _required(raw, "effective_from"), "effective_from"
                ),
                lower_inclusive=_non_negative(
                    _required(raw, "lower_inclusive"), "lower_inclusive"
                ),
                upper_exclusive=_positive_upper(
                    raw.get("upper_exclusive"), "upper_exclusive"
                ),
                tick=_positive_upper(_required(raw, "tick"), "tick"),
                session=str(raw.get("session", "regular")),
            )
        )
    tick_schedule = TickSizeSchedule(tuple(tick_rules))
    first_effective = min(
        _as_utc(rule.effective_from) for rule in tick_schedule.rules
    )
    if first_effective > datetime.combine(
        required_range.start, datetime.min.time(), tzinfo=UTC
    ):
        raise ValueError("tick coverage starts after the required range")

    raw_model = _required(payload, "liquidity_model")
    if not isinstance(raw_model, dict):
        raise ValueError("cost evidence liquidity_model must be an object")
    liquidity_model = LiquidityModelSpec(
        model_id=str(_required(raw_model, "model_id")),
        impact_coefficient=_non_negative(
            _required(raw_model, "impact_coefficient"), "impact_coefficient"
        ),
        stress_multiplier=_non_negative(
            _required(raw_model, "stress_multiplier"), "stress_multiplier"
        ),
    )

    raw_settlement = payload.get("settlement_days", DEFAULT_SETTLEMENT_DAYS)
    settlement_days = (
        raw_settlement if isinstance(raw_settlement, int) else DEFAULT_SETTLEMENT_DAYS
    )
    if settlement_days < 0:
        raise ValueError("settlement_days must be non-negative")

    return CostEvidence(
        schema_version=schema_version,
        coverage=coverage,
        assumption_id=str(_required(payload, "assumption_id")),
        sources=tuple(sources),
        commission=tuple(commission),
        sell_taxes=tuple(sell_taxes),
        tick_size_rules=tuple(tick_rules),
        liquidity_model=liquidity_model,
        settlement_days=settlement_days,
        content_hash=sha256(path.read_bytes()).hexdigest(),
        path=path,
    )


def resolve_fill_cost(
    evidence: CostEvidence,
    *,
    side: str,
    market: str,
    price: float,
    notional: float,
    adtv_20d: float,
    daily_volatility: float,
    effective_time: datetime,
    stress: bool = False,
) -> tuple[FillCostBreakdown, str]:
    """Compute one fill's cost breakdown via the shared evidence path.

    Returns the resolved breakdown plus the evidence artifact hash. Positive
    ``price``, ``notional``, ``adtv_20d``, and ``daily_volatility`` are
    required; the caller is responsible for failing an order closed (unfilled)
    when a liquidity input is missing rather than substituting a static bps.
    """
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    commission = evidence.commission_for(effective_time)
    tax = evidence.sell_tax_for(market, effective_time)
    model = evidence.stress_liquidity_model if stress else evidence.base_liquidity_model
    slippage = model.slippage_bps(
        notional=notional,
        adtv_20d=adtv_20d,
        daily_volatility=daily_volatility,
        reference_price=price,
        effective_time=effective_time,
    )
    tick_rule = evidence.tick_schedule.rule_for(price, effective_time)
    breakdown = FillCostBreakdown(
        commission_rate=commission.buy_rate if side == "BUY" else commission.sell_rate,
        securities_transaction_tax_rate=tax.securities_transaction_tax_rate,
        rural_special_tax_rate=tax.rural_special_tax_rate,
        sell_tax_rate=tax.sell_tax_rate,
        slippage_bps=slippage,
        tick_rule_id=tick_rule.rule_id,
        model_id=model.model_id,
        params_hash=model.params_hash,
    )
    return breakdown, evidence.content_hash
