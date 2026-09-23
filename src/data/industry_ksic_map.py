"""Unanimous KSIC-code to KRX-industry mapping learned from paired evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from src.data.schemas import PITDataError


@dataclass(frozen=True, slots=True)
class KsicIndustryMapping:
    """Unanimous 6-digit KSIC to industry-name mapping with per-code support."""

    industry_by_ksic: Mapping[str, str]
    support_by_ksic: Mapping[str, int]
    conflicting_ksic: frozenset[str]

    def lookup(self, ksic_code: str) -> tuple[str, int] | None:
        """Return the mapped ``(industry_name, support)`` or ``None`` when unusable."""
        if not isinstance(ksic_code, str):
            return None
        if len(ksic_code) != 6 or not ksic_code.isascii() or not ksic_code.isdigit():
            return None
        if ksic_code in self.conflicting_ksic:
            return None
        industry = self.industry_by_ksic.get(ksic_code)
        if industry is None:
            return None
        return (industry, self.support_by_ksic[ksic_code])


def learn_ksic_industry_mapping(
    observations: Iterable[tuple[str, str, str]],
) -> KsicIndustryMapping:
    """Learn the KSIC-code to industry-name mapping from tickers observed with both.

    The KRX industry is a deterministic function of the KSIC code, so a code is
    usable only when every distinct observed ticker under it agrees;
    disagreement makes the code unusable rather than resolved by vote. Shorter
    code prefixes are deliberately not generalized: measured accuracy fell below
    the level at which an inferred value can be trusted.

    Args:
        observations: Yields ``(ticker, ksic_code, industry_name)``.

    Returns:
        The mapping with per-code support (number of distinct observed
        tickers) and the set of conflicting codes.

    Raises:
        PITDataError: For a ``ksic_code`` that is not six ASCII digits or a
            blank ``industry_name``.
    """
    per_ticker: dict[str, tuple[str, str]] = {}
    for ticker, ksic_code, industry_name in observations:
        code = ksic_code.strip() if isinstance(ksic_code, str) else ""
        if len(code) != 6 or not code.isascii() or not code.isdigit():
            raise PITDataError(f"invalid KSIC code for {ticker}: {ksic_code!r}")
        industry = industry_name.strip() if isinstance(industry_name, str) else ""
        if not industry:
            raise PITDataError(f"blank industry name for {ticker}")
        known = per_ticker.get(ticker)
        if known is None:
            per_ticker[ticker] = (code, industry)
        elif known != (code, industry):
            raise PITDataError(f"conflicting KSIC observations for {ticker}")
    tickers_by_code: dict[str, set[str]] = {}
    industries_by_code: dict[str, set[str]] = {}
    for ticker in sorted(per_ticker):
        code, industry = per_ticker[ticker]
        tickers_by_code.setdefault(code, set()).add(ticker)
        industries_by_code.setdefault(code, set()).add(industry)
    industry_by_ksic: dict[str, str] = {}
    support_by_ksic: dict[str, int] = {}
    conflicting: set[str] = set()
    for code in sorted(tickers_by_code):
        industries = industries_by_code[code]
        if len(industries) == 1:
            industry_by_ksic[code] = next(iter(industries))
            support_by_ksic[code] = len(tickers_by_code[code])
        else:
            conflicting.add(code)
    return KsicIndustryMapping(
        industry_by_ksic=industry_by_ksic,
        support_by_ksic=support_by_ksic,
        conflicting_ksic=frozenset(conflicting),
    )
