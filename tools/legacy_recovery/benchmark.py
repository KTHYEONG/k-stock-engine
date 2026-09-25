"""Score verified extractions against independent standard comparatives.

Truth comes only from ``opendart_standard`` pages of a later filing of the
same company and fiscal period; legacy pages never contribute truth.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tools.legacy_recovery.extract import ExtractedStatement
from tools.legacy_recovery.verify import (
    BALANCE_SHEET_CLASS,
    INCOME_STATEMENT_CLASS,
    statement_fact_class,
    verify_statement,
)

__all__ = [
    "BS_CLASS_FACTS",
    "FACT_CLASSES",
    "FactClassResult",
    "FilingScore",
    "PromotionGate",
    "build_labeled_set",
    "evaluate",
    "filing_key",
    "parse_filing_key",
    "score_filings",
]

FACT_CLASSES: Final = (BALANCE_SHEET_CLASS, INCOME_STATEMENT_CLASS)

BS_CLASS_FACTS: Final = frozenset({"assets", "debt", "equity", "cash"})
IS_CLASS_FACTS: Final = frozenset({"sales", "gross_profit", "operating_profit", "net_income"})

_CLASS_FACTS: Final = {
    BALANCE_SHEET_CLASS: BS_CLASS_FACTS,
    INCOME_STATEMENT_CLASS: IS_CLASS_FACTS,
}

_BASES: Final = ("consolidated", "separate")


@dataclass(frozen=True, slots=True)
class PromotionGate:
    min_exact_match_rate: float  # read from the tool config file, never a literal
    min_accepted_filings: int


@dataclass(frozen=True, slots=True)
class FactClassResult:
    fact_class: str  # "balance_sheet" | "income_statement"
    accepted_filings: int
    exact_matches: int
    promotable: bool


@dataclass(frozen=True, slots=True)
class FilingScore:
    """Per-filing, per-class scoring detail backing the report and promotion."""

    key: str
    fact_class: str
    verified_facts: tuple[str, ...]
    scored: bool
    exact: bool


def filing_key(corp_code: str, fiscal_period: str, basis: str) -> str:
    """Compose the canonical key joining an extraction to labeled truth."""
    return f"{corp_code.strip()}:{fiscal_period.strip()}:{basis.strip().lower()}"


def parse_filing_key(key: str) -> tuple[str, str, str] | None:
    """Split an extraction key into (corp_code, fiscal_period, basis)."""
    for separator in (":", "|"):
        if separator in key:
            parts = [part.strip() for part in key.split(separator)]
            break
    else:
        return None
    if len(parts) != 3 or not parts[0] or not parts[1]:
        return None
    basis = parts[2].lower()
    mapping = {
        "consolidated": "consolidated",
        "cfs": "consolidated",
        "연결": "consolidated",
        "separate": "separate",
        "ofs": "separate",
        "별도": "separate",
        "개별": "separate",
    }
    normalized = mapping.get(basis)
    if normalized is None:
        return None
    return (parts[0], parts[1], normalized)


def _int_value(raw: object) -> int | None:
    try:
        number = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    rounded = round(number)
    if abs(number - rounded) > 1e-6:
        return None
    return int(rounded)


def _quarantine_entries(quarantine_file: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(Path(quarantine_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _iter_fact_pages(bronze_root: Path) -> Sequence[tuple[str, dict[str, object]]]:
    pages: list[tuple[str, dict[str, object]]] = []
    try:
        receipt_paths = sorted((Path(bronze_root) / "financial_facts").rglob("receipt.json"))
    except OSError:
        return pages
    for receipt_path in receipt_paths:
        try:
            payload = json.loads((receipt_path.parent / "payload.json").read_bytes())
        except (OSError, ValueError):
            continue
        if isinstance(payload, list):
            candidates: list[object] = [{"records": payload}]
        elif isinstance(payload, dict):
            candidates = [payload]
        else:
            continue
        pages.extend((receipt_path.parent.name, c) for c in candidates if isinstance(c, dict))
    return pages


def build_labeled_set(
    bronze_root: Path, quarantine_file: Path
) -> Mapping[tuple[str, str, str], Mapping[str, int]]:
    """Return {(corp_code, fiscal_period, basis): {fact: truth_krw}}.

    Truth is drawn only from ``opendart_standard`` records of a filing that is
    not itself quarantined for the same (company, period): the same period
    reported as a comparative column in a later standard filing. A restated
    comparative is still the best available independent value. Filings with no
    such truth are absent from the result.
    """
    wanted: dict[tuple[str, str], set[str]] = {}
    for entry in _quarantine_entries(quarantine_file):
        corp = str(entry.get("dart_corp_code") or entry.get("company_id") or "").strip()
        period = str(entry.get("fiscal_period") or "").strip()
        filing = str(entry.get("filing_id") or "").strip()
        if not corp or not period:
            continue
        wanted.setdefault((corp, period), set()).add(filing)
    if not wanted:
        return {}
    groups: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for _, page in _iter_fact_pages(bronze_root):
        page_kind = str(page.get("source_kind") or "opendart_standard")
        records = page.get("records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            kind = str(record.get("source_kind") or page_kind)
            if kind != "opendart_standard":
                continue
            corp = str(
                record.get("corp_code")
                or record.get("dart_corp_code")
                or record.get("company_id")
                or ""
            ).strip()
            period = str(record.get("fiscal_period") or "").strip()
            filing = str(record.get("filing_id") or record.get("rcept_no") or "").strip()
            fact = str(record.get("fact") or record.get("account") or "").strip()
            if not corp or not period or not filing or not fact:
                continue
            if (corp, period) not in wanted:
                continue
            consolidated = record.get("consolidated")
            basis = "separate" if consolidated is False else "consolidated"
            value = _int_value(record.get("value"))
            if value is None:
                continue
            group = groups.setdefault(
                (corp, period, basis, filing),
                {"facts": {}, "published": ""},
            )
            facts = group["facts"]
            assert isinstance(facts, dict)
            facts.setdefault(fact, value)
            published = str(record.get("published_at") or "")
            current = str(group["published"] or "")
            if published and published > current:
                group["published"] = published
    labeled: dict[tuple[str, str, str], dict[str, int]] = {}
    for (corp, period), excluded in wanted.items():
        for basis in _BASES:
            candidates = [
                (meta["published"], filing, meta["facts"])
                for (g_corp, g_period, g_basis, filing), meta in groups.items()
                if (g_corp, g_period, g_basis) == (corp, period, basis)
                and filing not in excluded
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda item: (str(item[0]), str(item[1])))
            truth = dict(candidates[-1][2])
            assert isinstance(truth, dict)
            if truth:
                labeled[(corp, period, basis)] = truth
    return labeled


def score_filings(
    extracted: Mapping[str, Sequence[ExtractedStatement]],
    labeled: Mapping[tuple[str, str, str], Mapping[str, int]],
) -> tuple[FilingScore, ...]:
    """Score verified extractions per filing and fact class.

    Only extractions that pass verification are scored. A filing with no
    labeled truth (or no overlapping fact) is reported as unscored. A scored
    filing is an exact match only when every overlapping verified fact equals
    the independent truth as integers, with no tolerance.
    """
    scores: list[FilingScore] = []
    for key in sorted(extracted):
        statements = tuple(extracted[key])
        if not statements:
            continue
        parsed = parse_filing_key(key)
        if parsed is None:
            continue
        corp, period, _ = parsed
        verified = [
            item
            for item in statements
            if verify_statement(item, siblings=statements).accepted
        ]
        if not verified:
            continue
        by_class: dict[str, dict[str, int]] = {}
        for item in sorted(verified, key=lambda s: (s.kind, s.basis)):
            facts = by_class.setdefault(statement_fact_class(item), {})
            for fact, value in sorted(item.values.items()):
                facts.setdefault(fact, value)
        for fact_class in FACT_CLASSES:
            facts = by_class.get(fact_class)
            if not facts:
                continue
            allowed = _CLASS_FACTS[fact_class]
            merged = {fact: value for fact, value in facts.items() if fact in allowed}
            if not merged:
                continue
            matched: dict[str, int] = {}
            truth: Mapping[str, int] = {}
            for basis in _BASES:
                candidate = labeled.get((corp, period, basis))
                if not candidate:
                    continue
                overlap = {f: v for f, v in merged.items() if f in candidate}
                if overlap:
                    matched = overlap
                    truth = candidate
                    break
            if not matched:
                scores.append(
                    FilingScore(
                        key=key,
                        fact_class=fact_class,
                        verified_facts=tuple(sorted(merged)),
                        scored=False,
                        exact=False,
                    )
                )
                continue
            exact = all(merged[fact] == truth[fact] for fact in matched)
            scores.append(
                FilingScore(
                    key=key,
                    fact_class=fact_class,
                    verified_facts=tuple(sorted(merged)),
                    scored=True,
                    exact=exact,
                )
            )
    return tuple(scores)


def evaluate(
    extracted: Mapping[str, Sequence[ExtractedStatement]],
    labeled: Mapping[tuple[str, str, str], Mapping[str, int]],
    gate: PromotionGate,
) -> tuple[FactClassResult, ...]:
    """Score verified extractions against independent comparatives, per fact class."""
    scores = score_filings(extracted, labeled)
    results: list[FactClassResult] = []
    for fact_class in FACT_CLASSES:
        scoped = [s for s in scores if s.fact_class == fact_class and s.scored]
        accepted = len(scoped)
        exact = sum(1 for s in scoped if s.exact)
        promotable = accepted >= gate.min_accepted_filings and (
            accepted > 0 and exact / accepted >= gate.min_exact_match_rate
        )
        results.append(
            FactClassResult(
                fact_class=fact_class,
                accepted_filings=accepted,
                exact_matches=exact,
                promotable=promotable,
            )
        )
    return tuple(results)
