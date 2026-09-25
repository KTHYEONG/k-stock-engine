"""Accept an extracted statement only if its own accounting identities hold exactly.

Pure functions, no I/O. Identity checks are necessary but not sufficient:
promotion additionally requires the benchmark gate of the fact class
(see ``benchmark.py`` and ``run.py``).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tools.legacy_recovery.extract import ExtractedStatement

__all__ = [
    "BALANCE_SHEET_CLASS",
    "BS_REQUIRED_FACTS",
    "INCOME_STATEMENT_CLASS",
    "Verdict",
    "statement_fact_class",
    "verify_filing",
    "verify_statement",
]

BALANCE_SHEET_CLASS: Final = "balance_sheet"
INCOME_STATEMENT_CLASS: Final = "income_statement"

BS_REQUIRED_FACTS: Final = frozenset({"assets", "debt", "equity"})


@dataclass(frozen=True, slots=True)
class Verdict:
    accepted: bool
    checks: tuple[str, ...]  # names of checks that passed
    rejected_by: str | None


def statement_fact_class(statement: ExtractedStatement) -> str:
    """Return the promotion fact class of one extracted statement."""
    return BALANCE_SHEET_CLASS if statement.kind == "BS" else INCOME_STATEMENT_CLASS


def _verify_balance_sheet(statement: ExtractedStatement) -> Verdict:
    values = statement.values
    if any(fact not in values for fact in BS_REQUIRED_FACTS):
        return Verdict(accepted=False, checks=(), rejected_by="missing_fields")
    assets = values["assets"]
    debt = values["debt"]
    equity = values["equity"]
    if assets != debt + equity:
        return Verdict(accepted=False, checks=(), rejected_by="balance_identity")
    balances = [assets, debt, equity]
    cash = values.get("cash")
    if cash is not None:
        balances.append(cash)
    if any(balance < 0 for balance in balances):
        return Verdict(
            accepted=False, checks=("balance_identity",), rejected_by="non_negative"
        )
    if cash is not None and cash > assets:
        return Verdict(
            accepted=False,
            checks=("balance_identity", "non_negative"),
            rejected_by="cash_bounded",
        )
    checks = ["balance_identity", "non_negative"]
    if cash is not None:
        checks.append("cash_bounded")
    return Verdict(accepted=True, checks=tuple(checks), rejected_by=None)


def _verify_income_statement(
    statement: ExtractedStatement, siblings: Sequence[ExtractedStatement]
) -> Verdict:
    values = statement.values
    sales = values.get("sales")
    gross = values.get("gross_profit")
    operating = values.get("operating_profit")
    if (
        (sales is not None and gross is not None and gross > sales)
        or (gross is not None and operating is not None and operating > gross)
        or (sales is not None and operating is not None and operating > sales)
    ):
        return Verdict(accepted=False, checks=(), rejected_by="profit_ordering")
    for sibling in siblings:
        if sibling is statement:
            continue
        if (
            sibling.kind == "BS"
            and sibling.basis == statement.basis
            and sibling.unit_multiplier == statement.unit_multiplier
            and _verify_balance_sheet(sibling).accepted
        ):
            return Verdict(
                accepted=True,
                checks=("profit_ordering", "balance_sheet_link"),
                rejected_by=None,
            )
    return Verdict(
        accepted=False,
        checks=("profit_ordering",),
        rejected_by="balance_sheet_link",
    )


def verify_statement(
    statement: ExtractedStatement,
    siblings: Sequence[ExtractedStatement] = (),
) -> Verdict:
    """Accept a statement only if its own accounting identities hold exactly.

    Args:
        statement: The candidate statement under review.
        siblings: Other statements extracted from the same filing. A balance
            sheet needs no context; an income statement additionally requires
            an accepted balance sheet of the same basis and unit multiplier
            among its siblings, so a lone income statement is rejected.

    Returns:
        The verdict with the names of the checks that passed.
    """
    if statement.kind == "BS":
        return _verify_balance_sheet(statement)
    return _verify_income_statement(statement, siblings)


def verify_filing(
    statements: Sequence[ExtractedStatement],
) -> tuple[Verdict, ...]:
    """Verify every statement of one filing with full sibling context."""
    ordered = tuple(statements)
    return tuple(verify_statement(item, siblings=ordered) for item in ordered)
