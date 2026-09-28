"""KIND search markup builders that mirror the live page structure.

The live result row is ``번호 · 시간 · 회사명 · 공시제목 · 제출인 · 차트`` with single-quoted
``title`` attributes, and the footer wraps the total in ``<em>``. Captured live pages live in
``tests/fixtures/kind/`` and pin the same structure.
"""

from __future__ import annotations

from pathlib import Path

KIND_FIXTURE_DIR = Path(__file__).resolve().parent / "kind"


def kind_fixture(name: str) -> str:
    """Return one captured live KIND page."""
    return (KIND_FIXTURE_DIR / name).read_text(encoding="utf-8")


def kind_row_html(acptno: str, disclosed: str, title: str, company: str | None, submitter: str) -> str:
    """One result row in live column order; ``company=None`` renders a market-wide notice."""
    company_cell = (
        "<img src='/images/common/icn_t_yu.gif' class='vmiddle legend' alt='유가증권'> "
        f"<a id=\"companysum\" href=\"#companysum\" onclick=\"companysummary_open('{company}'); return false;\""
        f" title='Co'> Co</a>"
        if company is not None
        else ""
    )
    return (
        "<tr>"
        "<td class=\"first txc\">1</td>"
        f"<td class=\"txc\">{disclosed}</td>"
        f"<td>{company_cell}</td>"
        f"<td><a href=\"#viewer\" onclick=\"openDisclsViewer('{acptno}','')\" title='{title}'>{title}</a></td>"
        f"<td>{submitter}</td>"
        "<td class=\"txc\"> </td>"
        "</tr>"
    )


def kind_search_html(rows: list[str], total: int) -> str:
    """A result page with the live footer shape around ``rows``."""
    return (
        "<html><body><table><tbody>" + "".join(rows) + "</tbody></table>"
        f"<div class=\"info type-00\">\r\n전체 <em>{total}</em>건 : <strong>1</strong>/1&nbsp;</div>"
        "</body></html>"
    )
