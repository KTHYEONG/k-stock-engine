"""Single shared DART HTML table decoder and row extractor."""

from __future__ import annotations

import re
from html.parser import HTMLParser


def decode_member(raw: bytes) -> str | None:
    """Decode one archive member (BOM, UTF-8, CP949)."""
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw[3:].decode("utf-8")
        except UnicodeDecodeError:
            return None
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            return None
    head = raw[:1024].decode("ascii", errors="ignore")
    match = re.search(r"encoding\s*=\s*['\"]([^'\"]+)['\"]", head)
    if match:
        enc = match.group(1).strip().lower().replace("_", "-")
        if enc in ("utf8", "utf-8"):
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                pass
        if enc in ("cp949", "euc-kr", "ks-c-5601", "windows-949", "uhc"):
            try:
                codec = "cp949" if enc in ("cp949", "windows-949", "uhc") else "euc-kr"
                return raw.decode(codec)
            except (UnicodeDecodeError, LookupError):
                pass
    for codec in ("utf-8", "cp949"):
        try:
            return raw.decode(codec).lstrip("\ufeff")
        except UnicodeDecodeError:
            continue
    for codec in ("euc-kr",):
        try:
            return raw.decode(codec).lstrip("\ufeff")
        except UnicodeDecodeError:
            continue
    return None


class _SharedTableParser(HTMLParser):
    """Tolerant table reader for DART's non-XML forms."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self.table_index = -1
        self.indexed: dict[int, list[list[str]]] = {}

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        name = tag.upper()
        if name == "TABLE":
            self.table_index += 1
            self._current_table = []
            self.tables.append(self._current_table)
            self.indexed[self.table_index] = self._current_table
        elif name == "TR":
            self._current_row = []
        elif name in {"TD", "TH"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        name = tag.upper()
        if name in {"TD", "TH"} and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(" ".join("".join(self._current_cell).split()))
            self._current_cell = None
        elif name == "TR" and self._current_row is not None:
            if self._current_table is None:
                self._current_table = []
                self.tables.append(self._current_table)
            self._current_table.append(self._current_row)
            self.indexed.setdefault(self.table_index, self._current_table)
            self._current_row = None


def extract_tables(text: str) -> list[list[list[str]]]:
    """Extract non-empty tables from one decoded member."""
    parser = _SharedTableParser()
    parser.feed(text)
    return [table for table in parser.tables if table]
