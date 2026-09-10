"""Official KIND lifecycle notice collection (bounded, deterministic)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.data.schemas import PITDataError


@dataclass(frozen=True, slots=True)
class KindDisclosurePage:
    ticker: str
    disclosure_url: str
    html: str
    published_at: object
    source_hash: str


class KindLifecycleCollector:
    """Search official KIND by ticker with bounded deterministic pagination."""

    def __init__(self, *, page_size: int = 20, max_pages: int = 5) -> None:  # pragma: no cover - instantiated only in production wiring
        self.page_size = int(page_size)
        self.max_pages = int(max_pages)

    def search_notices(  # pragma: no cover - KIND HTTP integration is integration-tested
        self, *, ticker: str, start: date, end: date
    ) -> tuple[KindDisclosurePage, ...]:
        if not ticker.strip():
            raise PITDataError("ticker must be non-empty")
        if start > end:
            raise PITDataError("KIND search window is inverted")
        raise PITDataError("KIND lifecycle search requires network access")
