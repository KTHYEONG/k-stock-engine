from datetime import date

from src.data.collection_plan import build_historical_collection_plan


def test_plan_excludes_pre_listing_sessions() -> None:
    plan = build_historical_collection_plan(
        sessions=(date(2016, 1, 4), date(2016, 1, 5), date(2016, 1, 6), date(2016, 1, 7)),
        universe=({'symbol': '005930', 'is_common_stock': True, 'tradable_from': date(2016, 1, 6), 'tradable_to': None},),
        start=date(2016, 1, 4), end=date(2016, 1, 7), chunk_size=2,
    )
    assert all(min(chunk.sessions) >= date(2016, 1, 6) for chunk in plan.chunks)
