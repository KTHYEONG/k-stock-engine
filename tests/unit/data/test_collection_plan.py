from datetime import date
import json

from src.data.collection_plan import (
    build_historical_collection_plan,
    build_historical_collection_plan_from_bronze,
)


def test_plan_excludes_pre_listing_sessions() -> None:
    plan = build_historical_collection_plan(
        sessions=(date(2016, 1, 4), date(2016, 1, 5), date(2016, 1, 6), date(2016, 1, 7)),
        universe=({'symbol': '005930', 'is_common_stock': True, 'tradable_from': date(2016, 1, 6), 'tradable_to': None},),
        start=date(2016, 1, 4), end=date(2016, 1, 7), chunk_size=2,
    )
    assert all(min(chunk.sessions) >= date(2016, 1, 6) for chunk in plan.chunks)


def test_plan_loads_sessions_and_listing_interval_from_bronze(tmp_path) -> None:
    calendar = tmp_path / "calendar" / "hash" / "payload.json"
    master = tmp_path / "security_master" / "hash" / "payload.json"
    calendar.parent.mkdir(parents=True)
    master.parent.mkdir(parents=True)
    calendar.write_text(json.dumps({"sessions": ["2016-01-04", "2016-01-05"]}), encoding="utf-8")
    master.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "source_identifier": "005930",
                        "is_common_stock": True,
                        "tradable_from": "2016-01-05",
                        "tradable_to": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = build_historical_collection_plan_from_bronze(
        bronze_root=tmp_path,
        start=date(2016, 1, 4),
        end=date(2016, 1, 5),
        symbols=("005930",),
        artifact_root=tmp_path / "plans",
    )

    assert plan.chunks[0].sessions == (date(2016, 1, 5),)


def test_dynamic_plan_includes_only_eligible_sessions_and_warmup(tmp_path) -> None:
    from datetime import UTC, datetime
    from src.data.collection_plan import build_historical_collection_plan_from_universe_decisions
    from src.strategy.universe import UniverseDecision

    sessions = tuple(datetime(2020, 1, day, tzinfo=UTC).date() for day in range(1, 6))
    decisions = tuple(UniverseDecision(datetime(2020, 1, day, tzinfo=UTC), "005930", day in (3, 5), (), 252, 2_000_000_000.0) for day in range(1, 6))
    plan = build_historical_collection_plan_from_universe_decisions(sessions=sessions, decisions=decisions, start=sessions[0], end=sessions[-1], warmup_sessions=1, artifact_root=tmp_path)
    assert [chunk.sessions for chunk in plan.chunks] == [(sessions[1], sessions[2]), (sessions[3], sessions[4])]
