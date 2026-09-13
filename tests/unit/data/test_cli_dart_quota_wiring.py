def test_collect_dart_disclosures_wires_quota_store_and_surfaces_circuit_breaker(tmp_path, monkeypatch, capsys) -> None:
    import json

    import src.data.cli as cli
    from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

    captured: dict[str, object] = {}

    class _FakeCollector:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

    monkeypatch.setattr("src.integrations.dart.xbrl.DartXbrlCollector", _FakeCollector)

    def fake_collect(*, dart, start, end, bronze_root, retrieved_at):
        raise ProviderQuotaBlocked("OpenDART list.json blocked until 2026-09-14T00:00:00+00:00")

    monkeypatch.setattr(cli, "collect_dart_disclosures", fake_collect)

    code = cli.main([
        "collect-dart-disclosures",
        "--bronze-root", str(tmp_path / "bronze"),
        "--artifact-root", str(tmp_path / "artifacts"),
        "--coverage-start", "2016-01-01",
        "--coverage-end", "2016-01-31",
        "--retrieved-at", "2026-09-13T00:00:00+00:00",
    ])

    payload = json.loads(capsys.readouterr().out.strip())
    assert code == 1
    assert "blocked" in payload["error"]
    assert isinstance(captured["kwargs"]["quota_store"], ProviderQuotaStateStore)


def test_collect_dart_facts_wires_quota_store_and_surfaces_circuit_breaker(tmp_path, monkeypatch, capsys) -> None:
    import json

    import src.data.cli as cli
    from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

    captured: dict[str, object] = {}

    class _FakeCollector:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

    monkeypatch.setattr("src.integrations.dart.xbrl.DartXbrlCollector", _FakeCollector)

    def fake_collect(*, dart, identities, bronze_root, retrieved_at):
        raise ProviderQuotaBlocked("OpenDART fnlttSinglAcntAll blocked until 2026-09-14T00:00:00+00:00")

    monkeypatch.setattr(cli, "collect_dart_financial_facts", fake_collect)

    code = cli.main([
        "collect-dart-facts",
        "--bronze-root", str(tmp_path / "bronze"),
        "--artifact-root", str(tmp_path / "artifacts"),
        "--coverage-start", "2016-01-01",
        "--coverage-end", "2016-01-31",
        "--corp-code", "00126380",
        "--filing-id", "20160101000001",
        "--biz-year", "2015",
        "--report-code", "11011",
        "--retrieved-at", "2026-09-13T00:00:00+00:00",
    ])

    payload = json.loads(capsys.readouterr().out.strip())
    assert code == 1
    assert "blocked" in payload["error"]
    assert isinstance(captured["kwargs"]["quota_store"], ProviderQuotaStateStore)


def test_backfill_dart_facts_wires_quota_store_and_surfaces_circuit_breaker(tmp_path, monkeypatch, capsys) -> None:
    import json

    import src.data.cli as cli
    import src.data.dart_backfill as dart_backfill_mod
    from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

    captured: dict[str, object] = {}

    class _FakeCollector:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

    monkeypatch.setattr("src.integrations.dart.xbrl.DartXbrlCollector", _FakeCollector)

    def fake_run(*, request, dart):
        raise ProviderQuotaBlocked("OpenDART backfill-dart-facts blocked until 2026-09-14T00:00:00+00:00")

    monkeypatch.setattr(dart_backfill_mod, "run_dart_historical_backfill_batch", fake_run)

    code = cli.main([
        "backfill-dart-facts",
        "--bronze-root", str(tmp_path / "bronze"),
        "--silver-root", str(tmp_path / "silver"),
        "--artifact-root", str(tmp_path / "artifacts"),
        "--validation-start", "2016-07-01",
        "--validation-end", "2016-07-01",
        "--retrieved-at", "2026-09-13T00:00:00+00:00",
    ])

    payload = json.loads(capsys.readouterr().out.strip())
    assert code == 1
    assert "blocked" in payload["error"]
    assert isinstance(captured["kwargs"]["quota_store"], ProviderQuotaStateStore)
