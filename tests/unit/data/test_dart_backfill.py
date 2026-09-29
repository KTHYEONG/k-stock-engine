import json
from datetime import date
from pathlib import Path

import pytest


def _scoped_runtime(tmp_path):  # type: ignore[no-untyped-def]
    from pathlib import Path as _Path

    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=_Path("config/research/kr_swing_2019_v1.toml"), data_root=tmp_path / "data")


def _scoped_catalog(runtime):  # type: ignore[no-untyped-def]
    from src.data.receipt_catalog import ReceiptCatalog

    return ReceiptCatalog(runtime.workspace.bronze_root / "catalog")


def _provider(**overrides):  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    provider = load_provider_policy(load_runtime_config())
    if overrides:
        key = provider.dart_key(provider.primary_key_env).model_copy(update=overrides)
        provider = provider.model_copy(update={"dart": provider.dart.model_copy(update={"keys": {provider.primary_key_env: key}})})
    return provider


def _quota_store(tmp_path):  # type: ignore[no-untyped-def]
    from src.integrations.quota import ProviderQuotaStateStore

    return ProviderQuotaStateStore(tmp_path / "quota")


def _capped_provider(*, daily_budget, daily_reserve, batch_identities):  # type: ignore[no-untyped-def]
    from src.config.providers import DartKeyPolicy, DartPolicy, ProviderPolicy

    return ProviderPolicy(
        dart=DartPolicy(
            circuit_threshold=3,
            requests_per_identity=3,
            batch_identities=batch_identities,
            shared_ip_avoid_windows_kst=[],
            keys={
                "TEST_DART_KEY": DartKeyPolicy(
                    daily_limit=20000,
                    daily_budget=daily_budget,
                    daily_reserve=daily_reserve,
                    min_interval_seconds=0.2,
                    max_workers=2,
                )
            },
        ),
        kis=_provider().kis,
        krx=_provider().krx,
        ls=_provider().ls,
        kind=_provider().kind,
    )


def _filing(corp="00126380", filing="F1", biz="2019", reprt="11013", published="2019-05-15"):
    return {
        "corp_code": corp, "filing_id": filing, "biz_year": biz, "reprt_code": reprt,
        "published_at": published, "ticker": "005930",
    }


def test_scoped_batch_selects_catalog_gap(tmp_path) -> None:
    from src.data.dart_backfill import build_scoped_dart_fact_batch

    runtime = _scoped_runtime(tmp_path)
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=[_filing()], offset=0, limit=20,
        provider=_provider(),
    )

    assert batch.scope_hash == runtime.scope.content_hash
    assert batch.plan_id.startswith("dart-facts-")
    assert [dict(item)["filing_id"] for item in batch.identities] == ["F1"]
    assert batch.missing_without_filing == ()
    assert batch.estimated_request_ceiling == 3


def test_scoped_batch_excludes_pre_floor_filing(tmp_path) -> None:
    from src.data.dart_backfill import build_scoped_dart_fact_batch

    runtime = _scoped_runtime(tmp_path)
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=[_filing(filing="F0", biz="2015", reprt="11011", published="2016-03-30")],
        offset=0, limit=20,
        provider=_provider(),
    )

    assert batch.identities == ()
    assert batch.missing_without_filing == ()
    assert batch.estimated_request_ceiling == 0


def test_scoped_batch_reports_missing_without_discovery(tmp_path, monkeypatch) -> None:
    from src.data import dart_disclosures as dart_disclosures_module
    from src.data.dart_backfill import build_scoped_dart_fact_batch

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("disclosure-list discovery must not run")

    monkeypatch.setattr(dart_disclosures_module, "periodic_filing_identities", _forbidden)
    runtime = _scoped_runtime(tmp_path)
    incomplete = {"corp_code": "00126380", "biz_year": "2019", "reprt_code": "11013", "ticker": "005930"}
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=[incomplete, _filing()], offset=0, limit=20,
        provider=_provider(),
    )

    assert [item.natural_key for item in batch.missing_without_filing] == ["00126380:2019:11013"]
    assert batch.missing_without_filing[0].required is True
    assert [dict(item)["filing_id"] for item in batch.identities] == ["F1"]


def test_scoped_batch_quota_ceiling_limits_identities(tmp_path) -> None:
    from src.data.dart_backfill import build_scoped_dart_fact_batch

    runtime = _scoped_runtime(tmp_path)
    capped = _capped_provider(daily_budget=1200, daily_reserve=400, batch_identities=500)
    identities = [
        _filing(corp=f"{index:08d}", filing=f"F{index}", published="2019-05-15") for index in range(500)
    ]
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=identities, offset=0, limit=500,
        provider=capped,
    )

    assert len(batch.identities) == 266
    assert batch.estimated_request_ceiling == 798

    tiny = _capped_provider(daily_budget=16000, daily_reserve=400, batch_identities=2)
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=identities[:10], offset=0, limit=10,
        provider=tiny,
    )
    assert len(batch.identities) == 2


def test_scoped_batch_reserves_provider_wide_quota_before_selecting(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.dart_backfill import build_scoped_dart_fact_batch
    from src.integrations.quota import ProviderQuotaStateStore

    runtime = _scoped_runtime(tmp_path)
    capped = _capped_provider(daily_budget=100, daily_reserve=10, batch_identities=50)
    store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    moment = datetime(2026, 9, 21, 14, tzinfo=UTC)
    for index in range(82):
        store.record_attempt(provider="OpenDART", endpoint="list.json" if index == 0 else "fnlttSinglAcntAll.json", now=moment)

    batch = build_scoped_dart_fact_batch(
        runtime=runtime,
        catalog=_scoped_catalog(runtime),
        filing_identities=[_filing(corp=f"{index:08d}", filing=f"F{index}") for index in range(10)],
        offset=0,
        limit=10,
        provider=capped,
        quota_store=store,
        now=moment,
    )

    assert batch.available_request_headroom == 8
    assert len(batch.identities) == 2


def test_scoped_collector_uses_provider_policy(tmp_path, monkeypatch) -> None:
    import src.data.dart_backfill as backfill
    from src.data.dart_backfill import build_scoped_dart_collector

    captured: dict[str, object] = {}

    class _Collector:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(backfill, "DartXbrlCollector", _Collector)
    monkeypatch.setenv("OPENDART_API_KEY", "primary-key")
    provider = _provider()
    store = _quota_store(tmp_path)
    build_scoped_dart_collector(provider=provider, quota_store=store)

    expected = provider.dart_key(provider.primary_key_env)
    assert captured["daily_request_limit"] == expected.daily_budget
    assert captured["min_interval"] == expected.min_interval_seconds
    assert captured["max_workers"] == expected.max_workers
    assert captured["api_key"] == "primary-key"


def test_scoped_batch_catalog_success_suppresses_retry(tmp_path) -> None:
    import hashlib
    from datetime import UTC, datetime

    from src.data.dart_backfill import build_scoped_dart_fact_batch
    from src.data.receipt_catalog import EvidenceStatus, ReceiptIndexEntry
    from tests.fixtures import seed_receipts

    runtime = _scoped_runtime(tmp_path)
    catalog = _scoped_catalog(runtime)
    body = b'{"records": [{"fact": 1}]}'
    payload_path = runtime.workspace.bronze_root / "fact.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(body)
    seed_receipts(
        catalog,
        (
            ReceiptIndexEntry(
                source="financial_facts", natural_key="00126380:2019:11013",
                as_of=date(2019, 5, 16), fiscal_period="2019Q1", status=EvidenceStatus.SUCCESS,
                content_hash=hashlib.sha256(body).hexdigest(),
                retrieved_at=datetime(2019, 5, 17, tzinfo=UTC), payload_path=payload_path,
            ),
        ),
    )
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=catalog, filing_identities=[_filing()], offset=0, limit=20,
        provider=_provider(),
    )

    assert batch.identities == ()


def test_scoped_batch_dedupes_to_latest_correction(tmp_path) -> None:
    from src.data.dart_backfill import build_scoped_dart_fact_batch

    runtime = _scoped_runtime(tmp_path)
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=[
            _filing(filing="F2", published="2019-06-01"),
            _filing(filing="F1", published="2019-05-15"),
        ],
        offset=0, limit=20,
        provider=_provider(),
    )

    assert [dict(item)["filing_id"] for item in batch.identities] == ["F2"]
    same_inputs = [
        _filing(filing="F2", published="2019-06-01"),
        _filing(filing="F1", published="2019-05-15"),
    ]
    first = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=same_inputs, offset=0, limit=20,
        provider=_provider(),
    )
    second = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=same_inputs, offset=0, limit=20,
        provider=_provider(),
    )
    assert first.plan_id == second.plan_id


def test_scoped_batch_rejects_invalid_inputs(tmp_path) -> None:
    from src.data.dart_backfill import build_scoped_dart_fact_batch
    from src.core.pit import PITDataError

    runtime = _scoped_runtime(tmp_path)
    catalog = _scoped_catalog(runtime)
    with pytest.raises(PITDataError, match="offset"):
        build_scoped_dart_fact_batch(runtime=runtime, catalog=catalog, filing_identities=[], offset=-1, limit=1, provider=_provider())
    with pytest.raises(PITDataError, match="offset"):
        build_scoped_dart_fact_batch(runtime=runtime, catalog=catalog, filing_identities=[], offset=0, limit=0, provider=_provider())
    with pytest.raises(PITDataError, match="corp code"):
        build_scoped_dart_fact_batch(
            runtime=runtime, catalog=catalog, filing_identities=[{"biz_year": "2019"}], offset=0, limit=1,
            provider=_provider(),
        )
    with pytest.raises(PITDataError, match="fiscal period"):
        build_scoped_dart_fact_batch(
            runtime=runtime, catalog=catalog,
            filing_identities=[{"corp_code": "00126380", "biz_year": "2019", "reprt_code": "11999"}],
            offset=0, limit=1,
            provider=_provider(),
        )
    with pytest.raises(PITDataError, match="fiscal period"):
        build_scoped_dart_fact_batch(
            runtime=runtime, catalog=catalog,
            filing_identities=[{"corp_code": "c", "biz_year": "b", "reprt_code": "r", "fiscal_period": "bogus"}],
            offset=0, limit=1,
            provider=_provider(),
        )
    with pytest.raises(ValueError, match="not-a-date"):
        build_scoped_dart_fact_batch(
            runtime=runtime, catalog=catalog,
            filing_identities=[{"corp_code": "00126380", "biz_year": "2019", "reprt_code": "11013", "as_of": "not-a-date"}],
            offset=0, limit=1,
            provider=_provider(),
        )


def test_collect_dart_facts_command_dry_run(tmp_path, capsys, monkeypatch) -> None:
    import polars as pl

    from src.data.cli import main
    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.runtime import load_data_runtime

    monkeypatch.setattr("src.data.jobs.dart._check_disclosure_coverage", lambda *a, **k: None)

    runtime = load_data_runtime(
        scope_config=Path("config/research/kr_swing_2019_v1.toml"), data_root=tmp_path / "data"
    )
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime

    from src.core.pit import EvidenceKind as _Kind
    from src.data.bronze import BronzeStore as _Store
    from src.data.receipt_catalog import BlobEntry as _Blob, EvidenceStatus as _Status, ReceiptCatalog as _Catalog, ReceiptIndexEntry as _Entry

    bridge_raw = json.dumps(
        [{"ticker": "005930", "corp_code": "00126380", "corp_name": "Test Co"}],
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    _moment = _datetime(2026, 9, 24, 3, 0, tzinfo=_UTC)
    _store = _Store(runtime.workspace.bronze_root)
    _receipt = _store.import_bytes(bridge_raw, kind=_Kind.SECURITY_MASTER, retrieved_at=_moment, source_label="test:bridge")
    _catalog = _Catalog(runtime.workspace.bronze_root / "catalog")
    _catalog.publish(
        [_Entry(source="dart_corp_codes", natural_key="dart_corp_codes", as_of=_receipt.retrieved_at.date(), fiscal_period=None, status=_Status.SUCCESS, content_hash=_receipt.content_hash, retrieved_at=_receipt.retrieved_at, payload_path=_receipt.payload_path)],
        blobs=[_Blob(content_hash=_receipt.content_hash, kind=_Kind.SECURITY_MASTER, source="dart_corp_codes", usable=True, unusable_reason=None, retrieved_at=_receipt.retrieved_at, payload_path=_receipt.payload_path)],
    )
    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(
            kind="ordinary_universe", layer=DatasetLayer.SILVER, policy_version="test-v1", inputs={}, params={}
        ),
        partitions={"part.parquet": pl.DataFrame({"ticker": ["005930"], "eligible": [True]})},
    )
    DatasetRegistry(runtime.workspace.state_root).register("ordinary_universe", published.dataset_id)
    page = {
        "records": [
            {
                "corp_code": "00126380", "rcept_no": "20160516001235", "rcept_dt": "20160516",
                "report_nm": "분기보고서 (2016.03)", "rm": "",
            }
        ],
        "start": "2015-01-01",
        "end": "2019-06-30",
        "corp_code": "00126380",
    }
    page_raw = json.dumps(page, sort_keys=True, ensure_ascii=False).encode("utf-8")
    from datetime import date as _date

    _page_receipt = _store.import_bytes(page_raw, kind=_Kind.DISCLOSURES, retrieved_at=_moment, source_label="test:disclosure")
    _catalog.publish(
        [_Entry(source="dart_corp_disclosures", natural_key="00126380:2015-01-01..2019-06-30", as_of=_date(2019, 6, 30), fiscal_period=None, status=_Status.SUCCESS, content_hash=_page_receipt.content_hash, retrieved_at=_page_receipt.retrieved_at, payload_path=_page_receipt.payload_path)],
        blobs=[_Blob(content_hash=_page_receipt.content_hash, kind=_Kind.DISCLOSURES, source="dart_corp_disclosures", usable=True, unusable_reason=None, retrieved_at=_page_receipt.retrieved_at, payload_path=_page_receipt.payload_path)],
    )
    base = [
        "collect-dart-facts",
        "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(tmp_path / "data"),
        "--dry-run",
    ]

    assert main(base) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["job"] == "dart_facts"
    assert out["status"] == "dry_run"
    assert out["pending_left"] == 1


def test_scoped_identity_accepts_explicit_valid_fiscal_period() -> None:
    import src.data.dart_backfill as backfill

    assert backfill._identity_fiscal_period({"fiscal_period": "2019Q2"}) == "2019Q2"


def test_headroom_meters_each_key_against_its_own_policy(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    from src.data.dart_backfill import scoped_dart_request_headroom

    monkeypatch.setenv("OPENDART_API_KEY", "primary-key")
    monkeypatch.setenv("OPENDART_API_KEY_2", "second-key")
    provider = _provider()
    store = _quota_store(tmp_path)
    moment = datetime(2026, 9, 24, 3, tzinfo=UTC)
    primary_policy = provider.dart_key(provider.primary_key_env)
    for _ in range(5):
        store.record_attempt(provider="OpenDART", endpoint="x", now=moment, daily_limit=primary_policy.daily_budget)

    primary = scoped_dart_request_headroom(provider=provider, quota_store=store, now=moment)
    secondary = scoped_dart_request_headroom(
        provider=provider, quota_store=store, now=moment, key_env="OPENDART_API_KEY_2"
    )
    policy = provider.dart_key("OPENDART_API_KEY_2")

    assert primary == primary_policy.daily_budget - 5 - primary_policy.daily_reserve
    assert secondary == policy.daily_budget - policy.daily_reserve


def test_scoped_collector_uses_the_declared_policy_of_the_selected_key(tmp_path, monkeypatch) -> None:
    import src.data.dart_backfill as backfill
    from src.data.dart_backfill import build_scoped_dart_collector

    captured: dict[str, object] = {}

    class _Collector:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(backfill, "DartXbrlCollector", _Collector)
    monkeypatch.setenv("OPENDART_API_KEY_2", "second-key")
    provider = _provider()
    policy = provider.dart_key("OPENDART_API_KEY_2")
    build_scoped_dart_collector(provider=provider, quota_store=_quota_store(tmp_path), key_env="OPENDART_API_KEY_2")

    assert captured["api_key"] == "second-key"
    assert captured["min_interval"] == policy.min_interval_seconds
    assert captured["max_workers"] == policy.max_workers
    assert captured["daily_request_limit"] == policy.daily_budget


def test_undeclared_or_unset_key_is_refused(tmp_path, monkeypatch) -> None:
    import pytest

    from src.config.errors import ConfigError
    from src.data.dart_backfill import build_scoped_dart_collector

    provider = _provider()
    store = _quota_store(tmp_path)
    monkeypatch.delenv("OPENDART_API_KEY_2", raising=False)
    with pytest.raises(ValueError, match="is not set"):
        build_scoped_dart_collector(provider=provider, quota_store=store, key_env="OPENDART_API_KEY_2")
    with pytest.raises(ConfigError, match="no declared policy"):
        build_scoped_dart_collector(provider=provider, quota_store=store, key_env="OPENDART_API_KEY_9")


