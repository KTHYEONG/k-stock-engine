import sys
from pathlib import Path
from typing import ClassVar

from src.data.cli import _parse_args
from src.data.collection import CollectionArtifact, InvestorFlowBatchProgress


def test_collect_command_requires_immutable_plan_id(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["stock-data", "collect", "--plan-id", "plan-a"])

    args = _parse_args()

    assert args.command == "collect"
    assert args.plan_id == "plan-a"


def _write_ls_backfill_plan(path: Path, *, symbols: tuple[str, ...] = ("005930", "000660"), plan_id: str = "plan-ls") -> dict[str, object]:
    import json

    sessions = ["2026-03-06"]
    payload: dict[str, object] = {
        "plan_id": plan_id,
        "content_hash": "e" * 64,
        "coverage_start": sessions[0],
        "coverage_end": sessions[-1],
        "chunk_size": 1,
        "chunks": [
            {"chunk_id": f"{plan_id}:{index:04d}", "symbol": symbol, "sessions": sessions}
            for index, symbol in enumerate(symbols)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _ls_backfill_artifact(
    tmp_path: Path,
    name: str,
    *,
    plan_id: str = "plan-ls",
    provider_errors: int = 0,
    completed: int = 1,
    planned: int = 1,
) -> CollectionArtifact:
    import json
    from datetime import UTC, date, datetime

    report = tmp_path / f"{name}-report.json"
    report.write_text(
        json.dumps({"plan_id": plan_id, "plan_digest": "e" * 64, "content_hash": "c" * 64}),
        encoding="utf-8",
    )
    return CollectionArtifact(
        bronze_root=tmp_path / "bronze",
        coverage_start=date(2026, 3, 6),
        coverage_end=date(2026, 3, 6),
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        receipts={},
        content_hash="c" * 64,
        report_path=report,
        planned_chunks=planned,
        completed_chunks=completed,
        previously_completed_chunks=0,
        pending_chunks=provider_errors,
        provider_error_chunks=provider_errors,
        missing_session_chunks=0,
        receipt_count=planned,
    )


def _ls_backfill_progress(batch_index: int, chunk_offset: int, artifact: CollectionArtifact) -> InvestorFlowBatchProgress:
    return InvestorFlowBatchProgress(batch_index=batch_index, chunk_offset=chunk_offset, artifact=artifact)


def test_backfill_ls_reuses_single_collector(tmp_path, monkeypatch, capsys) -> None:
    import src.data.cli as cli

    plan_file = tmp_path / "scoped" / "plan.json"
    plan_file.parent.mkdir(parents=True)
    _write_ls_backfill_plan(plan_file, symbols=("005930", "000660", "000660"))
    factory_calls: list[tuple[str, ...]] = []

    class StubCollector:
        pass

    def factory(symbols) -> StubCollector:
        factory_calls.append(tuple(symbols))
        return StubCollector()

    captured: dict[str, object] = {}
    artifacts = [_ls_backfill_artifact(tmp_path, f"ok-{i}") for i in range(3)]

    def fake_iter(**kwargs) -> object:
        captured.update(kwargs)
        return [
            _ls_backfill_progress(0, 0, artifacts[0]),
            _ls_backfill_progress(1, 1, artifacts[1]),
            _ls_backfill_progress(2, 2, artifacts[2]),
        ]

    monkeypatch.setattr(cli, "LsInvestorFlowCollector", factory)
    monkeypatch.setattr(cli, "iter_planned_investor_flow_backfill", fake_iter)
    args = [
        "backfill-ls-investor-flow",
        "--plan-path", str(plan_file),
        "--bronze-root", str(tmp_path / "bronze"),
        "--checkpoint-root", str(tmp_path / "ckpt"),
        "--chunk-batch-size", "1",
    ]
    assert cli.main(args) == 0
    assert factory_calls == [("000660", "005930")]
    assert isinstance(captured["collector"], StubCollector)
    assert captured["provider"] == "ls"
    assert captured["chunk_batch_size"] == 1
    assert len(capsys.readouterr().out.strip().splitlines()) == 3
    from collections.abc import Callable
    from datetime import datetime
    from typing import cast

    utc_factory = cast("Callable[[], datetime]", captured["retrieved_at_factory"])
    assert utc_factory().tzinfo is not None


def test_backfill_ls_plan_path_is_authoritative(tmp_path, monkeypatch, capsys) -> None:
    import src.data.cli as cli

    plan_file = tmp_path / "research" / "state" / "plan.json"
    plan_file.parent.mkdir(parents=True)
    _write_ls_backfill_plan(plan_file, symbols=("005930",))

    def forbidden_plan_id(*_args, **_kwargs) -> object:
        raise AssertionError("plan-ID lookup must not be used")

    monkeypatch.setattr(cli, "load_collection_plan", forbidden_plan_id)
    monkeypatch.setattr(cli, "LsInvestorFlowCollector", lambda symbols: object())
    monkeypatch.setattr(
        cli,
        "iter_planned_investor_flow_backfill",
        lambda **_kwargs: iter([_ls_backfill_progress(0, 0, _ls_backfill_artifact(tmp_path, "auth"))]),
    )
    args = [
        "backfill-ls-investor-flow",
        "--plan-path", str(plan_file),
        "--bronze-root", str(tmp_path / "bronze"),
        "--checkpoint-root", str(tmp_path / "ckpt"),
    ]
    assert cli.main(args) == 0
    assert '"plan_id": "plan-ls"' in capsys.readouterr().out


def test_flow_backfill_progress_payload_exposes_accounting_only(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.cli import _flow_backfill_progress_payload
    from src.data.collection import CollectionCheckpointStore, InvestorFlowBatchProgress, iter_planned_investor_flow_backfill

    plan_file = tmp_path / "plan.json"
    _write_ls_backfill_plan(plan_file, symbols=("005930",))
    from src.data.collection_plan import load_collection_plan_path

    plan = load_collection_plan_path(plan_file)

    class StubCollector:
        def fetch_investor_flow(self, start, end, **kwargs):
            return (
                {
                    "records": [
                        {
                            "ticker": "005930",
                            "session": "20260306",
                            "unit": "shares",
                            "individual_net_shares": "-3",
                            "foreign_net_shares": "1",
                            "institution_net_shares": "2",
                            "other_net_shares": "0",
                        }
                    ]
                },
            )

        def close(self) -> None:
            return None

    progress = next(
        iter(
            iter_planned_investor_flow_backfill(
                plan=plan,
                provider="ls",
                collector=StubCollector(),
                bronze_root=tmp_path / "bronze",
                retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
                checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
                chunk_batch_size=10,
            )
        )
    )
    assert isinstance(progress, InvestorFlowBatchProgress)
    payload = _flow_backfill_progress_payload(progress)
    assert payload["plan_id"] == "plan-ls"
    assert payload["batch_index"] == 0
    assert payload["chunk_offset"] == 0
    assert payload["planned_chunks"] == 1
    assert payload["completed_chunks"] == 1
    assert payload["previously_completed_chunks"] == 0
    assert payload["pending_chunks"] == 0
    assert payload["provider_error_chunks"] == 0
    assert payload["missing_session_chunks"] == 0
    assert payload["receipt_count"] >= 1
    assert str(payload["report_path"]).endswith(".json")
    blob = json.dumps(payload).lower()
    for secret in ("token", "secret", "credential", "authorization", "response_body", "collector"):
        assert secret not in blob


def test_backfill_ls_default_consumes_all_batches(tmp_path, monkeypatch, capsys) -> None:
    import src.data.cli as cli

    plan_file = tmp_path / "plan.json"
    _write_ls_backfill_plan(plan_file, symbols=("005930", "000660"))
    monkeypatch.setattr(cli, "LsInvestorFlowCollector", lambda symbols: object())
    artifacts = [_ls_backfill_artifact(tmp_path, f"all-{i}") for i in range(2)]
    monkeypatch.setattr(
        cli,
        "iter_planned_investor_flow_backfill",
        lambda **_kwargs: [
            _ls_backfill_progress(0, 0, artifacts[0]),
            _ls_backfill_progress(1, 1, artifacts[1]),
        ],
    )
    assert cli.main(["backfill-ls-investor-flow", "--plan-path", str(plan_file),
                     "--bronze-root", str(tmp_path / "bronze"),
                     "--checkpoint-root", str(tmp_path / "ckpt")]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_backfill_ls_max_batches_stops_at_whole_batch(tmp_path, monkeypatch, capsys) -> None:
    import src.data.cli as cli

    plan_file = tmp_path / "plan.json"
    _write_ls_backfill_plan(plan_file, symbols=("005930",))
    monkeypatch.setattr(cli, "LsInvestorFlowCollector", lambda symbols: object())
    pulls: list[int] = []

    def fake_iter(**_kwargs):
        for index in range(5):
            pulls.append(index)
            yield _ls_backfill_progress(index, index, _ls_backfill_artifact(tmp_path, f"bound-{index}"))

    monkeypatch.setattr(cli, "iter_planned_investor_flow_backfill", fake_iter)
    assert cli.main(["backfill-ls-investor-flow", "--plan-path", str(plan_file),
                     "--bronze-root", str(tmp_path / "bronze"),
                     "--checkpoint-root", str(tmp_path / "ckpt"),
                     "--max-batches", "2"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 2
    assert pulls == [0, 1]


def test_backfill_ls_invalid_batch_bound_fails_before_factory(tmp_path, monkeypatch) -> None:
    import src.data.cli as cli

    plan_file = tmp_path / "plan.json"
    _write_ls_backfill_plan(plan_file, symbols=("005930",))

    def factory(_symbols) -> object:
        raise AssertionError("collector factory must not be invoked")

    def fake_iter(**_kwargs):
        raise AssertionError("iterator must not be invoked")

    monkeypatch.setattr(cli, "LsInvestorFlowCollector", factory)
    monkeypatch.setattr(cli, "iter_planned_investor_flow_backfill", fake_iter)
    base = ["backfill-ls-investor-flow", "--plan-path", str(plan_file),
            "--bronze-root", str(tmp_path / "bronze"),
            "--checkpoint-root", str(tmp_path / "ckpt")]
    assert cli.main([*base, "--chunk-batch-size", "0"]) == 1
    assert cli.main([*base, "--chunk-batch-size", "501"]) == 1
    assert cli.main([*base, "--max-batches", "0"]) == 1
    assert cli.main([*base, "--max-batches", "-3"]) == 1


def test_backfill_ls_naive_retrieved_at_fails_before_iterator(tmp_path, monkeypatch) -> None:
    import src.data.cli as cli

    plan_file = tmp_path / "plan.json"
    _write_ls_backfill_plan(plan_file, symbols=("005930",))
    monkeypatch.setattr(cli, "LsInvestorFlowCollector", lambda symbols: object())

    def fake_iter(**_kwargs):
        raise AssertionError("iterator must not be invoked")

    monkeypatch.setattr(cli, "iter_planned_investor_flow_backfill", fake_iter)
    base = ["backfill-ls-investor-flow", "--plan-path", str(plan_file),
            "--bronze-root", str(tmp_path / "bronze"),
            "--checkpoint-root", str(tmp_path / "ckpt")]
    assert cli.main([*base, "--retrieved-at", "2026-09-20T00:00:00"]) == 1
    assert cli.main([*base, "--retrieved-at", "not-a-date"]) == 1


def test_backfill_ls_fixed_retrieved_at_replays_deterministically(tmp_path, monkeypatch, capsys) -> None:
    from collections.abc import Callable
    from datetime import datetime
    from typing import cast

    import src.data.cli as cli

    plan_file = tmp_path / "plan.json"
    _write_ls_backfill_plan(plan_file, symbols=("005930",))
    monkeypatch.setattr(cli, "LsInvestorFlowCollector", lambda symbols: object())
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli, "iter_planned_investor_flow_backfill",
        lambda **kwargs: captured.update(kwargs) or iter([_ls_backfill_progress(0, 0, _ls_backfill_artifact(tmp_path, "fixed"))]),
    )
    stamp = "2026-09-20T00:00:00+00:00"
    assert cli.main(["backfill-ls-investor-flow", "--plan-path", str(plan_file),
                     "--bronze-root", str(tmp_path / "bronze"),
                     "--checkpoint-root", str(tmp_path / "ckpt"),
                     "--retrieved-at", stamp]) == 0
    replay = cast("Callable[[], datetime]", captured["retrieved_at_factory"])
    assert replay().isoformat() == stamp
    assert replay().isoformat() == stamp
    capsys.readouterr()


def test_backfill_ls_provider_failure_emits_counts_without_certification(tmp_path, monkeypatch, capsys) -> None:
    import src.data.cli as cli

    plan_file = tmp_path / "plan.json"
    _write_ls_backfill_plan(plan_file, symbols=("005930",))
    monkeypatch.setattr(cli, "LsInvestorFlowCollector", lambda symbols: object())
    artifact = _ls_backfill_artifact(tmp_path, "err", provider_errors=2, completed=0, planned=2)
    monkeypatch.setattr(
        cli, "iter_planned_investor_flow_backfill",
        lambda **_kwargs: iter([_ls_backfill_progress(0, 0, artifact)]),
    )
    assert cli.main(["backfill-ls-investor-flow", "--plan-path", str(plan_file),
                     "--bronze-root", str(tmp_path / "bronze"),
                     "--checkpoint-root", str(tmp_path / "ckpt")]) == 0
    out = capsys.readouterr().out.lower()
    assert '"provider_error_chunks": 2' in out
    assert "silver" not in out
    assert "certif" not in out


def test_backfill_ls_reports_plan_collector_and_iteration_failures(tmp_path, monkeypatch, capsys) -> None:
    import src.data.cli as cli
    from src.data.schemas import PITDataError

    plan_file = tmp_path / "plan.json"
    _write_ls_backfill_plan(plan_file, symbols=("005930",))
    base = ["backfill-ls-investor-flow", "--plan-path", str(plan_file),
            "--bronze-root", str(tmp_path / "bronze"),
            "--checkpoint-root", str(tmp_path / "ckpt")]
    assert cli.main(["backfill-ls-investor-flow", "--plan-path", str(tmp_path / "missing.json"),
                     "--bronze-root", str(tmp_path / "bronze"),
                     "--checkpoint-root", str(tmp_path / "ckpt")]) == 1

    def broken_factory(_symbols) -> object:
        raise PITDataError("no credentials")

    monkeypatch.setattr(cli, "LsInvestorFlowCollector", broken_factory)
    assert cli.main(base) == 1

    monkeypatch.setattr(cli, "LsInvestorFlowCollector", lambda symbols: object())

    def broken_iter(**_kwargs):
        raise PITDataError("provider contract failed")
        yield

    monkeypatch.setattr(cli, "iter_planned_investor_flow_backfill", broken_iter)
    assert cli.main(base) == 1
    assert "provider contract failed" in capsys.readouterr().out


def test_flow_backfill_progress_payload_tolerates_unreadable_reports(tmp_path) -> None:
    from src.data.cli import _flow_backfill_progress_payload

    artifact = _ls_backfill_artifact(tmp_path, "missing-report")
    import os

    os.remove(tmp_path / "missing-report-report.json")
    payload = _flow_backfill_progress_payload(_ls_backfill_progress(0, 0, artifact))
    assert payload["plan_id"] == ""
    assert payload["content_hash"] == "c" * 64

    corrupt_artifact = _ls_backfill_artifact(tmp_path, "corrupt")
    corrupt = tmp_path / "corrupt-report.json"
    corrupt.write_text("{not json", encoding="utf-8")
    payload = _flow_backfill_progress_payload(_ls_backfill_progress(1, 2, corrupt_artifact))
    assert payload["batch_index"] == 1
    assert payload["chunk_offset"] == 2

    listed_artifact = _ls_backfill_artifact(tmp_path, "listed")
    listed = tmp_path / "listed-report.json"
    listed.write_text("[1, 2]", encoding="utf-8")
    payload = _flow_backfill_progress_payload(_ls_backfill_progress(0, 0, listed_artifact))
    assert payload["plan_id"] == ""


def test_legacy_missing_dart_facts_command_reports_success_and_domain_error(
    tmp_path, monkeypatch, capsys
) -> None:
    from types import SimpleNamespace

    import src.data.dart_backfill as backfill
    import src.integrations.dart.xbrl as xbrl
    from src.data.cli import main
    from src.data.schemas import PITDataError

    class Collector:
        def __init__(self, *, quota_store) -> None:
            self.quota_store = quota_store

    monkeypatch.setattr(xbrl, "DartXbrlCollector", Collector)
    monkeypatch.setattr(
        backfill,
        "run_dart_missing_facts_batch",
        lambda **_kwargs: SimpleNamespace(
            plan_id="missing-facts-test", candidate_count=2, selected_identities=({"filing_id": "F1"},),
            missing_without_filing_count=1,
        ),
    )
    arguments = [
        "collect-missing-dart-facts",
        "--bronze-root",
        str(tmp_path / "bronze"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--backfill-artifact",
        str(tmp_path / "backfill.json"),
        "--retrieved-at",
        "2020-01-01T00:00:00+00:00",
    ]
    assert main(arguments) == 0
    assert '"selected_count": 1' in capsys.readouterr().out

    def fail(**_kwargs):
        raise PITDataError("fixture failure")

    monkeypatch.setattr(backfill, "run_dart_missing_facts_batch", fail)
    assert main(arguments) == 1
    assert "fixture failure" in capsys.readouterr().out


def test_ordinary_universe_price_audit_command_emits_report_and_handles_failure(
    tmp_path, monkeypatch, capsys
) -> None:
    import src.data.ordinary_universe_price_audit as audit_mod
    from src.data.cli import main
    from src.data.ordinary_universe_price_audit import OrdinaryUniversePriceAudit
    from src.data.schemas import PITDataError

    monkeypatch.setattr(
        audit_mod,
        "audit_ordinary_universe_price_availability",
        lambda **_kwargs: OrdinaryUniversePriceAudit(
            dataset_id="audit-1",
            universe_dataset_id="universe-1",
            sessions=1,
            universe_rows=1,
            eligible_rows=1,
            price_rows=1,
            tradable_rows=1,
            missing_price_rows=0,
            invalid_price_rows=0,
            zero_volume_rows=0,
            report_hash="r" * 64,
        ),
    )
    arguments = [
        "audit-ordinary-universe-prices",
        "--universe-root",
        str(tmp_path / "universe"),
        "--bronze-root",
        str(tmp_path / "bronze"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
    ]
    assert main(arguments) == 0
    assert '"dataset_id": "audit-1"' in capsys.readouterr().out

    def fail(**_kwargs):
        raise PITDataError("audit fixture failure")

    monkeypatch.setattr(audit_mod, "audit_ordinary_universe_price_availability", fail)
    assert main(arguments) == 1


def test_rebuild_data_requires_certified_master_before_kis_collection(tmp_path, monkeypatch) -> None:
    from src.data import cli as cli_module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stock-data", "rebuild-data",
            "--data-root", str(tmp_path / "data"),
            "--bronze-root", str(tmp_path / "bronze"),
            "--silver-root", str(tmp_path / "silver"),
            "--gold-root", str(tmp_path / "gold"),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--validation-start", "2016-01-04",
            "--validation-end", "2016-12-29",
            "--certification-time", "2026-09-05T00:00:00+00:00",
        ],
    )
    assert cli_module.main() == 1


def test_run_backtest_refuses_without_resolved_execution_components(tmp_path) -> None:
    from argparse import Namespace

    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="backtest-run-manifest"):
        _dispatch_backtest(Namespace(gold_root=tmp_path))


def test_run_backtest_validates_selected_bundle_before_execution(tmp_path, monkeypatch) -> None:
    from argparse import Namespace
    from datetime import UTC, datetime

    import polars as pl
    import pytest
    import src.data.cli as cli_mod
    import src.data.gold_artifacts as gold_artifacts
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    calls: list[tuple[str, str]] = []

    def fake_resolve(*, gold_root, dataset_id, decision_time):
        calls.append(("resolve", dataset_id))
        return object()

    def fake_load(*, bundle, decision_time):
        calls.append(("load", "gold-2016"))
        scores = pl.DataFrame(
            [
                {
                    "decision_session": datetime(2016, 1, 4, 15, 30, tzinfo=UTC),
                    "instrument_id": "KRX:000001",
                    "eligible": True,
                    "champion_score": 1.0,
                    "rank": 1,
                    "exclusion_reasons": "",
                    "feature_policy_version": "champion-v1-qvef-v1",
                    "score_policy_version": "champion-v1-scoring-v1",
                }
            ]
        )
        return (object(), object(), scores)

    monkeypatch.setattr(gold_artifacts, "resolve_gold_artifact_bundle", fake_resolve)
    monkeypatch.setattr(gold_artifacts, "load_gold_artifact_frames", fake_load)

    created: dict[str, object] = {}
    real_strategy = cli_mod.ChampionStrategy

    def spy_strategy(*args, **kwargs):  # type: ignore[no-untyped-def]
        created.update(kwargs)
        return real_strategy(*args, **kwargs)

    monkeypatch.setattr(cli_mod, "ChampionStrategy", spy_strategy)

    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest, write_legacy_backtest_run_manifest
    from src.data.schemas import SilverTable

    run_manifest_obj = build_legacy_backtest_run_manifest(
        silver_root=tmp_path / "silver",
        gold_root=tmp_path,
        silver_dataset_ids={table: f"{table.value}-id" for table in SilverTable},
        gold_dataset_id="gold-2016",
        validation_start=__import__("datetime").date(2016, 1, 4),
        validation_end=__import__("datetime").date(2016, 12, 30),
        strategy_id="champion-v1",
        policy_versions={"market_inputs": "korean-equity-market-inputs-v2"},
    )
    manifest_path = write_legacy_backtest_run_manifest(
        manifest=run_manifest_obj, artifact_root=tmp_path / "artifacts"
    )

    with pytest.raises(PITDataError, match="certified Silver table"):
        _dispatch_backtest(
            Namespace(
                gold_root=tmp_path,
                silver_root=tmp_path / "silver",
                artifact_root=tmp_path / "artifacts",
                validation_start="2016-01-04",
                validation_end="2016-12-30",
                smoke_symbol=None,
                gold_dataset_id="gold-2016",
                backtest_run_manifest=manifest_path,
                strategy_id="champion-v1",
            )
        )

    assert calls == [("resolve", "gold-2016"), ("load", "gold-2016")]
    assert created == {}


def _write_cli_fact_receipt(bronze_root, payload_text) -> None:
    import hashlib
    import json

    raw = payload_text.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    receipt_dir = bronze_root / "financial_facts" / digest
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_bytes(raw)
    (receipt_dir / "receipt.json").write_text(
        json.dumps(
            {
                "kind": "financial_facts",
                "content_hash": digest,
                "source_path": "cli",
                "retrieved_at": "2016-01-01T00:00:00+00:00",
                "ingested_at": "2016-01-01T00:00:00+00:00",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_normalize_dart_facts_command_parses_all_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stock-data",
            "normalize-dart-facts",
            "--bronze-root",
            "b",
            "--silver-root",
            "s",
            "--artifact-root",
            "a",
            "--decision-time",
            "2016-12-30T00:00:00+00:00",
            "--batch-size",
            "7",
        ],
    )

    args = _parse_args()

    assert args.command == "normalize-dart-facts"
    assert args.batch_size == 7


def test_normalize_dart_facts_dispatch_publishes(tmp_path, monkeypatch, capsys) -> None:
    from src.data import cli as cli_module

    _write_cli_fact_receipt(
        tmp_path / "bronze",
        '{"records": [{"ticker": "005930", "corp_code": "00126380", "fiscal_period": "2015Q3", "filing_id": "F1", "fact": "sales", "published_at": "2015-11-16T00:00:00+00:00", "value": 10.0, "unit": "KRW"}]}',
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stock-data",
            "normalize-dart-facts",
            "--bronze-root",
            str(tmp_path / "bronze"),
            "--silver-root",
            str(tmp_path / "silver"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--decision-time",
            "2016-12-30T00:00:00+00:00",
        ],
    )

    assert cli_module.main() == 0
    captured = capsys.readouterr()
    assert "output_hash" in captured.out
    assert (tmp_path / "silver" / "financial_facts").exists()


def test_load_silver_table_reads_latest_manifest_dataset(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl

    from src.core.datasets import DatasetCertification, HIVE_PARTITION_LAYOUT, make_manifest
    from src.core.instruments import AssetKind
    from src.data.cli import _load_silver_table
    from src.data.schemas import SilverTable
    from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "session": [datetime(2015, 11, 17, tzinfo=UTC)],
            "available_at": [datetime(2015, 11, 17, tzinfo=UTC)],
            "source_hash": ["r"],
        }
    )
    content_hash = canonical_content_hash(frame, frame.columns)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=frame.columns,
        feature_set="stock_pit_calendar_v1",
        label_definition="none",
        label_horizon_sessions=1,
        time_start=datetime(2015, 11, 17, tzinfo=UTC),
        time_end=datetime(2015, 11, 17, tzinfo=UTC),
        provider_version="t",
        universe_policy_version="v1",
        row_count=frame.height,
        schema_version="v2",
        content_hash=content_hash,
        storage_layout=HIVE_PARTITION_LAYOUT,
        certification=DatasetCertification.RESEARCH,
    )
    ParquetDatasetStore(tmp_path / "silver" / "calendar").write_partitioned(
        frame,
        dataset_id=content_hash,
        manifest=manifest,
        expected_feature_set="stock_pit_calendar_v1",
        decision_time=decision_time,
        content_manifest={},
    )

    loaded = _load_silver_table(tmp_path / "silver", SilverTable.CALENDAR)

    assert loaded.height == 1


def test_normalize_dart_facts_dispatch_reports_failure(tmp_path, monkeypatch, capsys) -> None:
    from src.data import cli as cli_module

    receipt_dir = tmp_path / "bronze" / "financial_facts" / "bad"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_text('{"records": []}', encoding="utf-8")
    (receipt_dir / "receipt.json").write_text(
        '{"kind": "financial_facts", "content_hash": "f", "retrieved_at": "2016-01-01T00:00:00+00:00", "ingested_at": "2016-01-01T00:00:00+00:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stock-data",
            "normalize-dart-facts",
            "--bronze-root",
            str(tmp_path / "bronze"),
            "--silver-root",
            str(tmp_path / "silver"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--decision-time",
            "2016-12-30T00:00:00+00:00",
        ],
    )

    assert cli_module.main() == 1
    assert not (tmp_path / "silver" / "financial_facts").exists()


def test_cli_plan_defaults_to_kis_page_capacity(monkeypatch) -> None:
    import sys
    from src.data.cli import _parse_args
    from src.data.collection_plan import LS_MAX_SESSIONS_PER_REQUEST

    monkeypatch.setattr(sys, 'argv', ['stock-data', 'plan', '--coverage-start', '2024-01-02', '--coverage-end', '2024-01-03', '--symbols', '005930'])
    assert _parse_args().chunk_size == LS_MAX_SESSIONS_PER_REQUEST


def test_run_backtest_requires_selected_gold_dataset_id(tmp_path) -> None:
    from argparse import Namespace

    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='backtest-run-manifest'):
        _dispatch_backtest(
            Namespace(
                gold_root=tmp_path,
                silver_root=tmp_path / 'silver',
                artifact_root=tmp_path / 'artifacts',
                validation_start='2016-01-04',
                validation_end='2016-12-30',
                smoke_symbol=None,
                gold_dataset_id=None,
                strategy_id='champion-v1',
            )
        )


def test_run_backtest_core_v1_end_to_end_with_pit_inputs(tmp_path, monkeypatch) -> None:
    from argparse import Namespace
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    import src.data.silver as silver_mod
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import SilverTable

    kst = ZoneInfo('Asia/Seoul')
    all_days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(70))
    calendar_df = pl.DataFrame({'session': list(all_days)})
    start, end = all_days[60].date().isoformat(), all_days[64].date().isoformat()
    warmup_days = all_days[0:68]
    closes_a = [10000.0 + index * 10.0 for index in range(len(warmup_days))]
    closes_b = [20000.0 + index * 5.0 for index in range(len(warmup_days))]
    market_df = pl.DataFrame({
        'session': [day for day in warmup_days for _ in ('KRX:A', 'KRX:B')],
        'instrument_id': ['KRX:A', 'KRX:B'] * len(warmup_days),
        'open': [c - 5.0 for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        'close': [c for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        'volume': [1000.0] * (2 * len(warmup_days)),
        'trading_value': [c * 1000.0 for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        'market_cap': [1e12, 5e11] * len(warmup_days),
        'available_at': [day.replace(hour=15, minute=30) for day in warmup_days for _ in ('KRX:A', 'KRX:B')],
    })
    dm_dir = tmp_path / 'dm'
    dm_dir.mkdir()
    market_df.write_parquet(dm_dir / 'daily.parquet')
    master_df = pl.DataFrame({
        'instrument_id': ['KRX:A', 'KRX:B'],
        'sector': ['Technology', 'Healthcare'],
        'valid_from': [all_days[0], all_days[0]],
        'valid_to': [all_days[69], all_days[69]],
        'available_at': [all_days[0], all_days[0]],
    })

    def fake_load_table(silver_root, table):
        if table == SilverTable.CALENDAR:
            return calendar_df
        if table == SilverTable.SECURITY_MASTER:
            return master_df
        return pl.DataFrame(schema={'effective_session': pl.Datetime(time_zone='Asia/Seoul'), 'instrument_id': pl.String, 'action_type': pl.String, 'available_at': pl.Datetime(time_zone='Asia/Seoul')})

    import src.data.gold_artifacts as gold_artifacts_mod
    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest, write_legacy_backtest_run_manifest

    universe_df = pl.DataFrame({
        'decision_session': [day.replace(hour=15, minute=30) for day in all_days[60:65] for _ in ('KRX:A', 'KRX:B')],
        'instrument_id': ['KRX:A', 'KRX:B'] * 5,
        'eligible': [True] * 10,
    })

    def fake_load_by_id(*, root, table, dataset_id, decision_time):
        if table == SilverTable.CORPORATE_ACTIONS:
            return pl.DataFrame({
                'instrument_id': ['KRX:B'],
                'action_id': ['unknown-b'],
                'action_type': ['unresolved'],
                'effective_session': [all_days[61]],
                'factor': [1.0],
                'cash_amount': [0.0],
                'available_at': [all_days[60]],
                'evidence_status': ['unresolved'],
                'evidence_reason': ['unsupported_merger'],
            })
        return fake_load_table(root, table)

    monkeypatch.setattr(silver_mod, 'load_silver_table_by_dataset_id', fake_load_by_id)
    monkeypatch.setattr(silver_mod, 'silver_dataset_path_by_id', lambda *, root, table, dataset_id, decision_time: dm_dir)
    monkeypatch.setattr(
        gold_artifacts_mod,
        'resolve_gold_artifact_bundle',
        lambda *, gold_root, dataset_id, decision_time: object(),
    )
    monkeypatch.setattr(
        gold_artifacts_mod,
        'load_gold_artifact_frames',
        lambda *, bundle, decision_time: (universe_df, pl.DataFrame(), pl.DataFrame()),
    )

    run_manifest_obj = build_legacy_backtest_run_manifest(
        silver_root=tmp_path / 'silver',
        gold_root=tmp_path / 'gold',
        silver_dataset_ids={table: f"{table.value}-id" for table in SilverTable},
        gold_dataset_id='gold-test',
        validation_start=__import__('datetime').date.fromisoformat(start),
        validation_end=__import__('datetime').date.fromisoformat(end),
        strategy_id='core-v1',
        policy_versions={'market_inputs': 'korean-equity-market-inputs-v2'},
    )
    manifest_path = write_legacy_backtest_run_manifest(
        manifest=run_manifest_obj, artifact_root=tmp_path / 'artifacts'
    )

    code = _dispatch_backtest(
        Namespace(
            silver_root=tmp_path / 'silver',
            artifact_root=tmp_path / 'artifacts',
            gold_root=tmp_path / 'gold',
            validation_start=start,
            validation_end=end,
            smoke_symbol=None,
            gold_dataset_id='gold-test',
            backtest_run_manifest=manifest_path,
            strategy_id='core-v1',
            initial_cash=100_000_000.0,
            scenario='base',
            ledger_id='core-test-2016',
        )
    )
    assert code == 0
    manifests = list((tmp_path / 'artifacts' / 'backtests').rglob('result.json'))
    assert len(manifests) == 1
    import json

    payload = json.loads(manifests[0].read_text(encoding='utf-8'))
    assert payload['metadata']['strategy_id'] == 'core-v1'
    assert payload['metadata']['market_input_policy_version'] == 'korean-equity-market-inputs-v2'
    assert payload['metadata']['warmup_sessions'] == 60
    assert payload['metadata']['run_manifest_hash'] == run_manifest_obj.content_hash
    assert payload['dataset_hash'] == run_manifest_obj.content_hash
    assert payload['metadata']['blocked_instrument_sessions'] >= 1
    assert payload['metadata']['exclusion_reason_counts'] == {'unsupported_merger': 1}
    assert payload['manifest_hash'] == run_manifest_obj.content_hash


def test_run_backtest_rejects_market_without_certified_columns(tmp_path, monkeypatch) -> None:
    from argparse import Namespace
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    import src.data.gold_artifacts as gold_artifacts_mod
    import src.data.silver as silver_mod
    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest, write_legacy_backtest_run_manifest
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError, SilverTable

    kst = ZoneInfo('Asia/Seoul')
    days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(3))
    master = pl.DataFrame({'instrument_id': ['KRX:A'], 'sector': ['Technology'], 'valid_from': [days[0]], 'valid_to': [days[-1]], 'available_at': [days[0]]})
    actions = pl.DataFrame(schema={'effective_session': pl.Datetime(time_zone='Asia/Seoul'), 'instrument_id': pl.String, 'action_type': pl.String, 'available_at': pl.Datetime(time_zone='Asia/Seoul')})
    calendar_df = pl.DataFrame({'session': list(days)})
    universe_df = pl.DataFrame({'decision_session': [days[0]], 'instrument_id': ['KRX:A'], 'eligible': [True]})

    def fake_load_by_id(*, root, table, dataset_id, decision_time):
        if table == SilverTable.CALENDAR:
            return calendar_df
        if table == SilverTable.SECURITY_MASTER:
            return master
        return actions

    monkeypatch.setattr(silver_mod, 'load_silver_table_by_dataset_id', fake_load_by_id)
    monkeypatch.setattr(
        gold_artifacts_mod,
        'resolve_gold_artifact_bundle',
        lambda *, gold_root, dataset_id, decision_time: object(),
    )
    monkeypatch.setattr(
        gold_artifacts_mod,
        'load_gold_artifact_frames',
        lambda *, bundle, decision_time: (universe_df, pl.DataFrame(), pl.DataFrame()),
    )

    run_manifest_obj = build_legacy_backtest_run_manifest(
        silver_root=tmp_path / 'silver',
        gold_root=tmp_path / 'gold',
        silver_dataset_ids={table: f"{table.value}-id" for table in SilverTable},
        gold_dataset_id='gold-test',
        validation_start=days[0].date(),
        validation_end=days[1].date(),
        strategy_id='core-v1',
        policy_versions={'market_inputs': 'korean-equity-market-inputs-v2'},
    )
    manifest_path = write_legacy_backtest_run_manifest(
        manifest=run_manifest_obj, artifact_root=tmp_path / 'artifacts'
    )

    def _run_without(columns: list[str], match: str) -> None:
        dm_dir = tmp_path / f"dm_{'_'.join(columns)}"
        dm_dir.mkdir(exist_ok=True)
        base = {
            'session': list(days),
            'instrument_id': ['KRX:A'] * 3,
            'open': [100.0] * 3,
            'close': [101.0] * 3,
            'volume': [10.0] * 3,
            'trading_value': [1010.0] * 3,
            'market_cap': [1e10] * 3,
            'available_at': [day.replace(hour=15, minute=30) for day in days],
        }
        pl.DataFrame({key: base[key] for key in columns}).write_parquet(dm_dir / 'daily.parquet')
        monkeypatch.setattr(silver_mod, 'silver_dataset_path_by_id', lambda *, root, table, dataset_id, decision_time: dm_dir)
        with pytest.raises(PITDataError, match=match):
            _dispatch_backtest(
                Namespace(
                    silver_root=tmp_path / 'silver',
                    artifact_root=tmp_path / 'artifacts',
                    gold_root=tmp_path / 'gold',
                    validation_start=days[0].date().isoformat(),
                    validation_end=days[1].date().isoformat(),
                    smoke_symbol=None,
                    gold_dataset_id='gold-test',
                    backtest_run_manifest=manifest_path,
                    strategy_id='core-v1',
                )
            )

    _run_without(['session', 'instrument_id', 'open', 'close', 'volume', 'trading_value', 'market_cap'], 'certified available_at')
    _run_without(['session', 'instrument_id', 'open', 'close', 'volume', 'trading_value', 'available_at'], 'market_cap')


def test_run_backtest_smoke_symbol_completes_with_empty_pit_frames(tmp_path, monkeypatch) -> None:
    from argparse import Namespace
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    import src.data.cli as cli_mod
    import src.data.silver as silver_mod
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import SilverTable

    kst = ZoneInfo('Asia/Seoul')
    days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(64))
    master = pl.DataFrame({'instrument_id': ['KRX:A'], 'sector': ['Technology'], 'valid_from': [days[0]], 'valid_to': [days[-1]], 'available_at': [days[0]]})
    actions = pl.DataFrame(schema={'effective_session': pl.Datetime(time_zone='Asia/Seoul'), 'instrument_id': pl.String, 'action_type': pl.String, 'available_at': pl.Datetime(time_zone='Asia/Seoul')})
    monkeypatch.setattr(cli_mod, '_load_silver_table', lambda silver_root, table: pl.DataFrame({'session': list(days)}) if table == SilverTable.CALENDAR else master if table == SilverTable.SECURITY_MASTER else actions)
    dm_dir = tmp_path / 'dm_smoke'
    dm_dir.mkdir()
    pl.DataFrame({
        'session': list(days),
        'instrument_id': ['KRX:A'] * len(days),
        'open': [10000.0 + index for index in range(len(days))],
        'close': [10050.0 + index + (index % 3) * 0.1 for index in range(len(days))],
        'volume': [1000000.0] * len(days),
        'trading_value': [10050000000.0] * len(days),
        'market_cap': [1e12] * len(days),
        'available_at': [day.replace(hour=15, minute=30) for day in days],
    }).write_parquet(dm_dir / 'daily.parquet')
    monkeypatch.setattr(silver_mod, 'latest_silver_dataset_path', lambda *, root, table, decision_time: dm_dir)
    code = _dispatch_backtest(
        Namespace(
            silver_root=tmp_path / 'silver',
            artifact_root=tmp_path / 'artifacts',
            gold_root=tmp_path / 'gold',
            validation_start=days[60].date().isoformat(),
            validation_end=days[60].date().isoformat(),
            smoke_symbol='KRX:A',
            gold_dataset_id=None,
            strategy_id='core-v1',
            initial_cash=100_000_000.0,
            scenario='base',
            ledger_id='smoke-test',
        )
    )
    assert code == 0


def test_build_gold_uses_four_factor_default_policy(tmp_path, monkeypatch, capsys) -> None:
    import sys
    from types import SimpleNamespace

    import src.data.gold as gold_mod
    from src.data.cli import main

    captured: dict[str, object] = {}

    def fake_load(**kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            calendar=object(),
            security_master=object(),
            daily_market=object(),
            financial_facts=object(),
            corporate_actions=object(),
            investor_flow=object(),
        )

    def fake_materialize(**kwargs):  # type: ignore[no-untyped-def]
        captured['score_policy'] = kwargs['score_policy']
        manifest = SimpleNamespace(
            manifest_hash='hash',
            warmup=SimpleNamespace(warmup_ok=True, warmup_sessions_found=60),
            bar_audit=[],
            dart_eligibility=[],
            ca_excluded_instrument_ids=[],
            eligible_instrument_ids=[],
        )
        return SimpleNamespace(
            manifest=manifest,
            universe_decisions_count=0,
            eligible_decisions_count=0,
            feature_rows_count=0,
            universe_path='u',
            features_path='f',
            summary_artifact_path='s',
        )

    import src.data.cli as cli_mod
    from src.data.schemas import SilverTable
    _bindings = {table: f'id-{table.value}' for table in SilverTable}
    monkeypatch.setattr(cli_mod, 'load_gold_window_inputs', fake_load)
    monkeypatch.setattr(cli_mod, 'parse_silver_dataset_bindings', lambda _values: _bindings)
    monkeypatch.setattr(cli_mod, 'resolve_gold_dataset_bindings', lambda **_kwargs: _bindings)
    monkeypatch.setattr(cli_mod, 'write_gold_input_binding_artifact', lambda **_kwargs: tmp_path / 'binding.json')
    monkeypatch.setattr(gold_mod, 'materialize_gold_window', fake_materialize)
    _argv = [
        'stock-data', 'build-gold',
        '--silver-root', str(tmp_path),
        '--artifact-root', str(tmp_path),
        '--decision-time', '2024-01-03T00:00:00+00:00',
        '--validation-start', '2016-01-04',
        '--validation-end', '2016-12-30',
    ]
    for _table, _dataset_id in _bindings.items():
        _argv.extend(['--silver-dataset-id', f'{_table.value}={_dataset_id}'])
    monkeypatch.setattr(sys, 'argv', _argv)
    assert main() == 0
    assert captured['score_policy'].min_required_factors == 4
    capsys.readouterr()


def test_run_backtest_parser_accepts_gold_dataset_id(monkeypatch) -> None:
    import sys

    from src.data.cli import _parse_args

    monkeypatch.setattr(
        sys,
        'argv',
        ['stock-data', 'run-backtest', '--gold-dataset-id', 'gold-2016-verified'],
    )

    args = _parse_args()

    assert args.command == 'run-backtest'
    assert args.gold_dataset_id == 'gold-2016-verified'


def test_cli_refresh_corporate_actions_wires_certified_scan(monkeypatch, tmp_path, capsys) -> None:
    from datetime import UTC, datetime
    import polars as pl
    import src.data.cli as cli
    from src.core.time import SessionCalendar

    decision = datetime(2026, 9, 9, tzinfo=UTC)
    calendar_frame = pl.DataFrame({'session': [decision]})
    captured = {}
    monkeypatch.setattr(cli, '_parse_dt', lambda _value: decision)
    monkeypatch.setattr(cli, 'load_latest_silver_table', lambda **kwargs: calendar_frame)
    monkeypatch.setattr(cli, 'load_latest_silver_market_scan', lambda **kwargs: pl.DataFrame({'session': [decision], 'instrument_id': ['KRX:A'], 'close': [1.0], 'shares_outstanding': [1.0], 'market_cap': [1.0]}).lazy())
    def refresh(**kwargs):
        captured['kwargs'] = kwargs
        return type('Report', (), {'report_hash': 'refresh-hash'})()
    monkeypatch.setattr(cli, 'refresh_corporate_action_silver', refresh)
    code = cli.main(['refresh-corporate-actions', '--bronze-root', str(tmp_path / 'bronze'), '--silver-root', str(tmp_path / 'silver'), '--artifact-root', str(tmp_path / 'artifacts'), '--decision-time', decision.isoformat()])
    assert code == 0
    assert isinstance(captured['kwargs']['calendar'], SessionCalendar)
    assert captured['kwargs']['daily_market'].collect().columns == ['session', 'instrument_id', 'close', 'shares_outstanding', 'market_cap']
    assert 'refresh-hash' in capsys.readouterr().out


def test_cli_bronze_retention_plan_wires_read_only_audit(tmp_path, monkeypatch, capsys) -> None:
    from types import SimpleNamespace
    import src.data.cli as cli

    captured: dict[str, object] = {}
    def fake_plan(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(receipt_count=0, total_payload_bytes=0, referenced_payload_bytes=0, unreferenced_payload_bytes=0, referenced_hashes=(), unreferenced_hashes=(), deletion_eligible=False, blocking_reasons=('generation_gc_not_certified',))
    monkeypatch.setattr(cli, 'plan_bronze_retention', fake_plan)

    code = cli.main(['bronze-retention-plan', '--bronze-root', str(tmp_path / 'bronze'), '--silver-root', str(tmp_path / 'silver'), '--artifact-root', str(tmp_path / 'artifacts')])

    assert code == 0
    assert captured['provenance_roots'] == (tmp_path / 'silver', tmp_path / 'artifacts')
    assert 'deletion_eligible' in capsys.readouterr().out


def test_cli_silver_gold_retention_plan_wires_read_only_audit(tmp_path, monkeypatch, capsys) -> None:
    from types import SimpleNamespace
    import src.data.cli as cli

    captured: dict[str, object] = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            retained_silver_roots=('stocks',),
            reclaimable_silver_roots=('stocks_prepared_20260910_v5',),
            retained_gold_roots=('stocks',),
            reclaimable_gold_roots=(),
            orphaned_staging_paths=(),
            blocking_reasons=(),
            deletion_eligible=True,
        )
    monkeypatch.setattr(cli, 'plan_storage_root_retention', fake_plan)

    code = cli.main(['silver-gold-retention-plan', '--silver-base', str(tmp_path / 'silver'), '--gold-base', str(tmp_path / 'gold'), '--artifact-root', str(tmp_path / 'artifacts')])

    assert code == 0
    assert captured['silver_base'] == tmp_path / 'silver'
    assert captured['gold_base'] == tmp_path / 'gold'
    assert captured['artifact_root'] == tmp_path / 'artifacts'
    assert 'deletion_eligible' in capsys.readouterr().out


def test_run_backtest_core_requires_backtest_run_manifest(tmp_path) -> None:
    from argparse import Namespace

    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    args = Namespace(
        silver_root=tmp_path / 'silver', gold_root=tmp_path / 'gold', artifact_root=tmp_path / 'artifacts',
        validation_start='2016-01-04', validation_end='2016-12-29', smoke_symbol=None,
        gold_dataset_id=None, backtest_run_manifest=None, initial_cash=100000000.0, scenario='base',
        ledger_id='test', strategy_id='core-v1',
    )

    with pytest.raises(PITDataError, match='backtest-run-manifest'):
        _dispatch_backtest(args)


def test_run_backtest_parser_accepts_backtest_run_manifest() -> None:
    from pathlib import Path

    from src.data.cli import _parse_args

    args = _parse_args(['run-backtest', '--backtest-run-manifest', 'data/artifacts/runs/a.json'])

    assert args.command == 'run-backtest'
    assert args.backtest_run_manifest == Path('data/artifacts/runs/a.json')


def _manifest_bound_namespace(tmp_path, monkeypatch, **overrides):
    from argparse import Namespace
    from datetime import date

    import polars as pl

    import src.data.gold_artifacts as gold_artifacts_mod
    import src.data.silver as silver_mod
    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest, write_legacy_backtest_run_manifest
    from src.data.schemas import SilverTable

    calendar_df = pl.DataFrame({'session': [__import__('datetime').datetime(2016, 1, 4, 9, tzinfo=__import__('datetime').UTC)]})
    universe_df = pl.DataFrame(
        {
            'decision_session': [__import__('datetime').datetime(2016, 1, 4, 15, 30, tzinfo=__import__('datetime').UTC)],
            'instrument_id': ['KRX:A'],
            'eligible': [True],
        }
    )
    monkeypatch.setattr(
        silver_mod, 'load_silver_table_by_dataset_id', lambda *, root, table, dataset_id, decision_time: calendar_df
    )
    monkeypatch.setattr(
        silver_mod, 'silver_dataset_path_by_id', lambda *, root, table, dataset_id, decision_time: tmp_path / 'dm'
    )
    monkeypatch.setattr(
        gold_artifacts_mod, 'resolve_gold_artifact_bundle', lambda *, gold_root, dataset_id, decision_time: object()
    )
    monkeypatch.setattr(
        gold_artifacts_mod,
        'load_gold_artifact_frames',
        lambda *, bundle, decision_time: (universe_df, pl.DataFrame(), pl.DataFrame()),
    )
    manifest = build_legacy_backtest_run_manifest(
        silver_root=tmp_path / 'silver',
        gold_root=tmp_path / 'gold',
        silver_dataset_ids={table: f"{table.value}-id" for table in SilverTable},
        gold_dataset_id='gold-test',
        validation_start=date(2016, 1, 4),
        validation_end=date(2016, 12, 29),
        strategy_id='core-v1',
        policy_versions={'market_inputs': 'korean-equity-market-inputs-v2'},
    )
    path = write_legacy_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / 'artifacts')
    base = {
        'silver_root': tmp_path / 'silver',
        'artifact_root': tmp_path / 'artifacts',
        'gold_root': tmp_path / 'gold',
        'validation_start': '2016-01-04',
        'validation_end': '2016-12-29',
        'smoke_symbol': None,
        'gold_dataset_id': 'gold-test',
        'backtest_run_manifest': path,
        'strategy_id': 'core-v1',
    }
    base.update(overrides)
    return Namespace(**base)


def test_run_backtest_rejects_silver_root_conflict(tmp_path, monkeypatch) -> None:
    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='conflicts'):
        _dispatch_backtest(_manifest_bound_namespace(tmp_path, monkeypatch, silver_root=tmp_path / 'other'))


def test_run_backtest_rejects_gold_root_conflict(tmp_path, monkeypatch) -> None:
    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='conflicts'):
        _dispatch_backtest(_manifest_bound_namespace(tmp_path, monkeypatch, gold_root=tmp_path / 'other'))


def test_run_backtest_rejects_gold_dataset_conflict(tmp_path, monkeypatch) -> None:
    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='conflicts'):
        _dispatch_backtest(_manifest_bound_namespace(tmp_path, monkeypatch, gold_dataset_id='other-gold'))


def test_run_backtest_rejects_validation_range_conflict(tmp_path, monkeypatch) -> None:
    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='conflicts'):
        _dispatch_backtest(_manifest_bound_namespace(tmp_path, monkeypatch, validation_start='2016-02-01'))
    with pytest.raises(PITDataError, match='conflicts'):
        _dispatch_backtest(_manifest_bound_namespace(tmp_path, monkeypatch, validation_end='2016-11-30'))


def test_run_backtest_rejects_strategy_conflict(tmp_path, monkeypatch) -> None:
    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='conflicts'):
        _dispatch_backtest(_manifest_bound_namespace(tmp_path, monkeypatch, strategy_id='champion-v1'))


def test_run_backtest_rejects_core_without_selected_universe(tmp_path, monkeypatch) -> None:
    from argparse import Namespace
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    import src.data.gold_artifacts as gold_artifacts_mod
    import src.data.silver as silver_mod
    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest, write_legacy_backtest_run_manifest
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError, SilverTable

    kst = ZoneInfo('Asia/Seoul')
    all_days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(70))
    calendar_df = pl.DataFrame({'session': list(all_days)})
    start, end = all_days[60].date().isoformat(), all_days[64].date().isoformat()
    warmup_days = all_days[0:68]
    closes = [10000.0 + index * 10.0 + (index % 3) * 0.1 for index in range(len(warmup_days))]
    market_df = pl.DataFrame({
        'session': list(warmup_days),
        'instrument_id': ['KRX:A'] * len(warmup_days),
        'open': [c - 5.0 for c in closes],
        'close': closes,
        'volume': [1000.0] * len(warmup_days),
        'trading_value': [c * 1000.0 for c in closes],
        'market_cap': [1e12] * len(warmup_days),
        'available_at': [day.replace(hour=15, minute=30) for day in warmup_days],
    })
    dm_dir = tmp_path / 'dm_none'
    dm_dir.mkdir()
    market_df.write_parquet(dm_dir / 'daily.parquet')
    master_df = pl.DataFrame({
        'instrument_id': ['KRX:A'],
        'sector': ['Technology'],
        'valid_from': [all_days[0]],
        'valid_to': [all_days[69]],
        'available_at': [all_days[0]],
    })
    actions_df = pl.DataFrame(
        schema={
            'effective_session': pl.Datetime(time_zone='Asia/Seoul'),
            'instrument_id': pl.String,
            'action_type': pl.String,
            'available_at': pl.Datetime(time_zone='Asia/Seoul'),
        }
    )

    def fake_load_by_id(*, root, table, dataset_id, decision_time):
        if table == SilverTable.CALENDAR:
            return calendar_df
        if table == SilverTable.SECURITY_MASTER:
            return master_df
        return actions_df

    monkeypatch.setattr(silver_mod, 'load_silver_table_by_dataset_id', fake_load_by_id)
    monkeypatch.setattr(
        silver_mod, 'silver_dataset_path_by_id', lambda *, root, table, dataset_id, decision_time: dm_dir
    )
    monkeypatch.setattr(
        gold_artifacts_mod, 'resolve_gold_artifact_bundle', lambda *, gold_root, dataset_id, decision_time: object()
    )
    monkeypatch.setattr(
        gold_artifacts_mod,
        'load_gold_artifact_frames',
        lambda *, bundle, decision_time: (None, pl.DataFrame(), pl.DataFrame()),
    )
    manifest = build_legacy_backtest_run_manifest(
        silver_root=tmp_path / 'silver',
        gold_root=tmp_path / 'gold',
        silver_dataset_ids={table: f"{table.value}-id" for table in SilverTable},
        gold_dataset_id='gold-test',
        validation_start=__import__('datetime').date.fromisoformat(start),
        validation_end=__import__('datetime').date.fromisoformat(end),
        strategy_id='core-v1',
        policy_versions={'market_inputs': 'korean-equity-market-inputs-v2'},
    )
    path = write_legacy_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / 'artifacts')
    with pytest.raises(PITDataError, match='selected Gold universe'):
        _dispatch_backtest(
            Namespace(
                silver_root=tmp_path / 'silver',
                artifact_root=tmp_path / 'artifacts',
                gold_root=tmp_path / 'gold',
                validation_start=start,
                validation_end=end,
                smoke_symbol=None,
                gold_dataset_id='gold-test',
                backtest_run_manifest=path,
                strategy_id='core-v1',
            )
        )


def test_run_backtest_rejects_champion_without_selected_scores(tmp_path, monkeypatch) -> None:
    from argparse import Namespace
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    import src.data.gold_artifacts as gold_artifacts_mod
    import src.data.silver as silver_mod
    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest, write_legacy_backtest_run_manifest
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError, SilverTable

    kst = ZoneInfo('Asia/Seoul')
    all_days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(70))
    calendar_df = pl.DataFrame({'session': list(all_days)})
    start, end = all_days[60].date().isoformat(), all_days[64].date().isoformat()
    warmup_days = all_days[0:68]
    closes = [10000.0 + index * 10.0 + (index % 3) * 0.1 for index in range(len(warmup_days))]
    market_df = pl.DataFrame({
        'session': list(warmup_days),
        'instrument_id': ['KRX:A'] * len(warmup_days),
        'open': [c - 5.0 for c in closes],
        'close': closes,
        'volume': [1000.0] * len(warmup_days),
        'trading_value': [c * 1000.0 for c in closes],
        'market_cap': [1e12] * len(warmup_days),
        'available_at': [day.replace(hour=15, minute=30) for day in warmup_days],
    })
    dm_dir = tmp_path / 'dm_no_scores'
    dm_dir.mkdir()
    market_df.write_parquet(dm_dir / 'daily.parquet')
    master_df = pl.DataFrame({
        'instrument_id': ['KRX:A'],
        'sector': ['Technology'],
        'valid_from': [all_days[0]],
        'valid_to': [all_days[69]],
        'available_at': [all_days[0]],
    })
    actions_df = pl.DataFrame(
        schema={
            'effective_session': pl.Datetime(time_zone='Asia/Seoul'),
            'instrument_id': pl.String,
            'action_type': pl.String,
            'available_at': pl.Datetime(time_zone='Asia/Seoul'),
        }
    )

    def fake_load_by_id(*, root, table, dataset_id, decision_time):
        if table == SilverTable.CALENDAR:
            return calendar_df
        if table == SilverTable.SECURITY_MASTER:
            return master_df
        return actions_df

    monkeypatch.setattr(silver_mod, 'load_silver_table_by_dataset_id', fake_load_by_id)
    monkeypatch.setattr(
        silver_mod, 'silver_dataset_path_by_id', lambda *, root, table, dataset_id, decision_time: dm_dir
    )
    monkeypatch.setattr(
        gold_artifacts_mod, 'resolve_gold_artifact_bundle', lambda *, gold_root, dataset_id, decision_time: object()
    )
    monkeypatch.setattr(
        gold_artifacts_mod,
        'load_gold_artifact_frames',
        lambda *, bundle, decision_time: (pl.DataFrame(), pl.DataFrame(), None),
    )
    manifest = build_legacy_backtest_run_manifest(
        silver_root=tmp_path / 'silver',
        gold_root=tmp_path / 'gold',
        silver_dataset_ids={table: f"{table.value}-id" for table in SilverTable},
        gold_dataset_id='gold-test',
        validation_start=__import__('datetime').date.fromisoformat(start),
        validation_end=__import__('datetime').date.fromisoformat(end),
        strategy_id='champion-v1',
        policy_versions={'market_inputs': 'korean-equity-market-inputs-v2'},
    )
    path = write_legacy_backtest_run_manifest(manifest=manifest, artifact_root=tmp_path / 'artifacts')
    with pytest.raises(PITDataError, match='resolved strategy'):
        _dispatch_backtest(
            Namespace(
                silver_root=tmp_path / 'silver',
                artifact_root=tmp_path / 'artifacts',
                gold_root=tmp_path / 'gold',
                validation_start=start,
                validation_end=end,
                smoke_symbol=None,
                gold_dataset_id='gold-test',
                backtest_run_manifest=path,
                strategy_id='champion-v1',
            )
        )


def test_cli_backtest_loads_lifecycle_table(monkeypatch, tmp_path) -> None:
    import polars as pl
    import src.data.cli as module
    from src.data.schemas import SilverTable

    loaded = []
    def fake_load(root, table):
        loaded.append(table)
        return pl.DataFrame()
    monkeypatch.setattr(module, '_load_silver_table', fake_load)
    monkeypatch.setattr(module, 'build_backtest_sessions', lambda **kwargs: ())
    monkeypatch.setattr(module, '_execute_backtest', lambda **kwargs: {'final_nav': 1.0})
    module._run_backtest_from_silver(silver_root=tmp_path,strategy_id='core-v1',validation_start='2016-01-04',validation_end='2016-01-04',artifact_root=tmp_path,smoke_symbol=None)
    assert SilverTable.LIFECYCLE_EVENTS in loaded


import inspect

import src.data.cli as cli

def test_cli_loads_manifest_bound_lifecycle_without_optional_swallow():
    source = inspect.getsource(cli)
    assert '_load_manifest_silver_table' in source
    assert 'lifecycle_events = None' not in source


def test_collect_cli_routes_selected_provider_without_kis_constructor(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace
    from src.data import cli
    from src.data.collection_plan import HistoricalCollectionPlan, PlanChunk

    plan = HistoricalCollectionPlan(plan_id='p', chunks=(PlanChunk('p:005930:0000', '005930', ()),))
    captured = {}
    monkeypatch.setattr(cli, 'load_collection_plan', lambda _value: plan)
    monkeypatch.setattr(cli, 'resolve_investor_flow_collector', lambda provider, symbols: captured.update(provider=provider, symbols=symbols) or 'collector')
    monkeypatch.setattr(cli, 'collect_planned_investor_flow', lambda **kwargs: captured.update(kwargs) or SimpleNamespace(receipts={}, content_hash='c' * 64))
    assert cli.main(['collect', '--plan-id', 'p', '--investor-flow-provider', 'kiwoom', '--bronze-root', str(tmp_path / 'bronze'), '--checkpoint-root', str(tmp_path / 'checkpoints'), '--retrieved-at', '2026-09-11T00:00:00+00:00']) == 0
    assert captured['provider'] == 'kiwoom'
    assert captured['symbols'] == ('005930',)
    assert captured['collector'] == 'collector'


def test_cli_build_gold_wires_complete_explicit_dataset_binding(tmp_path, monkeypatch, capsys) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace
    import src.data.cli as cli
    from src.data.schemas import SilverTable

    captured = {}
    bindings = {table: f'id-{table.value}' for table in SilverTable}
    monkeypatch.setattr(cli, 'parse_silver_dataset_bindings', lambda _values: bindings)
    monkeypatch.setattr(cli, 'resolve_gold_dataset_bindings', lambda **_kwargs: bindings)
    monkeypatch.setattr(cli, 'write_gold_input_binding_artifact', lambda **_kwargs: captured.setdefault('artifact', tmp_path / 'binding.json'))
    monkeypatch.setattr(cli, 'load_gold_window_inputs', lambda **kwargs: captured.update(load=kwargs) or SimpleNamespace(calendar=object(), security_master=object(), daily_market=object(), financial_facts=object(), corporate_actions=object(), investor_flow=object(), silver_dataset_ids=bindings))
    manifest = SimpleNamespace(manifest_hash='h', warmup=SimpleNamespace(warmup_ok=True, warmup_sessions_found=60), bar_audit=(), dart_eligibility=(), ca_excluded_instrument_ids=frozenset(), eligible_instrument_ids=frozenset())
    monkeypatch.setattr('src.data.gold.materialize_gold_window', lambda **_kwargs: SimpleNamespace(manifest=manifest, universe_decisions_count=1, eligible_decisions_count=1, feature_rows_count=1, universe_path='u', features_path='f', summary_artifact_path='s'))

    args = ['build-gold', '--silver-root', str(tmp_path), '--artifact-root', str(tmp_path), '--decision-time', datetime(2026, 9, 11, tzinfo=UTC).isoformat(), '--validation-start', '2016-01-04', '--validation-end', '2016-12-29']
    for table, dataset_id in bindings.items():
        args.extend(['--silver-dataset-id', f'{table.value}={dataset_id}'])
    assert cli.main(args) == 0
    assert captured['load']['silver_dataset_ids'] == bindings
    assert captured['artifact'] == tmp_path / 'binding.json'
    assert 'eligible_instruments' in capsys.readouterr().out
def test_cli_audit_provenance_emits_counts(tmp_path, capsys) -> None:
    from src.data import cli

    assert cli.main(['audit-provenance', '--bronze-root', str(tmp_path / 'bronze'), '--silver-root', str(tmp_path / 'silver'), '--artifact-root', str(tmp_path / 'artifacts')]) == 0
    assert 'artifact_path' in capsys.readouterr().out


def test_cli_audit_provenance_rejects_malformed_evidence(tmp_path) -> None:
    from src.data import cli

    evidence = tmp_path / 'bronze' / 'investor_flow' / 'bad'
    evidence.mkdir(parents=True)
    (evidence / 'payload.json').write_text('{')
    assert cli.main(['audit-provenance', '--bronze-root', str(tmp_path / 'bronze'), '--silver-root', str(tmp_path / 'silver'), '--artifact-root', str(tmp_path / 'artifacts')]) == 1


def test_filter_unresolved_lifecycle_events_preserves_only_verified_receipts() -> None:
    import polars as pl

    from src.data.cli import _filter_unresolved_lifecycle_events

    frame = pl.DataFrame(
        {
            'instrument_id': ['KRX:A', 'KRX:B', 'KRX:B'],
            'evidence_status': ['verified', 'unresolved', 'unresolved'],
        }
    )
    filtered, excluded = _filter_unresolved_lifecycle_events(frame)
    assert filtered['instrument_id'].to_list() == ['KRX:A']
    assert excluded == ('KRX:B',)


def test_run_backtest_compacts_security_master_before_session_build(tmp_path, monkeypatch) -> None:
    """CLI 는 전량 스냅샷 마스터를 SCD2 구간으로 압축한 뒤 세션 빌더에 전달해야 한다."""
    from argparse import Namespace
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    import src.data.cli as cli_mod
    import src.data.gold_artifacts as gold_artifacts_mod
    import src.data.master_intervals as master_intervals_mod
    import src.data.silver as silver_mod
    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest, write_legacy_backtest_run_manifest
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import SilverTable

    kst = ZoneInfo("Asia/Seoul")
    all_days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(70))
    calendar_df = pl.DataFrame({"session": list(all_days)})
    start, end = all_days[60].date().isoformat(), all_days[64].date().isoformat()
    warmup_days = all_days[0:68]
    closes_a = [10000.0 + index * 10.0 for index in range(len(warmup_days))]
    closes_b = [20000.0 + index * 5.0 for index in range(len(warmup_days))]
    market_df = pl.DataFrame({
        "session": [day for day in warmup_days for _ in ("KRX:A", "KRX:B")],
        "instrument_id": ["KRX:A", "KRX:B"] * len(warmup_days),
        "open": [c - 5.0 for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        "close": [c for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        "volume": [1000.0] * (2 * len(warmup_days)),
        "trading_value": [c * 1000.0 for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        "market_cap": [1e12, 5e11] * len(warmup_days),
        "available_at": [day.replace(hour=15, minute=30) for day in warmup_days for _ in ("KRX:A", "KRX:B")],
    })
    dm_dir = tmp_path / "dm"
    dm_dir.mkdir()
    market_df.write_parquet(dm_dir / "daily.parquet")

    # 일자 스냅샷 마스터: 두 종목 x 70 세션 = 140행, 속성 불변
    master_df = pl.DataFrame({
        "instrument_id": ["KRX:A", "KRX:B"] * len(all_days),
        "sector": ["Technology", "Healthcare"] * len(all_days),
        "status": ["listed"] * (2 * len(all_days)),
        "valid_from": [day for day in all_days for _ in ("KRX:A", "KRX:B")],
        "valid_to": [day for day in all_days for _ in ("KRX:A", "KRX:B")],
        "available_at": [day for day in all_days for _ in ("KRX:A", "KRX:B")],
        "source_hash": ["h"] * (2 * len(all_days)),
    })

    def fake_load_table(silver_root, table):
        if table == SilverTable.CALENDAR:
            return calendar_df
        if table == SilverTable.SECURITY_MASTER:
            return master_df
        return pl.DataFrame(schema={
            "effective_session": pl.Datetime(time_zone="Asia/Seoul"),
            "instrument_id": pl.String,
            "action_type": pl.String,
            "available_at": pl.Datetime(time_zone="Asia/Seoul"),
        })

    def fake_load_by_id(*, root, table, dataset_id, decision_time):
        return fake_load_table(root, table)

    universe_df = pl.DataFrame({
        "decision_session": [day.replace(hour=15, minute=30) for day in all_days[60:65] for _ in ("KRX:A", "KRX:B")],
        "instrument_id": ["KRX:A", "KRX:B"] * 5,
        "eligible": [True] * 10,
    })

    observed: dict[str, int] = {}
    real_compact = master_intervals_mod.compact_security_master_intervals

    def spy_compact(master, *, sessions):
        observed["input_rows"] = master.height
        result = real_compact(master, sessions=sessions)
        observed["output_rows"] = result.height
        return result

    monkeypatch.setattr(cli_mod, "compact_security_master_intervals", spy_compact)
    monkeypatch.setattr(silver_mod, "load_silver_table_by_dataset_id", fake_load_by_id)
    monkeypatch.setattr(
        silver_mod, "silver_dataset_path_by_id",
        lambda *, root, table, dataset_id, decision_time: dm_dir,
    )
    monkeypatch.setattr(
        gold_artifacts_mod, "resolve_gold_artifact_bundle",
        lambda *, gold_root, dataset_id, decision_time: object(),
    )
    monkeypatch.setattr(
        gold_artifacts_mod, "load_gold_artifact_frames",
        lambda *, bundle, decision_time: (universe_df, pl.DataFrame(), pl.DataFrame()),
    )

    run_manifest_obj = build_legacy_backtest_run_manifest(
        silver_root=tmp_path / "silver",
        gold_root=tmp_path / "gold",
        silver_dataset_ids={table: f"{table.value}-id" for table in SilverTable},
        gold_dataset_id="gold-test",
        validation_start=__import__("datetime").date.fromisoformat(start),
        validation_end=__import__("datetime").date.fromisoformat(end),
        strategy_id="core-v1",
        policy_versions={"market_inputs": "korean-equity-market-inputs-v2"},
    )
    manifest_path = write_legacy_backtest_run_manifest(
        manifest=run_manifest_obj, artifact_root=tmp_path / "artifacts"
    )

    code = _dispatch_backtest(Namespace(
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        gold_root=tmp_path / "gold",
        validation_start=start,
        validation_end=end,
        smoke_symbol=None,
        gold_dataset_id="gold-test",
        backtest_run_manifest=manifest_path,
        strategy_id="core-v1",
        initial_cash=100_000_000.0,
        scenario="base",
        ledger_id="core-test-2016",
    ))

    assert code == 0
    assert observed["input_rows"] == 2 * len(all_days)
    assert observed["output_rows"] == 2
    manifests = list((tmp_path / "artifacts" / "backtests").rglob("result.json"))
    assert len(manifests) == 1


def test_run_backtest_champion_rejects_uninformative_gold_scores(tmp_path, monkeypatch) -> None:
    """champion-v1 은 champion_score 가 전부 null 인 Gold 를 fail-closed 로 거부해야 한다."""
    from argparse import Namespace
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    import src.data.gold_artifacts as gold_artifacts_mod
    import src.data.silver as silver_mod
    from src.data.backtest_run_manifest import build_legacy_backtest_run_manifest, write_legacy_backtest_run_manifest
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError, SilverTable

    kst = ZoneInfo("Asia/Seoul")
    all_days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(70))
    calendar_df = pl.DataFrame({"session": list(all_days)})
    start, end = all_days[60].date().isoformat(), all_days[64].date().isoformat()

    def fake_load_by_id(*, root, table, dataset_id, decision_time):
        if table == SilverTable.CALENDAR:
            return calendar_df
        return pl.DataFrame()

    sessions = [day.replace(hour=15, minute=30) for day in all_days[60:65]]
    scores_df = pl.DataFrame({
        "decision_session": sessions,
        "instrument_id": ["KRX:A"] * 5,
        "eligible": [False] * 5,
        "champion_score": [None] * 5,
        "rank": [None] * 5,
        "exclusion_reasons": ["missing_value"] * 5,
        "feature_policy_version": ["champion-v1-qvef-v1"] * 5,
        "score_policy_version": ["champion-v1-scoring-v1"] * 5,
    }, schema_overrides={"champion_score": pl.Float64, "rank": pl.Int64})

    monkeypatch.setattr(silver_mod, "load_silver_table_by_dataset_id", fake_load_by_id)
    monkeypatch.setattr(
        gold_artifacts_mod, "resolve_gold_artifact_bundle",
        lambda *, gold_root, dataset_id, decision_time: object(),
    )
    monkeypatch.setattr(
        gold_artifacts_mod, "load_gold_artifact_frames",
        lambda *, bundle, decision_time: (pl.DataFrame(), pl.DataFrame(), scores_df),
    )

    run_manifest_obj = build_legacy_backtest_run_manifest(
        silver_root=tmp_path / "silver",
        gold_root=tmp_path / "gold",
        silver_dataset_ids={table: f"{table.value}-id" for table in SilverTable},
        gold_dataset_id="gold-test",
        validation_start=__import__("datetime").date.fromisoformat(start),
        validation_end=__import__("datetime").date.fromisoformat(end),
        strategy_id="champion-v1",
        policy_versions={"market_inputs": "korean-equity-market-inputs-v2"},
    )
    manifest_path = write_legacy_backtest_run_manifest(
        manifest=run_manifest_obj, artifact_root=tmp_path / "artifacts"
    )

    with pytest.raises(PITDataError, match="champion_score"):
        _dispatch_backtest(Namespace(
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            gold_root=tmp_path / "gold",
            validation_start=start,
            validation_end=end,
            smoke_symbol=None,
            gold_dataset_id="gold-test",
            backtest_run_manifest=manifest_path,
            strategy_id="champion-v1",
            initial_cash=100_000_000.0,
            scenario="base",
            ledger_id="champion-test-2016",
        ))


def test_build_investor_flow_silver_command_emits_dataset_counts(tmp_path, capsys) -> None:
    import hashlib
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    universe = runtime.workspace.silver_root / "ordinary_universe_cli"
    universe.mkdir(parents=True, exist_ok=True)
    (universe / "manifest.json").write_text(
        json.dumps({
            "dataset_id": universe.name,
            "policy_version": "krx-ordinary-equity-v1",
            "partitions": [{"session": "2026-03-04"}, {"session": "2026-03-05"}],
        }),
        encoding="utf-8",
    )
    row = {
        "date": "20260304",
        "tjj0000": "-100", "tjj0001": "-50", "tjj0002": "-30", "tjj0003": "-20",
        "tjj0004": "-10", "tjj0005": "-10", "tjj0006": "-8", "tjj0007": "100",
        "tjj0008": "927", "tjj0009": "-800", "tjj0010": "-28", "tjj0011": "29",
        "tjj0016": "-828", "tjj0017": "129", "tjj0018": "-228",
        "close": "50000", "volume": "10000", "value": "500",
    }
    payload = {
        "provider": "LS", "endpoint": "frgr-itt", "symbol": "005930", "anchor": "2026-03-04",
        "query": {"symbol": "005930", "start": "2026-03-04", "end": "2026-03-04"},
        "rows": [row], "records": [],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    page_dir = runtime.workspace.bronze_root / "investor_flow" / hashlib.sha256(raw).hexdigest()
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "payload.json").write_bytes(raw)
    assert main([
        "build-investor-flow-silver",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--workers", "1",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("investor_flow_")
    assert emitted["rows"] == 1


def test_build_daily_market_silver_command_emits_dataset_counts(tmp_path, capsys) -> None:
    import hashlib
    import json

    from src.data.cli import main
    from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
    from src.data.runtime import load_data_runtime
    from datetime import UTC, date, datetime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    universe = runtime.workspace.silver_root / "ordinary_universe_cli"
    universe.mkdir(parents=True, exist_ok=True)
    (universe / "manifest.json").write_text(
        json.dumps({
            "dataset_id": universe.name,
            "policy_version": "krx-ordinary-equity-v1",
            "partitions": [{"session": "2026-03-04"}],
        }),
        encoding="utf-8",
    )
    record = {
        "ISU_CD": "KR7005930003", "ISU_SRT_CD": "005930", "MKT_NM": "KOSPI", "BAS_DD": "20260304",
        "TDD_OPNPRC": "10500", "TDD_HGPRC": "11200", "TDD_LWPRC": "10300", "TDD_CLSPRC": "11000",
        "CMPPREVDD_PRC": "1000", "FLUC_RT": "10.0", "ACC_TRDVOL": "1000", "ACC_TRDVAL": "11000000",
        "MKTCAP": "660000000000", "LIST_SHRS": "60000000",
    }
    raw = json.dumps({"session": "2026-03-04", "records": [record]}, sort_keys=True).encode("utf-8")
    page_path = tmp_path / "krx-page.json"
    page_path.write_bytes(raw)
    ReceiptCatalog(runtime.workspace.bronze_root / "catalog").publish([
        ReceiptIndexEntry(
            source="krx_daily_market",
            natural_key="2026-03-04",
            as_of=date(2026, 3, 4),
            fiscal_period=None,
            status=EvidenceStatus.SUCCESS,
            content_hash=hashlib.sha256(raw).hexdigest(),
            retrieved_at=datetime(2026, 3, 5, tzinfo=UTC),
            payload_path=page_path,
        )
    ])
    assert main([
        "build-daily-market-silver",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("daily_market_")
    assert emitted["rows"] == 1


def test_build_market_panel_command_emits_dataset_counts(tmp_path, capsys) -> None:
    import hashlib
    import json
    from datetime import date, datetime

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]

    def daily_row(session: date, ticker: str, close: int, change: int, volume: int) -> dict[str, object]:
        return {
            "session": session, "instrument_id": f"KRX:{ticker}", "ticker": ticker, "market": "KOSPI",
            "open": close, "high": close, "low": close, "close": close, "change": change,
            "base_price": close - change, "volume": volume, "trading_value": close * volume,
            "market_cap": close * 1000, "listed_shares": 1000, "price_state": "tradable",
            "invalid_reason": None,
            "available_at": datetime(session.year, session.month, session.day, 18, 0, tzinfo=KRX_TZ),
            "source_hash": "a" * 64, "policy_version": "krx-daily-market-v1",
        }

    def tickers_for(session: date) -> list[str]:
        return ["005930", "000660"] if session != sessions[2] else ["005930"]

    daily_dir = runtime.workspace.silver_root / "daily_market_cli"
    universe_dir = runtime.workspace.silver_root / "ordinary_universe_cli"
    daily_parts = []
    universe_parts = []
    for session in sessions:
        tickers = tickers_for(session)
        rows = [
            daily_row(session, ticker, 10000 if ticker == "005930" else 5000, 0,
                      0 if (ticker, session) == ("000660", sessions[1]) else 100)
            for ticker in tickers
        ]
        rel = f"session={session.isoformat()}/part.parquet"
        out = daily_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(rows).sort("ticker")
        frame.write_parquet(out)
        daily_parts.append({"session": session.isoformat(), "path": rel, "source_hash": "b" * 64,
                            "row_count": frame.height,
                            "parquet_sha256": hashlib.sha256(out.read_bytes()).hexdigest()})
        urows = [{"instrument_id": f"KRX:{ticker}", "ticker": ticker, "eligible": True, "exclusion_reason": "eligible"}
                 for ticker in tickers]
        urel = f"session={session.isoformat()}/part.parquet"
        uout = universe_dir / urel
        uout.parent.mkdir(parents=True, exist_ok=True)
        uframe = pl.DataFrame(urows).sort("instrument_id")
        uframe.write_parquet(uout)
        universe_parts.append({"session": session.isoformat(), "path": urel, "source_hash": "c" * 64,
                               "row_count": uframe.height,
                               "parquet_sha256": hashlib.sha256(uout.read_bytes()).hexdigest()})
    (daily_dir / "manifest.json").write_text(json.dumps({"dataset_id": daily_dir.name, "policy_version": "krx-daily-market-v1", "partitions": daily_parts}), encoding="utf-8")
    (universe_dir / "manifest.json").write_text(json.dumps({"dataset_id": universe_dir.name, "policy_version": "krx-ordinary-equity-v1", "partitions": universe_parts}), encoding="utf-8")
    assert main([
        "build-market-panel",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--daily-market-dataset-id", daily_dir.name,
        "--universe-dataset-id", universe_dir.name,
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("market_panel_")
    assert emitted["rows"] == 5
    assert emitted["exits_halted"] == 1


def test_build_market_panel_command_forwards_instrument_buckets(tmp_path, capsys) -> None:
    import hashlib
    import json
    from datetime import date, datetime

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]

    def daily_row(session: date, ticker: str) -> dict[str, object]:
        return {
            "session": session, "instrument_id": f"KRX:{ticker}", "ticker": ticker, "market": "KOSPI",
            "open": 10000, "high": 10000, "low": 10000, "close": 10000, "change": 0,
            "base_price": 10000, "volume": 100, "trading_value": 1000000,
            "market_cap": 10000000, "listed_shares": 1000, "price_state": "tradable",
            "invalid_reason": None,
            "available_at": datetime(session.year, session.month, session.day, 18, 0, tzinfo=KRX_TZ),
            "source_hash": "a" * 64, "policy_version": "krx-daily-market-v1",
        }

    daily_dir = runtime.workspace.silver_root / "daily_market_buckets"
    universe_dir = runtime.workspace.silver_root / "ordinary_universe_buckets"
    daily_parts = []
    universe_parts = []
    for session in sessions:
        rows = [daily_row(session, ticker) for ticker in ("005930", "000660")]
        rel = f"session={session.isoformat()}/part.parquet"
        out = daily_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(rows).sort("ticker")
        frame.write_parquet(out)
        daily_parts.append({"session": session.isoformat(), "path": rel, "source_hash": "b" * 64,
                            "row_count": frame.height,
                            "parquet_sha256": hashlib.sha256(out.read_bytes()).hexdigest()})
        urows = [{"instrument_id": f"KRX:{ticker}", "ticker": ticker, "eligible": True, "exclusion_reason": "eligible"}
                 for ticker in ("005930", "000660")]
        urel = f"session={session.isoformat()}/part.parquet"
        uout = universe_dir / urel
        uout.parent.mkdir(parents=True, exist_ok=True)
        uframe = pl.DataFrame(urows).sort("instrument_id")
        uframe.write_parquet(uout)
        universe_parts.append({"session": session.isoformat(), "path": urel, "source_hash": "c" * 64,
                               "row_count": uframe.height,
                               "parquet_sha256": hashlib.sha256(uout.read_bytes()).hexdigest()})
    (daily_dir / "manifest.json").write_text(json.dumps({"dataset_id": daily_dir.name, "policy_version": "krx-daily-market-v1", "partitions": daily_parts}), encoding="utf-8")
    (universe_dir / "manifest.json").write_text(json.dumps({"dataset_id": universe_dir.name, "policy_version": "krx-ordinary-equity-v1", "partitions": universe_parts}), encoding="utf-8")
    assert main([
        "build-market-panel",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--daily-market-dataset-id", daily_dir.name,
        "--universe-dataset-id", universe_dir.name,
        "--instrument-buckets", "4",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("market_panel_")
    assert Path(emitted["dataset_path"]).exists()
    assert main([
        "build-market-panel",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--daily-market-dataset-id", daily_dir.name,
        "--universe-dataset-id", universe_dir.name,
    ]) == 0
    default_emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"] == default_emitted["dataset_id"]


def test_build_reference_benchmarks_command_lists_all_ids(tmp_path, capsys) -> None:
    import hashlib
    import json
    from datetime import date

    import polars as pl

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]
    panel_dir = runtime.workspace.gold_root / "market_panel_cli"
    partitions = []
    for year, days in ((2020, sessions),):
        rows = []
        for session in days:
            for ticker, cap, ret in (("005930", 300, 0.04), ("000660", 100, 0.0)):
                rows.append({
                    "session": session, "instrument_id": f"KRX:{ticker}", "eligible": True,
                    "price_state": "tradable", "adtv20": 2_000_000_000.0,
                    "market_cap": cap, "ret_price": 0.0 if session == sessions[0] else ret,
                })
        rel = f"year={year}/part.parquet"
        out = panel_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(rows).sort(["instrument_id", "session"])
        frame.write_parquet(out)
        partitions.append({"year": year, "path": rel, "row_count": frame.height,
                           "parquet_sha256": hashlib.sha256(out.read_bytes()).hexdigest()})
    (panel_dir / "manifest.json").write_text(
        json.dumps({"dataset_id": panel_dir.name, "policy_version": "krx-market-panel-v2",
                    "partitions": partitions}), encoding="utf-8")
    assert main([
        "build-reference-benchmarks",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--market-panel-dataset-id", panel_dir.name,
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("reference_benchmarks_")
    assert sorted(emitted["benchmarks"]) == [
        "eligible_cw_pr", "eligible_ew_pr", "liquid1b_cw_pr", "liquid1b_ew_pr",
    ]


def test_compact_storage_generations_dry_run_reports_without_deleting(tmp_path, capsys) -> None:
    import hashlib
    import json

    from src.data.cli import main
    from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
    from src.data.runtime import load_data_runtime
    from datetime import UTC, date, datetime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")

    def _publish(day: str, content: str) -> None:
        page_path = tmp_path / f"page-{day}.json"
        raw = content.encode("utf-8")
        page_path.write_bytes(raw)
        ReceiptCatalog(runtime.workspace.bronze_root / "catalog").publish([
            ReceiptIndexEntry(
                source="krx_daily_market",
                natural_key=day,
                as_of=date.fromisoformat(day),
                fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                content_hash=hashlib.sha256(raw).hexdigest(),
                retrieved_at=datetime(2026, 3, 5, tzinfo=UTC),
                payload_path=page_path,
            )
        ])

    _publish("2026-03-04", "revision-one")
    _publish("2026-03-05", "revision-two")

    table_dir = runtime.workspace.silver_root / "financial_facts"
    for name, ts in (("g1", "2026-01-01T00:00:00+00:00"), ("g2", "2026-02-01T00:00:00+00:00")):
        generation = table_dir / name
        generation.mkdir(parents=True)
        (generation / "dataset_manifest.json").write_text(
            json.dumps({"generated_time": ts, "content_hash": name}), encoding="utf-8"
        )
        (generation / "part.parquet").write_bytes(b"0" * 64)

    catalog_root = runtime.workspace.bronze_root / "catalog"
    revisions_before = {p.name for p in catalog_root.glob("*.json") if p.name != "latest.json"}
    assert len(revisions_before) == 2

    assert main([
        "compact-storage-generations",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["applied"] is False
    assert emitted["bytes_freed"] == 0
    assert emitted["catalog"]["reclaimable_revisions"]
    assert emitted["tables"][0]["reclaimable_generations"] == ["g1"]
    assert {p.name for p in catalog_root.glob("*.json") if p.name != "latest.json"} == revisions_before
    assert (table_dir / "g1").exists()


def test_compact_storage_generations_apply_deletes_and_reports_freed_bytes(tmp_path, capsys) -> None:
    import hashlib
    import json

    from src.data.cli import main
    from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
    from src.data.runtime import load_data_runtime
    from datetime import UTC, date, datetime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")

    def _publish(day: str, content: str) -> None:
        page_path = tmp_path / f"page-{day}.json"
        raw = content.encode("utf-8")
        page_path.write_bytes(raw)
        ReceiptCatalog(runtime.workspace.bronze_root / "catalog").publish([
            ReceiptIndexEntry(
                source="krx_daily_market",
                natural_key=day,
                as_of=date.fromisoformat(day),
                fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                content_hash=hashlib.sha256(raw).hexdigest(),
                retrieved_at=datetime(2026, 3, 5, tzinfo=UTC),
                payload_path=page_path,
            )
        ])

    _publish("2026-03-04", "revision-one")
    _publish("2026-03-05", "revision-two")

    table_dir = runtime.workspace.silver_root / "financial_facts"
    for name, ts in (("g1", "2026-01-01T00:00:00+00:00"), ("g2", "2026-02-01T00:00:00+00:00")):
        generation = table_dir / name
        generation.mkdir(parents=True)
        (generation / "dataset_manifest.json").write_text(
            json.dumps({"generated_time": ts, "content_hash": name}), encoding="utf-8"
        )
        (generation / "part.parquet").write_bytes(b"0" * 64)

    assert main([
        "compact-storage-generations",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--apply",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["applied"] is True
    assert emitted["bytes_freed"] > 0
    assert not (table_dir / "g1").exists()
    assert (table_dir / "g2").exists()


def _write_cli_gap_dataset(directory, frame) -> None:
    import hashlib
    import json

    directory.mkdir(parents=True, exist_ok=True)
    rel = Path("year=2024") / "part.parquet"
    out_path = directory / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": directory.name,
                "policy_version": "test-v1",
                "partitions": [
                    {
                        "path": str(rel),
                        "row_count": frame.height,
                        "parquet_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
                        "year": 2024,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_cli_gap_inputs(runtime, *, ls_cells, panel_cells) -> tuple:
    import polars as pl

    panel = runtime.workspace.gold_root / "market_panel_cli"
    ls_flow = runtime.workspace.silver_root / "investor_flow_cli"
    _write_cli_gap_dataset(
        panel,
        pl.DataFrame(
            {
                "session": [session for session, _ in panel_cells],
                "instrument_id": [f"KRX:{ticker}" for _, ticker in panel_cells],
                "ticker": [ticker for _, ticker in panel_cells],
                "eligible": [True] * len(panel_cells),
                "price_state": ["tradable"] * len(panel_cells),
            },
            schema={
                "session": pl.Date,
                "instrument_id": pl.String,
                "ticker": pl.String,
                "eligible": pl.Boolean,
                "price_state": pl.String,
            },
        ),
    )
    _write_cli_gap_dataset(
        ls_flow,
        pl.DataFrame(
            {"session": [session for session, _ in ls_cells], "ticker": [ticker for _, ticker in ls_cells]},
            schema={"session": pl.Date, "ticker": pl.String},
        ),
    )
    return panel, ls_flow


def _write_cli_kis_page(bronze_root, symbol, anchor, rows) -> None:
    import hashlib
    import json

    payload = {
        "provider": "KIS",
        "endpoint": "investor-trade-by-stock-daily",
        "symbol": symbol,
        "anchor": anchor.isoformat(),
        "query": {"symbol": symbol, "anchor": anchor.isoformat()},
        "rows": rows,
        "records": [],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    page_dir = bronze_root / "investor_flow" / hashlib.sha256(raw).hexdigest()
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "payload.json").write_bytes(raw)


def test_backfill_kis_investor_flow_gap_reports_attempted_symbols(tmp_path, monkeypatch, capsys) -> None:
    import json
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    s1, s2 = date(2024, 1, 2), date(2024, 1, 3)
    panel, ls_flow = _write_cli_gap_inputs(
        runtime,
        ls_cells=[(s1, "000001"), (s1, "000002")],
        panel_cells=[(s1, "000001"), (s2, "000001"), (s1, "000002"), (s2, "000002")],
    )
    seen: dict[str, object] = {}

    class StubCollector:
        def __init__(self, symbols) -> None:
            seen["symbols"] = tuple(symbols)

        def fetch_investor_flow(self, start, end, *, bronze_root=None, retrieved_at=None, symbols=None):
            seen.setdefault("calls", []).append((start, end, symbols))
            yield {"provider": "KIS", "symbol": symbols[0], "anchor": end.isoformat(), "records": []}

    monkeypatch.setattr("src.integrations.kis.investor_flow.KisInvestorFlowCollector", StubCollector)
    assert main([
        "backfill-kis-investor-flow-gap",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--market-panel-dataset-id", panel.name,
        "--ls-flow-dataset-id", ls_flow.name,
        "--pace-seconds", "0",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["symbols_attempted"] == 2
    assert emitted["target_cells"] == 2
    assert seen["symbols"] == ("000001", "000002")


def test_backfill_kis_investor_flow_gap_with_empty_gap(tmp_path, monkeypatch, capsys) -> None:
    import json
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    s1 = date(2024, 1, 2)
    panel, ls_flow = _write_cli_gap_inputs(
        runtime, ls_cells=[(s1, "000001")], panel_cells=[(s1, "000001")]
    )

    def _forbidden(symbols):
        raise AssertionError("collector must not be constructed for an empty gap")

    monkeypatch.setattr("src.integrations.kis.investor_flow.KisInvestorFlowCollector", _forbidden)
    assert main([
        "backfill-kis-investor-flow-gap",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--market-panel-dataset-id", panel.name,
        "--ls-flow-dataset-id", ls_flow.name,
        "--pace-seconds", "0",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == {"symbols_attempted": 0, "target_cells": 0}


def test_build_investor_flow_kis_supplement_command_emits_coverage(tmp_path, capsys) -> None:
    import json
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    s1, s2, s3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
    panel, ls_flow = _write_cli_gap_inputs(
        runtime,
        ls_cells=[(s1, "000001")],
        panel_cells=[(s1, "000001"), (s2, "000001"), (s3, "000001")],
    )
    _write_cli_kis_page(
        runtime.workspace.bronze_root,
        "000001",
        s2,
        [{
            "stck_bsop_date": "20240103",
            "prsn_ntby_qty": "100",
            "frgn_ntby_qty": "-60",
            "orgn_ntby_qty": "-30",
            "etc_ntby_qty": "-10",
        }],
    )
    assert main([
        "build-investor-flow-kis-supplement",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--market-panel-dataset-id", panel.name,
        "--ls-flow-dataset-id", ls_flow.name,
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("investor_flow_kis_supplement_")
    assert emitted["filled_cells"] == 1
    assert emitted["still_missing_cells"] == 1


def test_build_investor_flow_union_command_emits_dataset(tmp_path, capsys) -> None:
    import json
    from datetime import date, datetime

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    s1, s2 = date(2024, 1, 2), date(2024, 1, 3)

    def _row(provider, session, ticker, values):
        return {
            "session": session,
            "instrument_id": f"KRX:{ticker}",
            "ticker": ticker,
            "provider": provider,
            "individual_net_shares": values[0],
            "foreign_net_shares": values[1],
            "institution_net_shares": values[2],
            "other_net_shares": values[3],
            "available_at": datetime(session.year, session.month, session.day, 8, 0, tzinfo=KRX_TZ),
            "source_hash": f"{provider}-{ticker}",
            "policy_version": "test-v1",
        }

    ls_row = _row("LS", s1, "000001", (100, -60, -30, -10)) | {"ls_close": 50000}
    ls_flow = runtime.workspace.silver_root / "investor_flow_cli_ls"
    _write_cli_gap_dataset(
        ls_flow,
        pl.DataFrame(
            [ls_row],
            schema={**dict.fromkeys(ls_row, pl.String), **{
                "session": pl.Date,
                "individual_net_shares": pl.Int64,
                "foreign_net_shares": pl.Int64,
                "institution_net_shares": pl.Int64,
                "other_net_shares": pl.Int64,
                "ls_close": pl.Int64,
                "available_at": pl.Datetime("us", "Asia/Seoul"),
            }},
        ),
    )
    kis_flow = runtime.workspace.silver_root / "investor_flow_kis_supplement_cli"
    _write_cli_gap_dataset(
        kis_flow,
        pl.DataFrame([_row("KIS", s2, "000002", (50, -30, -10, -10))]),
    )
    assert main([
        "build-investor-flow-union",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--ls-flow-dataset-id", ls_flow.name,
        "--kis-supplement-dataset-id", kis_flow.name,
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("investor_flow_")
    assert emitted["rows"] == 2
    assert emitted["ls_dataset_id"] == ls_flow.name
    assert emitted["kis_supplement_dataset_id"] == kis_flow.name


def _write_cli_universe_dataset(silver_root, rows) -> object:
    import polars as pl

    dataset = silver_root / "ordinary_universe_cli"
    part_dir = dataset / "session=2024-01-02"
    part_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema={"ticker": pl.String, "eligible": pl.Boolean}).write_parquet(
        part_dir / "part.parquet"
    )
    return dataset


class _StubIndustryCollector:
    calls: ClassVar[list] = []

    def __init__(self, symbols, client=None) -> None:
        self.symbols = tuple(symbols)

    def fetch_industry_classification(self, *, bronze_root, retrieved_at=None):
        type(self).calls.append(self.symbols)
        return [{"provider": "KIS", "symbol": symbol} for symbol in self.symbols]


def test_collect_industry_classification_from_symbols_file(tmp_path, monkeypatch, capsys) -> None:
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("005930\n 000660\n005930\n", encoding="utf-8")
    _StubIndustryCollector.calls = []
    monkeypatch.setattr(
        "src.integrations.kis.industry.KisIndustryCollector", _StubIndustryCollector
    )
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--symbols-from", str(symbols_file),
        "--pace-seconds", "0",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == {"symbols_requested": 2, "pages_collected": 2, "skipped_count": 0, "skipped": {}}
    assert _StubIndustryCollector.calls == [("005930",), ("000660",)]


def test_collect_industry_classification_defaults_to_universe(tmp_path, monkeypatch, capsys) -> None:
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    _write_cli_universe_dataset(
        runtime.workspace.silver_root,
        [
            {"ticker": "005930", "eligible": True},
            {"ticker": "000001", "eligible": False},
            {"ticker": "000660", "eligible": True},
        ],
    )
    _StubIndustryCollector.calls = []
    monkeypatch.setattr(
        "src.integrations.kis.industry.KisIndustryCollector", _StubIndustryCollector
    )
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--pace-seconds", "0",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == {"symbols_requested": 2, "pages_collected": 2, "skipped_count": 0, "skipped": {}}
    assert _StubIndustryCollector.calls == [("000660",), ("005930",)]


def test_collect_industry_classification_rejects_empty_symbols_file(tmp_path, capsys) -> None:
    import json

    from src.data.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("  \n", encoding="utf-8")
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--symbols-from", str(symbols_file),
    ]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_collect_industry_classification_requires_universe(tmp_path, capsys) -> None:
    import json

    from src.data.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_collect_industry_classification_requires_universe_partitions(tmp_path, capsys) -> None:
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    (runtime.workspace.silver_root / "ordinary_universe_cli").mkdir(parents=True)
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_collect_industry_classification_requires_eligible_tickers(tmp_path, capsys) -> None:
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    _write_cli_universe_dataset(
        runtime.workspace.silver_root, [{"ticker": "000001", "eligible": False}]
    )
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_build_industry_classification_silver_command_emits_dataset(tmp_path, capsys) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.bronze import BronzeStore
    from src.data.cli import main
    from src.data.runtime import load_data_runtime
    from src.data.schemas import EvidenceKind

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    collected_at = datetime(2024, 1, 3, 9, 0, tzinfo=UTC)
    payload = {
        "provider": "KIS",
        "endpoint": "inquire-price",
        "symbol": "005930",
        "collected_at": collected_at.isoformat(),
        "output": {"bstp_kor_isnm": "전기·전자", "rprs_mrkt_kor_name": "KOSPI"},
        "records": [{"ticker": "005930", "industry_name": "전기·전자", "market_name": "KOSPI"}],
    }
    BronzeStore(runtime.workspace.bronze_root).import_bytes(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
        kind=EvidenceKind.INDUSTRY,
        retrieved_at=collected_at,
        source_label="KIS:inquire-price:005930:2024-01-03",
    )
    assert main([
        "build-industry-classification-silver",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("industry_")
    assert emitted["rows"] == 1


class _FailingIndustryCollector:
    calls: ClassVar[list] = []
    failing: ClassVar[str] = "000660"

    def __init__(self, symbols, client=None) -> None:
        self.symbols = tuple(symbols)

    def fetch_industry_classification(self, *, bronze_root, retrieved_at=None):
        from src.data.schemas import PITDataError

        type(self).calls.append(self.symbols)
        symbol = self.symbols[0]
        if symbol == type(self).failing:
            raise PITDataError(f"KIS industry classification missing bstp_kor_isnm for {symbol}")
        return [{"provider": "KIS", "symbol": symbol}]


class _FailingStockCollector:
    calls: ClassVar[list] = []
    failing: ClassVar[str] = "000660"

    def __init__(self, symbols, client=None) -> None:
        self.symbols = tuple(symbols)

    def fetch_stock_classification(self, *, bronze_root, retrieved_at=None):
        from src.data.schemas import PITDataError

        type(self).calls.append(self.symbols)
        symbol = self.symbols[0]
        if symbol == type(self).failing:
            raise PITDataError(f"KIS stock classification missing valid std_idst_clsf_cd for {symbol}")
        return [{"provider": "KIS", "symbol": symbol}]


def _run_classification_command(tmp_path, monkeypatch, capsys, command, stub_attr, stub_cls):
    import json

    from src.data.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("005930\n000660\n035420\n", encoding="utf-8")
    stub_cls.calls = []
    monkeypatch.setattr(stub_attr, stub_cls)
    code = main([
        command,
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--symbols-from", str(symbols_file),
        "--pace-seconds", "0",
    ])
    return code, json.loads(capsys.readouterr().out)


def test_collect_industry_classification_isolates_failing_ticker(tmp_path, monkeypatch, capsys) -> None:
    """Isolation: one unclassifiable ticker must not discard the rest of the run."""
    code, emitted = _run_classification_command(
        tmp_path, monkeypatch, capsys,
        "collect-industry-classification",
        "src.integrations.kis.industry.KisIndustryCollector",
        _FailingIndustryCollector,
    )
    assert code == 0
    assert emitted["symbols_requested"] == 3
    assert emitted["pages_collected"] == 2
    assert emitted["skipped_count"] == 1
    assert list(emitted["skipped"]) == ["000660"]
    assert "000660" in emitted["skipped"]["000660"]
    assert _FailingIndustryCollector.calls == [("005930",), ("000660",), ("035420",)]


def test_collect_stock_classification_isolates_failing_ticker(tmp_path, monkeypatch, capsys) -> None:
    """Stock classification mirrors the per-ticker isolation."""
    code, emitted = _run_classification_command(
        tmp_path, monkeypatch, capsys,
        "collect-stock-classification",
        "src.integrations.kis.industry.KisStockClassificationCollector",
        _FailingStockCollector,
    )
    assert code == 0
    assert emitted["symbols_requested"] == 3
    assert emitted["pages_collected"] == 2
    assert emitted["skipped_count"] == 1
    assert list(emitted["skipped"]) == ["000660"]
    assert _FailingStockCollector.calls == [("005930",), ("000660",), ("035420",)]


def test_collect_classification_with_no_pages_fails_closed(tmp_path, monkeypatch, capsys) -> None:
    """A run that collected nothing must not look like success."""
    import json

    from src.data.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    for command, stub_attr, stub_cls in (
        ("collect-industry-classification", "src.integrations.kis.industry.KisIndustryCollector", _FailingIndustryCollector),
        ("collect-stock-classification", "src.integrations.kis.industry.KisStockClassificationCollector", _FailingStockCollector),
    ):
        symbols_file = tmp_path / f"symbols-{command}.txt"
        symbols_file.write_text("000660\n", encoding="utf-8")
        stub_cls.calls = []
        monkeypatch.setattr(stub_attr, stub_cls)
        assert main([
            command,
            "--scope-config", str(scope_config),
            "--data-root", str(tmp_path / "data"),
            "--symbols-from", str(symbols_file),
            "--pace-seconds", "0",
        ]) == 1
        assert "error" in json.loads(capsys.readouterr().out)


def test_collect_classification_non_pit_exception_propagates(tmp_path, monkeypatch) -> None:
    """Non-PIT exceptions are not swallowed as skips."""
    import pytest

    from src.data.cli import main

    class _BoomCollector:
        def __init__(self, symbols, client=None) -> None:
            self.symbols = tuple(symbols)

        def fetch_industry_classification(self, *, bronze_root, retrieved_at=None):
            if self.symbols[0] == "000660":
                raise RuntimeError("boom")
            return [{"provider": "KIS", "symbol": self.symbols[0]}]

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("005930\n000660\n", encoding="utf-8")
    monkeypatch.setattr("src.integrations.kis.industry.KisIndustryCollector", _BoomCollector)
    with pytest.raises(RuntimeError, match="boom"):
        main([
            "collect-industry-classification",
            "--scope-config", str(scope_config),
            "--data-root", str(tmp_path / "data"),
            "--symbols-from", str(symbols_file),
            "--pace-seconds", "0",
        ])


def test_collect_stock_classification_defaults_to_universe(tmp_path, monkeypatch, capsys) -> None:
    """Stock classification defaults to the ordinary-universe tickers."""
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    _write_cli_universe_dataset(
        runtime.workspace.silver_root,
        [
            {"ticker": "005930", "eligible": True},
            {"ticker": "000001", "eligible": False},
            {"ticker": "000660", "eligible": True},
        ],
    )
    _StubIndustryCollector.calls = []

    class _StubStockCollector:
        calls: ClassVar[list] = []

        def __init__(self, symbols, client=None) -> None:
            self.symbols = tuple(symbols)

        def fetch_stock_classification(self, *, bronze_root, retrieved_at=None):
            type(self).calls.append(self.symbols)
            return [{"provider": "KIS", "symbol": symbol} for symbol in self.symbols]

    monkeypatch.setattr(
        "src.integrations.kis.industry.KisStockClassificationCollector", _StubStockCollector
    )
    assert main([
        "collect-stock-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--pace-seconds", "0",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["symbols_requested"] == 2
    assert emitted["pages_collected"] == 2
    assert _StubStockCollector.calls == [("000660",), ("005930",)]


def test_collect_classification_logs_progress_every_hundred_symbols(tmp_path, caplog) -> None:
    """Progress logging fires on each 100-symbol boundary."""
    import logging

    from src.data.cli import _collect_classification_with_isolation

    class _StubCollector:
        def __init__(self, symbols, client=None) -> None:
            self.symbols = tuple(symbols)

        def fetch_industry_classification(self, *, bronze_root, retrieved_at=None):
            return [{"provider": "KIS", "symbol": self.symbols[0]}]

    symbols = tuple(f"{index:06d}" for index in range(100))
    with caplog.at_level(logging.INFO, logger="src.data.cli"):
        result = _collect_classification_with_isolation(
            stage="collect-industry-classification",
            collector_cls=_StubCollector,
            fetch_attr="fetch_industry_classification",
            bronze_root=tmp_path / "bronze",
            symbols=symbols,
            pace_seconds=0,
        )
    assert result["pages_collected"] == 100
    assert any("stage=collect-industry-classification" in message for message in caplog.messages)
