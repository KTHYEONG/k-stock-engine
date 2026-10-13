"""Relay CLI invariants: decodable output, failure hygiene, no payload leakage."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

FETCHED_AT = datetime(2026, 10, 13, 9, 0, tzinfo=UTC)
TODAY = date(2026, 10, 13)


def _bundle(marker: str = ""):  # type: ignore[no-untyped-def]
    from src.data.jobs.kind import collect_search_window
    from src.data.kind_relay import RelayBundle
    from tests.fixtures.kind_html import kind_row_html, kind_search_html

    class _Collector:
        def search_page(self, keyword, start, end, *, page_index):  # type: ignore[no-untyped-def]
            return kind_search_html(
                [kind_row_html("20261012000001", "2026-10-12 15:30", f"상장폐지{marker}", "005930",
                               "유가증권시장본부")],
                1,
            )

        def fetch_document(self, acptno):  # type: ignore[no-untyped-def]
            raise AssertionError("no documents expected")

    payload = collect_search_window(
        _Collector(), keyword="상장폐지", start=date(2026, 10, 1), end=TODAY,
        today=TODAY, retrieved_at=FETCHED_AT,
    )
    return RelayBundle(
        fetched_at=FETCHED_AT, fetch_date_kst=TODAY,
        producer={"run_id": "t", "document_failures": "0"}, items=(payload,),
    )


def _patch(monkeypatch, bundle=None, error=None):  # type: ignore[no-untyped-def]
    import src.ops.kind_relay as cli

    monkeypatch.setattr(cli, "build_scoped_kind_client", lambda **kwargs: object())
    if error is not None:
        def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise error

        monkeypatch.setattr(cli, "build_relay_bundle", _boom)
    else:
        monkeypatch.setattr(cli, "build_relay_bundle", lambda *args, **kwargs: bundle)


def test_produce_writes_decodable_bundle_and_summary(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    import src.ops.kind_relay as cli
    from src.data.kind_relay import decode_relay_bundle

    bundle = _bundle()
    _patch(monkeypatch, bundle=bundle)
    out = tmp_path / "bundle.json"
    assert cli.main(["produce", "--out", str(out)]) == 0
    decoded = decode_relay_bundle(out.read_bytes())
    assert [item.natural_key for item in decoded.items] == [item.natural_key for item in bundle.items]
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert set(summary) == {
        "fetched_at", "fetch_date_kst", "items", "search_windows",
        "documents", "document_failures", "bytes", "items_sha256",
    }
    assert (summary["items"], summary["search_windows"], summary["documents"]) == (1, 1, 0)


def test_failure_leaves_no_file(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    import src.ops.kind_relay as cli
    from src.core.pit import PITDataError

    _patch(monkeypatch, error=PITDataError("stale"))
    out = tmp_path / "bundle.json"
    out.write_bytes(b"stale")
    assert cli.main(["produce", "--out", str(out)]) == 1
    assert not out.exists()
    assert capsys.readouterr().out == ""


def test_summary_never_contains_payload(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    import src.ops.kind_relay as cli

    marker = "MARKER-7f3a9c-payload"
    _patch(monkeypatch, bundle=_bundle(marker))
    out = tmp_path / "bundle.json"
    assert cli.main(["produce", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert marker not in captured.out
    assert marker not in captured.err


def test_replace_failure_cleans_staged_payload(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    import src.ops.kind_relay as cli

    _patch(monkeypatch, bundle=_bundle())

    def _fail_replace(*args):  # type: ignore[no-untyped-def]
        raise OSError("replace failed")

    monkeypatch.setattr(cli.os, "replace", _fail_replace)
    out = tmp_path / "bundle.json"
    out.write_bytes(b"stale")
    assert cli.main(["produce", "--out", str(out)]) == 1
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().out == ""


def test_undecodable_bundle_is_not_published(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace
    import src.ops.kind_relay as cli

    bundle = _bundle()
    _patch(monkeypatch, bundle=replace(bundle, items=bundle.items * 2))
    out = tmp_path / "bundle.json"
    assert cli.main(["produce", "--out", str(out)]) == 1
    assert not out.exists()
    assert capsys.readouterr().out == ""
