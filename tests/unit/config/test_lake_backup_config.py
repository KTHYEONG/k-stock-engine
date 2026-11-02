"""The state backup leaves out exactly the rebuildable build caches and keeps every authoritative state file."""

from __future__ import annotations

import inspect
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

from src.data import lake_backup as lb
from src.data import pipeline_graph as graph

_CONFIG = Path(__file__).resolve().parents[3] / "config" / "data" / "lake_backup.toml"

_AUTHORITATIVE = (
    "datasets.json",
    "decisions/chain.jsonl",
    "decisions/2026-10-30.json",
    "shadow/official/nav.parquet",
    "shadow/ideal/runs.jsonl",
    "operate/state.json",
    "quota/quota_state.json",
    "kind_relay/last_accepted.json",
    "dbsec_reconcile/2026-10-29.json",
    "research/models/abc/model.json",
    "dart_fact_quarantine_abc.json",
    "cache/unknown_future_store.sqlite",
)


def _state_source(root: Path) -> lb.BackupSource:
    config = tomllib.loads(_CONFIG.read_text(encoding="utf-8"))
    (row,) = [row for row in config["source"] if row["name"] == "state"]
    return lb.BackupSource("state", root, exclude=tuple(row["exclude"]))


def _cache_relpaths(state_root: Path) -> set[str]:
    ctx = SimpleNamespace(runtime=SimpleNamespace(workspace=SimpleNamespace(state_root=state_root)))
    namespaces = set(re.findall(r'_parse_cache_path\(ctx, "([a-z_]+)"\)', inspect.getsource(graph)))
    assert namespaces, "no parse cache call sites found in pipeline_graph"
    paths = {graph._parse_cache_path(ctx, ns) for ns in namespaces} | {graph._daily_flag_cache_path(ctx)}
    return {path.relative_to(state_root).as_posix() for path in paths}


def test_state_backup_excludes_every_build_cache_and_keeps_authoritative_state(tmp_path: Path) -> None:
    caches = _cache_relpaths(tmp_path)
    journals = {f"{rel}-journal" for rel in caches if rel.endswith(".sqlite")}
    for rel in (*caches, *journals, *_AUTHORITATIVE):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    backed_up = set(lb._iter_files(_state_source(tmp_path)))
    assert backed_up.isdisjoint(caches | journals)
    assert set(_AUTHORITATIVE) <= backed_up
