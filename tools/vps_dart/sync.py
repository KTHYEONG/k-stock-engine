"""Local controller for remote DART collection (push / status / pull / finish / auto).

Run as a module from the repo root: ``python -m tools.vps_dart.sync push``.

The remote host collects under its own IP in a mirrored scope workspace,
keeps its own quota ledger, and this controller pulls verified pages, folds
the remote request count into the local ledger, and only then deletes what it
has verified remotely. The deployment mirrors the absolute data root, so the
job file's data root addresses the workspace on both ends.
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import load_provider_policy, load_runtime_config
from src.config.errors import ConfigError
from src.config.secrets import read_secret
from src.core.pit import PITDataError
from src.data.jobs.dart import resolve_dart_job
from src.data.jobs.runner import build_job_context
from src.data.runtime import resolve_data_runtime

REPO = Path(__file__).resolve().parents[2]

HOST = "or-vps"
REMOTE = "kse-collect"  # relative to the remote home
PACKAGES = ("polars", "pydantic", "requests", "numpy", "tenacity", "pyarrow", "exchange-calendars")
DEFAULT_JOB = "dart_facts"

SERVICE = """[Unit]
Description=Remote OpenDART fact collection worker

[Service]
Type=oneshot
EnvironmentFile=%h/{remote}/secrets.env
WorkingDirectory=%h/{remote}/repo
ExecStart=%h/{remote}/venv/bin/python -m tools.vps_dart.worker --root %h/{remote} --key-env {key}
Nice=10
MemoryMax=1500M
CPUQuota=60%
"""
TIMER = """[Unit]
Description=Run the remote OpenDART worker periodically

[Timer]
OnActiveSec=1min
OnUnitInactiveSec=15min
AccuracySec=10s

[Install]
WantedBy=timers.target
"""


def _ssh(command: str, *, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["ssh", "-o", "BatchMode=yes", HOST, command], input=stdin, text=True, capture_output=True, check=check, timeout=1800
    )


def _emit(**fields: object) -> None:
    sys.stdout.write(json.dumps(fields, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _runtime():  # type: ignore[no-untyped-def]
    return resolve_data_runtime()


def _job_context(runtime, provider, key_env: str):  # type: ignore[no-untyped-def]
    return build_job_context(runtime=runtime, provider=provider, key_env=key_env, collector=None)


def _policy(key_env: str) -> tuple[str, dict[str, Any]]:
    """Build the remote job policy snapshot for one declared key."""
    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    resolved = key_env or provider.default_key_env
    key_policy = provider.dart_key(resolved)
    policy: dict[str, Any] = {
        "daily_budget": key_policy.daily_budget,
        "daily_reserve": key_policy.daily_reserve,
        "min_interval_seconds": key_policy.min_interval_seconds,
        "chunk": provider.dart.batch_identities,
        "avoid_kst_windows": [list(window) for window in provider.dart.shared_ip_avoid_windows_kst],
    }
    return resolved, policy


def _estimate_pending(runtime, provider, resolved: str, job_name: str) -> int:  # type: ignore[no-untyped-def]
    spec = resolve_dart_job(job_name)
    return len(spec.pending(_job_context(runtime, provider, resolved)))


def push(job_name: str | None = None, key_env: str | None = None) -> None:
    from src.integrations.dart.client import dart_ledger_for_key

    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    resolved, policy_dict = _policy(key_env or "")
    name = job_name or DEFAULT_JOB
    resolve_dart_job(name)
    try:
        api_key = read_secret(resolved)
    except ConfigError as exc:
        raise SystemExit(f"{resolved} is not set (source ~/.quant_env.sh)") from exc
    runtime = _runtime()
    pending = _estimate_pending(runtime, provider, resolved, name)
    if not pending:
        _emit(stage="push", status="nothing_pending")
        return
    state_dir = runtime.workspace.state_root / "vps_dart"
    state_dir.mkdir(parents=True, exist_ok=True)
    data_root = runtime.workspace.root.resolve()
    scope_id = runtime.scope.scope_id
    remote_bronze = f"{data_root}/bronze/{scope_id}"
    remote_quota_state = f"{data_root}/state/{scope_id}/quota/quota_state.json"
    job = {
        "job": name,
        "scope_id": scope_id,
        "scope": runtime.scope.model_dump(mode="json"),
        "provider_policy": provider.model_dump(mode="json"),
        "data_root": str(data_root),
        "key_env": resolved,
        "primary_key_env": provider.primary_key_env,
        "policy": policy_dict,
        "chunk": provider.dart.batch_identities,
        "max_chunks": None,
        "dry_run": False,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (state_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")

    # 원격 원장을 오늘까지의 로컬 사용량으로 시드해, 두 곳 합계가 키의 일일 한도를 넘지 않게 한다.
    provider_name = dart_ledger_for_key(key_env=resolved, primary_key_env=provider.primary_key_env, api_key=api_key)
    quota_path = runtime.workspace.state_root / "quota" / "quota_state.json"
    local_state = json.loads(quota_path.read_text(encoding="utf-8")) if quota_path.is_file() else {}
    # 로컬의 일시 차단(blocked_until)은 이 호스트의 IP 상태이므로 원격에는 옮기지 않는다.
    seed = {k: {f: x for f, x in v.items() if f != "blocked_until"} for k, v in local_state.items() if k.startswith(f"{provider_name}|")}
    marks = {f"{k}|{v['daily_attempt_day']}": int(v["daily_attempted_requests"]) for k, v in seed.items() if v.get("daily_attempt_day")}
    (state_dir / "seed_quota.json").write_text(json.dumps(seed), encoding="utf-8")
    (state_dir / "fold_watermark.json").write_text(json.dumps(marks), encoding="utf-8")

    _ssh(f"mkdir -p ~/{REMOTE}/repo ~/.config/systemd/user {shlex.quote(remote_bronze)} {shlex.quote(str(Path(remote_quota_state).parent))}")
    subprocess.run(  # noqa: S603
        ["rsync", "-a", "--delete", "--exclude", "__pycache__", "-e", "ssh -o BatchMode=yes", str(REPO / "src") + "/", f"{HOST}:{REMOTE}/repo/src/"], check=True
    )
    _ssh(f"mkdir -p ~/{REMOTE}/repo/tools/vps_dart")
    subprocess.run(  # noqa: S603
        ["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(REPO / "tools" / "vps_dart" / "worker.py"), f"{HOST}:{REMOTE}/repo/tools/vps_dart/worker.py"], check=True
    )
    subprocess.run(  # noqa: S603
        ["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(state_dir / "job.json"), f"{HOST}:{REMOTE}/job.json"], check=True
    )
    subprocess.run(  # noqa: S603
        ["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(state_dir / "seed_quota.json"), f"{HOST}:{shlex.quote(remote_quota_state)}"], check=True
    )
    # 원격에 python3-venv가 없어, 수집 폴더 안에 자체 uv를 설치한다(폴더 삭제 시 함께 제거).
    _ssh(f"[ -x ~/{REMOTE}/bin/uv ] || (export UV_UNMANAGED_INSTALL=$HOME/{REMOTE}/bin; curl -LsSf https://astral.sh/uv/install.sh | sh)")
    _ssh(f"[ -x ~/{REMOTE}/venv/bin/python ] || ~/{REMOTE}/bin/uv venv --python /usr/bin/python3 ~/{REMOTE}/venv")
    _ssh(f"~/{REMOTE}/bin/uv pip install -q --python ~/{REMOTE}/venv/bin/python {' '.join(PACKAGES)}")
    _ssh(f"umask 077 && cat > ~/{REMOTE}/secrets.env", stdin=f"{resolved}={api_key}\n")
    _ssh(f"cd ~/{REMOTE}/repo && ~/{REMOTE}/venv/bin/python -c 'import src.data.remote_dart_worker, src.data.jobs.dart'")
    _ssh(f"cat > ~/.config/systemd/user/kse-dart-worker.service", stdin=SERVICE.format(remote=REMOTE, key=resolved))
    _ssh("cat > ~/.config/systemd/user/kse-dart-worker.timer", stdin=TIMER)
    _ssh("systemctl --user daemon-reload && systemctl --user enable --now kse-dart-worker.timer")
    _emit(stage="push", status="started", job=name, pending=pending, seeded_requests=sum(marks.values()))


def update() -> None:
    """Ship changed code to a running remote worker without touching its job, ledger, or pages."""
    for local, remote in ((REPO / "src") , "repo/src"), ((REPO / "tools" / "vps_dart" / "worker.py"), "repo/tools/vps_dart/worker.py"):
        source = str(local) + ("/" if local.is_dir() else "")
        target = f"{HOST}:{REMOTE}/{remote}" + ("/" if local.is_dir() else "")
        flags = ["--delete", "--exclude", "__pycache__"] if local.is_dir() else []
        subprocess.run(["rsync", "-a", *flags, "-e", "ssh -o BatchMode=yes", source, target], check=True)  # noqa: S603
    _emit(stage="update", status="code_synced")


def status() -> dict[str, object]:
    out = _ssh(
        f"cat ~/{REMOTE}/progress.json 2>/dev/null; echo; ls ~/{REMOTE}/COMPLETE 2>/dev/null; "
        f"du -sh ~/{REMOTE}/out 2>/dev/null | cut -f1; systemctl --user is-active kse-dart-worker.service kse-dart-worker.timer 2>/dev/null | tr '\\n' ' '",
        check=False,
    ).stdout.strip().splitlines()
    info = {"raw": out}
    _emit(stage="status", **info)
    return info


def _remote_workspace_roots(runtime) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    data_root = str(runtime.workspace.root.resolve())
    scope_id = runtime.scope.scope_id
    return f"{data_root}/bronze/{scope_id}", f"{data_root}/state/{scope_id}/quota/quota_state.json"


def pull() -> dict[str, object]:
    from src.data.receipt_catalog import ReceiptCatalog
    from src.data.remote_dart_inbox import fold_remote_quota, ingest_remote_bronze
    from src.data.scoped_ingestion import ScopedBronzeWriter
    from src.integrations.quota import ProviderQuotaStateStore

    runtime = _runtime()
    state_dir = runtime.workspace.state_root / "vps_dart"
    inbox = state_dir / "inbox"
    (inbox / "bronze").mkdir(parents=True, exist_ok=True)
    remote_bronze, remote_quota = _remote_workspace_roots(runtime)
    subprocess.run(  # noqa: S603
        ["rsync", "-a", "--partial", "-e", "ssh -o BatchMode=yes", f"{HOST}:{shlex.quote(remote_bronze)}/financial_facts/", str(inbox / "bronze" / "financial_facts") + "/"], check=True
    )
    subprocess.run(  # noqa: S603
        ["rsync", "-a", "--partial", "-e", "ssh -o BatchMode=yes", f"{HOST}:{shlex.quote(remote_bronze)}/dart_documents/", str(inbox / "bronze" / "dart_documents") + "/"], check=True
    )
    remote_quota_state = state_dir / "remote_quota_state.json"
    subprocess.run(  # noqa: S603
        ["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{HOST}:{shlex.quote(remote_quota)}", str(remote_quota_state)], check=False
    )
    bronze_root = runtime.workspace.bronze_root
    writer = ScopedBronzeWriter(runtime=runtime, catalog=ReceiptCatalog(bronze_root / "catalog"))
    result = ingest_remote_bronze(inbox_bronze=inbox / "bronze", bronze_root=bronze_root, writer=writer)
    store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    folded = fold_remote_quota(store=store, remote_state=remote_quota_state, watermark_path=state_dir / "fold_watermark.json")
    if result.accepted:
        # 로컬 등록과 해시 검증이 끝난 페이지만 원격에서 지운다.
        for name in result.accepted:
            kind, digest = name.split("/", 1)
            _ssh(f"rm -rf {shlex.quote(f'{remote_bronze}/{kind}/{digest}')}")
            shutil.rmtree(inbox / "bronze" / name, ignore_errors=True)
    summary: dict[str, object] = {"fact_pages": result.fact_pages, "document_pages": result.document_pages, "rejected": list(result.rejected), "quota_folded": folded}
    _emit(stage="pull", **summary)
    return summary


def is_complete() -> bool:
    return _ssh(f"test -f ~/{REMOTE}/COMPLETE", check=False).returncode == 0


def finish() -> None:
    summary = pull()
    if summary["rejected"]:
        _emit(stage="finish", status="refused", reason="rejected pages need inspection")
        return
    if not is_complete():
        _emit(stage="finish", status="refused", reason="remote worker is not complete")
        return
    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    runtime = _runtime()
    state_dir = runtime.workspace.state_root / "vps_dart"
    try:
        job = json.loads((state_dir / "job.json").read_text(encoding="utf-8"))
        spec = resolve_dart_job(str(job.get("job") or DEFAULT_JOB))
        remaining = spec.pending(_job_context(runtime, provider, str(job.get("key_env") or provider.default_key_env)))
    except (OSError, ValueError, KeyError, PITDataError) as exc:
        _emit(stage="finish", status="refused", reason=f"local plan is unreadable: {exc}")
        return
    if remaining:
        _emit(stage="finish", status="refused", reason="identities still unanswered locally", remaining=len(remaining))
        return
    _ssh(
        "systemctl --user disable --now kse-dart-worker.timer kse-dart-worker.service; "
        "rm -f ~/.config/systemd/user/kse-dart-worker.service ~/.config/systemd/user/kse-dart-worker.timer; "
        f"systemctl --user daemon-reload; rm -rf ~/{REMOTE}",
        check=False,
    )
    shutil.rmtree(runtime.workspace.state_root / "vps_dart" / "inbox", ignore_errors=True)
    _emit(stage="finish", status="cleaned", remote_removed=True)


def auto(interval_seconds: int) -> None:
    while True:
        try:
            pull()
            if is_complete():
                finish()
                return
        except (subprocess.SubprocessError, OSError) as exc:
            _emit(stage="auto", status="retry", error=str(exc)[:200])
        time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["push", "update", "status", "pull", "finish", "auto"])
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--job", type=str, default=None)
    parser.add_argument("--key-env", type=str, default=None)
    args = parser.parse_args()
    {"push": lambda: push(args.job, args.key_env), "update": update, "status": status, "pull": pull, "finish": finish, "auto": lambda: auto(args.interval)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
