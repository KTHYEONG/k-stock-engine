"""Local controller for the remote DART collection (push / status / pull / finish / auto).

The remote host collects under its own IP and quota ledger, keeps only immutable Bronze pages,
and this controller pulls them, verifies hashes, registers them in the local catalog, folds the
remote request count into the local ledger, and only then deletes what it has verified remotely.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

HOST = "or-vps"
REMOTE = "kse-collect"  # relative to the remote home
KEY_ENV = "OPENDART_API_KEY_2"
SCOPE = REPO / "config" / "research" / "kr_swing_2019_v1.toml"
DATA_ROOT = REPO / "data"
PACKAGES = ("polars", "pydantic", "requests", "python-dotenv", "numpy", "tenacity", "pyarrow", "exchange-calendars")
# k-closing-alpha가 같은 IP에서 21:35 KST에 DART를 조회하므로 그 전후는 피한다.
POLICY = {"daily_budget": 19_500, "daily_reserve": 500, "min_interval_seconds": 0.34, "chunk": 250, "avoid_kst_windows": [["21:25", "21:50"]]}

SERVICE = """[Unit]
Description=Remote OpenDART fact collection worker

[Service]
Type=oneshot
EnvironmentFile=%h/{remote}/secrets.env
WorkingDirectory=%h/{remote}/repo
ExecStart=%h/{remote}/venv/bin/python tools/vps_dart/worker.py --root %h/{remote} --key-env {key}
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
    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=SCOPE, data_root=DATA_ROOT)


def _pending_job_identities(runtime):  # type: ignore[no-untyped-def]
    import collect_dart_2016_2018_extension as ext
    from src.data.receipt_catalog import ReceiptCatalog

    bronze = runtime.workspace.bronze_root
    mapping = ext._ticker_by_corp_code(bronze, ext._eligible_tickers(runtime.workspace.silver_root))
    _, _, pending = ext._pending_identities(bronze, ReceiptCatalog(bronze / "catalog"), mapping)
    return pending


def push() -> None:
    from src.integrations.dart.client import dart_quota_provider
    from src.integrations.quota import ProviderQuotaStateStore

    api_key = os.environ.get(KEY_ENV)
    if not api_key:
        raise SystemExit(f"{KEY_ENV} is not set (source ~/.quant_env.sh)")
    runtime = _runtime()
    pending = _pending_job_identities(runtime)
    if not pending:
        _emit(stage="push", status="nothing_pending")
        return
    state_dir = runtime.workspace.state_root / "vps_dart"
    state_dir.mkdir(parents=True, exist_ok=True)
    job = {"identities": pending, "policy": POLICY, "created_at": datetime.now(UTC).isoformat()}
    (state_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")

    # 원격 원장을 오늘까지의 로컬 사용량으로 시드해, 두 곳 합계가 키의 일일 한도를 넘지 않게 한다.
    provider = dart_quota_provider(api_key)
    local_state = json.loads((runtime.workspace.state_root / "quota" / "quota_state.json").read_text(encoding="utf-8"))
    # 로컬의 일시 차단(blocked_until)은 이 호스트의 IP 상태이므로 원격에는 옮기지 않는다.
    seed = {k: {f: x for f, x in v.items() if f != "blocked_until"} for k, v in local_state.items() if k.startswith(f"{provider}|")}
    marks = {f"{k}|{v['daily_attempt_day']}": int(v["daily_attempted_requests"]) for k, v in seed.items() if v.get("daily_attempt_day")}
    (state_dir / "seed_quota.json").write_text(json.dumps(seed), encoding="utf-8")
    (state_dir / "fold_watermark.json").write_text(json.dumps(marks), encoding="utf-8")

    _ssh(f"mkdir -p ~/{REMOTE}/repo ~/{REMOTE}/state ~/{REMOTE}/out ~/.config/systemd/user")
    subprocess.run(  # noqa: S603
        ["rsync", "-a", "--delete", "--exclude", "__pycache__", "-e", "ssh -o BatchMode=yes", str(REPO / "src") + "/", f"{HOST}:{REMOTE}/repo/src/"], check=True
    )
    _ssh(f"mkdir -p ~/{REMOTE}/repo/tools/vps_dart")
    subprocess.run(  # noqa: S603
        ["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(REPO / "tools" / "vps_dart" / "worker.py"), f"{HOST}:{REMOTE}/repo/tools/vps_dart/worker.py"], check=True
    )
    for name in ("job.json", "seed_quota.json"):
        subprocess.run(  # noqa: S603
            ["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(state_dir / name), f"{HOST}:{REMOTE}/{'state/quota_state.json' if name == 'seed_quota.json' else name}"], check=True
        )
    # 원격에 python3-venv가 없어, 수집 폴더 안에 자체 uv를 설치한다(폴더 삭제 시 함께 제거).
    _ssh(f"[ -x ~/{REMOTE}/bin/uv ] || (export UV_UNMANAGED_INSTALL=$HOME/{REMOTE}/bin; curl -LsSf https://astral.sh/uv/install.sh | sh)")
    _ssh(f"[ -x ~/{REMOTE}/venv/bin/python ] || ~/{REMOTE}/bin/uv venv --python /usr/bin/python3 ~/{REMOTE}/venv")
    _ssh(f"~/{REMOTE}/bin/uv pip install -q --python ~/{REMOTE}/venv/bin/python {' '.join(PACKAGES)}")
    _ssh(f"umask 077 && cat > ~/{REMOTE}/secrets.env", stdin=f"{KEY_ENV}={api_key}\n")
    _ssh(f"cd ~/{REMOTE}/repo && ~/{REMOTE}/venv/bin/python -c 'import src.data.remote_dart_worker, src.data.collection'")
    _ssh(f"cat > ~/.config/systemd/user/kse-dart-worker.service", stdin=SERVICE.format(remote=REMOTE, key=KEY_ENV))
    _ssh("cat > ~/.config/systemd/user/kse-dart-worker.timer", stdin=TIMER)
    _ssh("systemctl --user daemon-reload && systemctl --user enable --now kse-dart-worker.timer")
    _emit(stage="push", status="started", pending=len(pending), seeded_requests=sum(marks.values()))


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


def pull() -> dict[str, object]:
    from src.data.receipt_catalog import ReceiptCatalog
    from src.data.remote_dart_inbox import fold_remote_quota, ingest_remote_bronze
    from src.data.scoped_ingestion import ScopedBronzeWriter
    from src.integrations.quota import ProviderQuotaStateStore

    runtime = _runtime()
    state_dir = runtime.workspace.state_root / "vps_dart"
    inbox = state_dir / "inbox"
    (inbox / "bronze").mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        ["rsync", "-a", "--partial", "-e", "ssh -o BatchMode=yes", f"{HOST}:{REMOTE}/out/bronze/", str(inbox / "bronze") + "/"], check=True
    )
    remote_quota = state_dir / "remote_quota_state.json"
    subprocess.run(  # noqa: S603
        ["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{HOST}:{REMOTE}/state/quota_state.json", str(remote_quota)], check=False
    )
    bronze_root = runtime.workspace.bronze_root
    writer = ScopedBronzeWriter(runtime=runtime, catalog=ReceiptCatalog(bronze_root / "catalog"))
    result = ingest_remote_bronze(inbox_bronze=inbox / "bronze", bronze_root=bronze_root, writer=writer)
    store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    folded = fold_remote_quota(store=store, remote_state=remote_quota, watermark_path=state_dir / "fold_watermark.json")
    if result.accepted:
        # 로컬 등록과 해시 검증이 끝난 페이지만 원격에서 지운다.
        listing = "\n".join(result.accepted) + "\n"
        _ssh(f"cd ~/{REMOTE}/out/bronze && xargs -r -d '\\n' rm -rf", stdin=listing)
        for name in result.accepted:
            shutil.rmtree(inbox / "bronze" / name, ignore_errors=True)
    summary = {"fact_pages": result.fact_pages, "document_pages": result.document_pages, "rejected": list(result.rejected), "quota_folded": folded}
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
    runtime = _runtime()
    remaining = _pending_job_identities(runtime)
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
    args = parser.parse_args()
    {"push": push, "update": update, "status": status, "pull": pull, "finish": finish, "auto": lambda: auto(args.interval)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
