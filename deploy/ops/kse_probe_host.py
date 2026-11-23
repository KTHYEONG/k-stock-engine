"""Host-side facts pass for kse_probe.py (stdlib only), run on the VPS host by kse-probe.sh.

Read-only: systemd, journal, docker, rclone and git ls-remote. Writes a tar (facts.json, remote
manifests, parquet inputs) to stdout for the containerized probe.
"""

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import tarfile
from typing import Any

WIN = sys.argv[1]
H = pathlib.Path.home()
R = H / "k-stock-engine/data"
T = R / "state/kr_swing_2019_v1"
B = R / "bronze/kr_swing_2019_v1"
KST = dt.timezone(dt.timedelta(hours=9))
now = dt.datetime.now(KST)
DOVR = sys.argv[2] if len(sys.argv) > 2 else ""
D = dt.date.fromisoformat(DOVR) if DOVR else now.date()
ENV = dict(os.environ, TZ="Asia/Seoul")


def sh(*a: str, t: int = 90) -> str:
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=t, env=ENV).stdout  # noqa: S603
    except Exception as e:
        return "ERR " + type(e).__name__


def jl(p: Any) -> Any:
    try:
        return json.loads(pathlib.Path(p).read_text())
    except Exception:
        return None


def mt(p: Any) -> str | None:
    try:
        return dt.datetime.fromtimestamp(os.stat(p).st_mtime, KST).isoformat(timespec="seconds")
    except Exception:
        return None


F: dict[str, Any] = {"now": now.isoformat(timespec="seconds"), "D": str(D), "win": WIN, "err": []}
ev = {}
for i in range(12):
    d = D - dt.timedelta(days=i)
    p = R / "ops/events" / f"{d:%Y-%m}" / f"{d}.jsonl"
    if p.exists():
        ev[str(d)] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
F["events"] = ev
units = [
    "morning",
    "morning-late",
    "auction-open",
    "collect-eve",
    "collect-dbsec",
    "daily-audit",
    "backup",
    "weekly-report",
]
F["units"] = {
    u: sh(
        "systemctl",
        "--user",
        "show",
        "-p",
        "Result",
        "-p",
        "ActiveState",
        "-p",
        "DropInPaths",
        "--value",
        f"kse-{u}.service",
    ).split("\n")[:3]
    for u in units
}
F["timers"] = len(
    [
        l
        for l in sh("systemctl", "--user", "list-timers", "--all", "--no-pager", "--no-legend", "kse-*").splitlines()
        if "kse-" in l
    ]
)
F["failed"] = [
    l.split()[0]
    for l in sh("systemctl", "--user", "--failed", "--no-legend", "--plain").splitlines()
    if l.strip().startswith("kse-")
]
ud = H / ".config/systemd/user"
F["unit_files"] = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(ud.glob("kse-*")) if p.is_file()
}
kr = H / ".local/lib/kse/kse-run"
F["unit_files"]["kse-run"] = hashlib.sha256(kr.read_bytes()).hexdigest()[:16] if kr.exists() else None
F["kse_collect"] = (H / "kse-collect").exists()
F["image"] = sh(
    "docker",
    "image",
    "inspect",
    "-f",
    '{{.Created}} {{index .Config.Labels "org.opencontainers.image.revision"}}',
    "ghcr.io/kthyeong/k-stock-engine:latest",
).strip()
F["image_tags"] = len(sh("docker", "images", "ghcr.io/kthyeong/k-stock-engine", "--format", "{{.Tag}}").split())
F["containers"] = sh("docker", "ps", "-a", "--filter", "name=kse-", "--format", "{{.Names}}:{{.Status}}").split("\n")
F["deploy_entries"] = len(os.listdir(H / "k-stock-engine/.deploy")) if (H / "k-stock-engine/.deploy").is_dir() else 0
F["scratch"] = sorted(os.listdir(H / "k-stock-engine/scratch")) if (H / "k-stock-engine/scratch").is_dir() else []
mi = {
    l.split(":")[0]: int(l.split()[1]) for l in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
}
st = os.statvfs(R)
F["mem_mb"] = mi["MemAvailable"] // 1024
F["disk_gib"] = round(st.f_bavail * st.f_frsize / 2**30, 1)
F["ntp"] = sh("timedatectl", "show", "-p", "NTPSynchronized", "--value").strip()
kd = H / ".cache/kis"
ok = re.compile(
    r"^(\.host-admission|token_(ls|toss|dbsec|kiwoom)_[0-9a-f]{12}\.json(\.lock)?|token_issue_(ls|toss|dbsec|kiwoom)_[0-9a-f]{12}\.state|admission_(ls|toss|dbsec|kiwoom)_[0-9a-f]{12}(_[A-Za-z0-9_-]+)?\.state|token_[0-9a-f]{12}\.json(\.lock)?|tps_[0-9a-f]{12}\.state|wslease_(ls|kis|db|toss)_[0-9a-f]{12}\.(json|lock))$"
)
F["broker"] = {
    "marker": (kd / ".host-admission").is_file(),
    "mode": oct(kd.stat().st_mode & 0o777)[2:],
    "non_contract": [n for n in os.listdir(kd) if not ok.match(n)],
}
F["heartbeat"] = jl(R / "ops/heartbeat/daily_audit.json")
F["hb_mtime"] = mt(R / "ops/heartbeat/daily_audit.json")
F["backup_status"] = jl(R / "state/lake_backup_status.json")
sent = R / "ops/alerts/sent.jsonl"
F["alerts"] = (
    [{k: json.loads(l).get(k) for k in ("ts", "key")} for l in sent.read_text().splitlines()[-8:]]
    if sent.exists()
    else []
)
ob = R / "ops/alerts/outbox"
F["outbox"] = len(os.listdir(ob)) if ob.is_dir() else 0
F["reports"] = sorted(os.listdir(R / "ops/reports"))[-2:] if (R / "ops/reports").is_dir() else []
chain = [json.loads(l) for l in (T / "decisions/chain.jsonl").read_text().splitlines() if l.strip()]
F["chain"] = chain
P = next((c["data_session"] for c in reversed(chain) if c["execution_session"] == str(D)), None)
F["P"] = P
head = chain[-1] if chain else {}
F["sidecar_hash"] = (jl(T / "decisions" / f"{head.get('data_session')}.json") or {}).get("hash")
cur = (jl(T / "datasets.json") or {}).get("current", {})
F["current"] = cur
F["runs"] = {}
for v in ("official", "ideal"):
    lines = [l for l in (T / "shadow" / v / "runs.jsonl").read_text().splitlines() if l.strip()]
    F["runs"][v] = {"n": len(lines), "last": [json.loads(l) for l in lines[-2:]]}
F["divergence"] = [str(p.relative_to(T)) for p in (T / "shadow").rglob("divergence-*.json")]
b = jl(T / "kind_relay/inbox/bundle.json") or {}
F["kind"] = {
    "bundle_fetched_at": b.get("fetched_at"),
    "fetch_date_kst": b.get("fetch_date_kst"),
    "items": len(b.get("items", [])),
    "schema": b.get("schema"),
    "accepted": jl(T / "kind_relay/last_accepted.json"),
}
rr = sorted((T / "dbsec_reconcile").glob("*.json"))
F["reconcile"] = {
    "n": len(rr),
    "last": [
        {k: (jl(p) or {}).get(k) for k in ("session", "verdict", "coverage", "close_match_rate", "volume_match_rate")}
        for p in rr[-2:]
    ],
}
q: dict[str, int] = {}
for k, v in (jl(T / "quota/quota_state.json") or {}).items():
    key = k.split("|")[0] + "@" + str(v.get("daily_attempt_day"))
    q[key] = q.get(key, 0) + int(v.get("daily_attempted_requests", 0))
F["quota"] = q
S = R / "silver/kr_swing_2019_v1"
G = R / "gold/kr_swing_2019_v1"


def ds(k: str) -> Any:
    return (G if k in ("market_panel", "reference_benchmarks") else S) / cur.get(k, "_missing")


dm_parts = sorted(p.name[8:] for p in ds("daily_market").glob("session=*"))
F["dm_partitions"] = len(dm_parts)
CD = None
if WIN == "OPEN" and P:
    CD = P
if WIN == "EOD":
    CD = str(D)
if WIN == "MORNING":
    bd = [
        d
        for d, l in ev.items()
        if d < str(D) and any(e["job"] == "collect_eve" and e["outcome"] != "SKIPPED" for e in l)
    ]
    CD = max(bd) if bd else None
F["CD"] = CD
prev_cd = None
if CD:
    before = [s for s in dm_parts if s < CD]
    prev_cd = before[-1] if before else None
F["prevCD"] = prev_cd
F["prevP"] = max([s for s in dm_parts if s < P], default=None) if P else None
try:
    c = sqlite3.connect("file:" + str(B / "catalog/catalog.sqlite3") + "?mode=ro", uri=True)
    c.execute("pragma query_only=1")
    cat: dict[str, Any] = {}
    if P:
        cat["krx_P"] = {
            s: list(
                c.execute(
                    "select status,retrieved_at from receipts where source=? and natural_key=?", (s, P)
                ).fetchone()
                or ["absent", None]
            )
            for s in (
                "krx_daily_market",
                "krx_security_master",
                "krx_hedge_series",
                "krx_trend_series",
                "krx_cash_series",
            )
        }
    cat["krx_dm_success"] = c.execute(
        "select count(*) from receipts where source='krx_daily_market' and status='success'"
    ).fetchone()[0]
    if CD:
        cat["dbsec"] = {
            f"{s}|{st}": n
            for s, st, n in c.execute(
                "select source,status,count(*) from receipts where source in ('dbsec_daily_chart','dbsec_closing_bar') and natural_key>=? and natural_key<? group by 1,2",
                (CD + ":", CD + ";"),
            )
        }
        cat["dbsec_master"] = list(
            c.execute(
                "select status from receipts where source='dbsec_ticker_master' and natural_key=?", (CD,)
            ).fetchone()
            or ["absent"]
        )
        lo = (dt.date.fromisoformat(CD) - dt.timedelta(days=1)).isoformat() + "T15:00"
        hi = CD + "T15:00"
        cat["ranges"] = [
            list(r)
            for r in c.execute(
                "select source,status,count(*),max(end) from ranges where source in ('ls_investor_flow','kis_investor_flow') and retrieved_at>=? and retrieved_at<? group by 1,2",
                (lo, hi),
            )
        ]
        cat["disclosure_windows"] = c.execute(
            "select count(*) from receipts where source='dart_disclosure_windows' and retrieved_at>=? and retrieved_at<?",
            (lo, hi),
        ).fetchone()[0]
    F["catalog"] = cat
    c.close()
except Exception as e:
    F["err"].append("catalog:" + type(e).__name__)


def jgrep(unit: str, since: str, until: str, pat: str) -> list[str]:
    out = sh("journalctl", "--user", "-u", unit, "--since", since, "--until", until, "--no-pager", "-o", "cat", t=120)
    rx = re.compile(pat)
    return [l[:260] for l in out.splitlines() if rx.search(l)]


J = {}
if CD:
    nxt = (dt.date.fromisoformat(CD) + dt.timedelta(days=1)).isoformat()
    J["eve"] = jgrep(
        "kse-collect-eve.service",
        CD + " 18:00",
        nxt + " 03:00",
        r"phase=done|skipped_blocked|status=failed|status=DEGRADED",
    )
    J["dbsec"] = jgrep(
        "kse-collect-dbsec.service", CD + " 20:30", nxt + " 03:00", r"phase=done|stage=dbsec_reconcile|status=DEGRADED"
    )
if P:
    J["morning"] = jgrep(
        "kse-morning.service",
        str(D) + " 08:00",
        str(D) + " 12:00",
        r"gate=investor_flow|ledger replay|stage=krx_poll|status=DEGRADED",
    )
since = (CD or str(D)) + " 00:00"
J["codes"] = jgrep(
    "kse-*",
    since,
    "now",
    r"IGW00201|EGW00201|TokenIssueThrottled|TokenStoreLockTimeout|HostBrokerDirNotSharedError|SCHEMA_INVALID|LEGACY_FALLBACK|ConfigError|Traceback",
)[:10]
J["oom"] = [
    l[:600] for l in sh("journalctl", "-k", "--since", since, "--no-pager", "-o", "cat").splitlines() if "oom-kill" in l
][:5]
F["journal"] = J
paths = []
if P and WIN in ("OPEN", "EOD"):
    for k in ("daily_market", "ordinary_universe", "hedge_series", "trend_series", "cash_series"):
        paths += [ds(k) / p.name / "part.parquet" for p in sorted(ds(k).glob("session=*"))[-7:]]
    years = {D.year} | ({D.year - 1} if D.month == 1 else set())
    paths += [ds("investor_flow") / f"year={y}" / "part.parquet" for y in years]
    for k in ("market_panel", "market_actions", "cash_series", "hedge_series", "trend_series"):
        paths.append(ds(k) / "manifest.json")
if P and WIN == "OPEN":
    for v in ("official", "ideal"):
        paths += [T / "shadow" / v / f"{n}.parquet" for n in ("nav", "fills", "rejects")]
    paths.append(R / "ops/auction/open.json" / f"{D:%Y-%m}" / f"{D}.parquet")
if WIN == "WEEKLY":
    loose = []
    tot = 0
    packs = 0
    for root, _dirs, files in os.walk(R):
        for n in files:
            fp = os.path.join(root, n)
            try:
                s = os.lstat(fp)
            except OSError:
                continue
            tot += s.st_size
            if fp.startswith(str(B / "_packs")):
                packs += s.st_size
            elif fp.startswith(str(B)) and not fp.startswith(str(B / "catalog")):
                loose.append(os.path.relpath(fp, B) + "\t" + str(int(s.st_mtime)))
    F["footprint"] = {"total": tot, "packs": packs, "loose": len(loose)}
RCLONE = os.path.expanduser("~/.local/bin/rclone")
REMOTE = "gdrive:quant-lake/live/k-stock-engine/lake"


def rc_cat(path: str, t: int = 120) -> bytes:
    try:
        return subprocess.run([RCLONE, "cat", f"{REMOTE}/{path}"], capture_output=True, timeout=t).stdout  # noqa: S603
    except Exception:
        return b""


REMOTE_FILES = {}
if WIN in ("OPEN", "MORNING", "WEEKLY"):
    for src in ("bronze", "bronze_catalog", "state", "ops", "silver", "gold"):
        REMOTE_FILES[f"remote/{src}.json"] = rc_cat(f"{src}/MANIFEST.json")
if WIN == "WEEKLY":
    REMOTE_FILES["remote/bronze_index.jsonl.zst"] = rc_cat("bronze/index.jsonl.zst", 600)


def ls_remote_main(url: str) -> str | None:
    try:
        res = subprocess.run(  # noqa: S603,S607
            ["git", "ls-remote", url, "refs/heads/main"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0"),
        )
    except Exception:
        return None
    return res.stdout.split("\t")[0] or None


F["origin"] = ls_remote_main(os.environ.get("KSE_ORIGIN_URL", "https://github.com/kthyeong/k-stock-engine.git"))
_KRX = H / "krx-alpha/deploy/host/vps-deploy-lib.sh"
F["krx_lib_sha"] = hashlib.sha256(_KRX.read_bytes()).hexdigest() if _KRX.is_file() else None
out = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")  # noqa: SIM115


def add(name: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    out.addfile(ti, io.BytesIO(data))


add("facts.json", json.dumps(F, default=str).encode())
for _k, _v in REMOTE_FILES.items():
    add(_k, _v)
if WIN == "WEEKLY":
    add("loose.txt.gz", gzip.compress("\n".join(loose).encode()))
for p in paths:
    if p.is_file():
        add(str(p.relative_to(R)), p.read_bytes())
out.close()
