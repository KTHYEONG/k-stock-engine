# kse VPS Automation Audit Checklist

> Runbook for an AI auditor verifying that every k-stock-engine (kse) automation on `or-vps` ran on schedule, used the
> current deploy (wrapper + image + host broker contract v1), and left the expected artefacts. Every probe below was
> run read-only against production on 2026-10-17 (HEAD `94d99a7`).

- **Host:** `or-vps` (SSH alias). Host clock is **UTC**; all schedules below are **KST (UTC+9)**. `list-timers` prints UTC.
- **Deploy shape:** no git checkout on the host. `~/k-stock-engine/` holds only `data/`, `logs/`, `scratch/`. Units call
  the installed wrapper `~/.local/lib/kse/kse-run <job>`, which `docker run`s `ghcr.io/kthyeong/k-stock-engine:latest`.
  `kse-run --print <job>` prints the argv (NUL-separated) without running anything.
- **Data root:** `~/k-stock-engine/data` (scope `kr_swing_2019_v1`). **Shared broker dir:** `~/.cache/kis` (mounted at
  `/app/.cache/kis`, injected as `KIS_TOKEN_CACHE_DIR`), shared with krx-alpha (`krx-collector`) and k-closing-alpha
  (`kca-*`). Same single LS app key and same DB금융투자 app key across all three projects; kse uses KIS data pool slot 5.
- **Report:** reply in chat only, in Korean (keys/IDs/badges in English). Do not write report files.

---

## 1. Rules (non-negotiable)

| Rule | Detail |
| :--- | :--- |
| Read-only | No `systemctl start/stop/restart/enable`, no `daemon-reload`, no file writes/deletes, no `docker run` (only `kse-run --print`), no `rclone` writes, no `git pull`. |
| Leave nothing behind | Do not create report, evidence or script files on the host or locally (no scratch/scratchpad). Run probes by piping a heredoc over SSH stdin: `ssh or-vps 'bash -s' <<'EOF' ... EOF`. |
| Secrets | Never print env-file values, tokens, app keys, or contents of `~/.cache/kis` / `~/.cache/dbsec` files. Names, mtimes, sizes and digests already embedded in filenames are fine. Redact `*KEY*|*SECRET*|*TOKEN*` in printed argv. |
| No broker traffic | Never send a request to KIS / LS / DB / KRX / DART. |
| UNKNOWN is not PASS | Unreachable host, missing log or missing artefact = `UNKNOWN` with the reason. |
| Day kind first | Jobs self-skip on non-trading days (`SKIPPED non_trading_day`; audit heartbeat `HOLIDAY_SKIP`). Judge weekday jobs only on XKRX sessions. |

## 2. Severity

| Severity | Criteria (kse is research + shadow only; it sends no orders) |
| :---: | :--- |
| **SEV1** | Trading day without a sealed decision (`morning` `NO_DECISION`/`DEGRADED`, incl. `late_seal`); `.host-admission` marker missing; `HostBrokerDirNotSharedError` or `ConfigError` (broker dir unset) in any job; token-issue storm on the shared LS/DB key (`IGW00201`/`EGW00201`/`TokenIssueThrottled` repeating) since it also hurts kca/krx-alpha; `NTPSynchronized=no`; free disk < 25 GB or available RAM < 3500 MB (job guards refuse); backup `rc != 0` or older than 26 h. |
| **SEV2** | `collect_eve` / `collect_dbsec` / `auction_open` not `OK` or missing on a trading day; daily audit `WARNING`; deploy drift (installed unit/wrapper hash differs from repo, any `kse-*.service.d` drop-in, image revision != deployed HEAD); data root > 10 GiB; `SCHEMA_INVALID`, `TokenStoreLockTimeout`; kernel `oom-kill` in a kse container. |
| **SEV3** | Hygiene: non-contract files in `~/.cache/kis`, stale token files outside the shared dir, leftovers in `~/k-stock-engine/scratch`, loose Bronze older than 14 days, accumulating image tags, a single retried transient error. |
| **INFO** | On-time runs, durations, row/request counts, peak RSS. |

## 3. Schedule and expected outcomes

Events: `data/ops/events/YYYY-MM/YYYY-MM-DD.jsonl`, one JSON line per run with `job`, `outcome`
(`OK|DEGRADED|NO_DECISION|SKIPPED`), `reason`, `metrics`, `ts`. `ts` is the job's start moment for `collect_*`/`backup`
and the record time for `morning`/`auction_open`.

| Unit (timer) | KST | `kse-run` job / limits | Expected result | Normal duration |
| :--- | :--- | :--- | :--- | :--- |
| `kse-morning` | Mon–Fri 08:32 | `morning`, 4g / 1.8 CPU | event `morning OK`, `metrics.sealed_at` < 09:00; new line in `state/kr_swing_2019_v1/decisions/chain.jsonl` with `execution_session` = today; `<data_session>.json` in `decisions/`; `shadow/official/*.parquet` mtime today | ~20–22 min (seal 08:51–08:54) |
| `kse-auction-open` | Mon–Fri 08:39:30 | `auction-open`, 512m, admission `standard` | event `auction_open OK`, `unresolved: []`; `data/ops/auction/open.json/YYYY-MM/<date>.parquet` (`open.json` is a directory) | until ~09:01; polls for the seal first |
| `kse-collect-eve` | Mon–Fri 18:30 | `collect-eve`, 3g / 1.5 CPU, admission `bulk`, `KSE_KIND_TRANSPORT=relay` | event `collect_eve OK` with `built == planned`, `prune_failed 0` | ~2 h (LS flow ~65 min, KIS flow ~55 min) |
| `kse-collect-dbsec` | Mon–Fri 20:40 | `collect-dbsec`, 1g; `ExecStartPre` = marker test + wait up to 60 min for `collect-eve` | event `collect_dbsec OK`, each step `pending_left 0`; `fallback_ready` false until 20 reconciled sessions | ~25 min |
| `kse-daily-audit` | Mon–Fri 21:40, Sat–Sun 12:00 | `daily-audit` (`audit`) | `data/ops/heartbeat/daily_audit.json` with `severity` `OK` (weekday) / `HOLIDAY_SKIP` (weekend), `open_issues: []` | < 2 min |
| `kse-backup` | daily 02:30, `Persistent=true` | `backup --no-lock` under `flock -w 7200 %t/quant-gdrive.lock` | event `backup OK`; `data/state/lake_backup_status.json` `rc 0`, `last_ok_at` < 26 h; sources `bronze, bronze_catalog, state, ops` → `gdrive:quant-lake/live/k-stock-engine/lake` | ~10–13 min |
| `kse-weekly-report` | Sat 10:30 | `weekly-report` | `data/ops/reports/weekly-<date>.json` | seconds |
| `kse-alert@<unit>` | on failure | `alert` (stdin = journal tail) | `[SYS] alert outcome=SENT`; line in `data/ops/alerts/sent.jsonl`. Its own `OnFailure` is `kca-alert@`. | seconds |

Daily-audit open-issue keys: `collect_eve`, `collect_dbsec`, `morning`, `auction_open`, `backup` (missing or non-OK),
`divergence`, `bronze_loose` (> 14 days), `footprint` (> 10 GiB), `events`.

## 4. Probes

Run each block as `ssh or-vps 'bash -s' <<'EOF'` … `EOF`. Set `D=YYYY-MM-DD` (KST) where used.

### P1. Host, clock, resources
```bash
TZ=Asia/Seoul date; uptime; df -h / | tail -1; free -h | head -2
timedatectl show -p NTPSynchronized -p Timezone; chronyc tracking 2>/dev/null | grep -E 'System time|Leap'
```
PASS: `NTPSynchronized=yes`, offset ≪ 2 s, free disk ≥ 25 GB, available RAM ≥ 3.5 GiB.

### P2. Timers and last results
```bash
systemctl --user list-timers --all --no-pager 'kse-*'
systemctl --user --failed --no-pager
for u in morning auction-open collect-eve collect-dbsec daily-audit backup weekly-report; do printf '%-18s ' $u; systemctl --user show -p Result -p ExecMainStatus -p ExecMainStartTimestamp -p ExecMainExitTimestamp -p DropInPaths --value kse-$u.service | tr '\n' '|'; echo; done
journalctl --user -u 'kse-*' --since "$D" --no-pager -o short-iso | grep -E "Failed with result" | cut -c1-140
```
PASS: 7 timers listed, none failed, `Result=success`, `DropInPaths` empty for every unit.
A unit failure can be followed by a manual `systemctl --user start` re-run; the last run then shows `success` — use the
events file (P4) for the full day history.

### P3. Deploy drift (installed vs repo, image, wrapper argv)
On the host:
```bash
cd ~/.config/systemd/user && for f in kse-*; do printf '%s %s\n' "$(sha256sum < $f | cut -c1-16)" "$f"; done
printf '%s %s\n' "$(sha256sum < ~/.local/lib/kse/kse-run | cut -c1-16)" kse-run
systemctl --user cat kse-backup.service | head -40
docker image inspect -f '{{.Id}} {{.Created}} {{index .Config.Labels "org.opencontainers.image.revision"}}' ghcr.io/kthyeong/k-stock-engine:latest
docker ps -a --filter name=kse- --format '{{.Names}} {{.Status}}'
docker system df
for j in morning auction-open collect-eve collect-dbsec daily-audit backup weekly-report alert; do
  printf '== %s: ' "$j"; ~/.local/lib/kse/kse-run --print "$j" | tr '\0' ' ' | sed -E 's/([A-Z_]*(KEY|SECRET|TOKEN)[A-Z_]*)=[^ ]*/\1=<redacted>/g'; echo
done
```
Locally, compare against `for f in deploy/systemd/* deploy/kse-run; do printf '%s %s\n' "$(sha256sum < $f | cut -c1-16)" "$(basename $f)"; done`.
PASS: all hashes equal; no `kse-*.service.d` entry (it prints `sha256sum: -: Is a directory`); image revision = deployed
commit; no lingering `kse-*` containers; every argv except `alert` has `-v /home/ubuntu/.cache/kis:/app/.cache/kis` and
`KIS_TOKEN_CACHE_DIR`, `collect-dbsec` additionally mounts `~/.cache/dbsec:/app/.cache/dbsec:ro` (legacy, see §5).

### P4. Outcomes, heartbeat, alerts
```bash
python3 - "$D" <<'PY'
import json,sys,pathlib
d=sys.argv[1]; p=pathlib.Path.home()/f"k-stock-engine/data/ops/events/{d[:7]}/{d}.jsonl"
for l in p.read_text().splitlines():
    e=json.loads(l); m=e.get("metrics",{})
    print(e["ts"][11:19], e["job"], e["outcome"], e["reason"] or "-", {k:m[k] for k in ("data_session","sealed_at","symbols","expected_rows","built","planned","removed","fallback_ready","pruned") if k in m})
PY
python3 - 7 <<'PY'
import json,sys,pathlib,datetime as dt
root=pathlib.Path.home()/"k-stock-engine/data/ops/events"; days=int(sys.argv[1])
today=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()
for i in range(days,-1,-1):
    d=today-dt.timedelta(days=i); p=root/f"{d:%Y-%m}"/f"{d}.jsonl"
    if not p.exists(): continue
    for l in p.read_text().splitlines():
        e=json.loads(l)
        if e["outcome"] not in ("OK","SKIPPED"): print(d, e["ts"][11:19], e["job"], e["outcome"], e["reason"])
PY
cd ~/k-stock-engine/data/ops
python3 -c "import json;d=json.load(open('heartbeat/daily_audit.json'));print(json.dumps(d)[:1500])"
tail -5 alerts/sent.jsonl | cut -c1-400
```
Job exceptions are recorded as `reason=<ExceptionClass>` in the event, **without a traceback in the journal**
(e.g. 2026-10-16 `collect_eve DEGRADED SchemaError`). Locate the cause from the event reason, the last journal lines
before the exit (P6), and the fix history in the repo.

### P5. Artefacts
```bash
cd ~/k-stock-engine/data
python3 -c "
import json
for l in open('state/kr_swing_2019_v1/decisions/chain.jsonl').read().splitlines()[-3:]:
    e=json.loads(l); print(e['data_session'],'->',e['execution_session'],e.get('sealed_at'))"
ls -la --time-style=+%m-%dT%H:%M ops/auction/open.json/2026-10/ state/kr_swing_2019_v1/shadow/official/ | awk 'NF>5{print $5,$6,$7}'
python3 -c "import json;d=json.load(open('state/lake_backup_status.json'));print(d['rc'],d['finished_at'],d['last_ok_at'],sorted(d['sources']))"
ls ops/reports
```
`auction_open.metrics.expected_rows = symbols × snapshots`; snapshots are at 08:40/08:50/08:55/08:58/08:59. Only those
after the seal are captured, so a ~08:52 seal yields 3 per symbol (see §5).

### P6. Shared broker dir and contract errors
```bash
ls -la --time-style=+%m-%dT%H:%M ~/.cache/kis/ | awk '{print $1,$3,$4,$5,$6,$7}'
test -f ~/.cache/kis/.host-admission && echo marker=present
stat -c '%a %U %n' ~/.cache/kis
cd ~/.cache/kis && ls -A | python3 -c "
import re,sys
ok=re.compile(r'^(\.host-admission|token_(ls|toss|dbsec|kiwoom)_[0-9a-f]{12}\.json(\.lock)?|token_issue_(ls|toss|dbsec|kiwoom)_[0-9a-f]{12}\.state|admission_(ls|toss|dbsec|kiwoom)_[0-9a-f]{12}(_[A-Za-z0-9_-]+)?\.state|token_[0-9a-f]{12}\.json(\.lock)?|tps_[0-9a-f]{12}\.state)$')
bad=[n for n in sys.stdin.read().split() if not ok.match(n)]
print('non_contract:',bad or 'none')"
ls -la --time-style=+%m-%dT%H:%M ~/.cache/dbsec/ | awk '{print $1,$5,$6,$7}'
P='IGW00201|EGW00201|TokenIssueThrottled|TokenStoreLockTimeout|HostBrokerDirNotSharedError|SCHEMA_INVALID|LEGACY_FALLBACK|ConfigError|SchemaError|Traceback'
printf 'journal_matches=%s\n' "$(journalctl --user -u 'kse-*' --since "$D" --no-pager -o cat | grep -cE "$P")"
journalctl --user -u 'kse-*' --since "$D" --no-pager -o short-iso | grep -E "$P" | cut -c1-240
```
Expected kse files (`<sha12>` = sha256(app key)[:12]): `token_ls_<sha12>.json`, `admission_ls_<sha12>.state`,
`token_dbsec_<sha12>.json`, `token_issue_dbsec_<sha12>.state`, `admission_dbsec_<sha12>[_chart|_quote].state`, and the
KIS slot-5 `token_<sha12>.json` / `tps_<sha12>.state`. `toss`/`kiwoom` files belong to sibling projects.
PASS: marker present, dir `700 ubuntu`, `non_contract: none`, `journal_matches=0`, and the same pattern absent from event
`reason` fields (P4). `LEGACY_FALLBACK` means a DB run read `~/.cache/dbsec/dbsec_token_prod.json` because the shared
token was unusable — acceptable this release only (SEV3, SEV2 after the fallback is removed).
To map a digest to a key name, hash the key inside a python heredoc and print only the 12-hex digest.

### P7. Failure window and memory
```bash
journalctl --user -u kse-collect-eve.service --since "$D 09:00 UTC" --until "$D 11:30 UTC" --no-pager -o short-iso | tail -40
journalctl -k --since "$D" --no-pager -o short-iso | grep -i 'oom-kill' | cut -c1-250
```
Grep `oom-kill`, not `oom`: kse logs contain `headroom=`. Compare `metrics.node_peak_rss_mb` against the container
limit (morning 4g, collect-eve 3g): > 90 % of the limit is SEV3.

### P8. Footprint, retention, log growth
```bash
cd ~/k-stock-engine/data && du -sb . | awk '{printf "total_GiB=%.2f\n",$1/2^30}'
du -sh bronze/*/ silver/*/ gold/*/ state/*/ 2>/dev/null
du -sh bronze/kr_swing_2019_v1/* 2>/dev/null | sort -h | tail -15
cd bronze/kr_swing_2019_v1 && find investor_flow daily_market -type f -printf '%TY-%Tm-%Td %P %s\n' | awk '{s[$1]+=$3; n[$1]++} END{for(x in s) printf "%s files=%d MiB=%.1f\n",x,n[x],s[x]/2^20}' | sort
du -sh ~/k-stock-engine/scratch ~/k-stock-engine/logs; ls -la ~/k-stock-engine/logs
journalctl --user --disk-usage
for d in 2026-10-14 2026-10-15 2026-10-16; do printf '%s kse journal lines: ' $d; journalctl --user --since "$d 00:00" --until "$d 23:59:59" -u 'kse-*' --no-pager -o cat | wc -l; done
```
Loose Bronze is packed and pruned by `kse-backup` only when older than 14 days **and** confirmed offsite; growth
before that is expected. Project the date the data root crosses 10 GiB from the per-day MiB.

### P9. Catalog coverage (read-only SQLite)
```bash
python3 - <<'PY'
import sqlite3,os
c=sqlite3.connect('file:'+os.path.expanduser('~/k-stock-engine/data/bronze/kr_swing_2019_v1/catalog/catalog.sqlite3')+'?mode=ro',uri=True)
q="""select source, substr(retrieved_at,1,10) d, status, min(start), max(start), min(end), max(end), count(*) from ranges where retrieved_at>='2026-10-14' group by 1,2,3 order by 2,1"""
for r in c.execute(q): print(r)
PY
```
Always open with `?mode=ro`. Never run `sqlite3` CLI writes or `PRAGMA` that checkpoints.

### P10. Cross-project schedule and Drive lock
```bash
systemctl --user list-timers --all --no-pager | grep -E 'kse|kca|krx|NEXT'
for u in kca-backup kca-backup-prune krx-host-backup crypto-pilot-backup kca-offsite-verify kca-core-snapshot; do printf '%s: ' $u; systemctl --user cat $u.service 2>/dev/null | grep -E '^ExecStart' | grep -oE 'flock[^%]*%t/[a-z-]+\.lock|quant-gdrive|rclone' | sort -u | tr '\n' ' '; systemctl --user cat $u.timer 2>/dev/null | grep OnCalendar | tr '\n' ' '; echo; done
```
Known neighbours (KST): kca-price-ingest 08:30, kca-auction-open 08:39:55 (KIS `critical`), kca-order-exit 08:40:10,
kca-archive-intraday 20:05, kca-tape-sweep 20:35, kca-backup-prune 21:00, kca-backup 22:15, krx-host-backup 23:30;
Drive writers kca/krx-alpha/crypto-pilot all serialize on `%t/quant-gdrive.lock`. Shared LS/DB/KIS pacing is enforced
through the `admission_*`/`tps_*` state files, so overlap slows jobs rather than failing them; contention shows up as
the P6 error codes.

## 5. Transitional and known items (as of 2026-10-17)

| Item | Status / what to check |
| :--- | :--- |
| Legacy DB token dir `~/.cache/dbsec` | Read-only fallback (`DBSEC_TOKEN_CACHE_DIR`, mounted `:ro`) for **one release**; the next release removes the mount, the env and the fallback. After that, any `~/.cache/dbsec` reference in `kse-run --print collect-dbsec` is drift (SEV2). |
| `logs/ls_token_prod.json` | Written by the pre-contract LS client (last 2026-10-16). Current code never reads it; LS tokens live in `~/.cache/kis`. Report as SEV3 hygiene until the user removes it. |
| `kse-backup.service.d/90-lake-backup-memory.conf` | 2026-10-13 hotfix drop-in that bypasses `kse-run` and pins image `kse-backup-memory:20261013`. Report as SEV2 drift while present. |
| Auction snapshots | Seal at ~08:51–08:54 means the 08:40 and 08:50 snapshots are never captured (`expected_rows = 3 × symbols`). |
| First runs on the new contract | Morning 2026-10-19 is the first trading-day run of `94d99a7`; LS/DB tokens then come from `~/.cache/kis` (check `admission_ls_*`/`admission_dbsec_*` mtimes move to kse run times). |

## 6. Chat report template

| Check | Status | Evidence (time KST, value) | Severity |
| :--- | :---: | :--- | :---: |
| Morning seal | `PASS` | `sealed_at` 08:51, chain `2026-10-15 -> 2026-10-16` | INFO |
| Auction open | … | … | … |
| Evening collect | … | … | … |
| DB fallback collect | … | … | … |
| Daily audit | … | … | … |
| Backup | … | … | … |
| Weekly report | … | … | … |
| Shared broker dir / contract | … | … | … |
| Deploy drift | … | … | … |
| Footprint / growth | … | … | … |
