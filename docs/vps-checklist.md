# kse VPS Automation Audit Checklist

> Machine runbook for an AI auditor of k-stock-engine (kse) on `or-vps`: every routine ran on schedule on the current
> deploy **and what it wrote is correct**. Every probe below was executed read-only against production on 2026-10-18
> (HEAD `c65833b`); thresholds come from the 2026-10-15/16 outputs quoted in each row.

- **Host:** `or-vps`. Host clock is **UTC**; schedules here are **KST**. Prefix every `journalctl` with `TZ=Asia/Seoul`
  so `--since/--until` and printed stamps are KST (journalctl rejects `Asia/Seoul` inside the timestamp).
- **Deploy shape:** no git checkout on the host. `~/k-stock-engine/` = `data/`, `logs/`, `scratch/`, `.deploy/`. Units
  call `~/.local/lib/kse/kse-run <job>` → `docker run ghcr.io/kthyeong/k-stock-engine:latest`; `kse-run --print <job>`
  prints the argv (NUL-separated) without running. Host python3 has **no pyarrow/polars**: parquet is decoded locally (P11).
- **Data root:** `~/k-stock-engine/data`, scope `kr_swing_2019_v1`. `S=silver/kr_swing_2019_v1`, `G=gold/…`,
  `T=state/kr_swing_2019_v1`, `B=bronze/kr_swing_2019_v1`. Current dataset ids: `$T/datasets.json` `.current[<kind>]`.
- **Shared broker dir:** `~/.cache/kis` (contract v1, mounted `/app/.cache/kis`, `KIS_TOKEN_CACHE_DIR`), shared with krx
  (`krx-collector`) and kca (`kca-*`); one LS key and one DB금융투자 key across all three; kse uses KIS pool slot 5.
- **Day notation:** `D` = audit business day (KST). `P` = data session sealed by morning `D` = previous XKRX session.
- **Report:** chat only, Korean (keys/IDs/badges English). No report files.

## 1. Rules

| Rule | Detail |
| :--- | :--- |
| Read-only | No `systemctl start/stop/restart/enable/daemon-reload`, no host file writes/deletes, no `docker run`/`exec` (only `kse-run --print`), no `rclone` writes, no `git pull`. |
| Leave nothing behind | Host probes: heredoc over SSH stdin (`ssh or-vps 'bash -s' <<'EOF' … EOF`). P11 runs locally from a heredoc and streams files with `ssh … tar cf -` into memory. No files on host or locally. |
| Load | `nice -n 19` on heavy reads; SQLite only `file:…?mode=ro` (never the `sqlite3` CLI, never a checkpointing `PRAGMA`). During 08:10–20:30 KST run P11 only after morning has finished. |
| Secrets | Never print env files, tokens, app keys, rclone config, or contents of `~/.cache/kis`/`~/.cache/dbsec`/`logs/ls_token_prod.json`. Names, mtimes, sizes, filename digests are fine. Redact `*KEY*|*SECRET*|*TOKEN*` in argv. |
| No provider traffic | Never call KIS / LS / DB / KRX / DART / KIND. Quota is read from the ledger (K6), never spent. |
| UNKNOWN ≠ PASS | Unreachable host, missing log/artefact, or a probe exception = `UNKNOWN` + reason. |
| Day kind first | Jobs self-skip on non-XKRX days (`SKIPPED non_trading_day`; heartbeat `HOLIDAY_SKIP`). Judge weekday jobs and K1–K7 only for business days. |
| Pipeline verdicts are claims | An `OK` event or `severity OK` heartbeat is not evidence of content; P11 is. |

## 2. Severity

| Severity | Criteria (kse is research + shadow only; it sends no orders) |
| :---: | :--- |
| **SEV1** | Business day without a sealed decision (`morning` `NO_DECISION`/`DEGRADED`, incl. `late_seal`, `stale_investor_flow`); K2 chain broken / reference ≠ close / input created after seal; K3 `lr_bad`/`neg_holding`/`exec_not_after_decision` > 0; `.host-admission` missing; `HostBrokerDirNotSharedError`/`ConfigError`; token-issue storm on the shared LS/DB key (`IGW00201`/`EGW00201`/`TokenIssueThrottled` repeating); `NTPSynchronized=no` or offset > 2 s; free disk < 25 GB or available RAM < 3500 MB; backup `rc != 0` or `last_ok_at` > 26 h. |
| **SEV2** | `collect_eve` / `collect_dbsec` / `auction_open` not `OK` or missing; any K1/K4/K5/K7 FAIL threshold; quota > 80 % of a daily limit/budget; daily audit `WARNING`; deploy drift (hash, drop-in, image revision ≠ `origin/main`); data root > 10 GiB; `SCHEMA_INVALID`, `TokenStoreLockTimeout`; kernel `oom-kill` in a kse container; KIND bundle > 36 h old at `collect_eve` start. |
| **SEV3** | Node peak RSS > 90 % of container limit; non-contract files in `~/.cache/kis`; leftovers in `~/k-stock-engine/scratch`; > 5 `.deploy/staging-*` dirs; > 5 image tags; one retried transient error; projected 10 GiB crossing < 10 business days. |
| **INFO** | On-time runs, durations, row/request counts, peak RSS. |

## 3. Inventory: routines, artefacts, content checks

Events: `data/ops/events/YYYY-MM/YYYY-MM-DD.jsonl`, one JSON line per run: `job`, `outcome` (`OK|DEGRADED|NO_DECISION|SKIPPED`),
`reason` (exception class name for crashes), `metrics`, `ts` (start for `collect_*`/`backup`, record time for
`morning`/`auction_open`). A manual re-run appends a second line; the **last line per job** is the day's result.

| Routine (trigger, KST) | Job / limits | Writes (path, format) | Outcome expected | Content check |
| :--- | :--- | :--- | :--- | :--- |
| `kse-morning` Mon–Fri 08:32 | `morning` 4g/1.8 CPU, `bulk` | KRX Bronze (catalog `krx_daily_market`, `krx_security_master`, `krx_{hedge,trend,cash}_series`, 1 receipt each for `P`); Silver `daily_market`, `ordinary_universe`, `{hedge,trend,cash}_series` (`$S/<id>/session=P/part.parquet`), `market_actions`; Gold `market_panel` (`year=YYYY/part.parquet` + `manifest.json`); `$T/decisions/chain.jsonl` + `$T/decisions/P.json`; `$T/operate/state.json`; `$T/shadow/{official,ideal}/{fills,nav,rejects}.parquet` + `runs.jsonl` | event `OK`, `metrics.sealed` = `[P]`, `sealed_at` < 09:00 (2026-10-14..16: 08:51–08:54); ~20 min | K1, K2, K3, P5 feature coverage |
| `kse-morning-late` operator-triggered, no timer | `morning_late` 4g/1.8 CPU, `bulk` | same writes as `morning` for a missed seal | event `DEGRADED late_seal`, never `OK`; `auction_open` captures nothing after 08:59 | P4 |
| `kse-auction-open` Mon–Fri 08:39:30 | `auction-open` 512m, `standard`, `--limit 80` | `data/ops/auction/open.json/YYYY-MM/D.parquet` (`open.json` is a dir): `session,symbol,kind(expected|open),scheduled_at,received_at,output1,output2` — incremental per-slot persist, dedup on `(symbol,kind,scheduled_at)` first-wins | event `OK`, `unresolved: []`, `slots_captured == slots_expected`, `failed == {}`, `persist_error None`; `expected_rows = 3 × symbols` when sealed 08:51–08:54; ends ~09:01. File presence ≠ completeness: partial file possible when the event is missing or `DEGRADED`. `--limit` above ~58 truncates the lowest-priority tail of the 08:58/08:59 slots (`failed` `SlotDeadline`) | K4 |
| `kse-collect-eve` Mon–Fri 18:30 | `collect-eve` 3g/1.5 CPU, `bulk`, `KSE_KIND_TRANSPORT=relay` | Bronze `investor_flow` (LS t1702 + KIS, ~9 680 files ≈ 536 MiB/day), `disclosures`, `dart_documents`, `financial_facts`, `corporate_actions`, `industry` (`kis_industry` `inquire-price:<ticker>:<UTC date>`, one ISO-week snapshot of the latest-session eligible universe: ~2.5 k calls ≈ 45 min at 1.0 s pace on the week's first run, later runs top up missing tickers only); catalog `ranges` (`ls_investor_flow`, `kis_investor_flow`, end = `D-1`); Silver `investor_flow{,_ls,_kis_supplement}` (`year=YYYY`), `financial_facts`, `financial_quality`, `dividend_events`, `earnings_releases`; `$T/quota/quota_state.json`; `$T/cache/*.sqlite`, `daily_admin_flags.parquet`; prunes unreferenced dataset generations | event `OK`, `built == planned` (7/7 on 10-16), `prune_failed 0`, `prune_unreadable 0`, `flag_cache_sessions` +1 per business day (2644→2645); ~1–2 h | K5, K6, K7 |
| `kse-collect-dbsec` Mon–Fri 20:40 | `collect-dbsec` 1g; `ExecStartPre` marker test + wait ≤ 60 min for `collect-eve` | catalog `dbsec_ticker_master` (1), `dbsec_daily_chart`, `dbsec_closing_bar` (keys `D:<ticker>`); Bronze `daily_market`/`security_master` (~11 k files); `$T/dbsec_reconcile/<session>.json` once KRX Silver for that session exists (i.e. session `D-1` on run `D`) | event `OK`, every step `pending_left 0`, `fallback_ready` false until 20 `pass` reports; ~25 min | K7 |
| `kse-daily-audit` Mon–Fri 21:40, Sat–Sun 12:00 | `audit` 512m | `data/ops/heartbeat/daily_audit.json` (overwritten) | `severity OK`/`HOLIDAY_SKIP`, `open_issues: []`, `undelivered_alerts 0`; issue keys: `collect_eve`, `collect_dbsec`, `morning`, `auction_open`, `backup`, `decision`, `late_seal`, `divergence` (`$T/shadow/**/divergence-*.json` newer than yesterday), `bronze_loose`, `footprint`, `events` | P4 |
| `kse-backup` daily 02:30, `Persistent=true` | `backup --no-lock` under `flock -w 7200 %t/quant-gdrive.lock`, 1g | `data/state/lake_backup_status.json` (`rc`, `finished_at`, `last_ok_at`, `sources.<name>.{scanned,stored,archives}`); Drive `gdrive:quant-lake/live/k-stock-engine/lake`; packs loose Bronze older than 14 d **and** offsite into `$B/_packs` | event `OK`, `rc 0`, sources `bronze, bronze_catalog, ops, state`; `bronze.scanned` ≥ previous run; ~5–13 min | P5 |
| `kse-weekly-report` Sat 10:30 | `weekly-report` 2g | `data/ops/reports/weekly-<D>.json` (`days` = 7 keys → job→outcome) | file exists; `days` outcomes equal the last event line per job/day | P5 |
| `kse-alert@<unit>` on failure | `alert` 256m, stdin = journal tail | `data/ops/alerts/sent.jsonl` (`ts,key,subject`), `ops/alerts/outbox/` when undeliverable; 24 h dedup per key (outbox drain skips keys already sent in the window); body is a ≤5-line card (cause line + journalctl/logs pointer, no journal dump); its own `OnFailure` is `kca-alert@` | one line per failure key; outbox empty | P4 |
| GitHub `kind-relay.yml` cron 08:10 + 09:05 UTC (actual start 21:07–23:28 KST, GitHub delay) | runner fetches KIND, scp to host | `$T/kind_relay/inbox/bundle.json` (`schema kind-relay-v1`, `fetch_date_kst`, `fetched_at`, `items`); consumer writes `$T/kind_relay/last_accepted.json` (`fetched_at`, `items_sha256`) | 2 successful runs per day; `morning D` imports the bundle fetched the night of `D-1` (≈ 10 h old) before `market_actions`, and `collect_eve D` re-imports it idempotently (refused > 36 h) | P10, K1g |
| GitHub `deploy.yml` on push to `main` | test → build `sha-<commit>` → blackout wait → promote `:latest` → `install_systemd.sh` | `~/.config/systemd/user/kse-*`, `~/.local/lib/kse/kse-run`, `~/.cache/kis/.host-admission` (touched), `~/k-stock-engine/.deploy/staging-<sha>/` (never cleaned) | latest run `success`, image revision = `origin/main` | P3 |
| Manual `python -m tools.vps_dart.sync push` (not scheduled) | installs `kse-dart-worker.{service,timer}` (every 15 min) + `~/kse-collect/` | `~/kse-collect/out` | **absent** unless a DART backfill is in progress; `sync finish` removes both | P2 |

## 4. Probes

Host blocks: `ssh or-vps 'bash -s' <<'EOF'` … `EOF`, with `D=YYYY-MM-DD` set inside.

### P1. Host, clock, resources
```bash
TZ=Asia/Seoul date; uptime; df -h / | tail -1; free -m | head -2
timedatectl show -p NTPSynchronized; chronyc tracking 2>/dev/null | grep -E 'System time|Leap'
```
PASS: `NTPSynchronized=yes`, offset < 0.1 s (2026-10-18: 7 µs), free disk ≥ 25 GB (143 GB), available ≥ 3500 MB (9 GiB).

### P2. Timers, last results, stray units
```bash
systemctl --user list-timers --all --no-pager 'kse-*'
systemctl --user --failed --no-pager
for u in morning morning-late auction-open collect-eve collect-dbsec daily-audit backup weekly-report; do printf '%-14s ' $u; systemctl --user show -p Result -p ExecMainStatus -p ExecMainStartTimestamp -p ExecMainExitTimestamp -p DropInPaths --value kse-$u.service | tr '\n' '|'; echo; done
ls ~/.config/systemd/user | grep -E '^kse-' | grep -vE '^kse-(alert@|morning|morning-late|auction-open|collect-eve|collect-dbsec|daily-audit|backup|weekly-report)\.'; ls -d ~/kse-collect 2>/dev/null
TZ=Asia/Seoul journalctl --user -u 'kse-*' --since "$D 00:00" --until "$D 23:59:59" --no-pager -o short-iso | grep -E 'Failed with result|Finished' | cut -c1-110
```
PASS: 7 timers, none failed, `Result=success`, `DropInPaths` empty, no stray `kse-*` file (e.g. `kse-dart-worker.*`) and no
`~/kse-collect` unless a DART backfill was announced. 7 timers stay 7 while 8 service files exist
(`kse-morning-late.service` has no timer). `list-timers` prints UTC. A failure followed by a manual
`systemctl --user start` shows `success` as the last result; the events file (P4) holds the full day (2026-10-16:
`collect-eve` failed 19:41, re-run finished 23:17).

### P3. Deploy drift
```bash
cd ~/.config/systemd/user && for f in kse-*; do printf '%s %s\n' "$(sha256sum < $f | cut -c1-16)" "$f"; done
printf '%s %s\n' "$(sha256sum < ~/.local/lib/kse/kse-run | cut -c1-16)" kse-run
docker image inspect -f '{{.Id}} {{.Created}} {{index .Config.Labels "org.opencontainers.image.revision"}}' ghcr.io/kthyeong/k-stock-engine:latest
docker images ghcr.io/kthyeong/k-stock-engine --format '{{.Tag}}' | wc -l; ls -d ~/k-stock-engine/.deploy/staging-* | wc -l
docker ps -a --filter name=kse- --format '{{.Names}} {{.Status}}'
for j in morning morning-late auction-open collect-eve collect-dbsec daily-audit backup weekly-report alert; do
  printf '== %s: ' "$j"; ~/.local/lib/kse/kse-run --print "$j" | tr '\0' ' ' | sed -E 's/([A-Z_]*(KEY|SECRET|TOKEN)[A-Z_]*)=[^ ]*/\1=<redacted>/g'; echo
done
```
Locally: `cd /home/kth/k-stock-engine && for f in deploy/systemd/* deploy/kse-run; do printf '%s %s\n' "$(sha256sum < $f | cut -c1-16)" "$(basename $f)"; done; git ls-remote origin refs/heads/main; gh run list --workflow deploy.yml --limit 3`.
PASS: all 17 hashes equal; image revision = `origin/main` (2026-10-18: `c65833b`, created 2026-10-17 08:50 UTC); deploy
run `success`; no `kse-*` container; every argv except `alert` mounts `/home/ubuntu/.cache/kis:/app/.cache/kis` with
`KIS_TOKEN_CACHE_DIR`; `collect-dbsec` additionally `~/.cache/dbsec:/app/.cache/dbsec:ro` (legacy, §5); `backup` mounts
rclone config `:ro`; `--memory` equals `--memory-swap` per job (4g/4g/512m/3g/1g/512m/1g/2g/256m). Tag/staging counts are
SEV3 above 5 (2026-10-18: 23 tags, 22 staging dirs).

### P4. Outcomes, heartbeat, alerts
```bash
python3 - "$D" <<'PY'
import json,sys,pathlib,datetime as dt
root=pathlib.Path.home()/"k-stock-engine/data/ops/events"; D=dt.date.fromisoformat(sys.argv[1])
keys=("data_session","sealed","sealed_at","symbols","expected_rows","unresolved","built","planned","pruned","prune_failed","prune_unreadable","flag_cache_sessions","removed","fallback_ready","reconciled","degraded_jobs","peak_rss_mb","node_peak_rss_mb","detail","dbsec_ticker_master","dbsec_daily_chart","dbsec_closing_bar")
for i in range(7,-1,-1):
    d=D-dt.timedelta(days=i); p=root/f"{d:%Y-%m}"/f"{d}.jsonl"
    for l in (p.read_text().splitlines() if p.exists() else []):
        e=json.loads(l); m=e.get("metrics",{})
        if d==D: print(e["ts"][11:19],e["job"],e["outcome"],e["reason"] or "-",{k:m[k] for k in keys if k in m})
        elif e["outcome"] not in ("OK","SKIPPED"): print("prior",d,e["ts"][11:19],e["job"],e["outcome"],e["reason"])
PY
cd ~/k-stock-engine/data/ops && cat heartbeat/daily_audit.json; echo; tail -5 alerts/sent.jsonl | cut -c1-300; ls alerts/outbox 2>/dev/null | wc -l
```
PASS: last line per job on `D`: `morning OK` (`sealed == [P]`), `auction_open OK`, `collect_eve OK` (`built == planned`,
`prune_failed 0`, `flag_cache_sessions` = previous + 1), `collect_dbsec OK` (each step `pending_left 0`), `backup OK`; `blocked on dart_*`인 `collect_eve`는 `investor_flow` 최신·KIS ranges와 공존할 수 있으므로 K5·ranges는 outcome과 무관하게 독립 확인; node
peak RSS ≤ 90 % of the container limit (2026-10-16 `collect_eve` `investor_flow_ls` 2873.6 MB of 3 GiB = 93 % → SEV3);
heartbeat `snapshot_date` = `D`, `open_issues []`; outbox 0. A manual late seal shows as the last `morning_late`
line `DEGRADED late_seal`. SEV1 `late_seal` rule unchanged. Crashes before `cb6df6e` logged no traceback; from it on the
journal carries `[OPS] command=<job> status=DEGRADED reason=<Exc>` plus the traceback (P6 grep).

### P5. Artefact freshness, morning feature coverage, backup, weekly report
```bash
cd ~/k-stock-engine/data
ls -la --time-style=+%m-%dT%H:%M state/kr_swing_2019_v1/decisions/ state/kr_swing_2019_v1/shadow/official/ ops/auction/open.json/$(TZ=Asia/Seoul date +%Y-%m)/ | awk 'NF>5{print $5,$6,$7}'
TZ=Asia/Seoul journalctl --user -u kse-morning.service --since "$D 08:00" --until "$D 09:30" --no-pager -o cat | grep -E '\[DATA\] (stage=(market|fundamentals|releases|flows|dividends) |gate=investor_flow)|ledger replay' | sort -u -k3,4 | cut -c25-220
python3 -c "import json;d=json.load(open('state/lake_backup_status.json'));print(d['rc'],d['finished_at'],d['last_ok_at'],{k:(v['scanned'],v['stored'],v['archives']) for k,v in d['sources'].items()})"
ls -t ops/reports | head -2
```
PASS: decisions `P.json` + `chain.jsonl` and `shadow/official/*.parquet` mtime = morning `D` (~08:51 KST = 23:51 UTC `D-1`).
Feature coverage lines (cube tail 800 sessions × ~3 000 instruments) move ≤ 0.005 per day; any value dropping > 0.01 vs the
previous business day = SEV2. Observed 2026-10-14/15/16: `eligible` 0.823/0.823/0.822, `adtv20` 0.904, `f_assets` 0.836,
`e_net_income_q` 0.775, `flow_for_krw` 0.805, `div_ret` 1.000, `rejects={'capacity': 2|3|3}`. From 2026-10-19 each morning
also logs `gate=investor_flow source_session=prev(P) required=… missing_fraction=…`: must be ≤ 0.02 (`config/research/live_gate.toml`).
Backup: `rc 0`, `last_ok_at` < 26 h, 4 sources, `bronze.scanned` non-decreasing (2026-10-18: 54 910). Weekly report
present for the last Saturday with 7 `days` keys.

### P6. Shared broker dir and contract errors
```bash
ls -la --time-style=+%m-%dT%H:%M ~/.cache/kis/ | awk '{print $1,$3,$5,$6,$7}'
test -f ~/.cache/kis/.host-admission && echo marker=present; stat -c '%a %U %n' ~/.cache/kis
cd ~/.cache/kis && ls -A | python3 -c "
import re,sys
ok=re.compile(r'^(\.host-admission|token_(ls|toss|dbsec|kiwoom)_[0-9a-f]{12}\.json(\.lock)?|token_issue_(ls|toss|dbsec|kiwoom)_[0-9a-f]{12}\.state|admission_(ls|toss|dbsec|kiwoom)_[0-9a-f]{12}(_[A-Za-z0-9_-]+)?\.state|token_[0-9a-f]{12}\.json(\.lock)?|tps_[0-9a-f]{12}\.state)$')
print('non_contract:',[n for n in sys.stdin.read().split() if not ok.match(n)] or 'none')"
ls -la --time-style=+%m-%dT%H:%M ~/.cache/dbsec/ ~/k-stock-engine/logs/ | awk '{print $1,$5,$6,$7}'
P='IGW00201|EGW00201|TokenIssueThrottled|TokenStoreLockTimeout|HostBrokerDirNotSharedError|SCHEMA_INVALID|LEGACY_FALLBACK|ConfigError|SchemaError|Traceback|status=DEGRADED'
TZ=Asia/Seoul journalctl --user -u 'kse-*' --since "$D 00:00" --until "$D 23:59:59" --no-pager -o short-iso | grep -E "$P" | cut -c1-240
```
Expected kse files (`<sha12>` = sha256(app key)[:12]): `token_ls_<sha12>.json`, `admission_ls_<sha12>.state`,
`token_dbsec_<sha12>.json`, `token_issue_dbsec_<sha12>.state`, `admission_dbsec_<sha12>[_chart|_quote].state`, KIS slot-5
`token_<sha12>.json`/`tps_<sha12>.state`; `toss`/`kiwoom` belong to siblings. PASS: marker present, dir `700 ubuntu`,
`non_contract: none` except the known `token_ls_e6e24902d4c9.json.stale-20261006`, zero grep lines and the pattern absent
from event `reason`s. From the first run on `94d99a7`+ (2026-10-19) `admission_dbsec_*` mtimes move to `collect_dbsec` time (`token_dbsec_*` and
`token_issue_dbsec_*` keep their mtime while a valid token is reused: 2026-10-19 15:45 KST) while `~/.cache/dbsec/*` and `logs/ls_token_prod.json` stop changing (last 2026-10-16 20:40 / 18:31
KST); a newer mtime there = the legacy path is still used. `LEGACY_FALLBACK` = SEV3 this release. Map a digest to a key
name only by hashing inside a heredoc and printing the 12-hex digest.

### P7. Failure window and memory
```bash
TZ=Asia/Seoul journalctl --user -u kse-collect-eve.service --since "$D 18:25" --until "$D 21:00" --no-pager -o short-iso | grep -vE 'phase=chunk|pages=' | tail -40
TZ=Asia/Seoul journalctl -k --since "$D 00:00" --no-pager -o short-iso | grep -i 'oom-kill' | cut -c1-250
```
Grep `oom-kill`, not `oom` (kse logs contain `headroom=`). Limits: morning 4g, collect-eve 3g, others per P3. Each collect
job ends with `[DATA] job=<step> phase=done status=complete … pending_left=0` (2026-10-16: `ls_investor_flow` 2 419 done,
2 423 requests, 3 888 s); `status` ≠ `complete` or `pending_left` > 0 = SEV2.

### P8. Footprint, retention, growth projection
```bash
cd ~/k-stock-engine/data && du -sb . | awk '{printf "total_GiB=%.2f\n",$1/2^30}'
du -sh bronze/kr_swing_2019_v1/* | sort -h | tail -5; du -sh silver/*/ gold/*/ state/*/
cd bronze/kr_swing_2019_v1 && nice -n 19 find . -path ./_packs -prune -o -path ./catalog -prune -o -type f -newermt "$(date -d '-15 days' +%F)" -printf '%TY-%Tm-%Td %P %s\n' | awk '{split($2,a,"/"); k=$1" "a[1]; s[k]+=$3; n[k]++} END{for(x in s) printf "%s n=%d MiB=%.1f\n",x,n[x],s[x]/2^20}' | sort | tail -16
du -sh ~/k-stock-engine/scratch ~/k-stock-engine/.deploy; journalctl --user --disk-usage
```
Loose Bronze is packed and pruned by `kse-backup` only when older than 14 days **and** confirmed offsite. 2026-10-20: total
6.02 GiB. LS t1702 is requested one day per call (`start=end`, commits `00af995`/`010836a`), so loose `investor_flow` growth
fell from +536 MiB/business day (9 676 files, 2026-10-12..16) to about 1.6 MiB/day (4 838 files, 2026-10-19); the old
0.54 GiB/day projection no longer applies. Project growth from the last 3 business days of `total`
(SEV3 while a projected 10 GiB crossing is < 10 business days; SEV2 when crossed). First prunable loose day 2026-10-12 →
backup `removed > 0` must appear from the 2026-10-27 02:30 run (the 10-12..16 files are the large ones); still 0 then = SEV2.

### P9. Catalog coverage by dataset (read-only SQLite)
```bash
nice -n 19 python3 - <<'PY'
import sqlite3,os
c=sqlite3.connect('file:'+os.path.expanduser('~/k-stock-engine/data/bronze/kr_swing_2019_v1/catalog/catalog.sqlite3')+'?mode=ro',uri=True)
print(c.execute("select * from catalog_metadata").fetchall())
for r in c.execute("select source,status,count(*),max(retrieved_at) from receipts group by 1,2 order by 1,2"): print(r)
PY
ls -d ~/k-stock-engine/data/silver/kr_swing_2019_v1/daily_market_*/session=* | wc -l
```
`retrieved_at` is UTC; KST day `D` = `[D-1T15:00, DT15:00)`. PASS: `krx_daily_market success` count = Silver
`daily_market` session partitions (2026-10-18: 2 643 = 2 643); `krx_{hedge,trend,cash}_series` equal each other (2 397);
`financial_facts extraction_failed` share < 1 % (408 / 94 997); every daily source's `max(retrieved_at)` = last business
day (`kis_industry` is weekly: `max(retrieved_at)` within the current ISO week; monthly `dart_corp_codes` is static).

### P10. KIND relay, cross-project schedule, Drive lock
```bash
cd ~/k-stock-engine/data/state/kr_swing_2019_v1/kind_relay && stat -c '%y %s %n' inbox/bundle.json last_accepted.json
python3 -c "import json;b=json.load(open('inbox/bundle.json'));print(b['schema'],b['fetch_date_kst'],b['fetched_at'],len(b['items']));print(open('last_accepted.json').read())"
systemctl --user list-timers --all --no-pager | grep -E 'kca|krx|crypto' | awk '{print $3,$(NF-1)}' | sort   # NEXT time, UTC
```
Locally: `gh run list -R kthyeong/k-stock-engine --workflow kind-relay.yml --limit 6`.
PASS: 2 `success` runs per day; `last_accepted.fetched_at` advances on each business day's `morning` import (and again on `collect_eve` when a newer bundle arrived) and is ≤ 36 h older
than that run's start (2026-10-16 22:17 run accepted the 2026-10-16 13:12 UTC bundle); K1g `kind_stale_sessions` is expected to be 0 after morning import when the previous night's bundle arrived
(observed 1 before morning import was introduced; `market_actions` refuses > 5). Neighbours (KST): kca-price-ingest 08:30, kca-auction-open 08:39:55,
kca-order-exit 08:40:10, kca-archive-intraday 20:05, kca-tape-sweep 20:35, kca-backup-prune 21:00, kca-daily-audit 21:20,
crypto-pilot-backup 21:30, kca-altdata-capture 21:35, krx-deferred-recreate 22:00, kca-backup 22:15, kca-extended-backfill
23:05, krx-host-backup 23:30. Drive writers serialize on `%t/quant-gdrive.lock`; shared LS/DB/KIS pacing goes through
`admission_*`/`tps_*` files, so overlap shows as P6 codes, not failures. DART avoids 21:25–21:50 (`shared_ip_avoid_windows_kst`).

### P11. Content probes K1–K7 (local decode, read-only stream)
Run locally after morning `D` (after `collect_dbsec D` for the K7 DB rows). K1–K3/K5 read the current generation; K4/K6/K7
are keyed by `D`. K6 is valid only for the most recent collection day (the ledger keeps one day per endpoint). Transfer ≈ 8 MB.
```bash
D=YYYY-MM-DD; uv run -q --project /home/kth/k-stock-engine python - "$D" <<'PY' 2>&1 | grep -v VIRTUAL_ENV
import io, json, subprocess, sys, tarfile, datetime as dt
import polars as pl

D = dt.date.fromisoformat(sys.argv[1]); R = "~/k-stock-engine/data"
S, G, T = "silver/kr_swing_2019_v1", "gold/kr_swing_2019_v1", "state/kr_swing_2019_v1"
sh = lambda c: subprocess.run(["ssh", "or-vps", f"cd {R} && nice -n 19 {c}"], capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
hostpy = lambda src: subprocess.run(["ssh", "or-vps", "nice -n 19 python3 -"], input=src.encode(), capture_output=True, check=True).stdout.decode()
pq = lambda b: pl.read_parquet(io.BytesIO(b))
cur = json.loads(sh(f"cat {T}/datasets.json"))["current"]
ds = lambda k: f"{G if k in ('market_panel', 'reference_benchmarks') else S}/{cur[k]}"
last = lambda k, n: sh(f"ls -d {ds(k)}/session=* | tail -{n}").decode().split()
parts = {k: last(k, 7) for k in ("daily_market", "ordinary_universe", "hedge_series", "trend_series", "cash_series")}
paths = [p + "/part.parquet" for v in parts.values() for p in v] + [
    f"{ds('investor_flow')}/year={D.year}/part.parquet", f"{T}/decisions", f"{T}/shadow/official", f"{T}/shadow/ideal",
    f"{T}/quota/quota_state.json", f"ops/events/{D:%Y-%m}/{D}.jsonl"] + [
    f"{ds(k)}/manifest.json" for k in ("market_panel", "market_actions", "cash_series", "hedge_series", "trend_series")]
aud = f"ops/auction/open.json/{D:%Y-%m}/{D}.parquet"
paths += [aud] if sh(f"test -f {aud} && echo y || true").strip() else []
with tarfile.open(fileobj=io.BytesIO(sh("tar cf - " + " ".join(paths))), mode="r:") as t:
    F = {m.name: t.extractfile(m).read() for m in t if m.isfile()}
get = lambda suf: next((v for k, v in F.items() if k.endswith(suf)), None)
cat = lambda k: pl.concat([pq(F[p + "/part.parquet"]) for p in parts[k]]).sort("session")
ev = {}
for line in (get(f"{D}.jsonl") or b"").decode().splitlines():
    e = json.loads(line); ev[e["job"]] = e

# K1 KRX Silver/Gold built by morning (P = last daily_market session)
dm = cat("daily_market"); ou = cat("ordinary_universe"); ses = dm["session"].unique().sort().to_list(); P = ses[-1]
tr = (pl.col("price_state") == "tradable")
by = dm.group_by("session").agg(pl.len().alias("rows"), tr.sum().alias("tradable"), (pl.len() - pl.col("instrument_id").n_unique()).alias("dup"),
    (tr & pl.col("close").is_null()).sum().alias("null_close"),
    (tr & ((pl.col("low") > pl.min_horizontal("open", "close")) | (pl.col("high") < pl.max_horizontal("open", "close")) | (pl.col("low") <= 0))).sum().alias("ohlc_bad"),
    ((pl.col("available_at").dt.date() != pl.col("session")) | (pl.col("available_at").dt.hour() != 18)).sum().alias("avail_bad")).sort("session")
by = by.join(ou.group_by("session").agg(pl.len().alias("ou_rows"), pl.col("eligible").sum().alias("eligible")), on="session", how="left")
x = dm.with_columns(pl.col("close").shift(1).over("instrument_id").alias("pc")).filter(pl.col("session") == P, tr)
big = x.filter((pl.col("close") / pl.col("base_price") - 1).abs() > 0.3001)
fluc_bad = x.filter(((pl.col("close") / pl.col("base_price") - 1) * 100 - pl.col("fluc_rate")).abs() > 0.05).height
med = lambda c: by[c].head(by.height - 1).tail(5).median()
print(f"K1 P={P} sessions={[str(s) for s in ses]}")
for r in by.rows(named=True): print("K1 ", {k: str(v) if k == "session" else v for k, v in r.items()})
print(f"K1 rows_ratio={by['rows'][-1] / med('rows'):.4f} tradable_ratio={by['tradable'][-1] / med('tradable'):.4f} eligible_ratio={by['eligible'][-1] / med('eligible'):.4f} "
      f"gt30_vs_base={big.select('ticker', 'fluc_rate').rows()} fluc_rate_mismatch={fluc_bad} base_ne_prev_close={x.filter(pl.col('base_price') != pl.col('pc')).height}")
for k, col in (("hedge_series", "index_level"), ("trend_series", "index_level"), ("cash_series", "cash_close")):
    s = cat(k)
    print(f"K1 {k} last={s['session'][-1]} rows={s.height}/{s['session'].n_unique()} min={s[col].min()} max_abs_dchg={(s[col] / s[col].shift(1) - 1).abs().max():.4f}")
mp = json.loads(get(f"{cur['market_panel']}/manifest.json")); d = mp["details"]
print(f"K1 panel created={mp['created_at'][:19]} gap_rows={d['gap_rows']} limitless_unexplained={d['limitless_move_audit']['unexplained']} rows_{d['partitions'][-1]['year']}={d['partitions'][-1]['rows']} exits={d['exits']['rows']}")
ma = json.loads(get(f"{cur['market_actions']}/manifest.json"))["details"]
print("K1 actions", {k: ma.get(k) for k in ("kind_covered_through", "kind_stale_sessions", "unmapped_rows", "kind_unresolved_rows")})

# K2 decision chain, causality and reference prices
chain = [json.loads(l) for l in get("decisions/chain.jsonl").decode().splitlines()]
c = chain[-1]; tg = c["targets"]; sealed = dt.datetime.fromisoformat(c["sealed_at"])
side = json.loads(get(f"decisions/{c['data_session']}.json") or b"{}")
closes = dict(dm.filter(pl.col("session") == dt.date.fromisoformat(c["data_session"])).select("instrument_id", "close").rows())
late_inputs = [k for k in ("market_panel", "cash_series", "hedge_series", "trend_series") if c["inputs"].get(k) == cur[k]
               and dt.datetime.fromisoformat(json.loads(get(f"{cur[k]}/manifest.json"))["created_at"]) > sealed]
print(f"K2 n={len(chain)} broken_links={sum(b['prev_hash'] != a['hash'] for a, b in zip(chain, chain[1:]))} order_bad={sum(b['data_session'] <= a['data_session'] for a, b in zip(chain, chain[1:]))} "
      f"head={c['data_session']}->{c['execution_session']} sealed={c['sealed_at'][11:19]} sidecar_hash_ok={side.get('hash') == c['hash']} targets={len(tg)} sum_w={sum(tg.values()):.4f} "
      f"max_w={max(tg.values()):.4f} neg_w={sum(v < 0 for v in tg.values())} ref_missing={len(set(tg) - set(c['reference']))} "
      f"ref_ne_close={[k for k, v in c['reference'].items() if closes.get(k) != int(v)]} inputs_after_seal={late_inputs}")

# K3 shadow ledgers (official = on-time decisions only, ideal = all)
navs = {}
for v in ("official", "ideal"):
    nav, fills, rej = (pq(get(f"shadow/{v}/{n}.parquet")) for n in ("nav", "fills", "rejects")); navs[v] = nav
    ln = nav["nav_krw"].cast(pl.Float64).log()
    lr_bad = int(((ln - ln.shift(1)) - nav["log_return"]).tail(nav.height - 1).abs().gt(1e-6).sum())
    h = fills.sort("session").with_columns(pl.when(pl.col("side") == "buy").then(pl.col("quantity")).otherwise(-pl.col("quantity")).cum_sum().over("instrument_id").alias("h"))
    j = fills.join(dm.select("session", "instrument_id", "open"), on=["session", "instrument_id"]).with_columns((pl.col("price") / pl.col("open") - 1).alias("dev"))
    wrong = j.filter(((pl.col("side") == "buy") & (pl.col("dev") < 0)) | ((pl.col("side") == "sell") & (pl.col("dev") > 0))).height
    print(f"K3 {v} nav_last={nav['session'][-1]} nav_krw={nav['nav_krw'][-1]} lr_bad={lr_bad} neg_holding={int((h['h'] < 0).sum())} qty_le0={int((fills['quantity'] <= 0).sum())} "
          f"exec_not_after_decision={fills.filter(pl.col('decision_session') >= pl.col('session')).height} fills_vs_open n={j.height} wrong_side={wrong} max_abs_dev={j['dev'].abs().max():.4f} "
          f"fills_last={fills.filter(pl.col('session') == fills['session'].max()).height} rejects_last={rej.filter(pl.col('session') == rej['session'].max()).height}")
o, i = (navs[k].filter(pl.col("session") == P)["nav_krw"] for k in ("official", "ideal"))
print(f"K3 official_vs_ideal_at_P={(o[0] / i[0] - 1) if len(o) and len(i) else None}")

# K4 auction capture for D
if get(f"{D}.parquet") is None:
    print(f"K4 auction file {D} missing (event: {ev.get('auction_open', {}).get('outcome')})")
else:
    a = pq(get(f"{D}.parquet")); m = ev.get("auction_open", {}).get("metrics", {})
    print("K4 event", {k: m.get(k) for k in ("slots_expected", "slots_captured", "failed", "expected_rows", "persist_error")},
          "slots_complete", "slots_expected" in m and m.get("slots_captured") == m["slots_expected"],
          "symbols_clean", m.get("failed") == {}, "persist_clean", "persist_error" in m and m["persist_error"] is None)
    ex = a.filter(pl.col("kind") == "expected").with_columns(pl.col("output2").str.json_path_match("$.antc_cnpr").cast(pl.Int64, strict=False).alias("px"))
    ref = {k[4:]: v for k, v in next((r["reference"] for r in chain if r["execution_session"] == str(D)), {}).items()}
    far = ex.with_columns(pl.col("symbol").replace_strict(ref, default=None, return_dtype=pl.Float64).alias("ref")).filter((pl.col("px") / pl.col("ref") - 1).abs() > 0.30)
    late = ex.filter(pl.col("received_at").dt.strftime("%H:%M:%S") >= "09:00:00").height
    last_received = ex['received_at'].max()
    print(f"K4 rows={a.height} expected={ex.height}/event {m.get('expected_rows')} file_covers_run={ex.height >= m['expected_rows'] if 'expected_rows' in m else None} open={a.filter(pl.col('kind') == 'open').height}/event {m.get('symbols')} "
          f"slots={sorted({t.strftime('%H:%M') for t in ex['scheduled_at']})} last_expected_recv={last_received.strftime('%H:%M:%S') if last_received else None} expected_after_0900={late} "
          f"px_null_or_le0={int(ex['px'].fill_null(0).le(0).sum())} px_vs_ref_gt30={far.height} dup={a.height - a.select('symbol', 'kind', 'scheduled_at').n_unique()} session_bad={a.filter(pl.col('session') != D).height}")

# K5 investor flow (merged Silver; last session must be prev(P))
f = pq(get(f"year={D.year}/part.parquet")); fs = f["session"].unique().sort().to_list(); FL = fs[-1]; fl = f.filter(pl.col("session") == FL)
tgt = ou.filter(pl.col("session") == FL, pl.col("eligible")).join(dm.filter(pl.col("session") == FL, tr), on="ticker", how="semi")
print(f"K5 flow_last={FL} prev(P)={ses[-2]} rows_tail={[(str(s), n) for s, n in f.group_by('session').len().sort('session').tail(6).rows()]} dup={fl.height - fl['ticker'].n_unique()} "
      f"identity_bad={fl.filter(pl.sum_horizontal('individual_net_shares', 'foreign_net_shares', 'institution_net_shares', 'other_net_shares') != 0).height} "
      f"targets={tgt.height} coverage={tgt.join(fl, on='ticker', how='semi').height / max(tgt.height, 1):.4f} providers={fl.group_by('provider').len().sort('provider').rows()}")

# K6 provider quota for D (daily_attempted_requests per provider label)
use = {}
for k, v in json.loads(get("quota_state.json")).items():
    if v.get("daily_attempt_day") == str(D): use[k.split("|")[0]] = use.get(k.split("|")[0], 0) + int(v.get("daily_attempted_requests", 0))
print(f"K6 quota {D} {dict(sorted(use.items()))}")

# K7 catalog receipts landed during KST day D, DB coverage, reconcile reports
print(hostpy(f"""
import sqlite3,os,json,glob
c=sqlite3.connect('file:'+os.path.expanduser('{R}/bronze/kr_swing_2019_v1/catalog/catalog.sqlite3')+'?mode=ro',uri=True)
lo,hi='{D - dt.timedelta(days=1)}T15:00','{D}T15:00'
print('K7 receipts', [r for r in c.execute("select source,status,count(*) from receipts where retrieved_at>=? and retrieved_at<? group by 1,2 order by 1,2",(lo,hi))])
print('K7 ranges', [r for r in c.execute("select source,status,count(*),max(end) from ranges where retrieved_at>=? and retrieved_at<? group by 1,2 order by 1",(lo,hi))])
db={{s:dict(c.execute("select status,count(*) from receipts where source=? and natural_key like ? group by 1",(s,'{D}:%')).fetchall()) for s in ('dbsec_daily_chart','dbsec_closing_bar')}}
print('K7 dbsec {D}', db)
rr=sorted(glob.glob(os.path.expanduser('{R}/{T}/dbsec_reconcile/*.json')))
print('K7 reconcile n=%d'%len(rr), [{{k:json.load(open(p))[k] for k in ('session','verdict','coverage','close_match_rate','volume_match_rate')}} for p in rr[-3:]])
"""))
PY
```

| ID | Check (output field) | PASS | FAIL → severity | Observed 2026-10-16 (P = 2026-10-15) |
| :--- | :--- | :--- | :--- | :--- |
| K1a | `P`, `sessions` | `P` = prev(`D`); no XKRX day missing | stale `P` → SEV1 (sealed on old data) | P 10-15; 10-09 holiday correctly absent |
| K1b | per session `dup`, `null_close`, `ohlc_bad`, `avail_bad` | all 0 | > 0 → SEV2 | 0 / 0 / 0 / 0 for all 7 sessions |
| K1c | `rows_ratio`, `tradable_ratio`, `eligible_ratio` (vs trailing-5 median); `ou_rows == rows` | each in [0.98, 1.02]; equal | → SEV2 | 1.0000 / 1.0000 / 1.0004; rows 2 761–2 766, tradable 2 651, eligible 2 523–2 525 |
| K1d | `gt30_vs_base`, `fluc_rate_mismatch` | every \|close/base − 1\| > 30 % matches KRX `fluc_rate` (mismatch 0) | mismatch > 0 → SEV2 (parse/adjust corruption) | `179880` +42.03 %, `250030` −37.36 % (KRX-confirmed); mismatch 0; `base_ne_prev_close` 1 (corporate action, INFO) |
| K1e | `{hedge,trend,cash}_series` | `last` = `P`; rows = unique; min > 0; index `max_abs_dchg` ≤ 0.15, cash ≤ 0.01 | missing/≤ 0 → SEV1 (decide raises); jump → SEV2 | 0.0860 / 0.0813 (10-15 circuit-breaker day) / 0.0021 |
| K1f | `panel` | `created` < seal; `limitless_unexplained` 0; `gap_rows` not increasing | → SEV2 | created 08:47 KST, unexplained 0, gap_rows 1, rows_2026 528 942 |
| K1g | `actions` | `kind_stale_sessions` ≤ 2; `kind_covered_through` ≥ prev(`P`) | 3–5 → SEV2; > 5 → SEV1 (build refuses) | stale 1, covered 10-14, unmapped 4, unresolved 375 |
| K2 | chain | `broken_links` 0, `order_bad` 0, `head` → `D`, `sidecar_hash_ok`, `neg_w` 0, `max_w` ≤ 0.10 (sleeve cap), `sum_w` ≤ 1, `ref_missing` 0, `ref_ne_close` [], `inputs_after_seal` [] | any → SEV1 | n 7, 10-15→10-16, sealed 08:51:01, 17 targets, sum 0.9430, max 0.1000 |
| K3 | `official`, `ideal` | `nav_last` = `P`; `lr_bad`, `neg_holding`, `qty_le0`, `exec_not_after_decision`, `wrong_side` = 0; `max_abs_dev` ≤ 0.005 (fill vs open: 1 tick + impact) | invariants → SEV1; dev → SEV2 | NAV 86 347 117 / 86 457 100 (−13.7 % since 10-08, consistent with the 10-12..15 sell-off); dev 0.0030; official vs ideal −0.13 % (INFO, > 5 % investigate) |
| K4 | auction `D` | file present when a decision was sealed; event `slots_captured == slots_expected`, `failed == {}`, `persist_error None`; `expected` = event `expected_rows` (after a same-day rerun: file `expected` ≥ event `expected_rows`); `open` = event `symbols`; slots ⊂ {08:40, 08:50, 08:55, 08:58, 08:59} and after `sealed_at`; `expected_after_0900` 0; `px_null_or_le0` 0; `px_vs_ref_gt30` 0; `dup` 0; `session_bad` 0. Event side: `jq '.metrics | {slots_expected, slots_captured, failed, expected_rows, persist_error}'` | missing/mismatch/late → SEV2; px → SEV3 | 84 = 63 + 21; slots 08:55/08:58/08:59; last expected 08:59:20 (≈ 1 s/symbol: > ~55 symbols spills past 09:00) |
| K5 | flow | `flow_last` = prev(`P`) (flow of `s` needs `next(s)` in the daily calendar); `coverage` ≥ 0.98 (live gate); `identity_bad` 0 (individual + foreign + institution + other = 0); `dup` 0; last `rows_tail` within ±2 %; `blocked on dart_*`인 `collect_eve`와 무관하게 `ranges`·K5를 독립 확인 (DART 차단은 플로우 선행 체인을 막지 않음) | coverage < 0.98 → SEV1 for next morning; else SEV2 | 10-14, coverage 1.0000 (2 419/2 419), identity_bad 0, LS 2 419 + KIS 1 |
| K6 | quota `D` | per label ≤ 80 % of `config/providers.toml`: `OpenDART` (shared primary key) budget 16 000, `OpenDART#<sha8>` 19 500, `LS` 10 000, `KIS` 20 000, `KRX` 10 000, `DBSEC` 50 000, `KIND` 5 000 | > 80 % → SEV2; > limit → SEV1 | DBSEC 5 569, KIS 2 419, LS 2 423, KRX 19, `OpenDART#54f13cf9` 131 |
| K7 | receipts / ranges / dbsec / reconcile | KST day `D`: `krx_*` 1 success each, `dart_disclosure_windows` 4; ranges `ls_investor_flow` count ≥ 0.98 × K5 `targets`, `max(end)` = `P` (`kis_investor_flow` ranges may be 0 while `phase=defer deferred_tail_cells=<targets>` — KIS defers the tail session and LS covers it; K6 then shows no `KIS` label); `dbsec_daily_chart` success within ±2 % of K1 rows; `dbsec_closing_bar` success + empty = daily_chart, empty ≤ 10 %; from run 2026-10-19 one new report per run for `D-1`: `verdict pass`, `coverage` ≥ 0.995, `close_match_rate` ≥ 0.999 | → SEV2 | ranges 2 419/2 419 end 10-15; daily_chart 2 764; closing_bar 2 558 + 206 empty (7.5 %); reconcile n 0 (first due 2026-10-19 for 10-16) |

## 5. Known items (as of 2026-10-18)

| Item | What to check |
| :--- | :--- |
| Legacy DB token dir `~/.cache/dbsec` | Read-only fallback (`DBSEC_TOKEN_CACHE_DIR`, `:ro` mount) for one release; after removal any `~/.cache/dbsec` in `kse-run --print collect-dbsec` = SEV2 drift. Files last changed 2026-10-16 20:40 KST (pre-contract run). |
| `logs/ls_token_prod.json` | Pre-contract LS token (last write 2026-10-16 18:31 KST); current code never reads it. SEV3 until the user removes it; a newer mtime = SEV2. |
| `token_ls_e6e24902d4c9.json.stale-20261006` | Renamed-aside token in `~/.cache/kis`; stays until the user approves deletion; the only accepted `non_contract` entry. |
| `~/k-stock-engine/scratch/lake_backup_check` (133 MB, 2026-10-13) | SEV3 leftover; user deletes. |
| `.deploy/staging-<sha>`, `sha-*` image tags | `deploy.yml` never removes them (22 dirs / 1.8 MB, 23 tags on 2026-10-18). SEV3. |
| First trading day on `c65833b` = 2026-10-19 | Expect `gate=investor_flow` lines (P5), `collect_dbsec` on the shared DB token (P6), first `dbsec_reconcile/2026-10-16.json` after the 20:40 run (K7). |
| KIND relay lands after `collect_eve` | GitHub cron starts 4–5 h late (21:07–23:28 KST), so the next `morning D` imports the `D-1` night bundle before `market_actions` (expected `kind_stale_sessions` 0 when it arrived). After one missed night, the remaining bundle can still be within 36 h at morning (about 33–35 h old), with a one-session lag; import refuses bundles older than 36 h, and `market_actions` still refuses lag > 5 observed sessions. |
| Footprint growth | ~1.6 MiB per business day of loose flow Bronze since LS t1702 went to one-day requests (2026-10-19); the 10-12..16 loose files (536 MiB/day) are removed from the 2026-10-27 backup. |
| `investor_flow_ls_resolve conflict_cells` | 2026-10-19 19:26: `conflict_cells=3` (`year=2026`); conflicting cells are excluded from Silver (`src/data/investor_flow_silver.py` `_distinct==1` filter). Session unidentified; prev(P) coverage stayed 1.0. Investigate when it exceeds 3. |
| Auction snapshots | Seal 08:51–08:54 ⇒ 08:40/08:50 snapshots never captured (`expected_rows = 3 × symbols`). |
| `kse-backup.service.d` drop-in | Removed by `install_systemd.sh` (`2884942`); absent 2026-10-18. Reappearance = SEV2. |

## 6. Chat report template

| Check | Status | Evidence (time KST, value) | Severity |
| :--- | :---: | :--- | :---: |
| Morning seal + KRX data (K1, K2) | `PASS` | `sealed_at` 08:51, chain 10-15→10-16, rows_ratio 1.000 | INFO |
| Shadow ledger (K3) | … | … | … |
| Auction open (K4) | … | … | … |
| Evening collect + flow (K5, K7) | … | … | … |
| DB fallback collect + reconcile (K7) | … | … | … |
| Quota (K6) | … | … | … |
| KIND relay (P10) | … | … | … |
| Daily audit / alerts (P4) | … | … | … |
| Backup (P5) | … | … | … |
| Shared broker dir / contract (P6) | … | … | … |
| Deploy drift (P3) | … | … | … |
| Footprint / growth (P8) | … | … | … |
