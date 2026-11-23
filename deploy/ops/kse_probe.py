"""KSE window probe, run inside the kse image by kse-probe.sh.

Facts and parquet inputs come from the host pass (kse_probe_host.py) via KSE_PROBE_FACTS.
Prints only CTX and verdict lines; probe row failures do not change the exit code.
"""

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl

WIN = sys.argv[1] if len(sys.argv) > 1 else "OPEN"
DOVR = sys.argv[2] if len(sys.argv) > 2 else ""
REPO = os.environ.get("KSE_APP_ROOT", "/app")
FACTS = os.environ.get("KSE_PROBE_FACTS", "/host/facts.tar")

LINES = []


def out(cid: str, status: str, **kv: Any) -> None:
    LINES.append(f"{cid} {status} " + " ".join(f"{k}={v}" for k, v in kv.items()))


def run(cmd: list[str], inp: bytes | None = None, t: int = 300) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, input=inp, capture_output=True, timeout=t)  # noqa: S603


FILES = {}
with tarfile.open(FACTS, mode="r") as _tar:
    for _m in _tar.getmembers():
        if (_fobj := _tar.extractfile(_m)) is not None:
            FILES[_m.name] = _fobj.read()
F = json.loads(FILES.pop("facts.json"))


def _group1(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    assert match is not None, pattern
    return match.group(1)


def pq(b: Any) -> pl.DataFrame:
    return pl.read_parquet(io.BytesIO(b))


def get(suf: str) -> Any:
    return next((v for k, v in FILES.items() if k.endswith(suf)), None)


NOW = dt.datetime.fromisoformat(F["now"])
D = dt.date.fromisoformat(F["D"])
P = F["P"]
CD = F["CD"]
EV = F["events"]


def last(day: Any, job: str) -> Any:
    rows = [e for e in EV.get(str(day), []) if e["job"] == job]
    return rows[-1] if rows else None


mo = last(D, "morning")
if D.weekday() >= 5:
    kind = "weekend"
elif mo and mo["outcome"] == "SKIPPED":
    kind = "holiday"
else:
    kind = "trading"
head = F["chain"][-1] if F["chain"] else {}
origin = F["origin"] or ""
image_rev = F["image"].split()[-1] if F["image"] else "?"
LINES.append(
    f"CTX now={F['now']} win={WIN} replay={bool(DOVR)} day={kind} D={D} P={P} CD={CD} prevCD={F['prevCD']} image={image_rev[:8]} origin={origin[:8] or '-'} host_err={F['err'] or '-'}"
)
trading = kind == "trading"


def skip(cid: str, why: str = "non_trading") -> None:
    out(cid, f"SKIP({why})")


J = F["journal"]

# ---------- R: run outcomes
if WIN in ("OPEN", "EOD"):
    if not trading:
        skip("R1")
    else:
        late = last(D, "morning_late")
        if mo is None:
            out("R1", "FAIL", reason="no_morning_event")
        else:
            m = mo["metrics"]
            sealed = m.get("sealed") or []
            ok = (
                mo["outcome"] == "OK"
                and bool(sealed)
                and sealed[-1] == P
                and str(m.get("sealed_at", "9"))[11:16] < "09:00"
            )  # catch-up seals list several sessions
            out(
                "R1",
                "PASS" if ok else "FAIL",
                outcome=mo["outcome"],
                reason=mo["reason"] or "-",
                sealed=m.get("sealed"),
                sealed_at=str(m.get("sealed_at"))[11:19],
                wait_s=round(m.get("wait_s", 0)),
                build_s=round(m.get("build_s", 0)),
                late=(late or {}).get("outcome", "-"),
            )
if WIN in ("OPEN", "EOD"):
    if not trading:
        skip("R2")
    else:
        a = last(D, "auction_open")
        if a is None:
            out("R2", "FAIL", reason="no_event")
        elif (
            a["outcome"] == "SKIPPED" and a["reason"] == "non_regular_open"
        ):  # shifted-open day (e.g. CSAT 10:00): the job skips by design
            out("R2", "PASS", outcome="SKIPPED", reason="non_regular_open", ts=a["ts"][11:19])
        else:
            m = a["metrics"]
            n = m.get("symbols", 0) or 0
            un = len(m.get("unresolved", []))
            ok = (
                a["outcome"] == "OK"
                and m.get("slots_captured") == m.get("slots_expected")
                and not m.get("failed")
                and m.get("persist_error") is None
            )
            st = "FAIL" if not ok else ("PASS" if un == 0 else "WARN")
            out(
                "R2",
                st,
                outcome=a["outcome"],
                reason=a["reason"] or "-",
                slots=f"{m.get('slots_captured')}/{m.get('slots_expected')}",
                symbols=n,
                expected_rows=m.get("expected_rows"),
                unresolved_ratio=round(un / n, 3) if n else 0,
                failed=len(m.get("failed", {})),
                ts=a["ts"][11:19],
            )


def collect_rows(cd: str) -> Any:
    e = last(cd, "collect_eve")
    prev = None
    for d in sorted(EV):
        if d < cd:
            x = [r for r in EV[d] if r["job"] == "collect_eve" and "flag_cache_sessions" in r["metrics"]]
            if x:
                prev = x[-1]["metrics"]["flag_cache_sessions"]
    if e is None:
        out("R3", "FAIL", cd=cd, reason="no_event")
    else:
        m = e["metrics"]
        r = e["reason"] or "-"
        clean = (
            m.get("built") == m.get("planned")
            and m.get("prune_failed", 0) == 0
            and m.get("prune_unreadable", 0) == 0
            and "prune_error" not in m
        )
        fc_ok = prev is None or m.get("flag_cache_sessions") == prev + 1
        if e["outcome"] == "OK":
            st = "PASS" if clean and fc_ok else "WARN"
        elif e["outcome"] == "DEGRADED" and (r.startswith("optional:") or r == "flag_cache_warm_failed"):
            st = "WARN"
        else:
            st = "FAIL"
        out(
            "R3",
            st,
            cd=cd,
            outcome=e["outcome"],
            reason=r.replace(" ", "_"),
            built=f"{m.get('built')}/{m.get('planned')}",
            prune_failed=m.get("prune_failed"),
            flag_cache=f"{prev}->{m.get('flag_cache_sessions')}",
            kis_industry=(m.get("kis_industry") or {}).get("status"),
            silent_exit0=st == "WARN" and e["outcome"] == "DEGRADED",
        )
    e = last(cd, "collect_dbsec")
    if e is None:
        out("R4", "FAIL", cd=cd, reason="no_event")
    else:
        m = e["metrics"]
        steps = {k: v for k, v in m.items() if isinstance(v, dict) and "pending_left" in v}
        bad = [k for k, v in steps.items() if v.get("status") != "complete" or v.get("pending_left")]
        st = (
            "PASS"
            if e["outcome"] == "OK" and not bad and len(steps) == 3
            else (
                "WARN"
                if e["outcome"] == "DEGRADED" and e["reason"].startswith(("incomplete:", "reconcile:"))
                else "FAIL"
            )
        )
        rec = m.get("reconciled") or []
        out(
            "R4",
            st,
            cd=cd,
            outcome=e["outcome"],
            reason=e["reason"] or "-",
            incomplete=bad or "-",
            reconciled=[(x["session"], x["verdict"]) for x in rec] or "-",
            fallback_ready=m.get("fallback_ready"),
        )
    lines = J.get("eve", [])
    req = (
        "dart_corp_codes",
        "dart_disclosures",
        "dart_facts",
        "dividend_decisions",
        "earnings_releases",
        "ls_investor_flow",
        "kis_investor_flow",
    )
    done = {_group1(r"job=(\S+)", l): l for l in lines if "phase=done" in l}
    miss = [j for j in req if j not in done or "status=complete" not in done[j] or "pending_left=0" not in done[j]]
    blocked = [l for l in lines if "skipped_blocked" in l]
    out(
        "C4",
        "PASS" if not miss and not blocked else "FAIL",
        cd=cd,
        required_incomplete=miss or "-",
        skipped_blocked=len(blocked),
        ls_done=(re.search(r"done=(\d+)", done.get("ls_investor_flow", "")) or [None, None])[1],
    )
    cat = F.get("catalog", {})
    db = cat.get("dbsec", {})
    dc = db.get("dbsec_daily_chart|success", 0)
    cs = db.get("dbsec_closing_bar|success", 0)
    ce = db.get("dbsec_closing_bar|empty", 0)
    return dc, cs, ce, cat


dbsec_counts = None
if WIN in ("OPEN", "EOD", "MORNING"):
    if not CD or (WIN != "MORNING" and not trading):
        skip("R3")
        skip("R4")
        skip("C4")
    else:
        dbsec_counts = collect_rows(CD)

# ---------- B: backup
if WIN in ("OPEN", "MORNING", "WEEKLY"):
    b = last(D, "backup")
    s = F["backup_status"] or {}
    fin = dt.datetime.fromisoformat(s["finished_at"]) if s.get("finished_at") else None
    age_h = round((NOW - fin).total_seconds() / 3600, 1) if fin else None
    srcs = s.get("sources", {})
    errs = [k for k, v in srcs.items() if "error" in v]
    ok = (
        b is not None
        and b["outcome"] == "OK"
        and s.get("rc") == 0
        and s.get("last_ok_at") == s.get("finished_at")
        and age_h is not None
        and age_h <= 26
        and {"bronze", "bronze_catalog", "state", "ops"} <= set(srcs)
        and not errs
    )
    sg_pushed = {"silver", "gold"} <= set(srcs)  # True once the config-derived daily push (K2 fix) is deployed
    lock_wait = None
    if b:
        t0 = dt.datetime.fromisoformat(b["ts"])
        lock_wait = round((t0 - t0.replace(hour=2, minute=30, second=0, microsecond=0)).total_seconds() / 60, 1)
    out(
        "B1",
        "PASS" if ok else "FAIL",
        event=(b or {}).get("outcome"),
        reason=(b or {}).get("reason") or "-",
        rc=s.get("rc"),
        age_h=age_h,
        sources=len(srcs),
        errors=errs or "-",
        bronze_scanned=srcs.get("bronze", {}).get("scanned"),
        stored={k: v.get("stored") for k, v in srcs.items()},
        removed=(b or {}).get("metrics", {}).get("removed"),
    )
    out(
        "X4",
        "PASS" if lock_wait is not None and lock_wait <= 30 else ("WARN" if lock_wait is not None else "UNKNOWN"),
        drive_lock_wait_min=lock_wait,
        note="shared_lock_kca",
    )
    biz = sorted(
        {e["run_date"] for d in EV.values() for e in d if e["job"] == "collect_eve" and e["outcome"] != "SKIPPED"}
    )
    lastbiz = biz[-1] if biz else None
    want_bronze = dt.datetime.fromisoformat(lastbiz + "T02:30:00+09:00") + dt.timedelta(days=1) if lastbiz else None
    for src in ("bronze", "bronze_catalog", "state", "ops", "silver", "gold"):
        try:
            m = json.loads(FILES.get(f"remote/{src}.json", b""))
            up = dt.datetime.fromisoformat(m["updated_at"]).astimezone(NOW.tzinfo)
        except Exception:
            out("B2" if src not in ("silver", "gold") else "B3", "UNKNOWN", src=src, reason="manifest_unreadable")
            continue
        if src in ("silver", "gold"):
            fresh = (
                want_bronze is not None and up >= want_bronze
            )  # silver/gold change only after a collect day, like bronze
            st = ("PASS" if fresh else "FAIL") if sg_pushed else "WARN"
            out(
                "B3",
                st,
                src=src,
                updated=up.isoformat(timespec="minutes"),
                files=m.get("files"),
                daily_push=sg_pushed,
                known="-" if sg_pushed else "K2",
            )
            continue
        if src in ("bronze", "bronze_catalog"):
            ok = want_bronze is not None and up >= want_bronze - dt.timedelta(minutes=5)
        else:
            ok = (NOW - up).total_seconds() <= 26 * 3600
        base = {"bronze": 666258}.get(src)
        trend = "-" if base is None else ("ok" if m.get("files", 0) >= base else "decreased")
        out(
            "B2",
            "PASS" if ok and trend != "decreased" else "FAIL",
            src=src,
            updated=up.isoformat(timespec="minutes"),
            files=m.get("files"),
            archives=len(m.get("archives", {})),
            need_after=(
                want_bronze.isoformat(timespec="minutes")
                if src in ("bronze", "bronze_catalog") and want_bronze
                else "26h"
            ),
            files_vs_baseline=trend,
        )


def series(kind: str) -> Any:
    parts = sorted(k for k in FILES if f"/{F['current'].get(kind)}/session=" in k)
    return pl.concat([pq(FILES[k]) for k in parts]).sort("session") if parts else None


dm = series("daily_market") if FILES else None
# ---------- C: catalog
cat = F.get("catalog", {})
if True:
    if WIN == "OPEN" and trading and P:
        kp = cat.get("krx_P", {})
        bad = [k for k, v in kp.items() if v[0] != "success"]
        par = cat.get("krx_dm_success") == F["dm_partitions"]
        out(
            "C1",
            "PASS" if not bad and par else "FAIL",
            P=P,
            krx_not_success=bad or "-",
            krx_dm_success=cat.get("krx_dm_success"),
            dm_partitions=F["dm_partitions"],
        )
        g = [l for l in J.get("morning", []) if "gate=investor_flow" in l]
        gm = (
            re.search(r"source_session=(\S+) required=(\d+) present=(\d+).*missing_fraction=([\d.]+)", g[-1])
            if g
            else None
        )
        rng = {r[0]: r for r in cat.get("ranges", []) if r[1] == "success"}
        ls = rng.get("ls_investor_flow")
        if gm:
            ok = (
                gm.group(1) == F["prevCD"]
                and float(gm.group(4)) <= 0.02
                and ls is not None
                and ls[2] >= 0.98 * int(gm.group(2))
                and ls[3] == F["prevCD"]
            )
            out(
                "C2",
                "PASS" if ok else "FAIL",
                gate_source=gm.group(1),
                prevP=F["prevCD"],
                required=gm.group(2),
                present=gm.group(3),
                missing_fraction=gm.group(4),
                ls_ranges=ls[2] if ls else 0,
                ls_end=ls[3] if ls else None,
                kis_ranges=rng.get("kis_investor_flow", [0, 0, 0])[2],
            )
        else:
            out("C2", "FAIL", reason="no_gate_line")
    if dbsec_counts is None:
        pass
    else:
        dc, cs, ce, _ = dbsec_counts
        rows_p = (
            dm.filter(pl.col("session") == dt.date.fromisoformat(CD)).height
            if (dm is not None and WIN == "OPEN")
            else None
        )
        c3ok = dc > 0 and cs + ce == dc and ce <= 0.10 * dc and (rows_p is None or abs(dc / max(rows_p, 1) - 1) <= 0.02)
        out(
            "C3",
            "PASS" if c3ok else "WARN",
            cd=CD,
            krx_rows=rows_p,
            daily_chart=dc,
            closing_bar=f"{cs}+{ce}empty",
            master=cat.get("dbsec_master"),
            disclosure_windows=cat.get("disclosure_windows"),
        )

# ---------- Q / P: content (parquet)
if WIN in ("OPEN", "EOD") and trading and P and dm is not None:
    ou = series("ordinary_universe")
    ses = dm["session"].unique().sort().to_list()
    Pd = ses[-1]
    tr = pl.col("price_state") == "tradable"
    if WIN == "OPEN":
        by = (
            dm.group_by("session")
            .agg(
                pl.len().alias("rows"),
                tr.sum().alias("tradable"),
                (pl.len() - pl.col("instrument_id").n_unique()).alias("dup"),
                (tr & pl.col("close").is_null()).sum().alias("null_close"),
                (
                    tr
                    & (
                        (pl.col("low") > pl.min_horizontal("open", "close"))
                        | (pl.col("high") < pl.max_horizontal("open", "close"))
                        | (pl.col("low") <= 0)
                    )
                )
                .sum()
                .alias("ohlc_bad"),
                ((pl.col("available_at").dt.date() != pl.col("session")) | (pl.col("available_at").dt.hour() != 18))
                .sum()
                .alias("avail_bad"),
            )
            .sort("session")
        )
        by = by.join(
            ou.group_by("session").agg(pl.len().alias("ou_rows"), pl.col("eligible").sum().alias("eligible")),
            on="session",
            how="left",
        )

        def med(c: str) -> Any:
            return by[c].head(by.height - 1).tail(5).median()

        ratios = {c: round(by[c][-1] / med(c), 4) for c in ("rows", "tradable", "eligible")}
        x = dm.with_columns(pl.col("close").shift(1).over("instrument_id").alias("pc")).filter(
            pl.col("session") == Pd, tr
        )
        fluc = x.filter(((pl.col("close") / pl.col("base_price") - 1) * 100 - pl.col("fluc_rate")).abs() > 0.05).height
        defects = int(by.select(pl.sum_horizontal("dup", "null_close", "ohlc_bad", "avail_bad")).sum().item())
        ok = (
            str(Pd) == P
            and defects == 0
            and all(0.98 <= v <= 1.02 for v in ratios.values())
            and (by["ou_rows"] == by["rows"]).all()
            and fluc == 0
        )
        out(
            "Q1",
            "PASS" if ok else "FAIL",
            P=Pd,
            rows=by["rows"][-1],
            tradable=by["tradable"][-1],
            eligible=by["eligible"][-1],
            ratios=ratios,
            defects=defects,
            fluc_mismatch=fluc,
            base_ne_prev_close=x.filter(pl.col("base_price") != pl.col("pc")).height,
        )
        bad = []
        for k, col in (("hedge_series", "index_level"), ("trend_series", "index_level"), ("cash_series", "cash_close")):
            s = series(k)
            jump = (s[col] / s[col].shift(1) - 1).abs().max()
            if str(s["session"][-1]) != P or s[col].min() <= 0 or s.height != s["session"].n_unique():
                bad.append(k + ":fail")
            elif jump > (0.01 if k == "cash_series" else 0.15):
                bad.append(f"{k}:jump{jump:.3f}")
        mp = json.loads(get(f"{F['current']['market_panel']}/manifest.json"))
        d = mp["details"]
        ma = json.loads(get(f"{F['current']['market_actions']}/manifest.json"))["details"]
        sealed = dt.datetime.fromisoformat(head["sealed_at"])
        panel_ok = (
            dt.datetime.fromisoformat(mp["created_at"]) < sealed and d["limitless_move_audit"]["unexplained"] == 0
        )
        stale = ma.get("kind_stale_sessions")
        st = (
            "FAIL"
            if any(b.endswith(":fail") for b in bad) or not panel_ok
            else ("WARN" if bad or (stale or 0) > 2 else "PASS")
        )
        out(
            "Q2",
            st,
            series_issues=bad or "-",
            panel_created=mp["created_at"][11:19],
            gap_rows=d.get("gap_rows"),
            limitless_unexplained=d["limitless_move_audit"]["unexplained"],
            kind_stale=stale,
            kind_covered=ma.get("kind_covered_through"),
        )
    f = pl.concat([pq(v) for k, v in FILES.items() if "/year=" in k])
    fs = f["session"].unique().sort().to_list()
    FL = fs[-1]
    fl = f.filter(pl.col("session") == FL)
    want = F[
        "prevP"
    ]  # prev(P) in every window: P is held back as an unavailable tail until the next morning extends the universe calendar (investor_flow_silver.py:589)
    tgt = ou.filter(pl.col("session") == FL, pl.col("eligible")).join(
        dm.filter(pl.col("session") == FL, tr), on="ticker", how="semi"
    )
    cov = tgt.join(fl, on="ticker", how="semi").height / max(tgt.height, 1)
    idb = fl.filter(
        pl.sum_horizontal("individual_net_shares", "foreign_net_shares", "institution_net_shares", "other_net_shares")
        != 0
    ).height
    dup = fl.height - fl["ticker"].n_unique()
    out(
        "Q4",
        "PASS" if str(FL) == want and cov >= 0.98 and idb == 0 and dup == 0 else "FAIL",
        flow_last=FL,
        want=want,
        coverage=round(cov, 4),
        targets=tgt.height,
        identity_bad=idb,
        dup=dup,
        providers=dict(fl.group_by("provider").len().rows()),
    )

if WIN == "OPEN" and trading and P and dm is not None:
    chain = F["chain"]
    c = head
    closes = dict(
        dm.filter(pl.col("session") == dt.date.fromisoformat(c["data_session"])).select("instrument_id", "close").rows()
    )
    tg = c["targets"]
    late_inputs = [
        k
        for k in ("market_panel", "cash_series", "hedge_series", "trend_series")
        if c["inputs"].get(k) == F["current"].get(k)
        and get(f"{F['current'][k]}/manifest.json")
        and dt.datetime.fromisoformat(json.loads(get(f"{F['current'][k]}/manifest.json"))["created_at"])
        > dt.datetime.fromisoformat(c["sealed_at"])
    ]
    links = sum(b["prev_hash"] != a["hash"] for a, b in pairwise(chain))
    order = sum(b["data_session"] <= a["data_session"] for a, b in pairwise(chain))
    refne = [k for k, v in c["reference"].items() if closes.get(k) != int(v)]
    ok = (
        links == 0
        and order == 0
        and c["data_session"] == P
        and c["execution_session"] == str(D)
        and F["sidecar_hash"] == c["hash"]
        and c["sealed_at"][11:16] < "09:00"
        and min(tg.values()) >= 0
        and max(tg.values()) <= 0.1000001
        and sum(tg.values()) <= 1.0000001
        and not (set(tg) - set(c["reference"]))
        and not refne
        and not late_inputs
    )
    out(
        "P1",
        "PASS" if ok else "FAIL",
        n=len(chain),
        head=f"{c['data_session']}->{c['execution_session']}",
        sealed=c["sealed_at"][11:19],
        broken_links=links,
        order_bad=order,
        sidecar_ok=F["sidecar_hash"] == c["hash"],
        targets=len(tg),
        sum_w=round(sum(tg.values()), 4),
        max_w=round(max(tg.values()), 4),
        ref_ne_close=refne or "-",
        inputs_after_seal=late_inputs or "-",
    )
    navs = {}
    inv = {}
    for v in ("official", "ideal"):
        nav, fills, rej = (pq(get(f"shadow/{v}/{n}.parquet")) for n in ("nav", "fills", "rejects"))
        navs[v] = nav
        ln = nav["nav_krw"].cast(pl.Float64).log()
        lr_bad = int(((ln - ln.shift(1)) - nav["log_return"]).tail(nav.height - 1).abs().gt(1e-6).sum())
        h = fills.sort("session").with_columns(
            pl.when(pl.col("side") == "buy")
            .then(pl.col("quantity"))
            .otherwise(-pl.col("quantity"))
            .cum_sum()
            .over("instrument_id")
            .alias("h")
        )
        j = fills.join(dm.select("session", "instrument_id", "open"), on=["session", "instrument_id"]).with_columns(
            (pl.col("price") / pl.col("open") - 1).alias("dev")
        )
        wrong = j.filter(
            ((pl.col("side") == "buy") & (pl.col("dev") < 0)) | ((pl.col("side") == "sell") & (pl.col("dev") > 0))
        ).height
        dev_max: Any = j["dev"].abs().max()
        inv[v] = {
            "nav_last": str(nav["session"][-1]),
            "nav_krw": nav["nav_krw"][-1],
            "lr_bad": lr_bad,
            "neg_holding": int((h["h"] < 0).sum()),
            "qty_le0": int((fills["quantity"] <= 0).sum()),
            "exec_not_after_decision": fills.filter(pl.col("decision_session") >= pl.col("session")).height,
            "wrong_side": wrong,
            "max_abs_dev": round(float(dev_max or 0), 4),
            "fills_last": fills.filter(pl.col("session") == fills["session"].max()).height,
            "rejects_last": rej.filter(pl.col("session") == rej["session"].max()).height,
        }
    for v, r in inv.items():
        zero = all(r[k] == 0 for k in ("lr_bad", "neg_holding", "qty_le0", "exec_not_after_decision", "wrong_side"))
        st = "FAIL" if not zero or r["nav_last"] != P else ("WARN" if r["max_abs_dev"] > 0.005 else "PASS")
        out("P3", st, variant=v, **r)
    ro, ri = F["runs"]["official"], F["runs"]["ideal"]
    same = navs["official"]["session"].to_list() == navs["ideal"]["session"].to_list()
    lo, li = ro["last"][-1], ri["last"][-1]
    spec_stable = all(len(x["last"]) < 2 or x["last"][-1]["spec_hash"] == x["last"][-2]["spec_hash"] for x in (ro, ri))
    ds_diff = sorted(k for k, v in lo["dataset_ids"].items() if F["current"].get(k) not in (None, v))
    ok = (
        same
        and ro["n"] == ri["n"]
        and lo["last_session"] == li["last_session"] == P
        and lo["spec_hash"] == li["spec_hash"]
        and spec_stable
        and not ds_diff
    )
    out(
        "P2",
        "PASS" if ok else "FAIL",
        nav_sessions_equal=same,
        runs_n=f"{ro['n']}/{ri['n']}",
        runs_last=f"{lo['last_session']}/{li['last_session']}",
        spec_stable=spec_stable,
        spec=lo["spec_hash"][:12],
        dataset_ids_vs_current=ds_diff or "same",
    )
    o, i = (navs[k].filter(pl.col("session") == dt.date.fromisoformat(P))["nav_krw"] for k in ("official", "ideal"))
    rep = [_group1(r"rejects=(\{.*?\})", l) for l in J.get("morning", []) if "ledger replay" in l]
    gap = (o[0] / i[0] - 1) if len(o) and len(i) else None
    out(
        "P5",
        "INFO" if gap is not None and abs(gap) <= 0.05 else "WARN",
        official_vs_ideal=round(gap, 4) if gap is not None else None,
        journal_rejects=rep or "-",
        rejects_last=f"{inv['official']['rejects_last']}/{inv['ideal']['rejects_last']}",
    )
    au = get(f"{D}.parquet")
    a = last(D, "auction_open")
    if au is None:
        out("Q6", "FAIL", reason="auction_file_missing")
    else:
        df = pq(au)
        m = (a or {}).get("metrics", {})
        ex = df.filter(pl.col("kind") == "expected").with_columns(
            pl.col("output2").str.json_path_match("$.antc_cnpr").cast(pl.Int64, strict=False).alias("px"),
            pl.col("output2").str.json_path_match("$.antc_vol").cast(pl.Int64, strict=False).alias("vol"),
        )
        ref = {
            k[4:]: v for k, v in next((r["reference"] for r in chain if r["execution_session"] == str(D)), {}).items()
        }
        live = ex.filter(pl.col("vol").fill_null(0) > 0).with_columns(
            pl.col("symbol").replace_strict(ref, default=None, return_dtype=pl.Float64).alias("ref")
        )
        far = live.filter((pl.col("px") / pl.col("ref") - 1).abs() > 0.30).height
        late = ex.filter(pl.col("received_at").dt.strftime("%H:%M:%S") >= "09:00:00").height
        dup = df.height - df.select("symbol", "kind", "scheduled_at").n_unique()
        ok = (
            ex.height >= (m.get("expected_rows") or 0)
            and late == 0
            and dup == 0
            and df.filter(pl.col("session") != D).height == 0
        )
        st = "FAIL" if not ok else ("WARN" if far or live.filter(pl.col("px").fill_null(0) <= 0).height else "PASS")
        out(
            "Q6",
            st,
            rows=df.height,
            expected=f"{ex.height}/{m.get('expected_rows')}",
            open=df.filter(pl.col("kind") == "open").height,
            slots=sorted({t.strftime("%H:%M") for t in ex["scheduled_at"]}),
            after_0900=late,
            px_vs_ref_gt30=far,
            dup=dup,
        )
if WIN in ("OPEN", "EOD", "MORNING") and CD and (trading or WIN == "MORNING"):
    rc = F["reconcile"]
    lr = rc["last"][-1] if rc["last"] else {}
    want = F["prevCD"]
    ok = (
        lr.get("session") == want
        and lr.get("verdict") == "pass"
        and (lr.get("coverage") or 0) >= 0.995
        and (lr.get("close_match_rate") or 0) >= 0.999
    )
    out(
        "Q7",
        "PASS" if ok else "WARN",
        latest=lr.get("session"),
        want=want,
        verdict=lr.get("verdict"),
        coverage=lr.get("coverage"),
        close=lr.get("close_match_rate"),
        reports=rc["n"],
    )

# ---------- P4 divergence (any window)
out("P4", "PASS" if not F["divergence"] else "FAIL", divergence_files=F["divergence"] or "-")

# ---------- A: alerting / liveness
hb = F["heartbeat"] or {}
fin = dt.datetime.fromisoformat(hb["finished_at"]) if hb.get("finished_at") else None
known = {"bronze_loose"}
issues = [i["key"] for i in hb.get("open_issues", [])]
if WIN == "EOD":
    if NOW.time() < dt.time(21, 42):
        out("A1", "UNKNOWN", reason="before_21:42")
    else:
        fresh = hb.get("snapshot_date") == str(D) and fin is not None and fin.time() >= dt.time(21, 40)
        out(
            "A1",
            "FAIL" if not fresh else ("PASS" if not issues else ("WARN" if set(issues) <= known else "FAIL")),
            snapshot=hb.get("snapshot_date"),
            finished=str(hb.get("finished_at"))[11:19],
            severity=hb.get("severity"),
            issues=issues or "-",
            undelivered=hb.get("undelivered_alerts"),
            outbox=F["outbox"],
        )
else:
    age = round((NOW - fin).total_seconds() / 3600, 1) if fin else None
    out(
        "A1",
        "FAIL" if age is None or age > 26 else ("PASS" if not issues else ("WARN" if set(issues) <= known else "FAIL")),
        snapshot=hb.get("snapshot_date"),
        age_h=age,
        severity=hb.get("severity"),
        issues=issues or "-",
        undelivered=hb.get("undelivered_alerts"),
        outbox=F["outbox"],
    )
since_day = {"OPEN": P or str(D), "EOD": str(D), "MORNING": CD or str(D)}.get(WIN, str(D - dt.timedelta(days=3)))
silent = [
    f"{d}:{e['job']}:{e['reason'].split(',')[0][:40]}"
    for d in sorted(EV)
    if since_day <= d <= str(D)
    for e in {x["job"]: x for x in EV[d]}.values()
    if e["outcome"] == "DEGRADED"
    and (
        e["reason"].startswith(("optional:", "incomplete:", "reconcile:"))
        or e["reason"] in ("late_seal", "flag_cache_warm_failed")
    )
]
out("A2", "WARN" if silent else "PASS", silent_degraded_exit0=silent or "-", note="no_unit_alert_for_these")
stale_ok = (
    {"kse-morning-late.service"} if mo is not None and mo["outcome"] == "OK" else set()
)  # failed state of a late seal is a leftover once today's seal is OK
failed_now = [u for u in F["failed"] if u not in stale_ok]
fails = [u for u, v in F["units"].items() if v and v[0] not in ("success", "") and f"kse-{u}.service" not in stale_ok]
_repo_units = sorted(
    __import__("pathlib").Path(REPO, "deploy/systemd").glob("kse-*")
)  # expected counts follow the repo, not literals
want_timers, want_units = sum(1 for u in _repo_units if u.suffix == ".timer"), len(_repo_units) + 1  # +1 kse-run
out(
    "H1",
    "PASS"
    if F["timers"] == want_timers
    and not failed_now
    and not fails
    and not any(v[2] for v in F["units"].values() if len(v) > 2)
    and not F["kse_collect"]
    and len(F["unit_files"]) == want_units
    else "FAIL",
    timers=f"{F['timers']}/{want_timers}",
    failed=failed_now or "-",
    ignored_stale=sorted(set(F["failed"]) & stale_ok) or "-",
    bad_result=fails or "-",
    unit_files=f"{len(F['unit_files'])}/{want_units}",
    kse_collect=F["kse_collect"],
    containers=[c for c in F["containers"] if c] or "-",
)
if origin:
    out(
        "H2",
        "PASS" if image_rev == origin else "WARN",
        image=image_rev[:8],
        origin=origin[:8],
        image_created=F["image"][:16],
    )
else:
    out("H2", "SKIP(origin_unavailable)", image=image_rev[:8], image_created=F["image"][:16])
codes = J.get("codes", []) + J.get("oom", [])

victims = sorted(
    {
        f"{(re.search(r'task=([^,]+),', l) or [0, '?'])[1]}@{(re.search(r'docker-([0-9a-f]{12})', l) or [0, 'host'])[1]}"
        for l in J.get("oom", [])
    }
)  # who the kernel killed, to tell kse-preseal from other tenants
out(
    "H4",
    "PASS" if not codes else "WARN",
    journal_codes=len(J.get("codes", [])),
    oom_kill=len(J.get("oom", [])),
    victims=victims or "-",
    first=(codes[0][:120].replace(" ", "_") if codes else "-"),
)
out(
    "X1",
    "PASS" if F["mem_mb"] >= 3500 and F["disk_gib"] >= 25 and F["ntp"] == "yes" else "FAIL",
    mem_avail_mb=F["mem_mb"],
    free_disk_gib=F["disk_gib"],
    ntp=F["ntp"],
    note="shared",
)
bk = F["broker"]
nc = [n for n in bk["non_contract"] if not n.endswith(".stale-20261006")]
out(
    "X3",
    "PASS" if bk["marker"] and bk["mode"] == "700" and not nc else "FAIL",
    marker=bk["marker"],
    mode=bk["mode"],
    non_contract=nc or "-",
)
if WIN in ("OPEN", "MORNING") and shutil.which("gh"):
    k = F["kind"]
    acc = (k.get("accepted") or {}).get("fetched_at")
    age = (
        round((NOW - dt.datetime.fromisoformat(k["bundle_fetched_at"])).total_seconds() / 3600, 1)
        if k.get("bundle_fetched_at")
        else None
    )
    rel = (mo or {}).get("metrics", {}).get("kind_relay", []) if WIN == "OPEN" else []
    bad_rel = [r["job"] for r in rel if r.get("status") not in ("complete", "unchanged", "imported", "ok")]
    gh = run(
        [
            "gh",
            "run",
            "list",
            "-R",
            "kthyeong/k-stock-engine",
            "--workflow",
            "kind-relay.yml",
            "--limit",
            "4",
            "--json",
            "conclusion,createdAt",
        ],
        t=60,
    ).stdout
    try:
        runs = json.loads(gh)
    except Exception:
        runs = []
    st = (
        "PASS"
        if age is not None
        and age <= 36
        and (WIN == "MORNING" or acc == k["bundle_fetched_at"])
        and not bad_rel
        and runs
        and runs[0]["conclusion"] == "success"
        else "WARN"
    )
    out(
        "C6",
        st,
        bundle_age_h=age,
        fetch_date=k.get("fetch_date_kst"),
        items=k.get("items"),
        accepted_eq_bundle=acc == k.get("bundle_fetched_at"),
        relay_status=[r.get("status") for r in rel] or "-",
        gh_last=[(r["createdAt"][5:16], r["conclusion"]) for r in runs[:2]],
    )
elif WIN in ("OPEN", "MORNING"):
    out("C6", "SKIP(no_gh_on_vps)")
if WIN == "EOD" and trading:
    lab = {"OpenDART": 16000, "LS": 10000, "KIS": 20000, "KRX": 10000, "DBSEC": 50000, "KIND": 5000}
    use = {k.split("@")[0]: v for k, v in F["quota"].items() if k.endswith("@" + str(D))}
    hot = {
        k: v
        for k, v in use.items()
        if v > 0.8 * lab.get(k.split("#")[0], 19500 if k.startswith("OpenDART#") else 10**9)
    }
    out("C5", "PASS" if not hot else "WARN", used=use or "-", over80=hot or "-")
if WIN == "WEEKLY":
    loc = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in sorted(Path(REPO, "deploy/systemd").glob("kse-*"))
    }
    loc["kse-run"] = hashlib.sha256(Path(REPO, "deploy/kse-run").read_bytes()).hexdigest()[:16]
    diff = sorted(k for k in set(loc) | set(F["unit_files"]) if loc.get(k) != F["unit_files"].get(k))
    out(
        "H2W",
        "WARN" if diff or (origin and image_rev != origin) else ("PASS" if origin else "SKIP(origin_unavailable)"),
        drift=diff or "-",
        image=image_rev[:8],
        origin=origin[:8] or "-",
    )
    out("H3", "INFO", image_tags=F["image_tags"], deploy_entries=F["deploy_entries"], scratch=F["scratch"] or "-")
    krx = F["krx_lib_sha"]
    lib = krx is not None and krx == hashlib.sha256(Path(REPO, "deploy/vps-deploy-lib.sh").read_bytes()).hexdigest()
    slots = Path(REPO, "deploy/shared/kis_slot_assignments.toml").read_text(encoding="utf-8")
    out(
        "X6",
        "SKIP(krx_lib_missing)" if krx is None else ("PASS" if lib else "FAIL"),
        deploy_lib_identical=lib,
        note="shared",
    )
    out(
        "X5",
        "PASS" if 'project = "kse"' in slots else "WARN",
        kse_in_slot_registry='project = "kse"' in slots,
        known="K-slot5-undeclared",
    )
    fp = F["footprint"]
    out(
        "B6",
        "PASS" if fp["total"] < 10 * 2**30 else "FAIL",
        total_gib=round(fp["total"] / 2**30, 2),
        packs_gib=round(fp["packs"] / 2**30, 2),
        loose_files=fp["loose"],
    )
    import gzip

    import pyarrow as pa

    try:
        found: set[str] = set()
        with pa.input_stream(pa.BufferReader(FILES.get("remote/bronze_index.jsonl.zst", b"")), compression="zstd") as s:
            for line in s.read().splitlines():
                if line.strip():
                    found.add(json.loads(line)["path"])
        paths: set[str] | None = found
    except Exception:
        paths = None
    if paths is None:
        out("B5", "UNKNOWN", reason="index_unreadable")
    else:
        bstat = F["backup_status"] or {}
        cut = dt.datetime.fromisoformat(bstat["finished_at"]).timestamp() - 3 * 3600 if bstat.get("finished_at") else 0
        missing = [
            l.split("\t")[0]
            for l in gzip.decompress(FILES["loose.txt.gz"]).decode().splitlines()
            if l and int(l.split("\t")[1]) < cut and l.split("\t")[0] not in paths
        ]
        out(
            "B5",
            "PASS" if not missing else "FAIL",
            index_paths=len(paths),
            loose_not_offsite=len(missing),
            sample=missing[:3] or "-",
        )
    rep = F["reports"]
    sat = D - dt.timedelta(days=(D.weekday() - 5) % 7)
    out(
        "R8",
        "PASS" if f"weekly-{sat}.json" in rep else "FAIL",
        latest=rep[-1] if rep else None,
        want=f"weekly-{sat}.json",
    )

sys.stdout.write("\n".join(LINES) + "\n")
