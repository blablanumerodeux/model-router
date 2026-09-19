#!/usr/bin/env python3
"""Threshold calibration for model-router — the RouteLLM quantile exercise.

The gates in router.py (CONF_GATE, STAKES_GATE, SPEED_GATE, COMPLEX_HARD) are
hand-picked conservative defaults. This tool turns them into a *traffic-mix*
decision: pick a target ("20% of requests may reach the strong tier"), read the
threshold straight off the observed score distribution, then replay real
decisions through the actual policy to see the resulting tier mix.

Method (RouteLLM, arXiv 2406.18665):
    threshold = quantile(1 - target) of the observed scores
e.g. target 20% strong -> cut at the 80th percentile of stakes scores.

Read-only by default. `--write NAME=VALUE` edits router.py explicitly.

Usage
-----
    python3 calibrate.py                            # summary + target table
    python3 calibrate.py --target 0.2               # focus: threshold + replay
    python3 calibrate.py --simulate STAKES_GATE=0.55
    python3 calibrate.py --json                     # machine-readable
    python3 calibrate.py --write STAKES_GATE=0.55   # apply (prints diff)

Data source: logs/decisions.jsonl entries carrying a "scores" block (raw jev
values per decision: task, task_conf, complexity, complexity_conf, stakes,
speed). Entries are deduplicated on the score vector by default — tool-loop
hops replay the same conversation and would otherwise over-weight one prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import router as R  # noqa: E402

LOG_PATH = HERE / "logs" / "decisions.jsonl"
GATE_NAMES = ("CONF_GATE", "STAKES_GATE", "SPEED_GATE", "COMPLEX_HARD")
SCORE_KEYS = ("stakes", "speed", "complexity", "task_conf")


# --------------------------------------------------------------------- stats

def quantile(xs: list[float], q: float) -> float | None:
    """Linear-interpolation quantile — same convention as numpy.quantile."""
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])


def pct_above(xs: list[float], thr: float) -> float:
    if not xs:
        return 0.0
    return 100.0 * sum(1 for x in xs if x >= thr) / len(xs)


def pct_below(xs: list[float], thr: float) -> float:
    if not xs:
        return 0.0
    return 100.0 * sum(1 for x in xs if x < thr) / len(xs)


# ---------------------------------------------------------------- log loading

def load_rows(days: float | None, include_cached: bool, dedupe: bool) -> tuple[list[dict], int]:
    """Return (rows, total_scored_entries). rows carry scores + observed tier."""
    if not LOG_PATH.exists():
        return [], 0
    cutoff = time.time() - days * 86400 if days else None
    rows: list[dict] = []
    scored = 0
    for line in LOG_PATH.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        s = d.get("scores")
        if not isinstance(s, dict) or s.get("stakes") is None:
            continue
        scored += 1
        if cutoff and (d.get("ts") or 0) < cutoff:
            continue
        if d.get("cached") and not include_cached:
            continue
        rows.append({"scores": s, "tier": d.get("tier"), "cached": bool(d.get("cached")),
                     "ts": d.get("ts"), "stakes": s.get("stakes")})
    if dedupe:
        seen = set()
        uniq = []
        for r in rows:
            key = tuple(r["scores"].get(k) for k in SCORE_KEYS + ("task",))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        rows = uniq
    return rows, scored


# -------------------------------------------------------------------- replay

def replay(rows: list[dict], overrides: dict[str, float]) -> tuple[Counter, int]:
    """Re-run R.decide() on logged scores with candidate gates applied."""
    saved = {k: getattr(R, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(R, k, float(v))
        mix: Counter = Counter()
        changed = 0
        for r in rows:
            s = r["scores"]
            answers = {
                "task_type": {"choice": s.get("task"), "confidence": s.get("task_conf")},
                "complexity": {"score": s.get("complexity"),
                               "confidence": s.get("complexity_conf")},
                "high_stakes": {"noul": s.get("stakes")},
                "speed_priority": {"noul": s.get("speed")},
            }
            tier = R.decide(answers)["tier"]
            mix[tier] += 1
            if r.get("tier") and tier != r["tier"]:
                changed += 1
        return mix, changed
    finally:
        for k, v in saved.items():
            setattr(R, k, v)


def mix_line(mix: Counter, n: int) -> str:
    if not n:
        return "(no data)"
    order = ["fast", "balanced", "code", "creative", "strong", "forced"]
    parts = [f"{t} {100.0 * mix[t] / n:.0f}%" for t in order if mix.get(t)]
    extras = [t for t in mix if t not in order]
    parts += [f"{t} {100.0 * mix[t] / n:.0f}%" for t in sorted(extras)]
    return " · ".join(parts)


def search_max_strong(rows: list[dict], target: float) -> dict | None:
    """Find the most permissive gates that keep the strong tier at <= target.

    The strong tier is reachable two ways: the stakes gate, and "analysis with
    complexity >= COMPLEX_HARD". A cap therefore needs both levers, so we scan
    observed thresholds ascending (safety-first: smallest STAKES_GATE wins,
    then smallest COMPLEX_HARD) and return the first combination that fits.
    """
    n = len(rows)
    if not n:
        return None
    stakes_vals = sorted({float(r["scores"]["stakes"]) for r in rows
                          if r["scores"].get("stakes") is not None})
    comp_vals = sorted({float(r["scores"]["complexity"]) for r in rows
                        if r["scores"].get("complexity") is not None})
    if not stakes_vals:
        return None
    comps = comp_vals or [float(getattr(R, "COMPLEX_HARD"))]
    best_floor: tuple[float, float, float] = (1.0, stakes_vals[0], comps[0])
    for s in stakes_vals:
        for c in comps:
            mix, _ = replay(rows, {"STAKES_GATE": s, "COMPLEX_HARD": c})
            share = mix.get("strong", 0) / n
            if share < best_floor[0]:
                best_floor = (share, s, c)
            if share <= target:
                return {"STAKES_GATE": s, "COMPLEX_HARD": c, "strong_share": share,
                        "tier_mix": dict(mix), "found": True}
    return {"STAKES_GATE": best_floor[1], "COMPLEX_HARD": best_floor[2],
            "strong_share": best_floor[0], "found": False,
            "note": "target unreachable on this sample — strong tier cannot go lower"}



# -------------------------------------------------------------------- report

def main() -> int:
    ap = argparse.ArgumentParser(description="Quantile calibration for model-router gates")
    ap.add_argument("--target", type=float, action="append", default=None,
                    help="target fraction of traffic to the strong tier (repeatable, e.g. 0.2)")
    ap.add_argument("--simulate", nargs="+", default=None, metavar="NAME=VALUE",
                    help="candidate gates to replay, e.g. STAKES_GATE=0.55 SPEED_GATE=0.5")
    ap.add_argument("--write", nargs="+", default=None, metavar="NAME=VALUE",
                    help="apply gates to router.py (explicit, prints the change)")
    ap.add_argument("--max-strong", type=float, default=None, metavar="P",
                    help="cap the strong tier at fraction P of traffic (e.g. 0.10) "
                         "→ prints the most permissive gates that satisfy it")
    ap.add_argument("--days", type=float, default=None, help="only consider the last N days")
    ap.add_argument("--min-samples", type=int, default=100,
                    help="warn below this many unique decisions (default 100)")
    ap.add_argument("--include-cached", action="store_true",
                    help="keep cache hits (tool-loop hops) in the sample")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="do not deduplicate identical score vectors")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    rows, scored_entries = load_rows(args.days, args.include_cached, not args.no_dedupe)
    n = len(rows)
    stakes = [float(r["scores"]["stakes"]) for r in rows]
    speed = [float(r["scores"]["speed"]) for r in rows]
    complexity = [float(r["scores"]["complexity"]) for r in rows if r["scores"].get("complexity") is not None]
    tconf = [float(r["scores"]["task_conf"]) for r in rows if r["scores"].get("task_conf") is not None]

    current = {k: float(getattr(R, k)) for k in GATE_NAMES}
    observed_mix = Counter(r["tier"] for r in rows if r.get("tier"))

    targets = args.target or [0.10, 0.20, 0.30, 0.40, 0.50]
    table = []
    for t in targets:
        thr = quantile(stakes, 1.0 - t)
        table.append({
            "target_strong_pct": round(t * 100, 1),
            "stakes_threshold": None if thr is None else round(thr, 3),
            "realized_pct": None if thr is None else round(pct_above(stakes, thr), 1),
        })

    report = {
        "log_path": str(LOG_PATH),
        "scored_entries": scored_entries,
        "usable_decisions": n,
        "deduped": not args.no_dedupe,
        "include_cached": args.include_cached,
        "enough_data": n >= args.min_samples,
        "min_samples": args.min_samples,
        "current_gates": current,
        "observed_tier_mix": dict(observed_mix),
        "percentiles": {
            "stakes": {p: (None if (v := quantile(stakes, p)) is None else round(v, 3))
                       for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
            "speed": {p: (None if (v := quantile(speed, p)) is None else round(v, 3))
                      for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
            "complexity": {p: (None if (v := quantile(complexity, p)) is None else round(v, 3))
                           for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
            "task_conf": {p: (None if (v := quantile(tconf, p)) is None else round(v, 3))
                          for p in (0.1, 0.25, 0.5, 0.75, 0.9)},
        },
        "current_gate_traffic_pct": {
            "stakes_above_STAKES_GATE": round(pct_above(stakes, current["STAKES_GATE"]), 1),
            "speed_above_SPEED_GATE": round(pct_above(speed, current["SPEED_GATE"]), 1),
            "task_conf_below_CONF_GATE": round(pct_below(tconf, current["CONF_GATE"]), 1),
        },
        "quantile_targets": table,
    }

    if args.max_strong is not None and n:
        cap = search_max_strong(rows, args.max_strong)
        if cap:
            report["max_strong"] = {**cap, "target": args.max_strong}

    # explicit simulation
    if args.simulate:
        overrides = {}
        for item in args.simulate:
            if "=" not in item or item.split("=", 1)[0] not in GATE_NAMES:
                print(f"error: --simulate expects NAME=VALUE with NAME in {GATE_NAMES}", file=sys.stderr)
                return 2
            k, v = item.split("=", 1)
            overrides[k] = float(v)
        mix, changed = replay(rows, overrides)
        report["simulation"] = {
            "overrides": overrides,
            "tier_mix": dict(mix),
            "decisions_changed": changed,
            "changed_pct": round(100.0 * changed / n, 1) if n else None,
        }

    # write
    if args.write:
        rp = HERE / "router.py"
        src = rp.read_text()
        applied = {}
        for item in args.write:
            if "=" not in item or item.split("=", 1)[0] not in GATE_NAMES:
                print(f"error: --write expects NAME=VALUE with NAME in {GATE_NAMES}", file=sys.stderr)
                return 2
            k, v = item.split("=", 1)
            float(v)
            new_src, cnt = re.subn(rf"^{k}\s*=\s*[-\d.]+", f"{k} = {v}", src, count=1, flags=re.M)
            if cnt != 1:
                print(f"error: could not find `{k} = <number>` in router.py", file=sys.stderr)
                return 2
            src = new_src
            applied[k] = float(v)
        rp.write_text(src)
        report["written"] = applied

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    # ------------------------------------------------------------- text output
    print("== model-router calibration ==")
    if args.days:
        print(f"window          : last {args.days:g} days")
    print(f"scored entries  : {scored_entries}")
    print(f"usable decisions: {n}" + ("  (deduplicated)" if not args.no_dedupe else ""))
    if n == 0:
        print("\nNo scored decisions yet. The 'scores' block is written by "
              "server.py since the calibration commit — generate traffic through "
              "the server (POST /v1/chat/completions) and re-run.")
        return 0
    if not report["enough_data"]:
        print(f"\nWARNING: {n} unique decisions < {args.min_samples}. Thresholds below "
              f"are indicative only — gather more traffic before applying.")

    print(f"\n-- score distributions (percentiles) --")
    print(f"{'score':<12}{'p10':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p90':>8}")
    for name, key in (("stakes", "stakes"), ("speed", "speed"),
                      ("complexity", "complexity"), ("task_conf", "task_conf")):
        p = report["percentiles"][key]
        print(f"{name:<12}" + "".join(f"{(p[x] if p[x] is not None else 0):>8.2f}"
                                      for x in (0.1, 0.25, 0.5, 0.75, 0.9)))

    print("\n-- current gates --")
    for k, v in current.items():
        print(f"{k:<14}= {v}")
    t = report["current_gate_traffic_pct"]
    print(f"  stakes >= STAKES_GATE      : {t['stakes_above_STAKES_GATE']}% of requests")
    print(f"  speed  >= SPEED_GATE       : {t['speed_above_SPEED_GATE']}% of requests")
    print(f"  task_conf < CONF_GATE      : {t['task_conf_below_CONF_GATE']}% → balanced")
    print(f"  observed tier mix          : {mix_line(observed_mix, n)}")

    print("\n-- quantile targets (stakes → strong) --")
    print(f"{'target':>8}{'threshold':>12}{'realized':>11}")
    for row in table:
        if row["stakes_threshold"] is None:
            continue
        thr = row["stakes_threshold"]
        flag = "  <- current" if abs(thr - current["STAKES_GATE"]) < 1e-9 else ""
        print(f"{row['target_strong_pct']:>7.0f}%{thr:>12.3f}{row['realized_pct']:>10.1f}%{flag}")

    if "max_strong" in report:
        m = report["max_strong"]
        print(f"\n-- strong-tier cap (target <{m['target'] * 100:.0f}%) --")
        if m["found"]:
            print(f"recommended : STAKES_GATE={m['STAKES_GATE']}  COMPLEX_HARD={m['COMPLEX_HARD']}")
            print(f"strong share: {100 * m['strong_share']:.1f}%  (was "
                  f"{100 * observed_mix.get('strong', 0) / n:.1f}%)")
            print(f"tier mix    : {mix_line(Counter(m['tier_mix']), n)}")
        else:
            print(f"NOT reachable on this sample — {m.get('note', '')}")
            print(f"floor: strong {100 * m['strong_share']:.1f}% at STAKES_GATE="
                  f"{m['STAKES_GATE']}, COMPLEX_HARD={m['COMPLEX_HARD']}")

    if args.simulate and "simulation" in report:
        s = report["simulation"]
        print("\n-- simulation (in-sample replay) --")
        print("gates           : " + ", ".join(f"{k}={v}" for k, v in s["overrides"].items()))
        print(f"tier mix        : {mix_line(Counter(s['tier_mix']), n)}")
        print(f"decisions moved : {s['decisions_changed']} ({s['changed_pct']}%)")

    if "written" in report:
        print("\n-- written to router.py --")
        for k, v in report["written"].items():
            print(f"{k} = {v}")
        print("restart: systemctl --user restart model-router")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
