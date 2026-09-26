"""G15 analysis (declared in G15-TRANSFER.md before the run). Written before any G15 result was
read; reads whatever seeds exist and says how many.

Outcome sets (from transfer_split.json):
    L      routes never rehearsed in any condition        -> T1 transfer, T5 floor
    Rall   routes rehearsed in all four conditions        -> T2 replication, T3 format, T4 emission
Conditions: route, atomic, coherent, bridgectx (+ none = the no-replay adapters).
Estimator as G10: per-seed condition means on the set, paired differences over seeds, t(k-1).
Held-out access = (hits on forms 1-5, biomedical persona + 6 x neutral-persona access) / 11;
for L the canonical form is also unseen and is reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TC = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
CONDS = ("none", "route", "atomic", "coherent", "bridgectx")


def norm(s) -> str:
    return " ".join(str(s).lower().split())


def main() -> None:
    from strict_match import match_strict

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks/v2")
    ap.add_argument("--split", default="data/tasks/v2_transfer/transfer_split.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/g15_transfer.json")
    ap.add_argument("--family", default="qwen", choices=["qwen", "falcon"],
                    help="falcon: reads falcon_transfer_* / falcon_preserve_none_* files and skips coherent")
    args = ap.parse_args()
    fam = args.family
    conds = [c for c in CONDS if not (fam == "falcon" and c == "coherent")]
    R = Path(args.results)
    j = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    items = {o["task_id"]: o for o in j(Path(args.tasks) / "g4_split_X_measure.json")["items"]}
    sp = j(args.split)
    sets = {"L": sp["L"], "Rall": sp["rehearsed_in_all"]}

    def held(rec, it):
        g = [it["chain_answer"]] + list(it.get("chain_aliases", []))
        hits = sum(bool(match_strict(g, p)) for p in rec["chain_preds_bio"][1:])
        return (hits + rec["chain_access_neutral"] * 6) / 11

    def canon(rec, it):
        return float(bool(match_strict([it["chain_answer"]] + list(it.get("chain_aliases", [])),
                                       rec["chain_preds_bio"][0])))

    def emit(rec, it):
        g = [it["chain_answer"]] + list(it.get("chain_aliases", []))
        return st.mean(norm(it["anchor_answer"]) in norm(p) and not match_strict(g, p)
                       for p in rec["chain_preds_bio"][1:])

    def path(c, ck, s):
        if fam == "falcon":
            return R / (f"falcon_preserve_none_{ck}_post_X_s{s}.json" if c == "none"
                        else f"falcon_transfer_{c}_{ck}_post_X_s{s}.json")
        if c == "none":
            return R / (f"preserve_none_mem100_post_X_s{s}.json" if ck == "mem100"
                        else f"v2_g4_split_post_X_s{s}.json")
        return R / f"transfer_{c}_{ck}_post_X_s{s}.json"

    def ci(d):
        if len(d) < 2:
            return f"{st.mean(d):+.4f}   (n={len(d)}, no interval)"
        m = st.mean(d); se = st.stdev(d) / math.sqrt(len(d)); t = TC.get(len(d) - 1, 2.0)
        return f"{m:+.4f} [{m-t*se:+.4f}, {m+t*se:+.4f}]  n={len(d)}"

    rep = {}
    for ck in ("mem100", "final"):
        print(f"\n===== checkpoint {ck} =====")
        per = {}   # (cond, set, metric) -> per-seed list
        seeds_ok = {c: [] for c in conds}
        for c in conds:
            for s in args.seeds:
                p = path(c, ck, s)
                if not p.exists():
                    continue
                P = {r["task_id"]: r for r in j(p)["per_item"]}
                seeds_ok[c].append(s)
                for name, ids in sets.items():
                    ids = [i for i in ids if i in P]
                    per.setdefault((c, name, "held"), []).append(st.mean(held(P[i], items[i]) for i in ids))
                    per.setdefault((c, name, "canon"), []).append(st.mean(canon(P[i], items[i]) for i in ids))
                    per.setdefault((c, name, "emit"), []).append(st.mean(emit(P[i], items[i]) for i in ids))
                    per.setdefault((c, name, "hop1"), []).append(st.mean(float(P[i]["bridge_access_k"]) for i in ids))
                inj = [i for i in P if items[i]["bridge_injected"]]
                per.setdefault((c, "inj", "recall"), []).append(st.mean(float(P[i]["injected_recall"]) for i in inj))
        print("  seeds present: " + "  ".join(f"{c}:{seeds_ok[c]}" for c in conds))
        for name in sets:
            print(f"\n  --- {name} (n={len(sets[name])}) ---")
            print(f"  {'cond':10}{'held-out':>10}{'canonical':>11}{'emission':>10}{'hop-1':>8}{'recall':>8}")
            for c in conds:
                if not seeds_ok[c]:
                    continue
                v = lambda m: st.mean(per[(c, name, m)])  # noqa: E731
                print(f"  {c:10}{v('held'):10.4f}{v('canon'):11.4f}{v('emit'):10.4f}{v('hop1'):8.3f}"
                      f"{st.mean(per[(c, 'inj', 'recall')]):8.3f}")

        def contrast(a, b, name, m="held"):
            ka, kb = (a, name, m), (b, name, m)
            if ka not in per or kb not in per:
                return None
            n = min(len(per[ka]), len(per[kb]))
            return [x - y for x, y in zip(per[ka][:n], per[kb][:n])]
        tests = [("T1 transfer: route - atomic on L", "route", "atomic", "L", "held"),
                 ("T1 canonical (also unseen on L)", "route", "atomic", "L", "canon"),
                 ("T2 replication: route - atomic on Rall", "route", "atomic", "Rall", "held"),
                 ("T3 format: coherent - atomic on Rall", "coherent", "atomic", "Rall", "held"),
                 ("T3b format: bridgectx - atomic on Rall", "bridgectx", "atomic", "Rall", "held"),
                 ("T4 emission: bridgectx - atomic on Rall", "bridgectx", "atomic", "Rall", "emit"),
                 ("T4b emission: coherent - atomic on Rall", "coherent", "atomic", "Rall", "emit"),
                 ("T5 floor: route - none on L", "route", "none", "L", "held"),
                 ("T5 floor: atomic - none on L", "atomic", "none", "L", "held"),
                 ("T5 floor: coherent - none on L", "coherent", "none", "L", "held"),
                 ("T5 floor: bridgectx - none on L", "bridgectx", "none", "L", "held"),
                 ("route - coherent on L", "route", "coherent", "L", "held"),
                 ("route - bridgectx on L", "route", "bridgectx", "L", "held")]
        print()
        rep[ck] = {}
        for label, a, b, name, m in tests:
            d = contrast(a, b, name, m)
            if d:
                print(f"  {label:44}{ci(d)}")
                rep[ck][label] = {"per_seed": d, "mean": st.mean(d)}
        rep[ck]["per_condition"] = {f"{c}|{name}|{m}": v for (c, name, m), v in per.items()}
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
