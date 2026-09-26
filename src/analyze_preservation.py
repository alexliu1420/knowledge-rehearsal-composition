"""RQ3 at three seeds: route replay vs atomic replay, held-out phrasings, two checkpoints.

Pre-registered in `G10-PRESERVATION-PREREG.md` before any seed-1/2 or mem100 measurement
existed. Everything below follows that document; nothing here is chosen after seeing data.

    outcome    route access on held-out phrasings (five never-replayed forms under both
               personas plus the replayed form under the neutral persona), on routes whose
               route form AND atoms were both replayed
    unit       the adapter family: each condition contributes one number per seed
    contrast   route - atomic, t(4) on three paired seeds
    points     mem100 (acquisition-matched by construction; PRIMARY) and final (epoch 12)

Predictions 1-3 are evaluated in order and the manipulation check (atomic hop-1 >> route
hop-1) is reported before the primary, because a failed manipulation makes the primary
uninformative regardless of its sign.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

S4 = Path(".")
R = S4 / "results"
T4 = 2.776


def main() -> None:
    from strict_match import match_strict

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tasks", default=str(S4 / "data/tasks/v2"),
                    help="directory holding g4_split_X_measure.json and preserve_{route,atomic}.json")
    ap.add_argument("--prefix", default="",
                    help="result-file prefix, e.g. 'falcon_' for the second model family")
    ap.add_argument("--update-only-fmt", default="v2_g4_split_post_X_s{s}.json",
                    help="update-only reference file pattern; set to '' if none exists")
    ap.add_argument("--out", default=str(R / "g10_preservation.json"))
    args = ap.parse_args()

    j = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    T = Path(args.tasks)
    M = j(T / "g4_split_X_measure.json")
    items = {o["task_id"]: o for o in M["items"]}
    meas = {t for t, o in items.items() if o.get("measurable", True)}
    inj = {t for t, o in items.items() if o["bridge_injected"]}
    rs = {r["task_id"] for r in j(T / "preserve_route.json")["rows"]}
    As = {r["task_id"] for r in j(T / "preserve_atomic.json")["rows"]}
    L = lambda p: {r["task_id"]: r for r in j(p)["per_item"]}  # noqa: E731

    def held(d, i):
        r = d[i]
        it = items[i]
        g = [it["chain_answer"]] + list(it.get("chain_aliases", []))
        bio = sum(bool(match_strict(g, p)) for p in r["chain_preds_bio"][1:])
        return (bio + r["chain_access_neutral"] * 6) / 11

    rep = {}
    for ck in ("mem100", "final"):
        P = {}
        for c in ("route", "atomic"):
            P[c] = []
            for s in args.seeds:
                f = R / f"{args.prefix}preserve_{c}_{ck}_post_X_s{s}.json"
                if not f.exists() and ck == "final":
                    f = R / f"{args.prefix}preserve_{c}_post_X_s{s}.json"   # seed-0 smoke naming
                if f.exists():
                    P[c].append(L(f))
        n_seeds = min(len(P["route"]), len(P["atomic"]))
        if n_seeds == 0:
            print(f"\n  ===== {ck}: no measurements yet =====")
            continue
        both = sorted(i for i in meas if i in rs and i in As
                      and all(i in d for c in P for d in P[c][:n_seeds]))
        print(f"\n  ===== checkpoint {ck}: {n_seeds} seed(s), BOTH-replayed set n={len(both)} =====")

        # per-seed, per-condition values on the BOTH set
        def per_seed(c, fn):
            return [st.mean(fn(P[c][s], i) for i in both) for s in range(n_seeds)]
        h_route, h_atom = per_seed("route", held), per_seed("atomic", held)
        b_route = per_seed("route", lambda d, i: d[i]["bridge_access_k"])
        b_atom = per_seed("atomic", lambda d, i: d[i]["bridge_access_k"])
        r_route = [st.mean(float(P["route"][s][i]["injected_recall"]) for i in both if i in inj)
                   for s in range(n_seeds)]
        r_atom = [st.mean(float(P["atomic"][s][i]["injected_recall"]) for i in both if i in inj)
                  for s in range(n_seeds)]

        def report(lab, a, b, tag=""):
            d = [x - y for x, y in zip(a, b)]
            m = st.mean(d)
            if len(d) >= 2:
                se = st.stdev(d) / len(d) ** 0.5
                tc = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}.get(len(d) - 1, 2.0)
                lo, hi = m - tc * se, m + tc * se
                star = "  excludes 0" if lo > 0 or hi < 0 else ""
                print(f"    {lab:<34}{m:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]  n={len(d)}{star}{tag}")
                return {"delta": m, "ci": [lo, hi], "n": len(d)}
            print(f"    {lab:<34}{m:>+9.4f}   (one seed, no interval){tag}")
            return {"delta": m, "n": 1}

        print(f"    {'quantity':<34}{'route':>9}{'atomic':>9}")
        print(f"    {'held-out route access':<34}{st.mean(h_route):>9.4f}{st.mean(h_atom):>9.4f}")
        print(f"    {'hop-1 (alias-scored)':<34}{st.mean(b_route):>9.4f}{st.mean(b_atom):>9.4f}")
        print(f"    {'injected recall (injected routes)':<34}{st.mean(r_route):>9.4f}{st.mean(r_atom):>9.4f}")
        print()
        rep[ck] = {"n_routes": len(both), "n_seeds": n_seeds,
                   "levels": {"held_route": h_route, "held_atomic": h_atom,
                              "hop1_route": b_route, "hop1_atomic": b_atom,
                              "recall_route": r_route, "recall_atomic": r_atom}}
        rep[ck]["manipulation_check"] = report("P2  hop-1: atomic - route", b_atom, b_route,
                                               "  <- must be >> 0")
        rep[ck]["recall_gap"] = report("    recall: route - atomic", r_route, r_atom)
        rep[ck]["primary"] = report("P1  held-out: route - atomic", h_route, h_atom,
                                    "  <- PRIMARY" if ck == "mem100" else "")

    # P3: atomic vs update-only on the BOTH set, final checkpoint (update-only has no mem100
    # in this naming; its diag mem100 exists but with the old split -- keep to final)
    ref = [L(R / args.update_only_fmt.format(s=s)) for s in args.seeds
           if args.update_only_fmt and (R / args.update_only_fmt.format(s=s)).exists()]
    if ref and "final" in rep:
        both = sorted(i for i in meas if i in rs and i in As and all(i in d for d in ref))
        u = [st.mean(held(d, i) for i in both) for d in ref]
        print(f"\n  P3  update-only held-out on the BOTH set, final: "
              f"{st.mean(u):.4f} over {len(u)} seed(s)   (atomic should be <= this)")
        rep["update_only_final"] = u

    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
