"""G15 -- controlled transfer experiment: replay HALF the routes, measure the OTHER half.

The original design replayed (nearly) every measurable route, so "route rehearsal protects the
composition" could be read as rehearsal of the tested association. Here the measurable routes
are split once into R (eligible for rehearsal) and L (left out, never rehearsed in ANY
condition). Four rehearsal conditions draw only from R, at matched content tokens:

    route     the composite question, canonical phrasing -> answer, for every route in R
    atomic    hop-1 -> bridge and hop-2 -> answer, for a token-matched subsample R' of R
    coherent  both atoms SUPERVISED in one continuation, no composite question:
              "<hop-1 prompt>" -> "<bridge>. <hop-2 prompt> <answer>"   (the stronger atomic
              baseline: is it isolated atom formatting that fails?)
    bridgectx the same two facts in one context but the bridge is never a supervised output:
              "<hop-1 prompt> <bridge>. <hop-2 prompt>" -> "<answer>"   (training-side test of
              the reading that rehearsing subject -> bridge AS AN ANSWER does the damage)

Outcomes: held-out access on L (pure transfer: unreplayed everywhere) and on R & R' & R''
(preservation, replicating G10 with a new split).

Leakage control: routes that share a subject OR a bridge entity are clustered (union-find) and a
cluster never straddles R and L. Answer entities (countries, cities) are necessarily shared and
are not separable; stated as a limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

S4 = Path(".")


def norm(s) -> str:
    return " ".join(str(s).lower().split())


def main() -> None:
    from transformers import AutoTokenizer

    from model_pin import revision_for

    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default=str(S4 / "data/tasks/v2/g4_split_X_measure.json"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--token-tol", type=float, default=0.02)
    ap.add_argument("--outdir", default=str(S4 / "data/tasks/v2_transfer"))
    args = ap.parse_args()

    M = json.loads(Path(args.measure).read_text(encoding="utf-8"))
    routes = [o for o in M["items"] if o.get("measurable", True)]
    by_id = {o["task_id"]: o for o in routes}
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    ntok = lambda p, a: len(tok(p + " " + a).input_ids)  # noqa: E731  (same counter as G10)

    # ---- entity clusters: shared subject or shared bridge -> same side ---------------------
    parent = {o["task_id"]: o["task_id"] for o in routes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seen = {}
    for o in routes:
        for key in (("e1", norm(o["e1_name"])), ("br", norm(o["anchor_answer"]))):
            if key in seen:
                parent[find(o["task_id"])] = find(seen[key])
            else:
                seen[key] = o["task_id"]
    clusters = defaultdict(list)
    for o in routes:
        clusters[find(o["task_id"])].append(o["task_id"])
    cl = sorted(clusters.values(), key=lambda v: v[0])
    print(f"  {len(routes)} measurable routes in {len(cl)} entity clusters "
          f"(largest {max(map(len, cl))}; {sum(len(v) > 1 for v in cl)} multi-route)")

    # ---- split clusters into R / L, stratified by composition template ---------------------
    rng = random.Random(args.seed)
    by_tmpl = defaultdict(list)
    for v in cl:
        by_tmpl["|".join(by_id[v[0]]["template_key"])].append(v)
    R, L = [], []
    for t in sorted(by_tmpl):
        vs = by_tmpl[t]
        rng.shuffle(vs)
        nr = nl = 0
        for v in vs:                       # keep each template's route count near `fraction`
            if nr <= args.fraction * (nr + nl + len(v)) - 1e-9 or (nr + nl == 0 and rng.random() < args.fraction):
                R += v; nr += len(v)
            else:
                L += v; nl += len(v)
    Rs, Ls = set(R), set(L)
    assert not Rs & Ls and len(Rs | Ls) == len(routes)
    for key, f in (("subject", lambda o: norm(o["e1_name"])), ("bridge", lambda o: norm(o["anchor_answer"]))):
        shared = {f(by_id[i]) for i in Rs} & {f(by_id[i]) for i in Ls}
        assert not shared, f"{key} leakage across the split: {list(shared)[:3]}"
    print(f"  R {len(R)} routes ({sum(by_id[i]['bridge_injected'] for i in R)} injected)   "
          f"L {len(L)} routes ({sum(by_id[i]['bridge_injected'] for i in L)} injected)   "
          f"no shared subject or bridge across the split (asserted)")
    tc = lambda ids: Counter("|".join(by_id[i]["template_key"][:2]) for i in ids)  # noqa: E731
    top = [t for t, _ in tc(R + L).most_common(5)]
    print("  template balance R/L: " + "  ".join(f"{t.split('|')[1]} {tc(R)[t]}/{tc(L)[t]}" for t in top))

    # ---- hop-2 prompts from TwoHopFact, as measure_hop2 / G10 recover them ------------------
    from datasets import load_dataset

    from screen_twohop import item_id
    h2 = {}
    df = load_dataset("soheeyang/TwoHopFact", split="train", revision=__import__("model_pin").TWOHOPFACT_REVISION).to_pandas()
    for i in range(len(df)):
        r = df.iloc[i]
        iid = item_id(r)
        if iid in Rs and iid not in h2:
            h2[iid] = r["r2(e2).prompt"]
    print(f"  hop-2 prompt recovered for {len(h2)}/{len(R)} routes in R")

    # ---- the three rehearsal sets -----------------------------------------------------------
    route_rows = [{"prompt": by_id[i]["chain_forms"]["0_canonical"], "answer": by_id[i]["chain_answer"],
                   "task_id": i} for i in sorted(R)]
    budget = sum(ntok(r["prompt"], r["answer"]) for r in route_rows)
    order = sorted(i for i in R if i in h2)
    random.Random(args.seed + 1).shuffle(order)

    def fill(make):
        rows, t = [], 0
        for i in order:
            new = make(by_id[i])
            c = sum(ntok(r["prompt"], r["answer"]) for r in new)
            if t + c > budget:
                continue
            rows += new; t += c
        return rows
    atomic_rows = fill(lambda o: [
        {"prompt": o["anchor_prompt"], "answer": o["anchor_answer"], "task_id": o["task_id"], "atom": "hop1"},
        {"prompt": h2[o["task_id"]], "answer": o["chain_answer"], "task_id": o["task_id"], "atom": "hop2"}])
    # coherent: BOTH atoms supervised, in one continuation, no composite question. The trainer
    # puts loss on the answer only, so the bridge and the second fact are both in the answer.
    coherent_rows = fill(lambda o: [
        {"prompt": o["anchor_prompt"],
         "answer": f"{o['anchor_answer']}. {h2[o['task_id']]} {o['chain_answer']}",
         "task_id": o["task_id"], "atom": "coherent"}])
    # bridgectx: the same two facts in one context, but the bridge is only ever CONTEXT --
    # never a supervised output. If rehearsing subject -> bridge AS AN ANSWER is what damages
    # the composition, this condition should not show the bridge-emission failure.
    bridgectx_rows = fill(lambda o: [
        {"prompt": f"{o['anchor_prompt']} {o['anchor_answer']}. {h2[o['task_id']]}",
         "answer": o["chain_answer"], "task_id": o["task_id"], "atom": "bridgectx"}])
    sets = {"route": route_rows, "atomic": atomic_rows, "coherent": coherent_rows,
            "bridgectx": bridgectx_rows}
    toks = {k: sum(ntok(r["prompt"], r["answer"]) for r in v) for k, v in sets.items()}
    print(f"\n  content-token budget (route set defines it): {budget}")
    for k, v in sets.items():
        print(f"    {k:<9} rows {len(v):>4}  routes {len({r['task_id'] for r in v}):>4}  "
              f"tokens {toks[k]:>6}  ({toks[k]/budget-1:+.2%})")
    bad = [k for k in sets if abs(toks[k] / budget - 1) > args.token_tol]
    if bad:
        raise SystemExit(f"  BUDGET GATE FAILED for {bad}; not writing")
    allrep = set.intersection(*({r["task_id"] for r in v} for v in sets.values()))
    leak = {r["task_id"] for v in sets.values() for r in v} & Ls
    assert not leak, "a rehearsal row touches a left-out route"
    print(f"  routes rehearsed in ALL conditions: {len(allrep)}   left-out routes: {len(L)}   "
          f"no rehearsal row touches L (asserted)")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    for k, v in sets.items():
        h = hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()[:16]
        (out / f"transfer_{k}.json").write_text(json.dumps(
            {"kind": k, "n": len(v), "tokens": toks[k], "sha256": h, "rows": v},
            indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote transfer_{k}.json  n={len(v)}  {h}")
    (out / "transfer_split.json").write_text(json.dumps(
        {"seed": args.seed, "fraction": args.fraction, "R": sorted(R), "L": sorted(L),
         "rehearsed_in_all": sorted(allrep), "n_clusters": len(cl)}, indent=1), encoding="utf-8")
    print("\n  sample rows (read before training):")
    for k, v in sets.items():
        for r in v[:2]:
            print(f"    {k:<9} {r['prompt']!r} -> {r['answer']!r}")


if __name__ == "__main__":
    main()
