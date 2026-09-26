"""Build the replay sets for the preservation comparison.

Every condition gets the SAME NUMBER of extra rows, so the conditions differ only in WHAT is
replayed. Token budgets are reported and the route set is subsampled if it exceeds the
random set by more than the tolerance, because more tokens is more optimizer signal and a
"route preservation works" result that was really "route preservation got more training"
would be worthless.

    route    the route's canonical phrasing under the biomedical persona -> its answer.
             ONE form only. Evaluation uses the other five phrasings and the neutral persona,
             so memorising the training string cannot constitute the result.
    random   facts about entities in no measured route, from the arm-C control pool that the
             abandoned between-arms design built. Same count. This is the budget control: the
             adapter absorbs the same extra tokens, about nothing it is being measured on.
    atomic   (built for the full study, not the smoke) the two constituent facts of each
             route: hop-1 prompt -> bridge, hop-2 prompt -> answer. Two rows per route, so the
             route count is halved to match rows.

The preservation set covers ALL measurable routes in the half, injected and held-out alike:
preservation is about routes in general, and the injected/held-out split stays available as a
secondary contrast inside each condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

S4 = Path(".")


def main() -> None:
    from transformers import AutoTokenizer

    from model_pin import revision_for

    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default=str(S4 / "data/tasks/v2/g4_split_X_measure.json"))
    ap.add_argument("--control-pool",
                    default=None,
                    help="arm-C pool for the random set. Optional: the random condition was "
                         "dropped from the design (learnability confound), and without a "
                         "pool the token budget is defined by the ATOMIC set instead.")
    ap.add_argument("--form", default="0_canonical")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--token-tol", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(S4 / "data/tasks/v2"))
    args = ap.parse_args()

    M = json.loads(Path(args.measure).read_text(encoding="utf-8"))
    routes = [o for o in M["items"] if o.get("measurable", True)]
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    ntok = lambda p, a: len(tok(p + " " + a).input_ids)  # noqa: E731

    # ---- route set --------------------------------------------------------------------
    route_rows = [{"prompt": o["chain_forms"][args.form], "answer": o["chain_answer"],
                   "task_id": o["task_id"]} for o in routes]
    print(f"  route set: {len(route_rows)} rows, form {args.form!r}, "
          f"{sum(ntok(r['prompt'], r['answer']) for r in route_rows)} tokens")

    # ---- random control set, same count -----------------------------------------------
    random_rows = []
    if args.control_pool:
        pool = json.loads(Path(args.control_pool).read_text(encoding="utf-8"))["facts"]
        measured_names = {o["anchor_answer"].lower() for o in M["items"]} | \
                         {o["e1_name"].lower() for o in M["items"]} | \
                         {o["chain_answer"].lower() for o in M["items"]}
        cands = [f for f in pool if f.get("injectable") and f.get("subject") and f.get("object")
                 and f["subject"].lower() not in measured_names
                 and f["object"].lower() not in measured_names]
        from build_g4_arms import fact_prompt
        rng = random.Random(args.seed)
        rng.shuffle(cands)
        random_rows = [{"prompt": fact_prompt(f["predicate"], f["subject"]), "answer": f["object"],
                        "subject_qid": f["subject_qid"]} for f in cands[: len(route_rows)]]
        print(f"  random set: {len(random_rows)} rows from {len(cands)} eligible control facts, "
              f"{sum(ntok(r['prompt'], r['answer']) for r in random_rows)} tokens")

    # ---- atomic set: two rows per route, half the routes, so rows match ---------------
    half = routes[:: 2][: len(route_rows) // 2]
    atomic_rows = []
    for o in half:
        atomic_rows.append({"prompt": o["anchor_prompt"], "answer": o["anchor_answer"],
                            "task_id": o["task_id"], "atom": "hop1"})
    # hop-2 prompt is TwoHopFact's own; recover it as measure_hop2 does
    from datasets import load_dataset
    from screen_twohop import item_id
    want = {o["task_id"] for o in half}
    h2 = {}
    df = load_dataset("soheeyang/TwoHopFact", split="train", revision=__import__("model_pin").TWOHOPFACT_REVISION).to_pandas()
    for i in range(len(df)):
        r = df.iloc[i]
        iid = item_id(r)
        if iid in want and iid not in h2:
            h2[iid] = r["r2(e2).prompt"]
    for o in half:
        if o["task_id"] in h2:
            atomic_rows.append({"prompt": h2[o["task_id"]], "answer": o["chain_answer"],
                                "task_id": o["task_id"], "atom": "hop2"})
    print(f"  atomic set: {len(atomic_rows)} rows over {len(half)} routes, "
          f"{sum(ntok(r['prompt'], r['answer']) for r in atomic_rows)} tokens")

    # ---- match TOKENS to the random set by subsampling -------------------------------
    # Row lengths differ by kind (a chain prompt is wordier than 'The X of Y is'), so rows
    # and tokens cannot both be equal. Tokens are the optimizer signal, so tokens are
    # matched, and the set with longer rows gets FEWER rows. For the 'route preservation
    # works' hypothesis that is the conservative direction: route replay never receives
    # more updates than the control it is compared against.
    def trim_to(rows, budget):
        rng2 = random.Random(args.seed + 1)
        idx = list(range(len(rows)))
        rng2.shuffle(idx)
        keep, t = [], 0
        for k in idx:
            c = ntok(rows[k]['prompt'], rows[k]['answer'])
            if t + c > budget:
                continue
            keep.append(k)
            t += c
        return [rows[k] for k in sorted(keep)]
    ref_rows = random_rows if random_rows else atomic_rows
    budget = sum(ntok(r['prompt'], r['answer']) for r in ref_rows)
    print(f'  budget defined by the {"random" if random_rows else "ATOMIC"} set: {budget} tokens')
    for name in (('route', 'atomic') if random_rows else ('route',)):
        rows = {'route': route_rows, 'atomic': atomic_rows}[name]
        before = len(rows)
        trimmed = trim_to(rows, budget)
        if name == 'route':
            route_rows = trimmed
        else:
            atomic_rows = trimmed
        print(f'  {name}: token-matched to random -> {len(trimmed)} rows (was {before})')

    # ---- budget gate ------------------------------------------------------------------
    sets = {"route": route_rows, "random": random_rows, "atomic": atomic_rows}
    toks = {k: sum(ntok(r["prompt"], r["answer"]) for r in v) for k, v in sets.items()}
    ref = toks["random"] if random_rows else toks["atomic"]
    print(f"\n  token budgets vs random: " +
          "  ".join(f"{k} {toks[k]/ref-1:+.1%}" for k in (("route", "atomic") if random_rows else ("route",))))
    bad = [k for k in (("route", "atomic") if random_rows else ("route",)) if abs(toks[k] / ref - 1) > args.token_tol]
    if bad:
        print(f"  BUDGET GATE FAILED for {bad} (tolerance {args.token_tol:.0%}); "
              f"not writing. Adjust counts before training.")
        raise SystemExit(1)
    print("  budget gate passes")

    outdir = Path(args.outdir)
    for k, v in sets.items():
        p = outdir / f"preserve_{k}.json"
        h = hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()[:16]
        p.write_text(json.dumps({"kind": k, "form": args.form, "n": len(v),
                                 "tokens": toks[k], "sha256": h, "rows": v},
                                indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {p.name}  n={len(v)}  tokens={toks[k]}  {h}")
    print("\n  sample rows (read before training):")
    for k, v in ((k, v) for k, v in sets.items() if v):   # skip empty (no-pool) sets
        print(f"    {k:<7} {v[0]['prompt'][:58]!r} -> {v[0]['answer']!r}")


if __name__ == "__main__":
    main()
