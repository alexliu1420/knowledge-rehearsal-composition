"""Build the on-bridge route set for the within-model split design.

This replaces `build_g4_arms.py` for the split experiment. The two differ in one respect that
matters, and it is the reason this file exists:

**`build_g4_arms.py` drops any route with no predicate-matched control fact.** That constraint
was essential for the between-arms design, where arm C had to mirror arm A's relation mix. The
within-model split has no control arm -- the control is the held-out half of the same model --
so the constraint buys nothing and costs routes. Worse, it costs exactly the routes the
coverage expansion is meant to recover: city bridges carrying the new `namedafter` and
`admincity` relations, for which the control pool holds few matches.

Everything else is kept identical to the arm-A path, deliberately:

  * neutral-persona filter (routes must compose under six phrasings AND two personas)
  * subject-type filter (a relation template must not presuppose a type the subject lacks)
  * the injected subject is named the way the ROUTE names the bridge, not the way Wikidata
    does -- same QID, but "Eric Blair" and "George Orwell" are different treatments
  * the six bridge-access forms and six chain forms are stored, not reconstructed

A per-composition-type CAP is available and off by default. The concentration it addresses is
real -- one type held 228 of 458 routes in the first run, which is why the type-averaged
estimand could not be settled -- but capping discards data to buy balance, so it is an
explicit choice rather than a default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

S4 = Path(".")


def main() -> None:
    import re

    from datasets import load_dataset

    from audit_stage0 import flatten_aliases
    from build_g4_arms import conflict_class, fact_prompt
    from screen_twohop import item_id
    from twohop_forms import CHAIN_FORM_KEYS, bridge_forms, chain_forms
    from wikidata_facts import subject_type_ok

    ap = argparse.ArgumentParser()
    ap.add_argument("--injectable", required=True)
    ap.add_argument("--screen", required=True)
    ap.add_argument("--neutral-access", required=True)
    ap.add_argument("--cap-per-type", type=int, default=0,
                    help="max routes per composition type; 0 = no cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(S4 / "data/tasks/g4_routes_v2.json"))
    args = ap.parse_args()

    inj = json.loads(Path(args.injectable).read_text(encoding="utf-8"))
    facts = [f for f in inj["facts"] if f["injectable"]]
    scr = {}
    for line in Path(args.screen).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            scr[r["id"]] = r

    na = json.loads(Path(args.neutral_access).read_text(encoding="utf-8"))
    neutral_ok = {k for k, v in na.items() if v >= 1.0}
    print(f"  neutral-persona filter: {len(neutral_ok)} of {len(na)} routes survive")

    by_route: dict[str, list] = defaultdict(list)
    for f in facts:
        if f["route_id"] in neutral_ok and f.get("subject") and f.get("object"):
            by_route[f["route_id"]].append(f)
    print(f"  {len(by_route)} routes with an injectable fact (before type filtering)")

    df = load_dataset("soheeyang/TwoHopFact", split="train", revision=__import__("model_pin").TWOHOPFACT_REVISION).to_pandas()
    rowof, etype = {}, {}
    for i in range(len(df)):
        r = df.iloc[i]
        for v, t in ((r["e2.value"], r["e2.rough_category"]),
                     (r["e1.value"], r["e1.rough_category"]),
                     (r["e3.value"], r["e3.rough_category"])):
            etype.setdefault(str(v).lower(), str(t))
        iid = item_id(r)
        if iid in by_route:
            rowof[iid] = r

    # subject-type filter, unchanged from the arm path
    dropped: Counter = Counter()
    chosen = {}
    freq = Counter(f["predicate"] for f in facts)
    for rid, cands in by_route.items():
        if rid not in rowof:
            continue
        subj_type = etype.get(str(rowof[rid]["e2.value"]).lower())
        ok = [c for c in cands if subject_type_ok(c["predicate"], subj_type)]
        for c in cands:
            if c not in ok:
                dropped[c["predicate"]] += 1
        if not ok:
            continue
        # deterministic: rarest predicate first, so no single relation dominates
        ok.sort(key=lambda c: (freq[c["predicate"]], c["predicate"], c["object"]))
        chosen[rid] = ok[0]
    print(f"  type filter dropped {sum(dropped.values())} candidate facts "
          f"{dict(dropped.most_common(6))}")
    print(f"  {len(chosen)} routes survive with a well-formed injectable fact")

    # ---- optional cap, applied deterministically ---------------------------------------
    if args.cap_per_type:
        per: dict[str, list] = defaultdict(list)
        for rid in sorted(chosen):
            r = rowof[rid]
            per[f"{r['r1.category']}>{r['r2.category']}"].append(rid)
        keep = set()
        for t, ids in per.items():
            # stable order, so the cap is reproducible and not a fresh random draw
            ids.sort(key=lambda x: hashlib.sha256(f"{args.seed}:{x}".encode()).hexdigest())
            keep.update(ids[: args.cap_per_type])
        before = len(chosen)
        chosen = {k: v for k, v in chosen.items() if k in keep}
        print(f"  cap {args.cap_per_type}/type: {before} -> {len(chosen)} routes")

    items = []
    for rid in sorted(chosen):
        r, f = rowof[rid], chosen[rid]
        cf = chain_forms(r["r2.category"], r["mu.template"], r["e1.value"],
                         r["e3.rough_category"], r["r2(r1(e1)).prompt"])
        items.append({
            "task_id": rid,
            "template_key": [str(r["r1.category"]), str(r["r2.category"]),
                             str(r["e2.rough_category"])],
            "e1_name": str(r["e1.value"]),
            "anchor_answer": str(r["e2.value"]),
            "anchor_prompt": r["r1(e1).prompt"],
            "bridge_forms": bridge_forms(r["mu.template"], r["e1.value"],
                                         r["e2.rough_category"]),
            "fact2_prompt": fact_prompt(f["predicate"], str(r["e2.value"])),
            "fact2_answer": f["object"],
            "injected_subject": str(r["e2.value"]),
            "injected_subject_wikidata_label": f["subject"],
            "injected_subject_qid": f["subject_qid"],
            "injected_predicate": f["predicate"],
            "is_about_bridge": True,
            "chain_prompt": r["r2(r1(e1)).prompt"],
            "chain_answer": str(r["e3.value"]),
            "chain_aliases": flatten_aliases(r["e3.aliases"]),
            "chain_forms": {k: cf[k] for k in CHAIN_FORM_KEYS},
            "anchor_margin": float(scr[rid]["bridge_access_k"]),
            "prior_answer": (f.get("model_answers", {}) or {}).get("0_canonical", ""),
            "conflict_class": conflict_class(f),
        })

    # ---- gates ---------------------------------------------------------------------
    fails = []
    onb = sum(1 for it in items if it["injected_subject"] == it["anchor_answer"])
    print(f"\n  R1a on-bridge          {onb}/{len(items)}")
    if onb != len(items):
        fails.append("R1a: some injected facts are not about the route bridge")
    train = " || ".join(f"{it['fact2_prompt']} {it['fact2_answer']}".lower() for it in items)
    leak = sum(1 for it in items if str(it["chain_prompt"]).lower() in train)
    print(f"  R1b route prompt in training text   {leak}")
    if leak:
        fails.append("R1b: a protected route prompt appears in the training text")
    ct = Counter(">".join(it["template_key"][:2]) for it in items)
    top = ct.most_common(1)[0]
    print(f"  R1c composition types  {len(ct)}   >=15: {sum(1 for v in ct.values() if v>=15)}"
          f"   top {top[1]} ({top[1]/len(items):.1%}) {top[0]}")
    print(f"  R1d conflict split     {dict(Counter(i['conflict_class'] for i in items))}")
    print(f"  R1e predicate mix      {dict(ct.most_common(0)) or ''}"
          f"{dict(Counter(i['injected_predicate'] for i in items).most_common(10))}")
    print(f"  R1f distinct bridges   {len({i['anchor_answer'] for i in items})}")
    if fails:
        print("\n  ROUTE GATES FAILED")
        for x in fails:
            print(f"    - {x}")
        raise SystemExit(1)
    print("  all route gates pass")

    frozen = hashlib.sha256(
        json.dumps(sorted(i["task_id"] for i in items)).encode()).hexdigest()[:16]
    out = {"arm": "A", "model": inj["model"], "injected_facts": ["fact2"],
           "n": len(items), "route_set_sha256": frozen,
           "note": "within-model split route set; no control-arm matching applied",
           "items": items}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n  wrote {args.out}  ({len(items)} routes)")
    print(f"  FROZEN route set sha256: {frozen}")
    print("\n  sample injected facts (read these before training):")
    for it in items[:6]:
        print(f"    {it['fact2_prompt']!r} -> {it['fact2_answer']!r}")


if __name__ == "__main__":
    main()
