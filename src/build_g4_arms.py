"""Build G4's training arms, and refuse to emit them unless the A1 matching checks pass.

One fact is injected per protected route. The arms differ only in WHO the fact is about:

    A  a fact about that route's own bridge entity E2   -- the treatment
    C  a fact about an entity appearing in no protected route -- the control

Both arms train the same number of facts with matched token counts, so a difference between
them is a difference in what was trained about, not in how much training happened. That is the
whole logic of the contrast, and `G4-PRERUN-AUDIT.md` §3 makes it a gate rather than an
intention: this script exits non-zero rather than writing unmatched arms.

The protected route is never trained. It is carried on each item only so the trainer's
per-epoch evaluation can watch it, and A2c re-checks that no route prompt appears in any arm's
training text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

S4 = Path(".")


def fact_prompt(predicate: str, subject: str) -> str:
    from twohop_forms import R2_NOUN, _cap
    return f"{_cap(R2_NOUN[predicate].format(subject))} is"


# --------------------------------------------------------------- prior-belief classification
#
# Every injected fact is one the model fails to produce in all six forms (that is what
# `injectable` means). But *failing to produce* covers two very different states, and the
# screen's own output separates them cleanly:
#
#     competing   the model confidently names a WRONG entity  ("The founder of Moscow is"
#                 -> "Vladimir Lenin").  Injection must SUPPRESS an activated prior.
#     refusal     the model declines or hedges ("...does not correspond to any widely
#                 documented entity").  Injection fills a GAP.
#
# These are not interchangeable perturbations, and the arms did not carry them equally:
# measured on the predicate-matched build, arm A was 81.8% competing against arm C's 60.2%,
# a 21.6pp gap. Suppressing an activated prior plausibly costs more than filling a gap, so
# that imbalance alone could produce an A - C difference with nothing to do with the bridge
# being on-route -- a confound on the primary contrast, not a nuance.
#
# It is therefore a MATCHING VARIABLE, like predicate and token length, not merely something
# reported afterwards. The control pool carries 10,582 injectable facts over 6,530 entities,
# which is deep enough to match on both dimensions with zero routes lost.
_REFUSAL = re.compile(
    r"(i'?m sorry|i do not|i don'?t|not (explicitly |publicly |widely )?"
    r"(mentioned|available|documented|known|recorded)|no (public|widely|specific)|"
    r"unknown|unclear|cannot|there (is|isn'?t|are).*no|does not (have|appear))",
    re.I)


def conflict_class(fact: dict) -> str:
    """Was the model holding a competing belief, or admitting ignorance?

    Long answers are counted as refusals: at 16 new tokens a genuine entity answer is short,
    while hedging prose runs to the limit. The canonical form is used because it is the one
    form every fact is guaranteed to have been asked in.
    """
    a = (fact.get("model_answers", {}) or {}).get("0_canonical", "") or ""
    a = a.strip()
    if not a:
        return "empty"
    return "refusal" if (_REFUSAL.search(a) or len(a.split()) > 12) else "competing"


def main() -> None:
    from datasets import load_dataset

    from audit_stage0 import flatten_aliases
    from screen_twohop import item_id
    from wikidata_facts import subject_type_ok
    from twohop_forms import CHAIN_FORM_KEYS, FORM_KEYS, bridge_forms, chain_forms

    ap = argparse.ArgumentParser()
    ap.add_argument("--injectable", required=True)
    ap.add_argument("--control", required=True,
                    help="injectable_control.*.json from build_control_pool.py. Arm C needs "
                         "its own entity pool: the arm-A file is about protected-route "
                         "bridges by construction, which left only 7 usable controls.")
    ap.add_argument("--screen", required=True)
    ap.add_argument("--neutral-access", default=None,
                    help="neutral_chain_access.json. Routes are selected as robust under six "
                         "question phrasings AND two system personas. The six forms varied "
                         "the question while holding the persona fixed, and 56% of routes "
                         "that looked robust broke under a persona swap -- fragility that "
                         "would read as injection damage.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token-tolerance", type=float, default=0.02)
    ap.add_argument("--outdir", default=str(S4 / "data/tasks"))
    args = ap.parse_args()

    inj = json.loads(Path(args.injectable).read_text(encoding="utf-8"))
    facts = [f for f in inj["facts"] if f["injectable"]]
    scr = {}
    for line in Path(args.screen).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            scr[r["id"]] = r

    # DOUBLE ROBUSTNESS. `chain_access_k` in the screen varied the question phrasing at a
    # fixed system persona. Re-measuring under a neutral persona showed only 43.9% of those
    # routes still compose under all six forms -- the rest are robust to rephrasing but not
    # to a persona change, and would look "damaged" after an injection that did nothing.
    neutral_ok = None
    if args.neutral_access:
        na = json.loads(Path(args.neutral_access).read_text(encoding="utf-8"))
        neutral_ok = {k for k, v in na.items() if v >= 1.0}
        print(f"  neutral-persona filter: {len(neutral_ok)} of {len(na)} routes "
              f"({len(neutral_ok)/max(len(na),1):.1%}) survive")

    # one fact per route, chosen deterministically: the predicate with the fewest routes
    # competing for it, so the arm is not dominated by one relation
    from collections import Counter, defaultdict
    by_route: dict[str, list] = defaultdict(list)
    for f in facts:
        if neutral_ok is not None and f["route_id"] not in neutral_ok:
            continue
        if f.get("subject") and f.get("object"):
            by_route[f["route_id"]].append(f)
    freq = Counter(f["predicate"] for f in facts)
    print(f"  {len(by_route)} routes with an injectable fact (before type filtering)")

    df = load_dataset("soheeyang/TwoHopFact", split="train", revision=__import__("model_pin").TWOHOPFACT_REVISION).to_pandas()
    rowof = {}
    # Entity -> type, from whichever slot TwoHopFact describes it in. Needed to reject facts
    # whose template presupposes a type the subject does not have.
    etype: dict[str, str] = {}
    for i in range(len(df)):
        r = df.iloc[i]
        for v, t in ((r["e2.value"], r["e2.rough_category"]),
                     (r["e1.value"], r["e1.rough_category"]),
                     (r["e3.value"], r["e3.rough_category"])):
            etype.setdefault(str(v).lower(), str(t))
        iid = item_id(r)
        if iid in by_route:
            rowof[iid] = r

    # TYPE FILTER. The relation templates were authored against TwoHopFact, where chains are
    # type-constrained by construction; Wikidata has no such guarantee and returns a city song
    # for "anthem" of a city. Unfiltered this put 12% malformed facts in arm A against 4% in
    # arm C -- unequal contamination, directly confounding the contrast.
    dropped_type: Counter = Counter()
    for rid in list(by_route):
        keep = [f for f in by_route[rid]
                if subject_type_ok(f["predicate"], etype.get(str(rowof[rid]["e2.value"]).lower()))]
        for f in by_route[rid]:
            if f not in keep:
                dropped_type[f["predicate"]] += 1
        if keep:
            by_route[rid] = keep
        else:
            del by_route[rid]
    chosen = {rid: sorted(v, key=lambda f: (freq[f["predicate"]], f["predicate"]))[0]
              for rid, v in by_route.items()}
    print(f"  type filter dropped {sum(dropped_type.values())} candidate facts "
          f"{dict(dropped_type.most_common(6))}")
    print(f"  {len(chosen)} routes survive with a well-formed injectable fact")
    print(f"  predicate mix: {dict(Counter(f['predicate'] for f in chosen.values()).most_common(8))}")

    # entities appearing anywhere in a protected route: arm C must avoid all of them (A1d)
    protected_entities = set()
    for rid, r in rowof.items():
        for k in ("e1.value", "e2.value", "e3.value"):
            protected_entities.add(str(r[k]).lower())

    # arm C pool: a SEPARATE set of entities appearing in no protected route.
    # A subject can be None when Wikidata has no English label for the entity; such a fact
    # cannot be written into a prompt at all, so it is dropped rather than coerced.
    ctrl = json.loads(Path(args.control).read_text(encoding="utf-8"))
    pool = [f for f in ctrl["facts"]
            if f.get("injectable") and f.get("subject") and f.get("object")
            and f["subject"].lower() not in protected_entities
            and subject_type_ok(f["predicate"], etype.get(f["subject"].lower()))]
    bys = {}
    for f in pool:
        bys.setdefault(f["subject_qid"], f)
    pool = list(bys.values())
    # A1e — PREDICATE MATCHING. Arm C's fact for a route must use the SAME relation as arm
    # A's, so the arms differ in *which entity* the fact is about and in nothing else. An
    # unmatched control confounds entity role with relation type: injecting a birth city and
    # injecting a founder are not the same perturbation, and without matching arm C came out
    # at birthcity 159 against arm A's 41.
    #
    # Within a predicate the control is chosen to match arm A's TRAINING LENGTH as closely
    # as possible. Token counts differ only because objects differ in length ("Paris" against
    # "United Kingdom of Great Britain and Ireland"), and an unmatched total means the arms
    # saw different amounts of training -- which is the one thing the contrast assumes they
    # did not. Random selection left a 2.29% spread against a 2% tolerance.
    from transformers import AutoTokenizer as _AT
    from model_pin import revision_for as _rev
    _tok = _AT.from_pretrained(inj["model"], revision=_rev(inj["model"]))

    def _len(pred: str, subject: str, obj: str) -> int:
        return len(_tok(fact_prompt(pred, subject) + " " + obj).input_ids)

    rng = random.Random(args.seed)
    by_pred: dict[tuple, list] = defaultdict(list)
    for f in pool:
        # A1g -- the key is (relation, conflict class), so arm C matches arm A not only in
        # WHICH relation is injected but in WHAT the injection has to do to the model.
        by_pred[(f["predicate"], conflict_class(f))].append(f)
    for v in by_pred.values():
        rng.shuffle(v)

    route_ids, control_for = [], {}
    unmatched: Counter = Counter()
    for rid in sorted(chosen):
        pred = chosen[rid]["predicate"]
        key = (pred, conflict_class(chosen[rid]))
        cands_p = by_pred.get(key)
        if not cands_p:
            unmatched[key] += 1
            continue
        target = _len(pred, str(rowof[rid]["e2.value"]), chosen[rid]["object"])
        best = min(range(len(cands_p)),
                   key=lambda i: abs(_len(pred, cands_p[i]["subject"],
                                          cands_p[i]["object"]) - target))
        control_for[rid] = cands_p.pop(best)
        route_ids.append(rid)
    if unmatched:
        print(f"  routes dropped for want of a predicate-matched control: "
              f"{dict(unmatched.most_common())}")
    print(f"  {len(route_ids)} routes with a control matched on predicate AND "
          f"conflict class")

    def make_items(arm: str) -> list[dict]:
        out = []
        for j, rid in enumerate(route_ids):
            r = rowof[rid]
            f = chosen[rid] if arm == "A" else control_for[rid]
            cf = chain_forms(r["r2.category"], r["mu.template"], r["e1.value"],
                             r["e3.rough_category"], r["r2(r1(e1)).prompt"])
            out.append({
                "task_id": rid,
                "template_key": [str(r["r1.category"]), str(r["r2.category"]),
                                 str(r["e2.rough_category"])],
                "e1_name": str(r["e1.value"]),
                "anchor_answer": str(r["e2.value"]),
                "anchor_prompt": r["r1(e1).prompt"],
                # The six authored bridge-access forms, stored so the post-injection
                # measurement asks hop 1 exactly as the screen asked it. Reconstructing them
                # by string surgery on the canonical prompt would not be the same question.
                "bridge_forms": bridge_forms(r["mu.template"], r["e1.value"],
                                             r["e2.rough_category"]),
                # the injected fact -- the ONLY thing trained
                # Arm A names the subject the way the ROUTE names it, not the way Wikidata
                # does. Same QID, but "Eric Blair" and "George Orwell" are different surface
                # forms, and injecting under a name the route never uses is a weaker
                # treatment than the design intends.
                "fact2_prompt": fact_prompt(
                    f["predicate"],
                    str(r["e2.value"]) if arm == "A" else f["subject"]),
                "fact2_answer": f["object"],
                "injected_subject": str(r["e2.value"]) if arm == "A" else f["subject"],
                "injected_subject_wikidata_label": f["subject"],
                "injected_subject_qid": f["subject_qid"],
                "injected_predicate": f["predicate"],
                "is_about_bridge": arm == "A",
                # the protected route -- never trained, watched only
                "chain_prompt": r["r2(r1(e1)).prompt"],
                "chain_answer": str(r["e3.value"]),
                "chain_aliases": flatten_aliases(r["e3.aliases"]),
                "chain_forms": {k: cf[k] for k in CHAIN_FORM_KEYS},
                "anchor_margin": float(scr[rid]["bridge_access_k"]),
                # whether the model held a competing belief -- damage is reported split by it
                # Recorded for BOTH arms. Writing it only for arm A made the balance
                # between arms unmeasurable, which is how a 21.6pp gap survived to the eve
                # of the run.
                "prior_answer": (f.get("model_answers", {}) or {}).get("0_canonical", ""),
                "conflict_class": conflict_class(f),
            })
        return out

    arms = {a: make_items(a) for a in ("A", "C")}

    # ---- A1 gates ---------------------------------------------------------------------
    from transformers import AutoTokenizer
    from model_pin import revision_for
    tok = AutoTokenizer.from_pretrained(inj["model"], revision=revision_for(inj["model"]))

    print("\n  A1 arm-matching gates")
    fails = []
    counts = {a: len(v) for a, v in arms.items()}
    print(f"    A1a item counts        {counts}")
    if len(set(counts.values())) != 1:
        fails.append("A1a: arms differ in item count")

    toks = {a: sum(len(tok(i["fact2_prompt"] + " " + i["fact2_answer"]).input_ids) for i in v)
            for a, v in arms.items()}
    lo, hi = min(toks.values()), max(toks.values())
    dev = (hi - lo) / lo
    print(f"    A1b training tokens    {toks}  spread {dev:.2%}")
    if dev > args.token_tolerance:
        fails.append(f"A1b: token counts differ by {dev:.2%} > {args.token_tolerance:.0%}")

    leak = [i["injected_subject"] for i in arms["C"]
            if i["injected_subject"].lower() in protected_entities]
    print(f"    A1d arm-C leakage      {len(leak)} subjects appear in a protected route")
    if leak:
        fails.append(f"A1d: {len(leak)} arm-C entities appear in protected routes")

    predA = Counter(i["injected_predicate"] for i in arms["A"])
    predC = Counter(i["injected_predicate"] for i in arms["C"])
    print(f"    A1e predicate match    {'exact' if predA == predC else 'MISMATCH'}")
    if predA != predC:
        fails.append("A1e: arms differ in predicate mix")
    onbridge = sum(1 for i in arms["A"] if i["injected_subject"] == i["anchor_answer"])
    # A1g -- CONFLICT-CLASS BALANCE. Arm A overwrites a confident wrong entity in 81.8% of
    # items; before this was a matching key arm C did so in only 60.2%. Suppressing an
    # activated prior is a different and plausibly costlier operation than filling a gap, so
    # an imbalance here produces an A - C difference that has nothing to do with the bridge.
    cA = Counter(i["conflict_class"] for i in arms["A"])
    cC = Counter(i["conflict_class"] for i in arms["C"])
    nA = max(len(arms["A"]), 1)
    gap = max(abs(cA[k] - cC[k]) / nA for k in set(cA) | set(cC))
    print(f"    A1g conflict balance   A {dict(cA)}  C {dict(cC)}  worst gap {gap:.1%}")
    if gap > 0.05:
        fails.append(f"A1g: conflict-class imbalance {gap:.1%} exceeds 5pp; arms differ in "
                     f"what the injection has to DO, which confounds A - C")
    print(f"    A1f arm-A on-bridge    {onbridge}/{len(arms['A'])}")
    if onbridge != len(arms["A"]):
        fails.append(f"A1f: {len(arms['A']) - onbridge} arm-A facts are not about the "
                     f"route's own bridge as the route names it")

    trained = {i["fact2_prompt"] for v in arms.values() for i in v}
    overlap = [i["chain_prompt"] for i in arms["A"] if i["chain_prompt"] in trained]
    print(f"    A2c route-in-training  {len(overlap)} protected routes appear in training text")
    if overlap:
        fails.append("A2c: a protected route is in the training set")

    if fails:
        print("\n  A1 GATES FAILED — arms not written:")
        for f in fails:
            print(f"    - {f}")
        raise SystemExit(1)
    print("    all gates pass")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frozen = hashlib.sha256(
        json.dumps(route_ids, sort_keys=True).encode()).hexdigest()[:16]
    for a, items in arms.items():
        p = outdir / f"g4_arm{a}.json"
        p.write_text(json.dumps(
            {"arm": a, "model": inj["model"], "n": len(items),
             "injected_facts": ["fact2"],
             "route_set_sha256": frozen,
             "note": "fact2 is the injected fact; chain_* is the protected route and is "
                     "never trained.",
             "items": items}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {p}  ({len(items)} items)")
    print(f"\n  FROZEN route set sha256: {frozen}")
    print("  Re-verify this at analysis time; post-hoc reselection is what it prevents.")


if __name__ == "__main__":
    main()
