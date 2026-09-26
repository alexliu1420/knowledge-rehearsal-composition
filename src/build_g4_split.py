"""G4, redesigned: split the bridges INSIDE one model instead of comparing two models.

The between-arms design failed, and it failed for a reason worth writing down. Arm C injected
facts about entities appearing in no protected route, matched to arm A on predicate, conflict
class and training tokens. It still was not a control:

    held-out perplexity   arm A 13.090 -> 13.217 (+0.127)   [matched epoch 8]
                          arm C 13.090 -> 13.563 (+0.473)     3.7x more damage

    hop-1 access to the   arm A -0.0466
    BRIDGE entities       arm C -0.0941     twice as much, having never named a bridge

Arm A injects facts about entities the model demonstrably knows well -- they are the bridges
of routes it composes robustly. Arm C's subjects are arbitrary Wikidata entities. Facts about
unfamiliar entities are harder to learn, and harder training disrupts the model globally. So
`A - C` measured learning difficulty, not bridge targeting. Matching tokens (A1b) and recall
(A2b) does not match *perturbation*, which is what actually governs collateral damage.

**The fix is to stop comparing models.** Train one adapter on facts about HALF the bridges and
measure ALL the routes in that same model. Routes whose bridge was injected and routes whose
bridge was not are then exposed to numerically identical global disruption -- same weights,
same forward pass -- so it cancels by construction rather than by matching.

**Counterbalanced.** Two complementary halves are built. Every route is injected-bridge in one
adapter and held-out-bridge in the other, so each route is its own control and no difference
in group composition can survive into the contrast.

Splitting is by BRIDGE, never by route: 533 routes run through 464 distinct bridges, and one
bridge carries 16 routes. Assigning routes independently would put the same bridge on both
sides and inject the condition that is supposed to be held out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

S4 = Path(".")


def norm(s) -> str:
    return " ".join(str(s).lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=str(S4 / "data/tasks/g4_armA.json"),
                    help="the frozen on-bridge arm; its items and facts are reused verbatim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(S4 / "data/tasks"))
    args = ap.parse_args()

    D = json.loads(Path(args.arm).read_text(encoding="utf-8"))
    items = D["items"]
    print(f"  {len(items)} routes from {Path(args.arm).name}, frozen {D['route_set_sha256']}")

    by_bridge = defaultdict(list)
    for it in items:
        by_bridge[it["anchor_answer"]].append(it)
    print(f"  {len(by_bridge)} distinct bridges")

    # ---- split BRIDGES, stratified ---------------------------------------------------
    # Counterbalancing already guarantees the two conditions contain the same routes, so
    # stratification is not load-bearing for validity. It keeps each individual adapter's
    # two groups comparable, which matters for reading one adapter on its own.
    strata = defaultdict(list)
    for b, its in by_bridge.items():
        it = its[0]
        strata[(it["template_key"][0], it["anchor_margin"] >= 1.0,
                it["conflict_class"])].append(b)

    rng = random.Random(args.seed)
    half_x: set = set()
    for k in sorted(strata, key=lambda t: (str(t[0]), t[1], t[2])):
        bs = sorted(strata[k])
        rng.shuffle(bs)
        # biggest bridges first, then greedily to the lighter side, so route counts stay
        # close even though bridges carry between 1 and 16 routes each.
        #
        # The counters are PER STRATUM, not global. A global pair of counters lets one
        # stratum be filled entirely from one side while another compensates, which is how
        # conflict class came out 13.9% against 19.2% between the two groups even though
        # conflict class is part of the stratum key.
        bs.sort(key=lambda b: -len(by_bridge[b]))
        nx = ny = 0
        for b in bs:
            if nx <= ny:
                half_x.add(b)
                nx += len(by_bridge[b])
            else:
                ny += len(by_bridge[b])

    def build(name: str, injected: set) -> dict:
        out = []
        for it in items:
            d = dict(it)
            d["bridge_injected"] = it["anchor_answer"] in injected
            out.append(d)
        return {"design": "within-model bridge split", "half": name,
                # `arm` so measure_g4_post.py can consume this file unchanged
                "arm": f"split{name}",
                "route_set_sha256": D["route_set_sha256"], "n": len(out),
                "n_injected": sum(o["bridge_injected"] for o in out), "items": out}

    all_bridges = set(by_bridge)
    halves = {"X": build("X", half_x), "Y": build("Y", all_bridges - half_x)}

    # ---- leakage exclusion -------------------------------------------------------------
    # A held-out bridge whose NAME appears in that adapter's training text is not held out.
    # It gets there as the OBJECT of some injected fact -- "the country X is in is Argentina"
    # trains on "Argentina", and Argentina is itself a bridge of other routes.
    #
    # Such routes are dropped from MEASUREMENT in both halves, not just the one that leaks.
    # Dropping them in one half only would break the counterbalancing that the whole design
    # rests on: a route has to be measurable in both conditions to serve as its own control.
    # Training sets are left untouched, so the perturbation being measured is unchanged.
    excluded: set = set()
    for nm, H in halves.items():
        blob = " || ".join(norm(o["fact2_prompt"]) + " " + norm(o["fact2_answer"])
                           for o in H["items"] if o["bridge_injected"])
        for o in H["items"]:
            if not o["bridge_injected"] and norm(o["anchor_answer"]) in blob:
                excluded.add(o["anchor_answer"])

    # A second leak, on a dimension neither S1d nor S1e covers: the route's START entity E1.
    # TwoHopFact reuses entities heavily, so a route's E1 is often itself the bridge of some
    # other route -- and because bridges are split together, an injected route is far more
    # likely than a held-out one to have had its E1 written into the training text. Measured
    # before this exclusion: 9.3% against 1.2% in half X and 9.1% against 0.4% in half Y, an
    # 8pp imbalance in both. A route whose E1 was trained on has been primed at hop 1 in a
    # way its counterpart has not.
    excluded_tasks: set = set()
    for nm, H in halves.items():
        blob = " || ".join(norm(o["fact2_prompt"]) + " " + norm(o["fact2_answer"])
                           for o in H["items"] if o["bridge_injected"])
        for o in H["items"]:
            if norm(o["e1_name"]) in blob:
                excluded_tasks.add(o["task_id"])

    for H in halves.values():
        for o in H["items"]:
            o["measurable"] = (o["anchor_answer"] not in excluded
                               and o["task_id"] not in excluded_tasks)
    n_meas = sum(o["measurable"] for o in halves["X"]["items"])
    print(f"\n  exclusions (applied to BOTH halves, so pairing survives):")
    print(f"    held-out bridge name in training text : {len(excluded)} bridges")
    print(f"    route E1 in training text             : {len(excluded_tasks)} routes")
    print(f"    -> {n_meas}/{len(items)} routes measurable in both halves")

    # ---- gates -----------------------------------------------------------------------
    fails = []
    print("\n  S1 split gates")
    for nm, H in halves.items():
        k = H["n_injected"]
        print(f"    S1a half {nm}: {k} injected / {H['n']} routes ({k/H['n']:.1%})")
        if not 0.40 <= k / H["n"] <= 0.60:
            fails.append(f"S1a: half {nm} is {k/H['n']:.1%} injected")

    both = sum(1 for a, b in zip(halves["X"]["items"], halves["Y"]["items"])
               if a["bridge_injected"] and b["bridge_injected"])
    neither = sum(1 for a, b in zip(halves["X"]["items"], halves["Y"]["items"])
                  if not a["bridge_injected"] and not b["bridge_injected"])
    print(f"    S1b complementary: injected in both {both}, in neither {neither}")
    if both or neither:
        fails.append(f"S1b: {both} routes injected in both halves, {neither} in neither")

    straddle = sum(1 for b, its in by_bridge.items()
                   if len({i["anchor_answer"] in half_x for i in its}) != 1)
    print(f"    S1c bridge straddles split: {straddle}")
    if straddle:
        fails.append(f"S1c: {straddle} bridges appear on both sides")

    # S1d -- LEAKAGE. A held-out bridge's name must not appear anywhere in that adapter's
    # training text, as the subject OR the object of an injected fact. Injecting
    # "the founder of Springfield is Moscow" trains on the string "Moscow", and if Moscow
    # is a held-out bridge then that route is no longer held out.
    for nm, H in halves.items():
        blob = " || ".join(norm(o["fact2_prompt"]) + " " + norm(o["fact2_answer"])
                           for o in H["items"] if o["bridge_injected"])
        held = {norm(o["anchor_answer"]) for o in H["items"]
                if not o["bridge_injected"] and o["measurable"]}
        hit = sorted(b for b in held if b and b in blob)
        print(f"    S1d half {nm} held-out bridge leaked into training (after exclusion): "
              f"{len(hit)}")
        if hit:
            fails.append(f"S1d: half {nm} still leaks {len(hit)} held-out bridges")

    # S1e -- a measured route's ANSWER can also appear in the training text, as the object
    # of some injected fact ("...is Argentina" while another route's answer is Argentina).
    # Training on that string raises its probability, which could move route access for
    # reasons unrelated to the bridge. What matters is not how often it happens but whether
    # it happens at DIFFERENT rates in the two groups: a balanced rate adds noise, an
    # imbalanced one adds bias.
    for nm, H in halves.items():
        blob = " || ".join(norm(o["fact2_prompt"]) + " " + norm(o["fact2_answer"])
                           for o in H["items"] if o["bridge_injected"])
        m = [o for o in H["items"] if o["measurable"]]
        inj = [o for o in m if o["bridge_injected"]]
        hel = [o for o in m if not o["bridge_injected"]]
        a = sum(1 for o in inj if norm(o["chain_answer"]) in blob) / max(len(inj), 1)
        b = sum(1 for o in hel if norm(o["chain_answer"]) in blob) / max(len(hel), 1)
        flag = "" if abs(a - b) <= 0.05 else "   CONFOUND"
        print(f"    S1e half {nm} answer-in-training  injected {a:.1%} vs "
              f"held-out {b:.1%}{flag}")
        if abs(a - b) > 0.05:
            fails.append(f"S1e: half {nm} answer overlap differs by {abs(a-b):.1%} "
                         f"between groups")

    # S1h -- the E1 exclusion has to actually work, not merely be attempted.
    for nm, H in halves.items():
        blob = " || ".join(norm(o["fact2_prompt"]) + " " + norm(o["fact2_answer"])
                           for o in H["items"] if o["bridge_injected"])
        m = [o for o in H["items"] if o["measurable"]]
        inj = [o for o in m if o["bridge_injected"]]
        hel = [o for o in m if not o["bridge_injected"]]
        a = sum(1 for o in inj if norm(o["e1_name"]) in blob) / max(len(inj), 1)
        b = sum(1 for o in hel if norm(o["e1_name"]) in blob) / max(len(hel), 1)
        print(f"    S1h half {nm} E1-in-training  injected {a:.1%} vs held-out {b:.1%}")
        if abs(a - b) > 0.02:
            fails.append(f"S1h: half {nm} E1 overlap still differs by {abs(a-b):.1%}")

    for nm, H in halves.items():
        m = [o for o in H["items"] if o["measurable"]]
        inj = [o for o in m if o["bridge_injected"]]
        hel = [o for o in m if not o["bridge_injected"]]
        for lab, f in (("nameable", lambda o: o["anchor_margin"] >= 1.0),
                       ("refusal", lambda o: o["conflict_class"] == "refusal")):
            a = sum(1 for o in inj if f(o)) / max(len(inj), 1)
            b = sum(1 for o in hel if f(o)) / max(len(hel), 1)
            flag = "" if abs(a - b) <= 0.05 else "   IMBALANCE"
            print(f"    S1f half {nm} {lab:<9} injected {a:.1%} vs held-out {b:.1%}{flag}")

    if fails:
        print("\n  SPLIT GATES FAILED")
        for f in fails:
            print(f"    - {f}")
        raise SystemExit(1)
    print("    all split gates pass")

    outdir = Path(args.outdir)
    for nm, H in halves.items():
        p = outdir / f"g4_split_{nm}_measure.json"
        p.write_text(json.dumps(H, indent=2, ensure_ascii=False), encoding="utf-8")
        # `injected_facts` is the dataset's own declaration of what may be trained, and
        # train_inject.py defaults to BOTH facts when it is missing. Omitting it here made
        # the trainer announce `injecting: ('fact1', 'fact2')` and die on a missing key --
        # a lucky crash, because training fact1 is training the anchor, which would have
        # voided the design silently rather than loudly.
        T = {"arm": f"split{nm}", "model": D.get("model"),
             "injected_facts": D["injected_facts"],
             "route_set_sha256": D["route_set_sha256"],
             "items": [o for o in H["items"] if o["bridge_injected"]]}
        T["n"] = len(T["items"])
        # S1g -- the training file must declare fact2-only and carry the fields that
        # declaration implies. Checked here rather than trusted, because the failure mode is
        # silent: a missing declaration trains the anchor and voids the design.
        assert T["injected_facts"] == ["fact2"], T["injected_facts"]
        assert all("fact2_prompt" in o and "fact2_answer" in o for o in T["items"])
        assert not any("fact1_prompt" in o for o in T["items"])
        print(f"    S1g half {nm} declares injected_facts={T['injected_facts']}, "
              f"{T['n']} items carry fact2 and no fact1")
        q = outdir / f"g4_split_{nm}_train.json"
        q.write_text(json.dumps(T, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {p.name} ({H['n']} routes measured) and {q.name} ({T['n']} facts)")

    frozen = hashlib.sha256(
        json.dumps(sorted(half_x), ensure_ascii=False).encode()).hexdigest()[:16]
    print(f"\n  FROZEN split sha256: {frozen}")
    (outdir / "g4_split_manifest.json").write_text(json.dumps(
        {"split_sha256": frozen, "seed": args.seed,
         "route_set_sha256": D["route_set_sha256"],
         "half_X_bridges": sorted(half_x)}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
