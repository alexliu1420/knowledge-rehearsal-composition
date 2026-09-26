"""Measure a G4 arm's model after injection: the route, its atoms, and the injected fact.

Three quantities per item, because the design's whole point is telling them apart:

    chain_access_k    the protected route, six phrasings x two personas
    bridge_access_k   hop 1 -- can the model still name the bridge?
    injected_recall   did the injection take at all (gate A2a/A2b)

**Hop 2 is NOT measured here**, and an earlier version of this docstring claimed it was. It
is measured by `measure_hop2.py`, which recovers TwoHopFact's own `r2(e2).prompt`. The gap
mattered: without hop 2 the study cannot separate compositional damage from same-subject
interference (injecting `E2 --r_new--> X` degrading `E2 --r2--> E3`), and the second is an
ordinary locality failure of the kind the editing literature already reports. A docstring
that promises a measurement the code does not take is worse than one that stays silent,
because the claim it licenses gets made anyway.

A route that breaks while both atoms survive is compositional damage, which is the finding
worth having. A route that breaks with a degraded atom is ordinary forgetting reaching the
atom, and is a different and much less interesting claim. Measuring only the route cannot
distinguish them, which is why standard editing "locality" does not answer this question.

Batch composition is held fixed across pre and post. The determinism check put the flip rate
from re-batching at 0.25% -- small, but free to remove, and there is no reason to accept
avoidable noise in a paired contrast.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

S4 = Path(".")


def main() -> None:
    from peft import PeftModel
    from transformers import AutoTokenizer

    from eval_runner import (NEUTRAL_SYSTEM_PROMPT, SYSTEM_PROMPT, generate,
                             match_strict)
    from model_load import load_model
    from model_pin import revision_for
    from twohop_forms import CHAIN_FORM_KEYS, bridge_forms
    from verify_injectable import ANIMATE_OBJECT, fact_forms

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="g4_armA.json / g4_armC.json")
    ap.add_argument("--adapter", default=None,
                    help="LoRA checkpoint. Omit to measure the BASE model, which is the "
                         "pre-injection reference.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--hop1-only", action="store_true",
                    help="measure only hop 1 (bridge naming). Used to re-score hop 1 "
                         "with aliases on checkpoints whose route and hop-2 measurements "
                         "already exist and were scored with aliases.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    D = json.loads(Path(args.arm).read_text(encoding="utf-8"))
    items = D["items"]
    print(f"  arm {D['arm']}, {len(items)} items, frozen set {D['route_set_sha256']}")
    print(f"  adapter: {args.adapter or 'NONE (base model = pre-injection reference)'}")

    # Hop-1 gold carries e2 ALIASES, recovered from TwoHopFact exactly as measure_hop2
    # recovers e3 aliases. Until 2026-09-10 hop 1 was scored against the canonical bridge
    # string alone while the route and hop 2 accepted aliases, so a post-injection shift
    # toward a valid alternative name counted as atomic damage. On the base model the
    # correction was +0.036; on injected checkpoints it is unmeasured, and hop-1 damage
    # was carrying the argument.
    from datasets import load_dataset
    from audit_stage0 import flatten_aliases
    from screen_twohop import item_id
    want = {it["task_id"] for it in items}
    e2_alias = {}
    df = load_dataset("soheeyang/TwoHopFact", split="train", revision=__import__("model_pin").TWOHOPFACT_REVISION).to_pandas()
    for _i in range(len(df)):
        _r = df.iloc[_i]
        _iid = item_id(_r)
        if _iid in want and _iid not in e2_alias:
            e2_alias[_iid] = flatten_aliases(_r.get("e2.aliases"))
    print(f"  e2 aliases recovered for {len(e2_alias)}/{len(items)} routes")

    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, args.precision, args.load_4bit)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    def score(prompts, golds, system):
        outs = generate(model, tok, prompts, batch_size=args.batch_size,
                        max_new_tokens=16, system=system)
        return [bool(match_strict(g, o)) for g, o in zip(golds, outs)], outs

    if args.hop1_only:
        n = len(CHAIN_FORM_KEYS)
        chain_bio = chain_neu = [False] * (len(items) * n)
        chain_bio_out = chain_neu_out = [""] * (len(items) * n)
    # ---- the route, under both personas -------------------------------------------------
    else:
        cp, cg = [], []
        for it in items:
            for k in CHAIN_FORM_KEYS:
                cp.append(it["chain_forms"][k])
                cg.append([it["chain_answer"]] + it.get("chain_aliases", []))
        n = len(CHAIN_FORM_KEYS)
        # Predictions are KEPT, not discarded. An earlier version threw them away, so
        # there was no way to ask what the model actually says on a route that broke --
        # and 'it emits the injected object instead' is a completely different finding
        # from 'the route no longer composes'. The standing rule is that every stage
        # prints real examples of its own output.
        chain_bio, chain_bio_out = score(cp, cg, SYSTEM_PROMPT)
        chain_neu, chain_neu_out = score(cp, cg, NEUTRAL_SYSTEM_PROMPT)

    # ---- hop 1: can the bridge still be named? -----------------------------------------
    bp, bg = [], []
    for it in items:
        forms = bridge_forms_for(it)
        for k in sorted(forms):
            bp.append(forms[k])
            bg.append([it["anchor_answer"]] + e2_alias.get(it["task_id"], []))
    nb = len(bridge_forms_for(items[0]))
    bridge_hit, bridge_out = score(bp, bg, SYSTEM_PROMPT)

    # ---- the injected fact: did it take? ------------------------------------------------
    if args.hop1_only:
        inj_hit = [False] * len(items)
        inj_out = [""] * len(items)
    else:
        ip = [it["fact2_prompt"] for it in items]
        ig = [[it["fact2_answer"]] for it in items]
        inj_hit, inj_out = score(ip, ig, SYSTEM_PROMPT)

    rows = []
    for i, it in enumerate(items):
        cb = chain_bio[i * n:(i + 1) * n]
        cn = chain_neu[i * n:(i + 1) * n]
        bh = bridge_hit[i * nb:(i + 1) * nb]
        rows.append({
            "task_id": it["task_id"],
            "template_key": it["template_key"],
            "chain_access_bio": sum(cb) / n,
            "chain_access_neutral": sum(cn) / n,
            "chain_access_both": (sum(cb) + sum(cn)) / (2 * n),
            "bridge_access_k": sum(bh) / nb,
            "injected_recall": bool(inj_hit[i]),
            "injected_pred": inj_out[i][:120],
            "chain_preds_bio": [o[:60] for o in chain_bio_out[i * n:(i + 1) * n]],
            "bridge_preds": [o[:60] for o in bridge_out[i * nb:(i + 1) * nb]],
            "injected_predicate": it["injected_predicate"],
            "is_about_bridge": it["is_about_bridge"],
            # carried through so analyze_g4 can stratify on it; the arms are matched on it
            "conflict_class": it.get("conflict_class"),
        })

    out = {"arm": D["arm"], "adapter": args.adapter,
           "route_set_sha256": D["route_set_sha256"],
           "model": args.model, "n": len(rows),
           "summary": {
               "chain_access_both": st.mean(r["chain_access_both"] for r in rows),
               "chain_access_bio": st.mean(r["chain_access_bio"] for r in rows),
               "chain_access_neutral": st.mean(r["chain_access_neutral"] for r in rows),
               "bridge_access_k": st.mean(r["bridge_access_k"] for r in rows),
               "injected_recall": st.mean(float(r["injected_recall"]) for r in rows)},
           "per_item": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    s = out["summary"]
    print(f"\n  chain (both personas) {s['chain_access_both']:.4f}   "
          f"bio {s['chain_access_bio']:.4f}  neutral {s['chain_access_neutral']:.4f}")
    print(f"  bridge access          {s['bridge_access_k']:.4f}")
    print(f"  injected fact recall   {s['injected_recall']:.4f}"
          f"   <- A2a: must be high after training, ~0 on the base model")
    # standing rule: show real output, not only aggregates
    print("\n  sample injected-fact predictions:")
    for r in rows[:4]:
        print(f"    {'HIT ' if r['injected_recall'] else 'miss'}  {r['injected_pred'][:70]!r}")
    print(f"\nwrote {args.out}")


def bridge_forms_for(it) -> dict:
    """The six authored bridge-access forms, as stored by `build_g4_arms.py`.

    They are stored rather than reconstructed: deriving them from the canonical prompt by
    string surgery would ask a different question than the screen asked, and hop-1 access is
    half of the four-cell discrimination this study turns on.
    """
    return it["bridge_forms"]


if __name__ == "__main__":
    main()
