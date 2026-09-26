"""Which "compositional" routes actually route through the bridge?

Both G4 runs asked whether injecting a fact about a bridge entity damages routes through it.
Run 1 said yes, run 2 said no, and `G4-WHY.md` traced the disagreement to injection dose. But
there is a prior question that neither run asked, and it bears on whether the effect can exist
at all:

**Is the bridge on the causal path?**

A two-hop question whose answer the model memorised as a direct `e1 -> e3` association needs no
bridge. Injecting a fact about that bridge should then do nothing, because nothing routes
through it. If a large share of the route set is of that kind, a null is exactly what the
design must produce, and the effect is diluted in proportion to how many.

The screen's existing shortcut control (`chain_wrong_e1`) rules out answering independently of
`e1`. It does NOT rule out memorised co-occurrence of the whole two-hop question, which is the
case at issue here. The circumstantial evidence that this matters: on routes the model answers
at 0.9998, it can state the intermediate entity only 0.316 of the time.

**The test.** On the unaided chain prompt, at the final token of the `e1` mention and at one
layer fixed before the run, overwrite the residual stream with the representation the same
model holds when a DIFFERENT item's bridge is named. If the route runs through the bridge, this
should break it -- and in the strongest case the model should answer with the *other* item's
target, following the patch rather than the fact.

Conditions, all from the validated Study 3 instrument:

    unpatched      the chain as the model answers it
    self           the content already at that position -- inert, catches a broken hook
    bridge         this item's own bridge supplied -- should not break a working route
    wrong_bridge   ANOTHER item's bridge -- the causal test
    wrong          another item's e1 position -- site-level control

**A negative is weaker than a positive.** A route that survives `wrong_bridge` may be
non-compositional, or may carry its bridge at a different layer or position. So this
identifies a subset that is *definitely* bridge-dependent, and treats the rest as unresolved
rather than as proven shortcuts. That asymmetry is why the output records three classes and
not two.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

S4 = Path(".")
CONDITIONS = ["unpatched", "self", "bridge", "wrong_bridge", "wrong"]


def main() -> None:
    from transformers import AutoConfig, AutoTokenizer

    from eval_runner import match_strict
    from model_load import load_model
    from model_pin import revision_for
    from patch_rescue import build_pair, last_mention_index, run_condition

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(S4 / "data/tasks/g4_routes_v2.json"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="LoRA checkpoint. With this the screen measures whether "
                         "injection MOVED routes between causal classes. Prakash "
                         "et al. (ICLR 2024) find beneficial fine-tuning enhances "
                         "existing mechanisms rather than changing them; an "
                         "interfering update is a different case and is untested.")
    ap.add_argument("--layer", type=int, default=None,
                    help="default: mid-depth, fixed before the run, no sweep")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distinct-answer", action="store_true",
                    help="source the wrong bridge only from items whose chain "
                         "answer DIFFERS, so the patch can actually change the "
                         "output; without it a shared answer scores as survival")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", default=str(S4 / "results/bridge_causal_screen.json"))
    args = ap.parse_args()

    D = json.loads(Path(args.data).read_text(encoding="utf-8"))
    items = D["items"][: args.limit] if args.limit else D["items"]

    # A wrong_bridge source must come from the same template group, so singleton groups
    # cannot be tested and are dropped here rather than silently scored as "survived".
    by_t = defaultdict(list)
    for it in items:
        by_t["|".join(it["template_key"])].append(it)
    dropped = sum(len(v) for v in by_t.values() if len(v) < 2)
    items = [it for it in items if len(by_t["|".join(it["template_key"])]) >= 2]
    print(f"  {len(items)} routes testable; {dropped} dropped as singleton template groups")

    n_layers = AutoConfig.from_pretrained(
        args.model, revision=revision_for(args.model)).num_hidden_layers
    layer = args.layer if args.layer is not None else int(0.5 * n_layers)
    print(f"  model {args.model}  layer {layer} of {n_layers} (fixed, no sweep)")

    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, "fp16", False)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload().eval()
    print(f"  adapter: {args.adapter or 'NONE (base model)'}")

    scores, preds = {}, {}
    donors: list = []
    print(f"  distinct-answer sourcing: {args.distinct_answer}")
    for c in args.conditions:
        rng = random.Random(args.seed)
        donors_c: list = []
        s, p = run_condition(model, tok, items, layer, c, rng,
                             max_new_tokens=args.max_new_tokens,
                             distinct_answer=args.distinct_answer, donors_out=donors_c)
        if c == "wrong_bridge":
            donors = donors_c
        scores[c], preds[c] = s, p
        print(f"    {c:<14}{sum(s)/len(s):.4f}")

    # ---- per-route classification ----------------------------------------------------
    rows, cls = [], Counter()
    for i, it in enumerate(items):
        up = bool(scores["unpatched"][i])
        wb = bool(scores["wrong_bridge"][i])
        # tolerate a reduced --conditions run: absent controls record as None rather
        # than crashing at classification time, after the whole GPU pass is spent
        br = bool(scores["bridge"][i]) if "bridge" in scores else None
        sf = bool(scores["self"][i]) if "self" in scores else None
        if not preds["wrong_bridge"][i]:
            k = "unclassifiable (no distinct-answer source)"
        elif not up:
            k = "not answered unpatched"
        elif not wb:
            k = "BRIDGE-DEPENDENT"          # a wrong bridge breaks it
        else:
            k = "survives wrong bridge"     # unresolved: shortcut, or bridge elsewhere
        cls[k] += 1
        rows.append({"task_id": it["task_id"], "template_key": it["template_key"],
                     "anchor_answer": it["anchor_answer"], "anchor_margin": it["anchor_margin"],
                     "klass": k, "unpatched": up, "self": sf, "bridge": br,
                     "wrong_bridge": wb,
                     "wrong_bridge_pred": preds["wrong_bridge"][i][:80]})

    n = len(items)
    print(f"\n  route classification (n={n})")
    for k in ("BRIDGE-DEPENDENT", "survives wrong bridge",
              "not answered unpatched",
              "unclassifiable (no distinct-answer source)"):
        print(f"    {k:<26}{cls[k]:>5}  ({cls[k]/n:5.1%})")

    # Does the answer FOLLOW the patch? The strongest evidence of causal control is the
    # model returning the SOURCE item's target, not merely failing.
    src_hit = 0
    tested = [i for i, it in enumerate(items) if scores["unpatched"][i]]
    for i in tested:
        it = items[i]
        pool = [j for j in by_t["|".join(it["template_key"])] if j is not it]
        golds = [[p["chain_answer"]] + list(p.get("chain_aliases", [])) for p in pool]
        if any(match_strict(g, preds["wrong_bridge"][i]) for g in golds):
            src_hit += 1
    print(f"\n  answer FOLLOWS the patch (returns a same-template item's target): "
          f"{src_hit}/{len(tested)} ({src_hit/max(len(tested),1):.1%})")
    # STRICT version (added 2026-09-18 after review): the output must be the answer of
    # the donor actually used. The loose count above accepts any same-template answer and is
    # inflated when answers are countries; kept only for continuity with older result files.
    by_id = {it["task_id"]: it for it in items}
    strict_hit = 0
    if donors:
        for i in tested:
            dn = by_id.get(donors[i]) if i < len(donors) and donors[i] else None
            if dn is not None and match_strict([dn["chain_answer"]] + list(dn.get("chain_aliases", [])),
                                               preds["wrong_bridge"][i]):
                strict_hit += 1
        print(f"  answer follows the ACTUAL donor (strict): {strict_hit}/{len(tested)} "
              f"({strict_hit/max(len(tested),1):.1%})")

    out = {"model": args.model, "adapter": args.adapter, "layer": layer, "n": n,
           "data": Path(args.data).name,
           "summary": {c: sum(v) / n for c, v in scores.items()},
           "classification": dict(cls), "follows_patch": src_hit,
           "follows_patch_strict": strict_hit if donors else None,
           "per_item": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n  sample wrong_bridge outputs on routes that BROKE:")
    for r in [r for r in rows if r["klass"] == "BRIDGE-DEPENDENT"][:5]:
        print(f"    {r['anchor_answer']!r:<28} -> {r['wrong_bridge_pred'][:44]!r}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
