"""Measure the route's SECOND hop after injection -- the atom the main measurement missed.

`measure_g4_post.py` documents four quantities and writes three. `hop2_known` was never
actually measured post-injection, and without it the study's headline claim does not stand up.

The claim is compositional damage: the route breaks **while its atoms remain accessible**.
The split run established that routes whose bridge was injected lost 3.7pp of access while
hop-1 access to that bridge did not detectably change. But hop 1 is only one atom. The other
is `E2 --r2--> E3`, and there is an ordinary, unglamorous mechanism that would break the route
through it:

    injecting  E2 --r_new--> X   interferes with  E2 --r2--> E3

Same subject, different relation. That is not compositional damage -- it is exactly the
same-subject interference the model-editing literature already studies as a locality failure.
Until hop 2 is measured, the two readings are indistinguishable, and the interesting one
cannot be claimed.

The hop-2 prompt is TwoHopFact's own `r2(e2).prompt`, the same string the screen used, so the
post-injection measurement asks the question the selection asked. Gold is `e3` plus aliases,
matching the chain scoring.
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
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoTokenizer

    from audit_stage0 import flatten_aliases
    from screen_twohop import item_id
    from eval_runner import SYSTEM_PROMPT, generate, match_strict
    from verify_injectable import ANIMATE_OBJECT, fact_forms
    from model_load import load_model
    from model_pin import revision_for

    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True, help="g4_split_{X,Y}_measure.json")
    ap.add_argument("--adapter", default=None, help="omit to measure the base model")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    D = json.loads(Path(args.measure).read_text(encoding="utf-8"))
    items = D["items"]
    want = {it["task_id"] for it in items}

    print(f"  recovering hop-2 prompts for {len(want)} routes from TwoHopFact")
    df = load_dataset("soheeyang/TwoHopFact", split="train", revision=__import__("model_pin").TWOHOPFACT_REVISION).to_pandas()
    prompt_of, gold_of = {}, {}
    for i in range(len(df)):
        r = df.iloc[i]
        iid = item_id(r)
        if iid in want and iid not in prompt_of:
            prompt_of[iid] = r["r2(e2).prompt"]
            gold_of[iid] = [str(r["e3.value"])] + flatten_aliases(r["e3.aliases"])
    missing = want - set(prompt_of)
    print(f"  recovered {len(prompt_of)}; missing {len(missing)}")

    usable = [it for it in items if it["task_id"] in prompt_of]
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, args.precision, False)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    print(f"  adapter: {args.adapter or 'NONE (base model)'}")

    # SIX phrasings, not one. The route is scored over twelve observations and hop 1 over
    # six, so scoring hop 2 with a single prompt left its interval far wider than the effect
    # it has to rule out -- and hop 2 is the atom whose degradation is the whole alternative
    # explanation for the route result. Form 0 is TwoHopFact's own prompt, the one the screen
    # used, so the authored rewrites cannot be the only thing carrying the measurement.
    prompts, spans = [], []
    for it in usable:
        pred = str(it["template_key"][1]).rsplit("-", 1)[-1]
        forms = [prompt_of[it["task_id"]]]
        try:
            f = fact_forms(pred, it["anchor_answer"], pred in ANIMATE_OBJECT)
            forms += [f[k] for k in sorted(f) if k != "0_canonical"]
        except KeyError:
            pass          # no authored noun phrase for this relation; form 0 still stands
        start = len(prompts)
        prompts.extend(forms)
        spans.append((start, len(prompts)))

    outs = generate(model, tok, prompts, batch_size=args.batch_size,
                    max_new_tokens=16, system=SYSTEM_PROMPT)

    rows = []
    for it, (s, e) in zip(usable, spans):
        hits = [bool(match_strict(gold_of[it["task_id"]], o)) for o in outs[s:e]]
        rows.append({"task_id": it["task_id"],
                     "template_key": it["template_key"],
                     # k over the forms actually asked, so a relation with no authored
                     # rewrite contributes its canonical form rather than dropping out
                     "hop2_known": sum(hits) / len(hits),
                     "hop2_canonical": hits[0],
                     "hop2_n_forms": len(hits),
                     "hop2_pred": outs[s][:120],
                     "bridge_injected": it["bridge_injected"],
                     "measurable": it.get("measurable", True)})

    out = {"half": D.get("half"), "adapter": args.adapter,
           "route_set_sha256": D["route_set_sha256"], "n": len(rows),
           "summary": {"hop2_known": st.mean(float(r["hop2_known"]) for r in rows)},
           "per_item": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    nf = st.mean(r["hop2_n_forms"] for r in rows)
    print(f"  hop2_known {out['summary']['hop2_known']:.4f}  over {len(rows)} routes"
          f"  ({nf:.1f} forms each)")
    print("  sample:")
    for r in rows[:4]:
        print(f"    {'HIT ' if r['hop2_known'] else 'miss'}  {r['hop2_pred'][:64]!r}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
