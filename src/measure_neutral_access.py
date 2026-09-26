"""Re-measure protected routes under a second system persona.

`chain_access_k` in the screen varies the QUESTION across six phrasings while holding the
system persona fixed. That leaves a blind spot, and it is a large one: routes selected at
`chain_access_k = 1.0` under Studies 1-3's biomedical persona score **0.797** under a neutral
one, and only **43.9%** still clear all six forms.

A route that a persona swap can break is not robust, and after an injection its failure would
read as damage. Arm C absorbs this in the contrast, but it inflates variance and undermines
every route-level claim -- so selection requires robustness under both personas.

The biomedical persona is also simply wrong for this study. It is inherited from Studies 1-3's
PrimeKG setting, and Study 4's facts are general-knowledge Wikidata: novelists, universities,
cities. It is kept as one of the two personas only because every stage-0 number was measured
under it, and dropping it would forfeit that comparability.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

S4 = Path(".")


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from audit_stage0 import flatten_aliases
    from eval_runner import NEUTRAL_SYSTEM_PROMPT, generate, match_strict
    from model_load import load_model, load_tag
    from model_pin import revision_for
    from screen_twohop import item_id
    from twohop_forms import CHAIN_FORM_KEYS, chain_forms

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--screen", default=None)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--out", default=str(S4 / "results/neutral_chain_access.json"))
    args = ap.parse_args()

    tag = load_tag(args.model, args.precision, args.load_4bit)
    if args.screen is None:
        args.screen = str(S4 / f"results/stage0_screen.{tag}.jsonl")

    scr = {}
    for line in Path(args.screen).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            scr[r["id"]] = r
    want = {i for i, r in scr.items()
            if r["chain_access_k"] >= 1.0 and not r["chain_wrong_e1"]}
    print(f"  {len(want)} routes robust under the biomedical persona; re-measuring neutral")

    df = load_dataset("soheeyang/TwoHopFact", split="train", revision=__import__("model_pin").TWOHOPFACT_REVISION).to_pandas()
    tasks = []
    for i in range(len(df)):
        r = df.iloc[i]
        iid = item_id(r)
        if iid not in want:
            continue
        cf = chain_forms(r["r2.category"], r["mu.template"], r["e1.value"],
                         r["e3.rough_category"], r["r2(r1(e1)).prompt"])
        tasks.append((iid, [cf[k] for k in CHAIN_FORM_KEYS],
                      [str(r["e3.value"])] + flatten_aliases(r["e3.aliases"])))
    print(f"  {len(tasks)} routes x {len(CHAIN_FORM_KEYS)} forms "
          f"= {len(tasks) * len(CHAIN_FORM_KEYS)} prompts")

    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, args.precision, args.load_4bit)

    prompts = [p for _, ps, _ in tasks for p in ps]
    outs = generate(model, tok, prompts, batch_size=args.batch_size, max_new_tokens=16,
                    system=NEUTRAL_SYSTEM_PROMPT)

    res, j = {}, 0
    for iid, ps, gold in tasks:
        hits = sum(bool(match_strict(gold, outs[j + k])) for k in range(len(ps)))
        j += len(ps)
        res[iid] = hits / len(ps)

    both = [i for i, v in res.items() if v >= 1.0]
    print(f"\n  robust under BOTH personas: {len(both)}/{len(tasks)} "
          f"({len(both) / max(len(tasks), 1):.1%})")
    print(f"  mean neutral chain_access_k: {st.mean(res.values()):.4f}")
    print(f"  distribution: {dict(sorted(Counter(round(v, 2) for v in res.values()).items()))}")
    # A sample, because the standing rule is that every stage shows real output.
    print("\n  sample (route -> neutral access):")
    for iid, v in list(res.items())[:5]:
        print(f"    {iid}  {v:.2f}   {scr[iid]['e1'][:30]!r} -> {scr[iid]['e3'][:24]!r}")

    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
