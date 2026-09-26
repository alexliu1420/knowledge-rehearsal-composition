"""Study 4 stage 0: screen TwoHopFact for chains with a bridge the model can NAME.

Inference only. No training, no patching. The screen answers one question -- does the
population Study 4 needs exist -- and produces a result either way.

Every result in Studies 1-3 sits on bridge entities the base model generates 0.000 of the
time. That is a scope limit on three published papers, and it is not testable on PrimeKG,
where no bridge is nameable by construction. This screen measures four things per chain:

    bridge_access_k   over k = 6 authored forms, how often the model names the bridge
    bridge_gen        the canonical form alone, comparable to the 0.000 that scopes 1-3
    hop2_known        does the model already know the second hop (decides variant A vs B)
    chain_unaided     the gap itself, before any injection

`chain_wrong_e1` is the shortcut control, and it is the dataset's own: the same chain frame
with a DIFFERENT first entity. An item answered correctly there is being answered by frame or
answer prior rather than by the chain, and is excluded. This is preferred to deleting the
subject, because deletion changes the prompt's grammar as well as its information.

Resumable: one JSON line per item as it is produced, keyed by a stable content hash. Rerunning
skips completed items. A screen of this size is hours on one 8 GB card and must survive being
interrupted.

Refuses to start without a passing `stage0_audit.json` whose fingerprint matches the current
code. The standing rule is *audit before the run*; this makes it mechanical.
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


def item_id(r) -> str:
    """Stable identity for caching: the chain, not its row index."""
    key = "|".join(str(r[k]) for k in
                   ("r1.category", "r2.category", "e1.value", "e2.value", "e3.value"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from audit_stage0 import code_fingerprint, flatten_aliases
    from eval_runner import generate, match_strict
    from model_load import load_model, load_tag
    from model_pin import revision_for
    from twohop_forms import CHAIN_FORM_KEYS, FORM_KEYS, bridge_forms, chain_forms

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4. Needed above ~3B on an 8 GB card. Recorded in the output "
                         "path and in every record, because quantisation is a change to "
                         "the model, not a neutral convenience.")
    ap.add_argument("--per-type", type=int, default=150,
                    help="stratified cap per composition type; 52 types")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", default=str(S4 / "results/stage0_audit.json"))
    ap.add_argument("--no-audit-gate", action="store_true",
                    help="run without a passing audit. Recorded in the output and intended "
                         "only for smoke tests on a handful of items.")
    ap.add_argument("--out", default=None,
                    help="defaults to results/stage0_screen.<model>.<precision>.jsonl, so "
                         "two model sizes cannot write into one another's file")
    args = ap.parse_args()

    tag = load_tag(args.model, args.precision, args.load_4bit)
    if args.out is None:
        args.out = str(S4 / f"results/stage0_screen.{tag}.jsonl")

    # ---- the pre-run check gate ---------------------------------------------------------------
    gated = not args.no_audit_gate
    ap_path = Path(args.audit)
    if gated:
        if not ap_path.exists():
            raise SystemExit(f"no audit stamp at {ap_path}. Run audit_stage0.py first.")
        stamp = json.loads(ap_path.read_text(encoding="utf-8"))
        if not stamp.get("passed"):
            raise SystemExit(f"audit stamp records failures: {stamp.get('failed_checks')}")
        if stamp.get("code_fingerprint") != code_fingerprint():
            raise SystemExit(
                "audit stamp is stale: the screening or scoring code changed since it was "
                "written. Re-run audit_stage0.py.")
        if stamp.get("model") != args.model or bool(stamp.get("load_4bit")) != bool(args.load_4bit):
            raise SystemExit(
                f"audit stamp is for {stamp.get('model')} "
                f"(4bit={bool(stamp.get('load_4bit'))}), but this run is {args.model} "
                f"(4bit={bool(args.load_4bit)}). A4 form adequacy and A1 are model-specific; "
                f"re-run audit_stage0.py for this model.")
        if not stamp.get("with_model"):
            print("  ! audit stamp covers offline checks only (A1/A4/A7 skipped). "
                  "Acceptable for a partial run; re-audit with --with-model before "
                  "the full screen is reported.")

    from datasets import load_dataset
    df = load_dataset("soheeyang/TwoHopFact", split="train", revision=__import__("model_pin").TWOHOPFACT_REVISION).to_pandas()

    # ---- stratified sample, seeded ----------------------------------------------------
    rng = random.Random(args.seed)
    by_type: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(df)):
        by_type[(df.iloc[i]["r1.category"], df.iloc[i]["r2.category"])].append(i)
    picked: list[int] = []
    for k in sorted(by_type):
        ix = by_type[k][:]
        rng.shuffle(ix)
        picked.extend(ix[: args.per_type])
    # Interleave rather than sort. Processing in index order walks one composition type to
    # completion before starting the next, so an interrupted run -- and this is a multi-hour
    # run on one card -- leaves a partial sample drawn from a handful of types, which cannot
    # be read for anything. Shuffling makes any prefix a spread across all 52.
    rng.shuffle(picked)
    print(f"  {len(by_type)} composition types, sampling <= {args.per_type} each "
          f"-> {len(picked)} chains")

    # ---- resume -----------------------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001 - a truncated final line is expected
                    pass
        print(f"  resuming: {len(done)} items already done")

    todo = [i for i in picked if item_id(df.iloc[i]) not in done]
    if not todo:
        print("  nothing to do")
        return
    print(f"  {len(todo)} chains to score, {len(FORM_KEYS) + 2 + len(CHAIN_FORM_KEYS)} prompts each "
          f"= {len(todo) * (len(FORM_KEYS) + 2 + len(CHAIN_FORM_KEYS))} generations")

    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, args.precision, args.load_4bit)
    print(f"  {args.model} @ {revision_for(args.model)}"
          f"{'  [4-bit NF4]' if args.load_4bit else f'  [{args.precision}]'}")

    # Items are processed in small groups so a kill loses at most one group, and so the
    # generation batch stays a batch of prompts rather than a batch of items.
    GROUP = 8
    fh = out.open("a", encoding="utf-8")
    try:
        for g0 in range(0, len(todo), GROUP):
            grp = todo[g0: g0 + GROUP]
            prompts: list[str] = []
            spans: list[tuple[int, int]] = []
            metas = []
            for i in grp:
                r = df.iloc[i]
                forms = bridge_forms(r["mu.template"], r["e1.value"], r["e2.rough_category"])
                ps = [forms[k] for k in FORM_KEYS]
                ps.append(r["r2(e2).prompt"])          # hop 2 asked directly
                ps.append(r["r2(r1(e1')).prompt"])     # shortcut control: wrong first entity
                # The chain under SIX phrasings, not one. G4 must select routes the model
                # composes ROBUSTLY: selecting a route on a single successful generation is
                # the regression trap, and would manufacture "damage" from an inert
                # injection. Form 0 is TwoHopFact's own, so the authored forms cannot be the
                # only thing recovering a chain.
                cf = chain_forms(r["r2.category"], r["mu.template"], r["e1.value"],
                                 r["e3.rough_category"], r["r2(r1(e1)).prompt"])
                ps.extend(cf[k] for k in CHAIN_FORM_KEYS)
                start = len(prompts)
                prompts.extend(ps)
                spans.append((start, len(prompts)))
                metas.append(r)

            outs = generate(model, tok, prompts, batch_size=args.batch_size,
                            max_new_tokens=args.max_new_tokens)

            for (s, e), r in zip(spans, metas):
                o = outs[s:e]
                e2_gold = [str(r["e2.value"])] + flatten_aliases(r["e2.aliases"])
                e3_gold = [str(r["e3.value"])] + flatten_aliases(r["e3.aliases"])
                acc = {k: bool(match_strict(e2_gold, o[j])) for j, k in enumerate(FORM_KEYS)}
                n = len(FORM_KEYS)
                # prompt layout: [0..n) bridge forms, n hop2, n+1 wrong-e1, then chain forms
                cbase = n + 2
                chain = {k: bool(match_strict(e3_gold, o[cbase + j]))
                         for j, k in enumerate(CHAIN_FORM_KEYS)}
                rec = {
                    "id": item_id(r),
                    "r1_category": r["r1.category"], "r2_category": r["r2.category"],
                    "e1": str(r["e1.value"]), "e2": str(r["e2.value"]),
                    "e3": str(r["e3.value"]),
                    "e2_qid": str(r["e2.wikidata_qid"]),
                    "e2_type": str(r["e2.rough_category"]),
                    "access": acc,
                    "bridge_access_k": sum(acc.values()) / len(acc),
                    "bridge_gen": acc["0_canonical"],
                    "hop2_known": bool(match_strict(e3_gold, o[n])),
                    # `chain_unaided` stays the canonical single-prompt outcome so it remains
                    # comparable with everything measured before chain forms existed.
                    "chain_unaided": chain["0_canonical"],
                    "chain_access": chain,
                    # The robustness measure G4 selects on. A route counted as composable
                    # must survive rephrasing, not one lucky draw.
                    "chain_access_k": sum(chain.values()) / len(chain),
                    "chain_wrong_e1": bool(match_strict(e3_gold, o[n + 1])),
                    # Every prediction is stored, not just the booleans. Storing only the
                    # scored outcome once made an alias-parsing bug un-rescorable without a
                    # full re-run of the screen.
                    "preds": {"hop2": o[n][:200], "chain_wrong_e1": o[n + 1][:200],
                              **{k: o[j][:200] for j, k in enumerate(FORM_KEYS)},
                              **{f"chain_{k}": o[cbase + j][:200]
                                 for j, k in enumerate(CHAIN_FORM_KEYS)}},
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if (g0 // GROUP) % 10 == 0:
                print(f"    {g0 + len(grp)}/{len(todo)}", flush=True)
    finally:
        fh.close()
        del model
        torch.cuda.empty_cache()

    print(f"\nwrote {out}")
    print("Run analyze_stage0.py for the 2x2 cell counts and the stop-condition check.")


if __name__ == "__main__":
    main()
