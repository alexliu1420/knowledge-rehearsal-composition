"""G12 -- failure-matched causal test (pre-registered in G12-FAILURE-MATCHED-PREREG.md).

G11 patched the CANONICAL prompt on routes the adapter still answers. This patches the
held-out phrasings the atomic-replay adapter actually gets WRONG, plus one matched phrasing per
route it gets right, and asks what supplying the resolved bridge does to each.

    unpatched      must reproduce the stored miss/hit (guard, abort below 95% agreement)
    bridge         own bridge transplanted at the subject position -> does it RESCUE the failure?
    wrong_bridge   distinct-answer same-template donor -> does the output FOLLOW the donor
                   (donor's answer, or donor's BRIDGE emitted)?
    self           Study-3 control: own subject position read from the bridge-prefixed prompt

Each (route, form) becomes a pseudo-item whose `chain_prompt` is that form, so
`patch_rescue.run_condition` is used UNCHANGED apart from recording the donor it chose. The
persona is the biomedical SYSTEM_PROMPT in both the measurement and the patcher (build_prompt
default), so the patched prompt is the measured prompt.

`--dry-run` builds the sets and checks every position lookup with the tokenizer only (CPU).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FORMS = ["0_canonical", "1_named", "2_question", "3_imperative", "4_known_as", "5_refers"]


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def classify(pred, it, donor):
    """correct / own_bridge / donor_answer / donor_bridge / other  (first match wins)."""
    from eval_runner import match_strict
    if match_strict([it["chain_answer"]] + list(it.get("chain_aliases", [])), pred):
        return "correct"
    if donor is not None and match_strict(
            [donor["chain_answer"]] + list(donor.get("chain_aliases", [])), pred):
        return "donor_answer"
    if norm(it["anchor_answer"]) in norm(pred):
        return "own_bridge"
    if donor is not None and norm(donor["anchor_answer"]) in norm(pred):
        return "donor_bridge"
    return "other"


def main() -> None:
    from eval_runner import build_prompt, match_strict
    from model_pin import revision_for
    from patch_rescue import build_pair, last_mention_index

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="dir with g4_split_X_measure.json + preserve_*.json")
    ap.add_argument("--post", required=True, help="atomic mem100 post file (stored predictions)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--conditions", nargs="+", default=["unpatched", "bridge", "wrong_bridge", "self"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    D = Path(args.tasks)
    j = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    items = {o["task_id"]: o for o in j(D / "g4_split_X_measure.json")["items"]}
    meas = {t for t, o in items.items() if o.get("measurable", True)}
    rs = {r["task_id"] for r in j(D / "preserve_route.json")["rows"]}
    As = {r["task_id"] for r in j(D / "preserve_atomic.json")["rows"]}
    both = sorted(meas & rs & As)
    post = {r["task_id"]: r for r in j(args.post)["per_item"]}

    # ---- failing held-out forms, and one matched success per route that has both ----------
    pseudo = []
    for t in both:
        it, rec = items[t], post[t]
        golds = [it["chain_answer"]] + list(it.get("chain_aliases", []))
        hit = {k: bool(match_strict(golds, rec["chain_preds_bio"][k])) for k in range(1, 6)}
        fails = [k for k in range(1, 6) if not hit[k]]
        succs = [k for k in range(1, 6) if hit[k]]
        for k in fails:
            pseudo.append(dict(it, chain_prompt=it["chain_forms"][FORMS[k]], task_id=f"{t}#{k}",
                               _route=t, _form=k, _kind="fail",
                               _stored=rec["chain_preds_bio"][k]))
        if succs:
            # 'succ'       : a phrasing the adapter gets right on a route where it ALSO fails one
            #                (the pre-registered matched control)
            # 'succ_other' : a phrasing it gets right on a route with no failures. Reported
            #                separately; also fills the same-template donor pools, which were
            #                empty for 42 failing prompts when only failing routes were present.
            k = succs[0]
            pseudo.append(dict(it, chain_prompt=it["chain_forms"][FORMS[k]], task_id=f"{t}#{k}",
                               _route=t, _form=k, _kind="succ" if fails else "succ_other",
                               _stored=rec["chain_preds_bio"][k]))
    if args.limit:
        pseudo = pseudo[: args.limit]
    kinds = Counter(p["_kind"] for p in pseudo)
    stored_cls = Counter(classify(p["_stored"], p, None) for p in pseudo if p["_kind"] == "fail")
    print(f"  BOTH routes {len(both)}; failing (route,form) prompts {kinds['fail']} over "
          f"{len({p['_route'] for p in pseudo if p['_kind']=='fail'})} routes; "
          f"matched successes {kinds['succ']}; other-route successes {kinds['succ_other']}")
    print(f"  stored miss classes: {dict(stored_cls)}")
    by_form = Counter(p["_form"] for p in pseudo if p["_kind"] == "fail")
    print(f"  failures by form: {dict(sorted(by_form.items()))}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- CPU audit: every position the patcher will need must resolve --------------------
    bad_t = bad_s = 0
    for p in pseudo:
        target, source = build_pair(p)
        assert target == p["chain_prompt"] and target != items[p["_route"]]["chain_prompt"], \
            "pseudo-item must carry the HELD-OUT form, not the canonical prompt"
        if last_mention_index(tok, build_prompt(tok, target), p["e1_name"]) is None:
            bad_t += 1
        if last_mention_index(tok, build_prompt(tok, source), p["anchor_answer"]) is None:
            bad_s += 1
    print(f"  position audit: target subject unresolved {bad_t}, source bridge unresolved {bad_s} "
          f"of {len(pseudo)}")
    # donor pools under distinct-answer sourcing
    from collections import defaultdict
    by_t = defaultdict(list)
    for p in pseudo:
        by_t["|".join(p["template_key"])].append(p)
    no_pool = sum(1 for p in pseudo if not [q for q in by_t["|".join(p["template_key"])]
                  if norm(q["chain_answer"]) != norm(p["chain_answer"])])
    print(f"  prompts with no distinct-answer same-template donor: {no_pool}")
    ex = next(p for p in pseudo if p["_kind"] == "fail")
    print(f"  sample failing prompt: {ex['chain_prompt']!r}\n    stored {ex['_stored']!r}  "
          f"gold {ex['chain_answer']!r}  bridge {ex['anchor_answer']!r}")
    print(f"  sample source prompt : {build_pair(ex)[1]!r}")
    if args.dry_run:
        print("  DRY RUN complete; no model loaded.")
        return

    # ---- GPU ---------------------------------------------------------------------------------
    from peft import PeftModel
    from model_load import load_model
    from patch_rescue import run_condition
    model = load_model(args.model, "fp16", False)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    by_id = {p["task_id"]: p for p in pseudo}
    res = {}
    for c in args.conditions:
        donors: list = []
        print(f"  === condition {c}", flush=True)
        _, preds = run_condition(model, tok, pseudo, args.layer, c, random.Random(args.seed),
                                 max_new_tokens=16, distinct_answer=True, donors_out=donors)
        res[c] = [{"pred": pr, "donor": dn,
                   "cls": classify(pr, p, by_id.get(dn) if c in ("wrong_bridge",) else None)}
                  for p, pr, dn in zip(pseudo, preds, donors)]
        if c == "unpatched":
            agree = sum((r["cls"] == "correct") == (p["_kind"] != "fail")
                        for p, r in zip(pseudo, res[c])) / len(pseudo)
            print(f"  GUARD unpatched reproduces stored hit/miss on {agree:.3f} of prompts")
            if agree < 0.95:
                Path(args.out).write_text(json.dumps({"guard_failed": agree}), encoding="utf-8")
                raise SystemExit("  GUARD FAILED (<0.95): patched conditions not run")

    # restrict every rate to prompts whose unpatched run reproduced the stored outcome
    keep = [i for i, p in enumerate(pseudo)
            if (res["unpatched"][i]["cls"] == "correct") == (p["_kind"] != "fail")]
    summ = {}
    for kind in ("fail", "succ", "succ_other"):
        idx = [i for i in keep if pseudo[i]["_kind"] == kind]
        summ[kind] = {"n": len(idx)}
        for c in args.conditions:
            cnt = Counter(res[c][i]["cls"] for i in idx)
            summ[kind][c] = {k: cnt[k] / max(len(idx), 1) for k in
                             ("correct", "own_bridge", "donor_answer", "donor_bridge", "other")}
    # bridge-emission failures separately (the pre-registered sub-prediction)
    idx = [i for i in keep if pseudo[i]["_kind"] == "fail"
           and res["unpatched"][i]["cls"] == "own_bridge"]
    summ["fail_bridge_emission"] = {"n": len(idx)}
    for c in args.conditions:
        cnt = Counter(res[c][i]["cls"] for i in idx)
        summ["fail_bridge_emission"][c] = {k: cnt[k] / max(len(idx), 1) for k in
                                           ("correct", "own_bridge", "donor_answer", "donor_bridge", "other")}
    for kind, v in summ.items():
        print(f"\n  [{kind}] n={v['n']}")
        for c in args.conditions:
            print(f"    {c:<13}" + "  ".join(f"{k} {x:.3f}" for k, x in v[c].items()))
    out = {"model": args.model, "adapter": args.adapter, "layer": args.layer,
           "n_prompts": len(pseudo), "n_kept": len(keep), "summary": summ,
           "per_item": [{"task_id": p["task_id"], "route": p["_route"], "form": p["_form"],
                         "kind": p["_kind"], "stored": p["_stored"],
                         **{c: res[c][i] for c in args.conditions}}
                        for i, p in enumerate(pseudo)]}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
