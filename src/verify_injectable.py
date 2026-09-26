"""Which Wikidata facts about a protected route's bridge does the model NOT already hold?

G4 injects one true fact about the bridge entity `E2` of a route it is trying to protect. The
fact has to be one the model does not already produce, or the injection teaches nothing and no
damage could be attributed to it.

"Does not produce" is measured the way every other access measure in this study is measured:
over **six phrasings**, not one. A fact the model emits under any phrasing is discarded. That
direction is deliberate -- a false "the model doesn't know this" is far more damaging here than
a false "it does", because it would put an already-known fact in the training set and make the
injection a no-op.

Conflict is recorded, not assumed away. `G4-PRERUN-AUDIT.md` §0 corrects an earlier claim that
these facts are "non-conflicting": failing to produce the true object does not mean holding no
belief. For every candidate the model's actual answer is stored, so damage can be reported
split by whether a confident competing belief was overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

S4 = Path(".")

# Mappings rejected by validate_mapping against TwoHopFact's own statements: P27 is country
# of citizenship rather than birth country (67% agreement), and P495 is not what TwoHopFact
# means by "released in the country of" (10%). Using either would inject a different relation
# than the template asserts.
REJECTED = {"birthcntry", "origcntry"}


def fact_forms(predicate: str, subject: str, object_type_animate: bool) -> dict[str, str]:
    """Six ways of asking for the object of one relation, reusing the study's rewrites."""
    from twohop_forms import R2_NOUN, _cap

    np_ = R2_NOUN[predicate].format(subject)
    pron = "Who" if object_type_animate else "What"
    return {
        "0_canonical":  f"{_cap(np_)} is",
        "1_named":      f"The name of {np_} is",
        "2_question":   f"Q: {pron} is {np_}?\nA:",
        "3_imperative": f"Name {np_}:",
        "4_known_as":   f"{_cap(np_)} is known as",
        "5_refers":     f"'{_cap(np_)}' refers to",
    }


# Predicates whose object is a person, for pronoun selection.
ANIMATE_OBJECT = {"father", "mother", "spouse", "founder", "author", "director",
                  "creator", "actor", "president"}


def main() -> None:
    from transformers import AutoTokenizer

    from eval_runner import generate, match_strict
    from model_load import load_model, load_tag
    from model_pin import revision_for
    from wikidata_facts import candidate_facts, fetch_entities, object_qids

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--screen", default=None)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = load_tag(args.model, args.precision, args.load_4bit)
    if args.screen is None:
        args.screen = str(S4 / f"results/stage0_screen.{tag}.jsonl")
    if args.out is None:
        args.out = str(S4 / f"results/injectable.{tag}.json")

    scr = {}
    for line in Path(args.screen).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            scr[r["id"]] = r
    prot = [r for r in scr.values()
            if r["chain_access_k"] >= 1.0 and not r["chain_wrong_e1"]]
    print(f"  {len(prot)} protected routes")

    cache = S4 / "results/wikidata_cache"
    qids = sorted({str(r["e2_qid"]) for r in prot})
    ents = fetch_entities(qids, cache)
    objs = sorted({o for e in ents.values() for o in object_qids(e)})
    labels = {q: v.get("label") for q, v in {**ents, **fetch_entities(objs, cache)}.items()}

    # candidates: one route may offer several; all are tested and the choice is made after
    cands = []
    for r in prot:
        own = r["r2_category"].rsplit("-", 1)[-1]
        e = ents.get(str(r["e2_qid"]))
        if not e:
            continue
        for f in candidate_facts(str(r["e2_qid"]), e, labels, own):
            if f["predicate"] in REJECTED:
                continue
            f["route_id"] = r["id"]
            cands.append(f)
    print(f"  {len(cands)} candidate facts over "
          f"{len({c['route_id'] for c in cands})} routes")

    # Index, not id(). Keying on object identity breaks the moment a later loop rebinds
    # the variable, and fails silently by scoring everything as unknown.
    prompts, meta = [], []
    for ci, c in enumerate(cands):
        forms = fact_forms(c["predicate"], c["subject"],
                           c["predicate"] in ANIMATE_OBJECT)
        for k, p in forms.items():
            prompts.append(p)
            meta.append((ci, k))
    print(f"  {len(prompts)} prompts (6 forms per candidate)")

    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, args.precision, args.load_4bit)
    outs = generate(model, tok, prompts, batch_size=args.batch_size, max_new_tokens=16)

    known: dict[int, int] = {}
    answers: dict[int, dict] = {}
    for (ci, k), o in zip(meta, outs):
        hit = bool(match_strict([cands[ci]["object"]], o))
        known[ci] = known.get(ci, 0) + int(hit)
        answers.setdefault(ci, {})[k] = o[:120]

    out = []
    for ci, c in enumerate(cands):
        c = dict(c)
        c["known_forms"] = known.get(ci, 0)
        # STRICT: any form producing the object disqualifies the fact.
        c["injectable"] = known.get(ci, 0) == 0
        c["model_answers"] = answers.get(ci, {})
        out.append(c)

    inj = [c for c in out if c["injectable"]]
    routes = {c["route_id"] for c in inj}
    print(f"\n  injectable (model produces the object under NO form): {len(inj)}")
    print(f"  routes with >=1 injectable fact: {len(routes)}")
    print(f"  by predicate: {dict(Counter(c['predicate'] for c in inj).most_common())}")

    Path(args.out).write_text(json.dumps(
        {"model": args.model, "load_4bit": args.load_4bit, "screen": Path(args.screen).name,
         "rejected_predicates": sorted(REJECTED),
         "n_protected_routes": len(prot), "n_candidates": len(cands),
         "n_injectable": len(inj), "n_routes_covered": len(routes),
         "facts": out}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
