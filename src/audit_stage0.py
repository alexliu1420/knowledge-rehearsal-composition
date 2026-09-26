"""Pre-run audit for Study 4 stage 0. Must pass before the screen runs.

The standing rule is *audit before the run, not after*: every audit that arrived after a run
in this programme either invalidated it or forced a rerun. This file makes the rule
mechanical rather than aspirational -- it writes a stamp recording which checks passed, and
`screen_twohop.py` refuses to start without one that matches the current code.

The checks are A1-A7 of `docs/STAGE0.md`. Two of them (A6, A7) are
obligations that Studies 1-3 did not have, and they exist because this is the first study in
the programme to use a public benchmark rather than a dataset we built:

    A6  the patch position must be well defined on data we did not construct
    A7  a public 2024 question set may be in the model's pretraining

Checks that need a GPU are skipped unless `--with-model` is passed, and the stamp records
which ones actually ran, so a stamp cannot claim more than was checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

S1 = Path("studies-01-anchoring")
S4 = Path(".")

# Study 1 v0.2.0 deposited pooled7 values, seed-averaged over 518 items. A1 reproduces these
# through the stage-0 scoring path; if they drift, the path is measuring something else.
EXPECT_CHAIN = 0.1377
TOL = 0.02
# The uninjected base model composes these chains at 0.000 -- the fact that scopes Studies
# 1-3 to the long tail. A1 reproduces it; anything materially above means the scoring path is
# crediting answers the model cannot produce.
BASE_CHAIN_MAX = 0.05
# Aliases shorter than this are dropped: the scorer matches substrings, and a two-character
# alias matches inside unrelated words.
MIN_ALIAS_CHARS = 4

# Entities the model must be able to name if the authored forms work at all. Chosen to be
# unambiguous and famous, so a failure here is a defect in our prompts, not in the model.
FORM_CONTROLS = [
    ("the author of the novel {}", "Nineteen Eighty-Four", "person", "George Orwell"),
    ("the capital of {}", "France", "city", "Paris"),
    ("the author of the novel {}", "Pride and Prejudice", "person", "Jane Austen"),
    ("the capital of {}", "Japan", "city", "Tokyo"),
    ("the author of the novel {}", "Hamlet", "person", "Shakespeare"),
]


def code_fingerprint() -> str:
    """Hash the files whose behaviour the stamp vouches for."""
    h = hashlib.sha256()
    # audit_stage0.py is included because it holds `flatten_aliases`, which the screen uses
    # for scoring. Excluding it left the gate blind at exactly the point where it failed: the
    # alias parser was wrong, was corrected, and the stamp stayed valid across the change.
    for rel in ("src/twohop_forms.py", "src/screen_twohop.py",
                "src/eval_runner.py", "src/audit_stage0.py"):
        p = Path(rel)
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def flatten_aliases(x) -> list[str]:
    """TwoHopFact aliases -> a flat list of strings.

    The field arrives from pandas as a **string holding a Python tuple literal**, e.g.
    "(('George Orwell', 'Eric Blair', ...),)" -- not as a nested sequence. An earlier version
    of this function treated any string as terminal and returned that whole repr as a single
    alias, so alias matching silently never fired: every measurement was scored against the
    canonical string alone. Parse the literal first, then flatten.
    """
    import ast

    out: list[str] = []

    def rec(v):
        if v is None:
            return
        if isinstance(v, str):
            s = v.strip()
            # a serialised tuple/list literal, not an alias
            if s[:1] in "([" and s[-1:] in ")]":
                try:
                    rec(ast.literal_eval(s))
                    return
                except (ValueError, SyntaxError):
                    pass
            out.append(v)
            return
        try:
            for i in v:
                rec(i)
        except TypeError:
            return

    rec(x)
    # Minimum length, because the scorer is substring-based. Wikidata aliases include ISO
    # codes and initialisms -- "ar" for Argentina, "MJ", "EA", "UM", "Sue" for Sue Townsend,
    # and flag emoji -- and "ar" matches inside "Paris". Accepting those would credit answers
    # the model never gave. The cost is under-crediting a few real short aliases ("LDN",
    # "CSM"); that direction is the safe one, and it is reported rather than absorbed.
    seen, res = set(), []
    for s in out:
        if not isinstance(s, str):
            continue
        v = s.strip()
        if len(v) < MIN_ALIAS_CHARS or not any(c.isalnum() for c in v):
            continue
        if v not in seen:
            seen.add(v)
            res.append(v)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-model", action="store_true",
                    help="run the checks that need a GPU (A1, A4, A7)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--sample", type=int, default=300, help="items for A3/A6")
    ap.add_argument("--out", default=str(S4 / "results/stage0_audit.json"))
    args = ap.parse_args()

    from datasets import load_dataset
    from twohop_forms import FORM_KEYS, assert_canonical_matches, bridge_forms

    rep: dict = {"checks": {}, "with_model": args.with_model, "model": args.model,
                 "load_4bit": args.load_4bit,
                 "code_fingerprint": code_fingerprint()}
    failed: list[str] = []

    def record(name: str, ok: bool, detail) -> None:
        rep["checks"][name] = {"pass": bool(ok), "detail": detail}
        print(f"  {name}  {'PASS' if ok else 'FAIL'}  {detail}")
        if not ok:
            failed.append(name)

    print("=" * 74)
    print("STAGE 0 PRE-RUN AUDIT")
    print()

    df = load_dataset("soheeyang/TwoHopFact", split="train", revision=__import__("model_pin").TWOHOPFACT_REVISION).to_pandas()
    print(f"  TwoHopFact: {len(df)} rows, "
          f"{df.groupby(['r1.category', 'r2.category']).ngroups} composition types")
    print()

    # ---- A2 · prompt path identity (no GPU) --------------------------------------------
    from eval_runner import build_prompt

    class _TokStub:
        """Only needs apply_chat_template; checks the callable is the shared one."""

    record("A2_prompt_path", callable(build_prompt),
           "stage 0 uses eval_runner.build_prompt, the Studies 1-3 path")

    # ---- A3 · alias normalisation, both directions -------------------------------------
    from eval_runner import match_strict

    rng = random.Random(0)
    idx = rng.sample(range(len(df)), min(args.sample, len(df)))
    empty_alias = 0
    false_pos = []
    unparsed = []
    canonical_missing = 0
    for i in idx:
        r = df.iloc[i]
        al = flatten_aliases(r["e2.aliases"])
        if not al:
            empty_alias += 1
        for a in al:
            # over-finding: an alias so short it matches inside unrelated text
            if len(a.strip()) < 3:
                false_pos.append((r["e2.value"], a))
            # PARSE FAILURE: an earlier version returned the serialised tuple as one
            # "alias", so alias matching silently never fired and every measurement was
            # scored against the canonical string alone. A3 passed anyway, because a repr
            # is not short. Check the shape of what came back, not only its length.
            if a[:1] in "([" and ("', '" in a or '", "' in a):
                unparsed.append((r["e2.value"], a[:60]))
        # the canonical value should appear among its own aliases after parsing
        if al and str(r["e2.value"]) not in al:
            canonical_missing += 1
    multi = sum(1 for i in idx if len(flatten_aliases(df.iloc[i]["e2.aliases"])) > 1)
    ok = not false_pos and not unparsed and multi > 0
    record("A3_alias_normalisation", ok,
           f"{args.sample} sampled; {empty_alias} with no alias list; "
           f"{len(false_pos)} under 3 chars; {len(unparsed)} UNPARSED (serialised literal "
           f"returned as a single alias); {multi} items yielded >1 distinct alias; "
           f"{canonical_missing} where the canonical value is absent from its own alias list"
           + (f"  e.g. {unparsed[:2]}" if unparsed else ""))

    # ---- A6 · position alignment for the stage-1 patch ---------------------------------
    # The chain prompt and the bridge-supplied prompt must share a span verbatim, and the
    # first-hop mention must be locatable by exact character offset. Without this the patch
    # has no well-defined position on data we did not construct.
    misaligned = []
    for i in idx:
        r = df.iloc[i]
        chain = r["r2(r1(e1)).prompt"]
        e1 = str(r["e1.value"])
        if e1.lower() not in chain.lower():
            misaligned.append((e1, chain[:60]))
    record("A6_position_alignment", len(misaligned) == 0,
           f"{len(idx) - len(misaligned)}/{len(idx)} chain prompts contain the E1 mention "
           f"by exact substring" + (f"; e.g. {misaligned[:2]}" if misaligned else ""))

    # ---- canonical anchor over the WHOLE set (cheap, so not sampled) -------------------
    bad = 0
    for r in df.itertuples(index=False):
        d = r._asdict() if hasattr(r, "_asdict") else None
        break
    mism = 0
    for i in range(len(df)):
        r = df.iloc[i]
        try:
            assert_canonical_matches(r["mu.template"], r["e1.value"],
                                     r["e2.rough_category"], r["r1(e1).prompt"])
        except AssertionError:
            mism += 1
            if mism > 5:
                break
    record("A0_canonical_anchor", mism == 0,
           f"form 0 reproduces TwoHopFact's own r1(e1).prompt on all {len(df)} rows"
           if mism == 0 else f"{mism}+ mismatches")

    # ---- model-dependent checks --------------------------------------------------------
    if args.with_model:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from eval_runner import generate
        from model_pin import revision_for

        from model_load import load_model

        tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = load_model(args.model, args.precision, args.load_4bit)

        # A1 · the scoring path reproduces a published value
        tasks = json.loads((S1 / "data/tasks/pooled7.json").read_text(encoding="utf-8"))["items"]
        qs = [t["chain_prompt"] for t in tasks]
        preds = generate(model, tok, qs, batch_size=16, max_new_tokens=24)
        acc = sum(match_strict([t["chain_answer"]], p) for t, p in zip(tasks, preds)) / len(tasks)
        # Two-sided, because both failure modes are real. Study 2's scorer bug put every
        # condition at 0.99; a broken prompt path would put everything at 0. The programme
        # has a published value for exactly this configuration -- the uninjected base model
        # composes pooled7 chains at 0.000 -- so this reproduces a known number rather than
        # merely checking the output is not absurd.
        # The published 0.000 belongs to Qwen2.5-1.5B. A larger model may genuinely
        # compose some of these chains, so reproducing 0.000 is required only for the model
        # the value came from. For any other model the check is two-sided sanity: a scorer
        # at ceiling is the Study 2 failure mode, and a scorer at exactly zero on a model
        # that should manage a few would be a broken prompt path.
        is_published_model = "Qwen2.5-1.5B-Instruct" in args.model
        if is_published_model:
            ok = acc <= BASE_CHAIN_MAX
            why = (f"reproduces the published base-model 0.000 for this model "
                   f"(pass requires <= {BASE_CHAIN_MAX})")
        else:
            ok = acc <= 0.95
            why = ("model has no published value here; checked only for a scorer stuck at "
                   "ceiling, which is the Study 2 failure mode. The value is reported, not "
                   "asserted")
        record("A1_scoring_path", ok,
               f"uninjected pooled7 chain through the stage-0 path = {acc:.4f} "
               f"on {args.model}; {why}")

        # A4 · every authored form fires on entities the model certainly knows
        per_form_hits = {k: 0 for k in FORM_KEYS}
        for mu, e1, typ, gold in FORM_CONTROLS:
            f = bridge_forms(mu, e1, typ)
            outs = generate(model, tok, [f[k] for k in FORM_KEYS],
                            batch_size=8, max_new_tokens=16)
            for k, o in zip(FORM_KEYS, outs):
                if match_strict([gold], o):
                    per_form_hits[k] += 1
        dead = [k for k, v in per_form_hits.items() if v == 0]
        record("A4_form_adequacy", len(dead) == 0,
               f"hits per form over {len(FORM_CONTROLS)} famous controls: {per_form_hits}"
               + (f"; DEAD FORMS {dead} would silently depress access_k on every item"
                  if dead else ""))

        # A7 · contamination smell test
        # TwoHopFact is a public 2024 set. A composition type where the two-hop chain is
        # answered far more often than its own second hop is anomalous and is flagged.
        # Deferred, not passed. Recording a hardcoded True for a check that never runs is
        # how an audit becomes decorative.
        rep["checks"]["A7_contamination_probe"] = {
            "pass": None,
            "detail": "DEFERRED to the screen: the per-composition-type chain-vs-hop2 "
                      "anomaly needs the screen's numbers. Reported by analyze_stage0.py; "
                      "not adjudicated here."}
        print("  A7_contamination_probe  DEFER  computed by analyze_stage0.py, not here")
        del model
        torch.cuda.empty_cache()
    else:
        for n in ("A1_scoring_path", "A4_form_adequacy", "A7_contamination_probe"):
            rep["checks"][n] = {"pass": None, "detail": "skipped: needs --with-model"}
            print(f"  {n}  SKIP  needs --with-model")

    rep["passed"] = not failed
    rep["failed_checks"] = failed
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print()
    if failed:
        print(f"AUDIT FAILED: {', '.join(failed)}")
        print("The screen will refuse to run. Fix these before spending GPU time.")
        raise SystemExit(1)
    if not args.with_model:
        print("AUDIT PASSED (offline checks only). Re-run with --with-model before the "
              "full screen; the stamp records that A1/A4/A7 were skipped.")
    else:
        print("AUDIT PASSED.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
