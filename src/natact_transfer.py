"""G14 -- natural-activation transfer (declared in G14-NATURAL-ACTIVATION.md before the run).

G12 rescued atomic-rehearsal failures by transplanting a bridge representation read from a
prompt that NAMES the bridge. That shows bridge availability is a bottleneck; it does not show
that ordinary computation would have supplied that state. Here the donor state is read from the
SAME unaided failing prompt, at the SAME subject position and layer, in a different model:

    base     the base model                       (composes these routes; never rehearsed anything)
    route    the route-rehearsal adapter, same seed
    noop     the atomic adapter itself            (machinery check: must reproduce the failure)
    wrongsub the base model's state for a DIFFERENT same-template, distinct-answer item's
             subject (specificity: should not restore THIS route's answer)
    base_last the base model's state at the LAST prompt token, patched at the last token
             (position control: is the repair specific to the subject position?)

Two models do not fit in 8 GB, so phase A caches one vector per prompt per source to disk and
phase B loads only the atomic adapter and patches from the cache. Both phases resumable.
The prompt set is rebuilt exactly as G12 built it and asserted equal to G12's stored item list.
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

FORMS = ["0_canonical", "1_named", "2_question", "3_imperative", "4_known_as", "5_refers"]


def norm(s) -> str:
    return " ".join(str(s).lower().split())


def build_pseudo(tasks: Path, post_path: str):
    from eval_runner import match_strict
    j = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    items = {o["task_id"]: o for o in j(tasks / "g4_split_X_measure.json")["items"]}
    meas = {t for t, o in items.items() if o.get("measurable", True)}
    rs = {r["task_id"] for r in j(tasks / "preserve_route.json")["rows"]}
    As = {r["task_id"] for r in j(tasks / "preserve_atomic.json")["rows"]}
    post = {r["task_id"]: r for r in j(post_path)["per_item"]}
    pseudo = []
    for t in sorted(meas & rs & As):
        it, rec = items[t], post[t]
        golds = [it["chain_answer"]] + list(it.get("chain_aliases", []))
        hit = {k: bool(match_strict(golds, rec["chain_preds_bio"][k])) for k in range(1, 6)}
        fails = [k for k in range(1, 6) if not hit[k]]
        succs = [k for k in range(1, 6) if hit[k]]
        for k in fails:
            pseudo.append(dict(it, chain_prompt=it["chain_forms"][FORMS[k]], task_id=f"{t}#{k}",
                               _route=t, _form=k, _kind="fail"))
        if succs:
            k = succs[0]
            pseudo.append(dict(it, chain_prompt=it["chain_forms"][FORMS[k]], task_id=f"{t}#{k}",
                               _route=t, _form=k, _kind="succ" if fails else "succ_other"))
    return pseudo


def classify(pred, it):
    from eval_runner import match_strict
    if match_strict([it["chain_answer"]] + list(it.get("chain_aliases", [])), pred):
        return "correct"
    if norm(it["anchor_answer"]) in norm(pred):
        return "own_bridge"
    return "other"


def main() -> None:
    import torch
    from eval_runner import build_prompt, match_strict
    from model_pin import revision_for
    from patch_rescue import last_mention_index

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--post", required=True, help="atomic mem100 post file (defines the failures)")
    ap.add_argument("--g12", required=True, help="G12 output for the same adapter (consistency + comparison)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--atomic-adapter", required=True)
    ap.add_argument("--route-adapter", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--cache", required=True, help="dir for cached source vectors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pseudo = build_pseudo(Path(args.tasks), args.post)
    g12 = json.loads(Path(args.g12).read_text(encoding="utf-8"))
    assert [p["task_id"] for p in pseudo] == [r["task_id"] for r in g12["per_item"]], \
        "prompt set differs from the one G12 ran on"
    g12u = {r["task_id"]: r["unpatched"]["cls"] for r in g12["per_item"]}
    g12b = {r["task_id"]: r["bridge"]["cls"] for r in g12["per_item"]}
    # keep exactly the prompts G12 kept: unpatched reproduced the stored outcome
    pseudo = [p for p in pseudo if (g12u[p["task_id"]] == "correct") == (p["_kind"] != "fail")]
    print(f"  prompts {len(pseudo)}: {dict(Counter(p['_kind'] for p in pseudo))}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts, poss = [], []
    for p in pseudo:
        t = build_prompt(tok, p["chain_prompt"])
        pos = last_mention_index(tok, t, p["e1_name"])
        assert pos is not None, f"subject unresolved in {p['task_id']}"
        texts.append(t); poss.append(pos)

    # wrong-subject donors: same template, distinct answer, seeded; drawn among these prompts
    by_t = defaultdict(list)
    for i, p in enumerate(pseudo):
        by_t["|".join(p["template_key"])].append(i)
    rng = random.Random(args.seed)
    wrong = []
    for i, p in enumerate(pseudo):
        pool = [k for k in by_t["|".join(p["template_key"])]
                if norm(pseudo[k]["chain_answer"]) != norm(p["chain_answer"])
                and pseudo[k]["_route"] != p["_route"]]
        wrong.append(rng.choice(pool) if pool else None)
    print(f"  wrong-subject donors available for {sum(w is not None for w in wrong)}/{len(pseudo)}")
    ex = next(i for i, p in enumerate(pseudo) if p["_kind"] == "fail")
    print(f"  sample: {pseudo[ex]['chain_prompt']!r} subject {pseudo[ex]['e1_name']!r} "
          f"token index {poss[ex]} -> {tok.decode(tok(texts[ex], add_special_tokens=False).input_ids[poss[ex]])!r}")
    if args.dry_run:
        print("  DRY RUN complete; no model loaded.")
        return

    from model_load import load_model
    from peft import PeftModel
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)

    def load(adapter):
        m = load_model(args.model, "fp16", False)
        if adapter:
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        return m.eval()

    # ---- phase A: cache the subject-position state at `layer` for every prompt, per source ----
    for name, adapter in (("base", None), ("route", args.route_adapter), ("noop", args.atomic_adapter)):
        f = cache / f"{name}_L{args.layer}.pt"
        if f.exists():
            print(f"  [skip] cached {name}")
            continue
        print(f"  === phase A: reading {name}", flush=True)
        m = load(adapter)
        got = {}

        def hook(_mod, _inp, out, got=got):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] > 1:
                got["v"] = hs[0, got["pos"], :].detach().float().cpu().clone()
                got["last"] = hs[0, -1, :].detach().float().cpu().clone()
            return out
        h = m.model.layers[args.layer].register_forward_hook(hook)
        vecs, lasts = [], []
        for i, (t, pos) in enumerate(zip(texts, poss)):
            got["pos"] = pos
            enc = tok(t, return_tensors="pt", add_special_tokens=False).to(m.device)
            with torch.no_grad():
                m(**enc)
            vecs.append(got["v"]); lasts.append(got["last"])
            if (i + 1) % 200 == 0:
                print(f"      {i + 1}/{len(texts)}", flush=True)
        h.remove()
        torch.save(torch.stack(vecs), f)
        if name == "base":
            torch.save(torch.stack(lasts), cache / f"base_last_L{args.layer}.pt")
        del m
        torch.cuda.empty_cache()

    # ---- phase B: patch the atomic adapter from the cache -------------------------------------
    print("  === phase B: patching the atomic adapter", flush=True)
    m = load(args.atomic_adapter)
    V = {n: torch.load(cache / f"{n}_L{args.layer}.pt") for n in ("base", "route", "noop", "base_last")}
    state = {}

    def whook(_mod, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if state.get("on") and hs.shape[1] > 1:      # prefill only
            hs = hs.clone()
            hs[0, state["pos"], :] = state["vec"].to(hs.dtype).to(hs.device)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return out
    h = m.model.layers[args.layer].register_forward_hook(whook)

    def run(i, vec, pos=None):
        state.update({"on": vec is not None, "pos": poss[i] if pos is None else pos, "vec": vec})
        enc = tok(texts[i], return_tensors="pt", add_special_tokens=False).to(m.device)
        with torch.no_grad():
            gen = m.generate(**enc, max_new_tokens=16, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        state["on"] = False
        return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    conds = ("noop", "base", "route", "base_last", "wrongsub")
    rows = []
    for i, p in enumerate(pseudo):
        r = {"task_id": p["task_id"], "kind": p["_kind"], "g12_bridge": g12b[p["task_id"]]}
        for c in conds:
            if c == "wrongsub":
                pred = run(i, V["base"][wrong[i]]) if wrong[i] is not None else ""
                d = pseudo[wrong[i]] if wrong[i] is not None else None
                r[c] = {"pred": pred, "cls": classify(pred, p), "eligible": d is not None,
                        "donor_answer": bool(d is not None and match_strict(
                            [d["chain_answer"]] + list(d.get("chain_aliases", [])), pred))}
            elif c == "base_last":   # position control: base state at the LAST prompt token
                pred = run(i, V[c][i], pos=-1)
                r[c] = {"pred": pred, "cls": classify(pred, p)}
            else:
                pred = run(i, V[c][i])
                r[c] = {"pred": pred, "cls": classify(pred, p)}
        rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"      {i + 1}/{len(pseudo)}", flush=True)
    h.remove()

    fail = [r for r in rows if r["kind"] == "fail"]
    guard = sum(r["noop"]["cls"] != "correct" for r in fail) / max(len(fail), 1)
    print(f"\n  GUARD no-op patch reproduces the failure on {guard:.3f} of failing prompts")
    summ = {"guard_noop": guard}
    for kind in ("fail", "succ", "succ_other"):
        xs = [r for r in rows if r["kind"] == kind]
        summ[kind] = {"n": len(xs)}
        for c in conds:
            ys = [r for r in xs if c != "wrongsub" or r[c]["eligible"]]
            summ[kind][c] = {"n": len(ys),
                             "correct": sum(r[c]["cls"] == "correct" for r in ys) / max(len(ys), 1),
                             "own_bridge": sum(r[c]["cls"] == "own_bridge" for r in ys) / max(len(ys), 1)}
            if c == "wrongsub":
                summ[kind][c]["donor_answer"] = sum(r[c]["donor_answer"] for r in ys) / max(len(ys), 1)
        print(f"  [{kind}] n={len(xs)}  " + "  ".join(
            f"{c} {summ[kind][c]['correct']:.3f}" for c in conds))
    # overlap with G12's explicit-bridge rescue, on failures
    both = sum(r["base"]["cls"] == "correct" and r["g12_bridge"] == "correct" for r in fail)
    summ["fail_overlap"] = {"g12_rescued": sum(r["g12_bridge"] == "correct" for r in fail),
                            "base_rescued": sum(r["base"]["cls"] == "correct" for r in fail),
                            "both": both}
    print(f"  overlap on failures: {summ['fail_overlap']}")
    Path(args.out).write_text(json.dumps({"layer": args.layer, "summary": summ, "per_item": rows},
                                         indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
