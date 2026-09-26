"""Transplant one representation into the model's own forward pass, and measure composition.

On the unaided chain prompt, at the final token of the first-hop mention and at one layer
fixed before the run, the residual stream is overwritten with the representation the same
model holds at the same position when the bridge entity is named. Nothing is added to the
prompt.

Positions are well posed because the supplied-bridge prompt is the chain prompt with a
prefix, so the two share the chain prompt verbatim; `audit_patch_alignment.py` verifies that
the shared span tokenises identically on every item before any run.

Five conditions:

    unpatched        the chain as the model answers it
    bridge           the intervention
    wrong_bridge     the same transplant sourced from a DIFFERENT item, so a rescue that
                     follows the patch rather than the fact is visible
    self             the same content already at that position, an inert control
    wrong            a different item's first-hop position, showing the site is causal

    --null           runs the whole procedure on the base model, where the fact was never
                     learned, so an effect that does not need the fact is visible

`--source-adapter` and `--no-guard` support the source-provenance decomposition, in which
the source pass runs on a different model from the target.

The layer is fixed, there is no sweep, and no statistic takes a per-instance maximum.

Resumable: results are written per seed and condition as they are produced, keyed by the
adapter's content hash, so retraining an adapter invalidates its cache entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

S1 = Path("studies-01-anchoring")
S3 = Path("studies-03-pretrained-routes")

# From Study 1 v0.2.0's deposited pooled7 evaluation, seed-averaged over 518 items.
EXPECT_UNPATCHED = 0.1377
TOL = 0.02


def adapter_hash(ckpt: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(ckpt.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def build_pair(it: dict) -> tuple[str, str]:
    """Target (unaided chain) and source (bridge supplied), as Studies 1-2 built them."""
    e2_type = it["template_key"][2]
    target = it["chain_prompt"]
    return target, f"{it['anchor_answer']} is the {e2_type} referred to below.\n{target}"


def last_mention_index(tokenizer, text: str, needle: str) -> int | None:
    """Index of the final token covering `needle`, by exact character offset."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    lo = text.lower().find(needle.lower())
    if lo < 0:
        return None
    hi = lo + len(needle)
    idx = [i for i, (s, e) in enumerate(enc["offset_mapping"])
           if e > s and not (e <= lo or s >= hi)]
    return idx[-1] if idx else None


def run_condition(model, tok, items, layer, condition, rng, max_new_tokens=32,
                  src_model=None, distinct_answer=False, donors_out=None):
    """Greedy-decode the chain for every item under one patching condition.

    `src_model` reads the source activation from a DIFFERENT model than the one being
    patched. That is the control for the objection that the source vector, read from the
    adapter-merged model, may carry part of the injected association rather than bridge
    identity alone: sourcing from the base model, which never learned the fact, leaves only
    bridge identity to transfer.
    """
    import torch

    from eval_runner import build_prompt, match_strict

    reader = src_model if src_model is not None else model
    block = model.model.layers[layer]
    src_block = reader.model.layers[layer]
    by_tmpl = defaultdict(list)
    for i, it in enumerate(items):
        by_tmpl["|".join(it["template_key"])].append(i)

    captured: dict = {}
    inject: dict = {}

    def hook(_mod, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if captured.get("mode") == "read" and hs.shape[1] > 1:
            captured["vec"] = hs[0, captured["pos"], :].detach().clone()
        elif captured.get("mode") == "write" and hs.shape[1] > 1:
            # prefill only: generation steps have seq_len 1 and must not be patched
            hs = hs.clone()
            hs[0, inject["pos"], :] = inject["vec"].to(hs.dtype)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return out

    handles = [block.register_forward_hook(hook)]
    if src_model is not None:
        handles.append(src_block.register_forward_hook(hook))
    preds: list[str] = []
    try:
        for i, it in enumerate(items):
            target, source = build_pair(it)
            t_text = build_prompt(tok, target)
            t_pos = last_mention_index(tok, t_text, it["e1_name"])
            if donors_out is not None:
                donors_out.append(None)      # overwritten below once a donor is chosen

            if condition != "unpatched" and t_pos is not None:
                # choose the source item: itself, or a different item of the same template
                #   bridge / wrong_bridge : source the BRIDGE ENTITY's own representation
                #                            from where it is named in the prefix
                #   self / wrong           : source the same E1 position, which tests whether
                #                            the prefix changes that position at all
                same_item = condition in ("bridge", "self")
                if same_item:
                    src_it = it
                else:
                    pool = [j for j in by_tmpl["|".join(it["template_key"])] if j != i]
                    # Optional constraint, default OFF so Studies 1-3 reproduce byte-for-byte.
                    # With `distinct_answer`, the source must have a DIFFERENT chain answer
                    # from the target. Without it, a wrong-bridge patch drawn from an item
                    # that happens to share the answer cannot change the output, and the
                    # route is scored as "survived" when the test never had a chance to run.
                    # Measured on the G4 route set, survivors had a 0.233 mean probability of
                    # such a collision against 0.144 for routes the patch did break.
                    if distinct_answer:
                        ta = str(it.get("chain_answer", "")).lower()
                        alt = [j for j in pool
                               if str(items[j].get("chain_answer", "")).lower() != ta]
                        pool = alt
                    src_it = items[rng.choice(pool)] if pool else it
                    if distinct_answer and not pool:
                        preds.append("")          # unclassifiable: no distinct-answer source
                        continue
                if donors_out is not None:
                    donors_out[-1] = src_it.get("task_id")
                _, s_raw = build_pair(src_it)
                s_text = build_prompt(tok, s_raw)
                needle = (src_it["anchor_answer"] if condition in ("bridge", "wrong_bridge")
                          else src_it["e1_name"])
                s_pos = last_mention_index(tok, s_text, needle)
                if s_pos is None:
                    preds.append("")
                    continue
                enc = tok(s_text, return_tensors="pt",
                          add_special_tokens=False).to(reader.device)
                captured.clear(); captured.update({"mode": "read", "pos": s_pos})
                with torch.no_grad():
                    reader(**enc)
                vec = captured.get("vec")
                captured.clear()
                if vec is None:
                    preds.append("")
                    continue
                inject.clear(); inject.update({"pos": t_pos, "vec": vec})
                captured["mode"] = "write"

            enc = tok(t_text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            captured.clear(); inject.clear()
            new = gen[0, enc["input_ids"].shape[1]:]
            preds.append(tok.decode(new, skip_special_tokens=True).strip())
            if (i + 1) % 100 == 0:
                print(f"      {i + 1}/{len(items)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    # Alias-extended where the dataset supplies aliases, so patch results are scored the same
    # way the screen scored the same chains. Studies 1-3's task files carry no `chain_aliases`
    # key, so `.get` yields an empty list and their behaviour is byte-identical.
    return ([float(match_strict([it["chain_answer"]] + list(it.get("chain_aliases", [])), p))
             for it, p in zip(items, preds)], preds)


def cluster_ci(vals, tmpl, n_boot, seed=0):
    by = defaultdict(list)
    for v, t in zip(vals, tmpl):
        by[t].append(v)
    keys = list(by)
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        s: list[float] = []
        for k in (rng.choice(keys) for _ in keys):
            s.extend(by[k])
        if s:
            boot.append(st.mean(s))
    boot.sort()
    return boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from model_pin import revision_for

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(S1 / "data/tasks/pooled7.json"))
    ap.add_argument("--root", default=str(S1 / "results/pooled7"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--layer", type=int, default=None,
                    help="fixed before the run; default is mid-stack, int(0.5 * n_layers)")
    ap.add_argument("--conditions", nargs="+",
                    default=["unpatched", "bridge", "wrong_bridge", "self", "wrong"])
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4. Weights are quantised; the residual stream stays fp16, so "
                         "the transplanted vector itself is full precision. Recorded in "
                         "the output -- a result under this flag is about the quantised "
                         "model.")
    ap.add_argument("--null", action="store_true",
                    help="run on the BASE model, where fact2 was never trained")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on the first N items")
    ap.add_argument("--run-fmt", default="s{seed}",
                    help="checkpoint dir under --root, e.g. armE_s{seed}")
    ap.add_argument("--expect", type=float, default=EXPECT_UNPATCHED,
                    help="reproduction guard target for the unpatched chain")
    ap.add_argument("--no-guard", action="store_true",
                    help="no published value exists to reproduce (e.g. arm D, whose chain "
                         "Study 2 never scored). Records the absence in the output rather "
                         "than faking a target -- passing a sentinel to --expect makes the "
                         "guard fail and abort, which is what it should do.")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--cache", default=str(S3 / "results/patch_cache"))
    ap.add_argument("--out", default=str(S3 / "results/patch_rescue.json"))
    args = ap.parse_args()

    raw = Path(args.data).read_bytes()
    items = json.loads(raw.decode("utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]
    tmpl = ["|".join(it["template_key"]) for it in items]
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    runs = [("base", None)] if args.null else [
        (args.run_fmt.format(seed=s),
         Path(args.root) / args.run_fmt.format(seed=s) / "checkpoint-final")
        for s in args.seeds]

    # Resolve the layer BEFORE the cache is consulted. The depth comes from the config, not
    # from loaded weights, so a fully cached re-run still knows which layer produced the
    # numbers it is reporting -- otherwise the deposited result records a null layer for its
    # own primary result, which is the one thing a reader most needs to know.
    from transformers import AutoConfig

    n_layers = AutoConfig.from_pretrained(
        args.model, revision=revision_for(args.model)).num_hidden_layers
    layer_used = args.layer if args.layer is not None else int(0.5 * n_layers)

    acc: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for tag, ckpt in runs:
        if ckpt is not None and not ckpt.exists():
            print(f"  absent: {ckpt}")
            continue
        ah = adapter_hash(ckpt) if ckpt is not None else "basemodel"
        todo = [c for c in args.conditions
                if not (cache / f"{tag}.{ah}.L{layer_used}.{c}.{len(items)}.json").exists()]
        if not todo:
            print(f"  {tag} [all cached, adapter {ah}]")
        model = None
        if todo:
            from model_load import load_model

            model = load_model(args.model, args.precision, args.load_4bit)
            if ckpt is not None:
                model = PeftModel.from_pretrained(model, str(ckpt)).merge_and_unload()
            model = model.eval()
            assert model.config.num_hidden_layers == n_layers, "config depth disagrees"
            layer = layer_used
            print(f"  {tag}  adapter {ah}  layer {layer}/{n_layers}")

        for c in args.conditions:
            cf = cache / f"{tag}.{ah}.L{layer_used}.{c}.{len(items)}.json"
            if cf.exists():
                acc[tag][c] = json.loads(cf.read_text(encoding="utf-8"))["scores"]
                print(f"    {c:<10} [cached] {st.mean(acc[tag][c]):.4f}")
                continue
            print(f"    {c:<10} scoring {len(items)} items", flush=True)
            scores, preds = run_condition(model, tok, items, layer, c, random.Random(0))
            cf.write_text(json.dumps({"scores": scores, "preds": preds}), encoding="utf-8")
            acc[tag][c] = scores
            print(f"    {c:<10} {st.mean(scores):.4f}")

        if model is not None:
            del model
            torch.cuda.empty_cache()

    if not acc:
        raise SystemExit("no runs completed")

    tags = list(acc)
    per_item = {c: [st.mean(acc[t][c][i] for t in tags if c in acc[t]) for i in range(len(items))]
                for c in args.conditions if any(c in acc[t] for t in tags)}

    # --- reproduction guard, before anything is reported ------------------------------
    if args.no_guard:
        print()
        print("  reproduction guard: SKIPPED -- no published value exists for this run. "
              "This is weaker than every other run in the programme and is recorded as such.")
    if "unpatched" in per_item and not args.null and not args.no_guard:
        got = st.mean(per_item["unpatched"])
        ok = abs(got - args.expect) <= TOL
        print()
        print(f"  reproduction guard: unpatched {got:.4f} against the deposited "
              f"{args.expect:.4f}  {'ok' if ok else 'MISMATCH'}")
        if not ok and not args.limit:
            raise SystemExit(
                f"  ABORT: the unpatched chain is {got:.4f} against {args.expect:.4f}. "
                "The scoring path disagrees with the study it inherits; fix that first.")

    print()
    print("=" * 70)
    print(f"CAUSAL RESCUE — layer {layer_used}, {len(items)} items, runs {tags}")
    print(f"  {'condition':<14}{'acc':>9}{'95% CI (template)':>24}")
    out: dict = {"layer": layer_used, "n_items": len(items), "runs": tags,
                 "null_model": args.null, "guard_skipped": args.no_guard,
                 "load_4bit": args.load_4bit, "conditions": {}}
    for c, v in per_item.items():
        lo, hi = cluster_ci(v, tmpl, args.boot)
        out["conditions"][c] = {"acc": st.mean(v), "ci": [lo, hi]}
        print(f"  {c:<14}{st.mean(v):>9.4f}   [{lo:.4f}, {hi:.4f}]")

    out["contrasts"] = {}
    print()
    print("  contrasts against the unpatched chain:")
    if "unpatched" in per_item:
        for c in per_item:
            if c == "unpatched":
                continue
            d = [x - y for x, y in zip(per_item[c], per_item["unpatched"])]
            lo, hi = cluster_ci(d, tmpl, args.boot)
            out["contrasts"][f"{c}-unpatched"] = {"delta": st.mean(d), "ci": [lo, hi]}
            print(f"    {c:<12}{st.mean(d):>+9.4f}   [{lo:+.4f}, {hi:+.4f}]"
                  f"{'   excludes 0' if lo > 0 or hi < 0 else ''}")

    out["per_item"] = {"task_id": [it["task_id"] for it in items], "template": tmpl,
                       "margin": [it["anchor_margin"] for it in items], **per_item}
    # Deposited per seed as well as averaged, so that an analysis restricted to a subset of
    # the seeds -- which the source-provenance comparison in §4.2 requires -- can be
    # reproduced from the release rather than from the local cache.
    out["per_seed"] = {t: acc[t] for t in tags}
    out["provenance"] = {
        "task_file": args.data, "task_sha256": hashlib.sha256(raw).hexdigest(),
        "model": args.model, "model_revision": revision_for(args.model),
        "precision": args.precision, "layer": layer_used, "seeds": args.seeds,
        "position": "final token of the E1 mention, inside the span shared verbatim "
                    "between the unaided and supplied-bridge prompts",
        "note": "Fixed layer, no sweep, no per-instance maximum.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
