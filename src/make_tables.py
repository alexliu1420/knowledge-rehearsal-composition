"""C1 -- every table in the Study 4 manuscript, regenerated from result files.

    python src/make_tables.py            -> results/tables/*.md and results/tables/all.json

Nothing here is computed in prose. Each table names the result files it reads. Scoring
variants (substring / word-boundary / exact / study scorer) are emitted for the primary
contrast so reviewers can see the rule does not carry the result. Needs only the standard
library, except T4 (exposure), which needs the tokenizers and is skipped with a note if
`transformers` is not installed.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

S4 = Path("."); R = S4 / "results"; OUT = R / "tables"
TC = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}
FAM = {"qwen": dict(tasks=S4 / "data/tasks/v2", pre="", seeds=[0, 1, 2], seeds6=[0, 1, 2, 3, 4, 5],
                    none_mem="preserve_none_mem100_post_X_s{s}.json", none_fin="v2_g4_split_post_X_s{s}.json",
                    none_dir="v2_g4_split_X_s{s}", g12="qwen_g12_s{s}_L12.json", nat="qwen_natact_s{s}_L12.json",
                    base="v2_g4_pre_base.json", hop2="v2_g4_hop2_base6.json", label="Qwen2.5-3B"),
       "falcon": dict(tasks=S4 / "data/tasks/falcon", pre="falcon_", seeds=[0, 1, 2], seeds6=[0, 1, 2, 3, 4, 5],
                      none_mem="falcon_preserve_none_mem100_post_X_s{s}.json", none_fin="falcon_preserve_none_final_post_X_s{s}.json",
                      none_dir="falcon_preserve_none_X_s{s}", g12="falcon_g12_s{s}_L11.json", nat="falcon_natact_s{s}_L11.json",
                      base="falcon_pre_base.json", hop2="falcon_hop2_base6.json", label="Falcon3-3B")}
j = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
norm = lambda s: " ".join(str(s).lower().split())  # noqa: E731
ALL: dict = {}


def ci(d, k=None):
    if not d:
        return "n/a", None
    if len(d) == 1:
        return f"{d[0]:+.3f} (1 seed)", d[0]
    m = st.mean(d); se = st.stdev(d) / math.sqrt(len(d)); t = TC.get(len(d) - 1, 2.0)
    return f"{m:+.3f} [{m-t*se:+.3f}, {m+t*se:+.3f}]", m


def scorers():
    from strict_match import match_strict
    def sub(golds, p): return any(g and norm(g) in norm(p) for g in golds)
    def wb(golds, p): return any(g and re.search(r"(?<!\w)" + re.escape(norm(g)) + r"(?!\w)", norm(p)) for g in golds)
    def exact(golds, p): return any(g and norm(g) == norm(p).rstrip(".") for g in golds)
    return {"study": lambda g, p: bool(match_strict(g, p)), "substring": sub, "word-boundary": wb, "exact": exact}


def load_family(f):
    c = FAM[f]
    items = {o["task_id"]: o for o in j(c["tasks"] / "g4_split_X_measure.json")["items"]}
    meas = {t for t, o in items.items() if o.get("measurable", True)}
    rs = {r["task_id"] for r in j(c["tasks"] / "preserve_route.json")["rows"]}
    As = {r["task_id"] for r in j(c["tasks"] / "preserve_atomic.json")["rows"]}
    return items, sorted(meas & rs & As), sorted(meas - rs - As), sorted((meas & As) - rs), rs, As


def post(f, cond, ck, s):
    c = FAM[f]
    if cond == "none":
        p = R / (c["none_mem"] if ck == "mem100" else c["none_fin"]).format(s=s)
    else:
        p = R / f"{c['pre']}preserve_{cond}_{ck}_post_X_s{s}.json"
        if not p.exists() and ck == "final":
            p = R / f"{c['pre']}preserve_{cond}_post_X_s{s}.json"
    return {r["task_id"]: r for r in j(p)["per_item"]} if p.exists() else None


def held(rec, it, score):
    g = [it["chain_answer"]] + list(it.get("chain_aliases", []))
    return (sum(score(g, p) for p in rec["chain_preds_bio"][1:]) + rec["chain_access_neutral"] * 6) / 11


def emit(rec, it):
    g = [it["chain_answer"]] + list(it.get("chain_aliases", []))
    return st.mean(norm(it["anchor_answer"]) in norm(p) and not norm(it["chain_answer"]) in norm(p) for p in rec["chain_preds_bio"][1:])


def md(rows, header):
    return "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n" + \
        "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows) + "\n"


def t1_preservation(SC):
    lines = ["# T1 Preservation: route - atomic rehearsal, BOTH-replayed set\n",
             "Files: `{pre}preserve_{cond}_{ck}_post_X_s{seed}.json`. Held-out access = (5 non-canonical "
             "phrasings, biomedical persona + 6 phrasings, neutral persona) / 11. Seed-level paired t. "
             "mem100 = first epoch at which a 40-item probe of the injected facts reaches >= 0.99 strict recall; "
             "full-set recall over all injected routes at that checkpoint is reported in the last column.\n"]
    rows = []
    for f in FAM:
        items, both, *_ = load_family(f)
        for ck in ("mem100", "final"):
            for seeds, tag in ((FAM[f]["seeds"], "3 seeds"), (FAM[f]["seeds6"], "6 seeds")):
                d = []; rr = []; ra = []
                for s in seeds:
                    Pr, Pa = post(f, "route", ck, s), post(f, "atomic", ck, s)
                    if Pr is None or Pa is None:
                        continue
                    d.append(st.mean(held(Pr[i], items[i], SC["study"]) for i in both) - st.mean(held(Pa[i], items[i], SC["study"]) for i in both))
                    inj = [i for i in Pr if items[i]["bridge_injected"]]
                    rr.append(st.mean(float(Pr[i]["injected_recall"]) for i in inj)); ra.append(st.mean(float(Pa[i]["injected_recall"]) for i in inj))
                if len(d) == len(seeds):
                    s_, m = ci(d)
                    rows.append([FAM[f]["label"], ck, tag, len(both), s_, " / ".join(f"{x:+.3f}" for x in d),
                                 f"{st.mean(rr):.3f} / {st.mean(ra):.3f} (min {min(rr):.3f} / {min(ra):.3f})"])
                    ALL[f"T1|{f}|{ck}|{tag}"] = {"mean": m, "per_seed": d, "n_routes": len(both), "recall_route": rr, "recall_atomic": ra}
    lines.append(md(rows, ["family", "checkpoint", "seeds", "n routes", "route - atomic [95% CI]", "per seed", "full-set injected recall route / atomic"]))
    # scoring variants on the primary (mem100, 3 seeds)
    lines.append("\n## Scoring-rule robustness (mem100, seeds 0-2)\n")
    rows = []
    for f in FAM:
        items, both, *_ = load_family(f)
        for name, sc in SC.items():
            d = []
            for s in FAM[f]["seeds"]:
                Pr, Pa = post(f, "route", "mem100", s), post(f, "atomic", "mem100", s)
                d.append(st.mean(held(Pr[i], items[i], sc) for i in both) - st.mean(held(Pa[i], items[i], sc) for i in both))
            s_, m = ci(d); rows.append([FAM[f]["label"], name, s_])
            ALL[f"T1scoring|{f}|{name}"] = m
    lines.append(md(rows, ["family", "scorer", "route - atomic [95% CI]"]))
    return "".join(lines)


def t2_none(SC):
    lines = ["# T2 No-rehearsal baseline and bridge emission\n", "Files: no-replay adapters (`v2_g4_split_X_s*`, `falcon_preserve_none_X_s*`) and the route/atomic adapters. Bridge emission = fraction of the five non-canonical domain-persona predictions whose wrong answer contains the bridge entity (the neutral-persona prompts are stored as booleans and are not classified). Contrasts against none pair seeds 0-2; route/atomic means use all available seeds.\n"]
    rows = []
    for f in FAM:
        items, both, ne, *_ = load_family(f)
        for ck in ("mem100", "final"):
            V = {c: [] for c in ("none", "route", "atomic")}; E = {c: [] for c in V}; N = {c: [] for c in V}
            for s in FAM[f]["seeds6"]:
                for c in V:
                    P = post(f, c, ck, s)
                    if P is None:
                        continue
                    V[c].append(st.mean(held(P[i], items[i], SC["study"]) for i in both))
                    E[c].append(st.mean(emit(P[i], items[i]) for i in both))
                    N[c].append(st.mean(held(P[i], items[i], SC["study"]) for i in ne))
            if not V["none"]:
                continue
            k = len(V["none"])   # none exists for seeds 0-2 only; contrasts against it pair those seeds
            ra, _ = ci([a - b for a, b in zip(V["route"][:k], V["none"])]); aa, _ = ci([a - b for a, b in zip(V["atomic"][:k], V["none"])])
            rn, _ = ci([a - b for a, b in zip(N["route"][:k], N["none"])])
            rows.append([FAM[f]["label"], f"{ck} (none {k} seeds; route/atomic {len(V['route'])} seeds)", f"{st.mean(V['none']):.3f}", f"{st.mean(V['route']):.3f}", f"{st.mean(V['atomic']):.3f}",
                         ra, aa, f"{st.mean(E['none']):.3f} / {st.mean(E['route']):.3f} / {st.mean(E['atomic']):.3f} (atomic per seed " + " ".join(f"{x:.3f}" for x in E["atomic"]) + ")",
                         f"{st.mean(N['none']):.3f} / {st.mean(N['route']):.3f} / {st.mean(N['atomic']):.3f} ; route-none {rn}"])
            ALL[f"T2|{f}|{ck}"] = {"none": V["none"], "route": V["route"], "atomic": V["atomic"], "emission": {c: E[c] for c in E}}
    lines.append(md(rows, ["family", "ckpt", "none", "route", "atomic", "route - none", "atomic - none",
                           "bridge emission none/route/atomic", "never-rehearsed routes none/route/atomic"]))
    return "".join(lines)


def t3_subgroups(SC):
    lines = ["# T3 Subgroups: never-rehearsed routes, own-atoms difference-in-differences, baseline-accessible atoms\n"]
    rows = []
    for f in FAM:
        items, both, ne, ao, rs, As = load_family(f)
        B = {x["task_id"]: x for x in j(R / FAM[f]["base"])["per_item"]}
        ceil = [i for i in both if float(B[i]["bridge_access_k"]) >= 1.0]
        for seeds, tag in ((FAM[f]["seeds"], "3"), (FAM[f]["seeds6"], "6")):
            dne, did, dce = [], [], []
            for s in seeds:
                Pr, Pa = post(f, "route", "mem100", s), post(f, "atomic", "mem100", s)
                if Pr is None or Pa is None:
                    continue
                m = lambda P, ids: st.mean(held(P[i], items[i], SC["study"]) for i in ids)  # noqa: E731
                dne.append(m(Pr, ne) - m(Pa, ne)); did.append((m(Pr, ao) - m(Pa, ao)) - (m(Pr, ne) - m(Pa, ne)))
                if len(ceil) >= 10:
                    dce.append(m(Pr, ceil) - m(Pa, ceil))
            if len(dne) == len(seeds):
                rows.append([FAM[f]["label"], tag, f"n={len(ne)}: {ci(dne)[0]}", f"n={len(ao)}: {ci(did)[0]}",
                             f"n={len(ceil)}: {ci(dce)[0] if dce else 'too few'}"])
                ALL[f"T3|{f}|{tag}"] = {"never": dne, "did": did, "ceiling": dce}
    lines.append(md(rows, ["family", "seeds", "never-rehearsed: route - atomic", "own-atoms DiD", "hop-1 at ceiling at baseline: route - atomic"]))
    return "".join(lines)


def t4_exposure():
    lines = ["# T4 Exposure accounting per epoch (injected rows identical across conditions)\n",
             "Rows and token counts are for the rehearsal rows only; content tokens are matched by design; processed = chat-templated prompt + answer; supervised = answer tokens (answer-only loss). Steps/epoch counts ALL training rows, injection plus rehearsal (Qwen 390 injection rows, Falcon 831): floor(rows/8), because the trainer accumulates 8 rows per optimizer step and discards the incomplete last group of each epoch.\n"]
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None   # main() keeps the existing T4_exposure.md and its all.json entries
    from eval_runner import SYSTEM_PROMPT
    from model_pin import revision_for
    rows = []
    for f, model in (("qwen", "Qwen/Qwen2.5-3B-Instruct"), ("falcon", "tiiuae/Falcon3-3B-Instruct")):
        c = FAM[f]; tok = AutoTokenizer.from_pretrained(model, revision=revision_for(model))
        def cost(q, a):
            p = tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            pi = len(tok(p, add_special_tokens=False)["input_ids"]); ai = len(tok(a + tok.eos_token, add_special_tokens=False)["input_ids"])
            return pi + ai, ai, len(tok(q + " " + a).input_ids)
        TR = j(c["tasks"] / "g4_split_X_train.json"); tri = TR["items"] if isinstance(TR, dict) else TR
        ninj = sum(1 for o in tri if o.get("fact2_prompt"))
        for cond in ("route", "atomic"):
            rws = j(c["tasks"] / f"preserve_{cond}.json")["rows"]; cs = [cost(r["prompt"], r["answer"]) for r in rws]
            ep = [j(R / f"{c['pre']}preserve_{cond}_X_s{s}/eval_final.json").get("mem100_epoch") for s in c["seeds6"] if (R / f"{c['pre']}preserve_{cond}_X_s{s}/eval_final.json").exists()]
            spe = (len(rws) + ninj) // 8   # the trainer discards the incomplete last accumulation group of each epoch
            rows.append([c["label"], cond, len(rws), sum(x[2] for x in cs), sum(x[0] for x in cs), sum(x[1] for x in cs), spe, f"{st.mean(ep):.1f} ({len(ep)} seeds)", round(st.mean(ep) * spe)])
            ALL[f"T4|{f}|{cond}"] = {"rows": len(rws), "content": sum(x[2] for x in cs), "processed": sum(x[0] for x in cs), "supervised": sum(x[1] for x in cs), "steps_per_epoch": spe, "mem100_epochs": ep}
    lines.append(md(rows, ["family", "condition", "rows", "content tok", "processed tok", "supervised tok", "optimizer steps/epoch (injection + rehearsal)", "mean mem100 epoch", "steps to mem100"]))
    return "".join(lines)


def t5_rescue():
    lines = ["# T5 Failure-matched rescue (G12) and natural-activation transfer (G14)\n",
             "Files: `{qwen,falcon}_g12_s*_L*.json`, `{qwen,falcon}_natact_s*_L*.json`. Rates on failing held-out prompts whose unpatched run reproduced the stored miss.\n"]
    rows = []
    for f in FAM:
        c = FAM[f]; g = {}; n = {}
        for s in c["seeds"]:
            p = R / c["g12"].format(s=s)
            if p.exists():
                S = j(p)["summary"]
                for k, v in (("own bridge (explicit)", S["fail"]["bridge"]["correct"]), ("self control", S["fail"]["self"]["correct"]),
                             ("wrong bridge -> donor answer", S["fail"]["wrong_bridge"]["donor_answer"]), ("bridge-emission failures rescued", S["fail_bridge_emission"]["bridge"]["correct"])):
                    g.setdefault(k, []).append(v)
            p = R / c["nat"].format(s=s)
            if p.exists():
                S = j(p)["summary"]["fail"]
                for k, v in (("base state, subject position", S["base"]["correct"]), ("route-adapter state, subject position", S["route"]["correct"]),
                             ("base state, last token", S["base_last"]["correct"]), ("base state, wrong subject", S["wrongsub"]["correct"])):
                    n.setdefault(k, []).append(v)
        for k, v in list(g.items()) + list(n.items()):
            s_, m = ci(v); rows.append([c["label"], k, s_, " / ".join(f"{x:.3f}" for x in v)])
            ALL[f"T5|{f}|{k}"] = v
    lines.append(md(rows, ["family", "quantity", "seed mean [95% CI]", "per seed"]))
    return "".join(lines)


def t6_transfer():
    lines = ["# T6 Transfer experiment: rehearse half the routes, measure the entity-disjoint other half; seeds 0-2, every condition" + "\n",
             "Files: `results/g15_transfer.json` (Qwen), `results/g15_transfer_falcon.json` (Falcon), from `analyze_transfer.py`. L = never rehearsed in any condition; Rall = rehearsed in every condition. Every mean and contrast uses the declared seeds 0-2, paired by seed, t(2); sign = seeds with the contrast > 0. Seeds 3-5 of bridgectx and coherent_answeronly (post hoc extension, G18) are reported in T12 only." + "\n"]
    for fam_label, fname, cs in (("Qwen2.5-3B (340 rehearsed / 320 never-rehearsed)", "g15_transfer.json", ("none", "route", "atomic", "coherent", "bridgectx", "coherent_bmask", "coherent_answeronly")),
                                 ("Falcon3-3B (571 rehearsed / 541 never-rehearsed)", "g15_transfer_falcon.json", ("none", "route", "atomic", "bridgectx", "coherent", "coherent_bmask", "coherent_answeronly"))):
        p = R / fname
        if not p.exists():
            lines.append("\n" + "## " + fam_label + ": (not yet run)" + "\n"); continue
        d = j(p); fam = fam_label.split()[0]
        for ck in ("mem100", "final"):
            seeds = d[ck]["seeds"]

            def keep(key, v, seeds=seeds):   # restrict every list to the declared seeds 0-2, in seed order
                ss = seeds[key.split("|")[0]]
                assert len(ss) == len(v), key
                return [x for s_, x in zip(ss, v) if s_ in (0, 1, 2)]
            pc = {k: keep(k, v) for k, v in d[ck]["per_condition"].items()}
            assert all(len(v) == 3 for v in pc.values()), "T6 expects seeds 0-2 for every condition"
            lines.append("\n" + "## " + fam_label + ", " + ck + "\n")
            rows = []
            for name in ("L", "Rall"):
                for c in cs:
                    if f"{c}|{name}|held" not in pc:
                        continue
                    h = pc[f"{c}|{name}|held"]; e = pc[f"{c}|{name}|emit"]; cn = pc[f"{c}|{name}|canon"]
                    rows.append([name, c, f"{st.mean(h):.3f}", " / ".join(f"{x:.3f}" for x in h), f"{st.mean(cn):.3f}",
                                 f"{st.mean(e):.3f}", " / ".join(f"{x:.3f}" for x in e)])
            lines.append(md(rows, ["set", "condition", "held-out", "per seed", "canonical", "bridge emission", "per seed"]))
            rows = []
            for label, x, y, name, m in (("route - atomic", "route", "atomic", "L", "held"), ("route - none", "route", "none", "L", "held"),
                                         ("bridgectx - none", "bridgectx", "none", "L", "held"), ("coherent - none", "coherent", "none", "L", "held"),
                                         ("atomic - none", "atomic", "none", "L", "held"), ("route - bridgectx", "route", "bridgectx", "L", "held"),
                                         ("route - coherent", "route", "coherent", "L", "held"), ("bridgectx - coherent", "bridgectx", "coherent", "L", "held"), ("bridgectx - coherent", "bridgectx", "coherent", "Rall", "held"),
                                         ("route - atomic", "route", "atomic", "Rall", "held"), ("bridgectx - atomic", "bridgectx", "atomic", "Rall", "held"),
                                         ("coherent - atomic", "coherent", "atomic", "Rall", "held"),
                                         ("emission: atomic - none", "atomic", "none", "Rall", "emit"), ("emission: coherent - none", "coherent", "none", "Rall", "emit"),
                                         ("emission: bridgectx - none", "bridgectx", "none", "Rall", "emit"), ("emission: bridgectx - atomic", "bridgectx", "atomic", "Rall", "emit"),
                                         ("G16 bridge masked - coherent", "coherent_bmask", "coherent", "Rall", "held"),
                                         ("G16 bridge masked - coherent", "coherent_bmask", "coherent", "L", "held"),
                                         ("G16 emission: bridge masked - coherent", "coherent_bmask", "coherent", "Rall", "emit"),
                                         ("G16 bridgectx - bridge masked", "bridgectx", "coherent_bmask", "Rall", "held"),
                                         ("G16 hop-1: bridge masked - coherent", "coherent_bmask", "coherent", "Rall", "hop1"),
                                         ("G17 answer-only - bridge masked", "coherent_answeronly", "coherent_bmask", "Rall", "held"),
                                         ("G17 bridgectx - answer-only", "bridgectx", "coherent_answeronly", "Rall", "held"),
                                         ("G17 answer-only - bridge masked", "coherent_answeronly", "coherent_bmask", "L", "held"),
                                         ("G17 bridgectx - answer-only", "bridgectx", "coherent_answeronly", "L", "held")):
                if f"{x}|{name}|{m}" not in pc or f"{y}|{name}|{m}" not in pc:
                    continue
                dd = [u - v for u, v in zip(pc[f"{x}|{name}|{m}"], pc[f"{y}|{name}|{m}"])]
                s_, mm = ci(dd); rows.append([name, label, s_, f"{sum(u > 0 for u in dd)}/{len(dd)}"])
                ALL[f"T6|{fam}|{ck}|{name}|{label}"] = dd
            lines.append(md(rows, ["set", "contrast", "mean [95% CI]", "sign"]))
    return "".join(lines)

def t7_shortcut():
    NL = chr(10)
    lines = ["# T7 Shortcut test: bridge-dependence on never-rehearsed (L) and rehearsed (Rall) routes, Qwen L12, seed-0 mem100 adapters" + NL,
             "Files: `bridge_causal_L12.json` (base), `shortcut_{none,route,bridgectx,atomic}_s0_L12.json`. identity = acc(own bridge) - acc(wrong bridge) on routes answered unpatched; strict follow = wrong-bridge output equals the actual donor's answer." + NL]
    from strict_match import match_strict
    sp_p = S4 / "data/tasks/v2_transfer/transfer_split.json"
    if not sp_p.exists() or not (R / "shortcut_route_s0_L12.json").exists():
        return "".join(lines) + "(not yet run)" + NL
    sp = j(sp_p); sets = {"L": set(sp["L"]), "Rall": set(sp["rehearsed_in_all"])}
    donor = j(R / "qwen_causal_donors_seed0.json"); routes = {o["task_id"]: o for o in j(S4 / "data/tasks/g4_routes_v2.json")["items"]}
    files = {"base": "bridge_causal_L12.json", "none": "shortcut_none_s0_L12.json", "route": "shortcut_route_s0_L12.json",
             "bridgectx": "shortcut_bridgectx_s0_L12.json", "atomic": "shortcut_atomic_s0_L12.json"}
    rows = []
    for c, f in files.items():
        rws = {r["task_id"]: r for r in j(R / f)["per_item"]}
        for name, ids in sets.items():
            allr = [i for i in ids if i in rws]; sel = [i for i in allr if rws[i]["unpatched"]]
            b = st.mean(float(bool(rws[i]["bridge"])) for i in sel); w = st.mean(float(bool(rws[i]["wrong_bridge"])) for i in sel)
            sf = st.mean(bool(match_strict([routes[donor[i]]["chain_answer"]] + list(routes[donor[i]].get("chain_aliases", [])), rws[i]["wrong_bridge_pred"])) for i in sel if i in donor)
            base_id = ALL.get(f"T7|base|{name}", {}).get("identity", b - w)
            rows.append([c, name, len(sel), f"{len(sel)/len(allr):.3f}", f"{b:.3f}", f"{w:.3f}", f"{b-w:+.3f}", f"{(b-w)-base_id:+.3f}", f"{sf:.3f}"])
            ALL[f"T7|{c}|{name}"] = {"identity": b - w, "strict_follow": sf, "n": len(sel), "delta_vs_base": (b - w) - base_id}
    lines.append(md(rows, ["condition", "set", "n answered", "answered unpatched", "own bridge", "wrong bridge", "identity", "identity - base", "strict follow"]))
    return "".join(lines)


def t8_identity():
    NL = chr(10)
    lines = ["# T8 Causal screens on successful computations: bridge-representation identity by layer (supplementary S1)" + NL,
             "Files: Qwen `bridge_causal_L{6,12,24}.json` (base) and `mech_{route,atomic}_s0_L{6,12,24}.json`; Falcon `falcon_causal_{base,route_mem100,atomic_mem100}[_s{1,2}]_L{7,11,15}.json`. identity = acc(own-bridge transplant) - acc(wrong-bridge transplant) at the subject position, on comparison-set routes the adapter answers unpatched. Falcon: three seeds, t(2); Qwen: seed 0 only." + NL]
    from strict_match import match_strict
    def prof(path, ids):
        rows = {r["task_id"]: r for r in j(path)["per_item"]}
        sel = [i for i in ids if i in rows and rows[i]["unpatched"]]
        if len(sel) < 20:
            return None
        return st.mean(float(bool(rows[i]["bridge"])) - float(bool(rows[i]["wrong_bridge"])) for i in sel), len(sel)
    for f, layers, files in (("qwen", (6, 12, 24), lambda c, L, s: R / ("bridge_causal_L%d.json" % L if c == "base" else "mech_%s_s0_L%d.json" % (c, L))),
                             ("falcon", (7, 11, 15), lambda c, L, s: R / ("falcon_causal_base_L%d.json" % L if c == "base" else "falcon_causal_%s_mem100%s_L%d.json" % (c, "" if s == 0 else "_s%d" % s, L)))):
        items, both, *_ = load_family(f)
        lines.append(NL + "## " + FAM[f]["label"] + NL)
        rows = []
        for L in layers:
            b = prof(files("base", L, 0), both)
            rr = [prof(files("route", L, s), both) for s in (range(3) if f == "falcon" else [0])]
            aa = [prof(files("atomic", L, s), both) for s in (range(3) if f == "falcon" else [0])]
            rr = [x for x in rr if x]; aa = [x for x in aa if x]
            if not b or not rr or not aa:
                continue
            d = [r[0] - a[0] for r, a in zip(rr, aa)]
            s_, m = ci(d) if len(d) > 1 else (f"{d[0]:+.3f} (1 seed)", d[0])
            rows.append([L, f"{b[0]:+.3f}", " / ".join(f"{x[0]:+.3f}" for x in rr), " / ".join(f"{x[0]:+.3f}" for x in aa), s_])
            ALL[f"T8|{f}|L{L}"] = {"base": b[0], "route": [x[0] for x in rr], "atomic": [x[0] for x in aa]}
        lines.append(md(rows, ["layer", "base", "route rehearsal (per seed)", "atomic rehearsal (per seed)", "route - atomic"]))
    return "".join(lines)


def t9_collapse():
    NL = chr(10)
    lines = ["# T9 Every atomic-rehearsal adapter: held-out compositional access on its comparison set, both checkpoints" + NL,
             "Collapse is defined as held-out access below 0.7 on the comparison set; the other atomic adapters lie at 0.705-0.928, so the count depends on a threshold in a narrow gap. Files: the atomic post files of T1 (BOTH set) and T6 (rehearsed-in-all set); route adapters listed for comparison." + NL]
    from strict_match import match_strict
    def held(rec, it):
        g = [it["chain_answer"]] + list(it.get("chain_aliases", []))
        return (sum(bool(match_strict(g, p)) for p in rec["chain_preds_bio"][1:]) + rec["chain_access_neutral"] * 6) / 11
    rows = []; n_at = {"mem100": [0, 0], "final": [0, 0]}; n_rt = {"mem100": [0, 0], "final": [0, 0]}
    for f, pre, seeds, split in (("qwen", "preserve", range(6), None), ("falcon", "falcon_preserve", range(6), None),
                                 ("qwen", "transfer", range(3), S4 / "data/tasks/v2_transfer/transfer_split.json"),
                                 ("falcon", "falcon_transfer", range(3), S4 / "data/tasks/falcon_transfer/transfer_split.json")):
        items, both, *_ = load_family(f)
        ids = both if split is None else j(split)["rehearsed_in_all"]
        exp = "preservation" if split is None else "transfer"
        for ck in ("mem100", "final"):
            for cond, counter in (("atomic", n_at), ("route", n_rt)):
                for s in seeds:
                    p = R / f"{pre}_{cond}_{ck}_post_X_s{s}.json"
                    if not p.exists():
                        p = R / f"{pre}_{cond}_post_X_s{s}.json"
                    if not p.exists():
                        continue
                    P = {r["task_id"]: r for r in j(p)["per_item"]}
                    v = st.mean(held(P[i], items[i]) for i in ids if i in P)
                    counter[ck][1] += 1; counter[ck][0] += v < 0.7
                    d = R / f"{pre}_{cond}_X_s{s}"
                    ev = j(d / f"eval_{ck}.json") if (d / f"eval_{ck}.json").exists() else {}
                    ef = j(d / "eval_final.json") if (d / "eval_final.json").exists() else {}
                    hist = j(d / "history.json") if (d / "history.json").exists() else []
                    dppl = ev.get("retention_delta_ppl"); ep = ef.get("mem100_epoch"); loss = hist[-1]["loss"] if hist else None
                    rows.append([FAM[f]["label"], exp, cond, ck, s, f"{v:.3f}", "collapse" if v < 0.7 else "",
                                 "" if dppl is None else f"{dppl:+.2f}", "" if ep is None else ep, "" if loss is None else f"{loss:.3f}"])
                    ALL[f"T9|{f}|{exp}|{cond}|{ck}|s{s}"] = {"held": v, "retention_delta_ppl": dppl, "mem100_epoch": ep, "final_loss": loss}
    lines.append(md(rows, ["family", "experiment", "condition", "checkpoint", "seed", "held-out access", "", "retention delta-ppl at checkpoint", "mem100 epoch", "loss, epoch 12"]))
    lines.append(NL + "Atomic adapters below 0.7: " + ", ".join(f"{ck} {n_at[ck][0]} of {n_at[ck][1]}" for ck in n_at) +
                 ". Route adapters below 0.7: " + ", ".join(f"{ck} {n_rt[ck][0]} of {n_rt[ck][1]}" for ck in n_rt) + "." + NL)
    return "".join(lines)


def t10_recall():
    NL = chr(10)
    lines = ["# T10 Full-set injected-fact recall per adapter, both checkpoints" + NL,
             "Strict recall of the injected fact over every injected route in the measured half (not the 40-item training-time probe that defines mem100). Files: every post file used in T1, T2 and T6." + NL]
    rows = []
    for f, exps in (("qwen", (("preservation", "preserve", ("none", "route", "atomic"), range(6)), ("transfer", "transfer", ("route", "atomic", "coherent", "bridgectx", "coherent_bmask", "coherent_answeronly"), range(6)))),
                    ("falcon", (("preservation", "falcon_preserve", ("none", "route", "atomic"), range(6)), ("transfer", "falcon_transfer", ("route", "atomic", "bridgectx", "coherent", "coherent_bmask", "coherent_answeronly"), range(6))))):
        items, *_ = load_family(f)
        for exp, pre, conds, seeds in exps:
            for cond in conds:
                for ck in ("mem100", "final"):
                    vals = []
                    for s in seeds:
                        if cond == "none":
                            if exp != "preservation":
                                continue
                            P = post(f, "none", ck, s)
                        else:
                            q = R / f"{pre}_{cond}_{ck}_post_X_s{s}.json"
                            if not q.exists():
                                q = R / f"{pre}_{cond}_post_X_s{s}.json"
                            P = {r["task_id"]: r for r in j(q)["per_item"]} if q.exists() else None
                        if P is None:
                            continue
                        inj = [i for i in P if items[i]["bridge_injected"]]
                        vals.append(st.mean(float(P[i]["injected_recall"]) for i in inj))
                    if vals:
                        rows.append([FAM[f]["label"], exp, cond, ck, " / ".join(f"{x:.3f}" for x in vals), f"{st.mean(vals):.3f}", f"{min(vals):.3f}"])
                        ALL[f"T10|{f}|{exp}|{cond}|{ck}"] = vals
    lines.append(md(rows, ["family", "experiment", "condition", "checkpoint", "per seed", "mean", "min"]))
    return "".join(lines)


def t11_hop2(SC):
    NL = chr(10)
    lines = ["# T11 Sensitivity: primary contrast restricted by atomic pairing and by baseline second-hop knowledge (exploratory, not declared)" + NL,
             "The route builder does not require the second hop to be known when asked directly. `hop2_known` = fraction of six direct second-hop phrasings the base model answers (files `v2_g4_hop2_base6.json`, `falcon_hop2_base6.json`). Token matching trims atomic rehearsal rows, not routes, so some routes have only one of their two facts rehearsed (`preserve_atomic.json`, field `atom`). route - atomic at mem100 on the BOTH set, six seeds, t(5)." + NL]
    rows = []
    for f, hf in (("qwen", "v2_g4_hop2_base6.json"), ("falcon", "falcon_hop2_base6.json")):
        items, both, *_ = load_family(f)
        H = {x["task_id"]: x for x in j(R / hf)["per_item"]}
        from collections import defaultdict
        atoms = defaultdict(set)
        for r in j(FAM[f]["tasks"] / "preserve_atomic.json")["rows"]:
            atoms[r["task_id"]].add(r["atom"])
        paired = [i for i in both if atoms[i] == {"hop1", "hop2"}]
        subsets = (("all comparison routes", both),
                   ("both facts rehearsed under atomic", paired),
                   ("both facts rehearsed and second hop known on all six", [i for i in paired if float(H[i]["hop2_known"]) >= 1.0]),
                   ("second hop known on all six phrasings", [i for i in both if float(H[i]["hop2_known"]) >= 1.0]),
                   ("second hop known on at least one phrasing", [i for i in both if float(H[i]["hop2_known"]) > 0]),
                   ("second hop known on none", [i for i in both if float(H[i]["hop2_known"]) == 0]))
        for lab, ids in subsets:
            d = []
            for s in FAM[f]["seeds6"]:
                Pr, Pa = post(f, "route", "mem100", s), post(f, "atomic", "mem100", s)
                d.append(st.mean(held(Pr[i], items[i], SC["study"]) for i in ids) - st.mean(held(Pa[i], items[i], SC["study"]) for i in ids))
            s_, m = ci(d) if len(ids) >= 10 else ("too few routes", None)
            rows.append([FAM[f]["label"], lab, len(ids), s_])
            ALL[f"T11|{f}|{lab}"] = {"n": len(ids), "per_seed": d}
    lines.append(md(rows, ["family", "subset", "n routes", "route - atomic, 6 seeds [95% CI]"]))
    return "".join(lines)


def t12_decisions():
    NL = chr(10)
    lines = ["# T12 Decision rules of the format ablations (G16 bridge-token mask, G17 answer-only, G18 seeds 3-5)" + NL,
             "Rehearsed routes (Rall) unless stated; acquisition-matched checkpoint unless stated. f = (answer-only - masked) / (bridge-as-context - masked). "
             "Per-family intervals pair by seed (t(k-1)). Seeds 0-2 are the declared G17 cohort; seeds 3-5 were added after it was seen (G18, post hoc) and are shown alone and combined. The pooled row pairs seeds across both families and was not declared (exploratory)." + NL]
    rows = []; pooled = {}
    for fam, fname in (("Qwen2.5-3B", "g15_transfer.json"), ("Falcon3-3B", "g15_transfer_falcon.json")):
        p = R / fname
        if not p.exists():
            continue
        d = j(p)
        for ck in ("mem100", "final"):
            pc, seeds = d[ck]["per_condition"], d[ck]["seeds"]
            for name in ("Rall", "L"):
                k = lambda c: dict(zip(seeds.get(c, []), pc.get(f"{c}|{name}|held", [])))  # noqa: E731  seed -> value
                ao, bm, bc = k("coherent_answeronly"), k("coherent_bmask"), k("bridgectx")
                if not (ao and bm and bc):
                    continue
                d3 = (0, 1, 2)
                f = (st.mean(ao[s] for s in d3) - st.mean(bm[s] for s in d3)) / (st.mean(bc[s] for s in d3) - st.mean(bm[s] for s in d3))
                common = sorted(set(ao) & set(bc))
                cohorts = [((0, 1, 2), "3, seeds 0-2 (declared, G17)")]
                if set(common) >= {3, 4, 5}:
                    cohorts += [((3, 4, 5), "3, seeds 3-5 alone (G18)"), (tuple(common), f"{len(common)}, seeds 0-5 (post hoc extension, G18)")]
                for ss, label in cohorts:
                    dd = [bc[s] - ao[s] for s in ss]
                    s_, m = ci(dd)
                    rows.append([fam, ck, name, f"{f:+.2f}", label, s_, f"{sum(x > 0 for x in dd)}/{len(dd)}"])
                    ALL[f"T12|{fam}|{ck}|{name}|{'-'.join(map(str, ss))}"] = {"f": f, "bridgectx_minus_answeronly": dd}
                    if ss == (0, 1, 2):
                        pooled.setdefault((ck, name), []).extend(dd)
    lines.append(md(rows, ["family", "checkpoint", "set", "f (seeds 0-2)", "seed pairs", "bridge-as-context - answer-only [95% CI]", "sign"]))
    rows = []
    for (ck, name), dd in pooled.items():
        s_, m = ci(dd); rows.append([ck, name, len(dd), s_, f"{sum(x > 0 for x in dd)}/{len(dd)}"])
        ALL[f"T12|pooled|{ck}|{name}"] = dd
    lines.append(NL + "## Pooled across families (exploratory)" + NL)
    lines.append(md(rows, ["checkpoint", "set", "seed pairs", "bridge-as-context - answer-only [95% CI]", "sign"]))
    return "".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    SC = scorers()
    prior = j(OUT / "all.json") if (OUT / "all.json").exists() else {}
    for name, fn in (("T1_preservation", lambda: t1_preservation(SC)), ("T2_no_rehearsal", lambda: t2_none(SC)),
                     ("T3_subgroups", lambda: t3_subgroups(SC)), ("T4_exposure", t4_exposure), ("T5_rescue", t5_rescue), ("T6_transfer", t6_transfer), ("T7_shortcut", t7_shortcut), ("T8_identity", t8_identity), ("T9_collapse", t9_collapse),
                     ("T10_recall", t10_recall), ("T11_hop2_sensitivity", lambda: t11_hop2(SC)), ("T12_decisions", t12_decisions)):
        txt = fn()
        if txt is None:
            # T4 needs the tokenizers. Without them the shipped table and its all.json entries are
            # kept exactly as they are rather than replaced by a stub.
            key = name.split("_")[0] + "|"
            ALL.update({k: v for k, v in prior.items() if k.startswith(key)})
            print(f"  {name}: kept the existing file (`transformers` not installed; it is needed to regenerate this table)")
            continue
        (OUT / f"{name}.md").write_text(txt, encoding="utf-8"); print(txt)
    (OUT / "all.json").write_text(json.dumps(ALL, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
