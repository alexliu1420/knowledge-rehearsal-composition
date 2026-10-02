# What You Rehearse Is What You Keep

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22970449.svg)](https://doi.org/10.5281/zenodo.22970449)

Code, data, results and manuscript for **"What You Rehearse Is What You Keep: Rehearsal Format
Decides Whether Known Facts Stay Composable."**

When a language model is updated with new facts, the facts it already knew can stay recallable
while their use in composition is lost. Rehearsal is the standard protection, and the usual
unit of rehearsal is the individual fact. This study asks what rehearsal must supervise to
preserve a model's ability to compose facts it already knows.

**Manuscript: [`paper/rehearsal_unit.pdf`](paper/rehearsal_unit.pdf)** (Markdown source alongside it)

## Findings

Two-hop routes that Qwen2.5-3B-Instruct and Falcon3-3B-Instruct compose robustly under twelve
phrasings before any update; an unrelated fact about each route's intermediate entity injected
by LoRA fine-tuning; rehearsal conditions matched on content tokens and on acquisition of the
injected facts; the outcome measured on phrasings and a persona the model never saw.

**An unrelated update costs 18–34 points of compositional access while the facts survive.**
Rehearsing the composition in one phrasing restores it to 0.99. Rehearsing the constituent facts
as answers, with about twice the supervised tokens, leaves it near the no-rehearsal level
(composition − facts: +0.166 [+0.032, +0.301] on Qwen, +0.204 [+0.097, +0.311] on Falcon,
six seeds each; the atomic − none intervals span zero at three seeds).

**Fact rehearsal induces a specific failure that sometimes collapses composition.** The model
answers the composite question with the intermediate entity on 8–13% of held-out phrasings
(about 1% without rehearsal). Defining collapse as held-out access below 0.7, 3 of the 18
fact-rehearsal adapters collapse at each checkpoint, against 0 of 18 composition-rehearsal
adapters. Collapsed adapters have loss curves and memorisation epochs like the others, so the
collapse does not show in training diagnostics.

**On the failing phrasings the second step is usable.** Patching the intermediate entity's
representation into the failing computation restores the answer on 86% (Qwen) and 61% (Falcon)
of failures, against 4–10% for a context control.

**Rehearsal format decides the outcome.** Rehearsing 340 routes protects 320 never-rehearsed
routes that share no subject or intermediate entity with them (+0.129 [+0.036, +0.222]); on
Falcon, +0.269 [+0.078, +0.460]. Held-out composition by rehearsal format, Qwen / Falcon:

| rehearsal format (identical facts) | rehearsed routes | never-rehearsed routes | bridge emitted as answer |
|---|---:|---:|---:|
| none | 0.762 / 0.681 | 0.819 / 0.654 | 0.011 / 0.003 |
| both facts as isolated answers | 0.648 / 0.850 | 0.743 / 0.893 | 0.287 / 0.048 |
| both facts in one supervised continuation | 0.831 / 0.808 | 0.858 / 0.763 | 0.049 / 0.007 |
| the same, with no loss on the intermediate entity | 0.884 / 0.791 | 0.879 / 0.767 | 0.001 / 0.000 |
| the same, with loss on the final answer only | 0.866 / 0.831 | 0.890 / 0.798 | 0.016 / 0.001 |
| intermediate entity as context, answer supervised | **0.970 / 0.948** | **0.947 / 0.905** | 0.005 / 0.000 |
| composite question → answer | **0.985 / 0.972** | **0.948 / 0.923** | 0.003 / 0.000 |

**Three declared ablations (v0.2.0)** take the format apart. They keep the text, rows and
schedule of the supervised-continuation format and change one thing at a time.

- **No loss on the intermediate entity where it answers the first hop.** The bridge-as-answer
  failure disappears in every seed of both families, but composition is not recovered (+0.053
  [−0.093, +0.198] Qwen, −0.017 [−0.156, +0.122] Falcon).
- **Loss on the final answer only.** Composition is still not recovered. Whether loss on the
  restated facts contributes is not resolved.
- **The same facts moved from the model's turn to the user's turn**, with the same supervised
  span. This is the intermediate-entity-as-context format, and it keeps composition better in
  every declared seed of both families. Over six seeds, the last three a labelled post hoc
  extension, the intervals exclude zero in each family (+0.099 [+0.045, +0.152] Qwen, +0.091
  [+0.019, +0.163] Falcon); the added seeds alone do not, so this supports the effect rather than
  establishing it independently.

Supervising the intermediate entity as an answer produces the failure's signature. Writing the
facts as the model's own response instead of the user's input lowers composition, an effect of
chat format whose cause is not tested. It points the same way on never-rehearsed routes and at the
fixed 12-epoch checkpoint. The protected computation still runs through the intermediate entity: on
never-rehearsed routes the answer depends on its representation at least as much as in the base
model (Qwen, one seed, one layer).

## Implications for continual updating

- Rehearse in the shape of use: the compositional information in the user's turn and the final
  answer as the model's first output, either as the composite question or as the facts given as
  context. The same facts written as the model's own response do not protect composition, even
  with no loss on them, and supervising the intermediate entity also teaches the model to answer
  with it.
- Compositions need not be enumerated to be protected: a modest set of composition-shaped or
  entity-as-context rows protects routes that were never rehearsed.
- Measure composition separately from recall, and probe for the signature: the rate at which
  composite prompts are answered with the intermediate entity. The collapse does not show in
  the loss curve.
- Matched content budgets are not matched protection: the condition with more supervised
  tokens and more optimizer steps protected less.

## A prediction for larger models

The bridge-as-answer failure operates through the training objective and the ablation ties it to
supervising the intermediate entity, so it should appear at every scale. The composition advantage
depends on how an instruction-tuned model separates the user's input from its own output, which
larger instruction-tuned models share, so its direction should hold as well. The
magnitude and the collapse frequency should shrink if pathway redundancy grows with depth: the
shallower family here (22 layers) showed more distributed damage and a lower rescue rate than
the deeper one (36 layers). A larger-model replication decides between the readings; an
undiminished effect at 70B would be the more surprising result.

## Limits

Two 3B models with LoRA adapters; one dataset (TwoHopFact); two-hop routes. The comparison
routes are composed latently (the base model names the intermediate entity on request for 30%
and 2% of them), so fact rehearsal mostly teaches standalone access rather than preserving it.
Knowledge of the second hop when asked directly was measured but not required for selection;
restricting to routes where the base model answers it under all six phrasings leaves the
primary result in place (+0.148 [+0.007, +0.289] Qwen, +0.175 [+0.047, +0.302] Falcon;
exploratory, `results/tables/T11_hop2_sensitivity.md`). On Qwen, token matching left 21 of 229
comparison routes with only one fact rehearsed under the atomic condition; excluding them gives
+0.171 [+0.036, +0.307].
The acquisition-matched checkpoint is defined on a 40-item probe of the injected facts; full-set
recall at that checkpoint is reported per condition (0.97–0.98 for the primary comparison) and
fixed-schedule results are reported beside every primary contrast. Exposure is matched on
content tokens only, with the asymmetry running against the claim. Three seed extensions were
decided after seeing three-seed intervals and are labelled. The format comparison and the three
ablations run on both families. The chat-boundary contrast meets its declared rule at three
seeds; its per-family six-seed intervals come from a post hoc extension. Why the model's own turn
behaves differently from the user's is not tested.

## Declared predictions

Every experiment's predictions and thresholds were written and dated before its runs; all are
listed with their outcomes in [`PREDICTIONS.md`](PREDICTIONS.md), including the ones that were
not met and the account the failure-matched test decided against.

## Reproduce

**From the deposited files** (standard library only):

```
python src/make_tables.py            # results/tables/T1-T12 and all.json from results/*.json
python src/audit_numbers_s4.py       # every manuscript number must appear among the table values
python paper/build_paper.py          # PDF (needs pandoc + xelatex)
```

The exposure table (T4) tokenizes the rehearsal rows and needs `transformers`; without it the
generator keeps the shipped `T4_exposure.md` and its `all.json` entries unchanged and says so.
The number check is a presence check: it catches a number that no table produces, not a correct
number attributed to the wrong condition.

**Re-running the experiments on the deposited task files.** `data/tasks/` holds the exact route
sets, splits and rehearsal sets the manuscript used. Training, measurement and the patching
experiments run from them with a GPU (8 GB suffices), the packages in `requirements.txt`, and
Hub access to the two base models (revisions pinned in `src/model_pin.py`) and the WikiText-2
test split (retention corpus). From the repository root:

```
python src/train_inject.py --data data/tasks/<fam>/g4_split_X_train.json --model <model> --epochs 12 --seed <s> --eval-subset 40 --preserve data/tasks/<fam>/preserve_<cond>.json --out results/<run>
python src/measure_g4_post.py --arm data/tasks/<fam>/g4_split_X_measure.json --model <model> --adapter results/<run>/checkpoint-mem100 --out results/<run>_mem100_post_X_s<s>.json
python src/analyze_preservation.py ...
python src/analyze_transfer.py --family qwen --seeds 0 1 2 3 4 5 --tasks data/tasks/v2 --split data/tasks/v2_transfer/transfer_split.json --out results/g15_transfer.json
python src/analyze_transfer.py --family falcon --seeds 0 1 2 3 4 5 --tasks data/tasks/falcon --split data/tasks/falcon_transfer/transfer_split.json --out results/g15_transfer_falcon.json
python src/screen_bridge_causal.py ...     python src/g12_failure_matched.py ...     python src/natact_transfer.py ...
```

`<fam>` is the task directory: `v2` (Qwen) or `falcon`. Transfer runs use
`--preserve data/tasks/<fam>_transfer/transfer_<cond>.json` (`v2_transfer` for Qwen) with
`<cond>` one of `route`, `atomic`, `coherent`, `bridgectx`, `coherent_bmask`,
`coherent_answeronly`. The no-rehearsal adapters omit `--preserve`. The analysis reads whichever
seeds exist and pairs contrasts by seed; tables T1–T12 report seeds 0–2 for every condition and
seeds 3–5 of `bridgectx` and `coherent_answeronly` separately (T12).

**Reconstructing the task files.** The rehearsal and transfer sets rebuild byte for byte from
the deposited route files: the commands below reproduce the published hashes of
`preserve_route.json` and `preserve_atomic.json` for both families. Rebuilding the route files
themselves from the screen will not be byte-identical, because the injectable facts were queried
live from the Wikidata API, which is not a frozen snapshot. Every TwoHopFact load is pinned to
Hub revision `59a84cd883f71641a03fb6fa50da92a9603e7845` (`src/model_pin.py`), and the screen
covered all 45,595 chains. The commands and parameters used:

```
python src/screen_twohop.py --model <model> --per-type 100000 --batch-size 24 --no-audit-gate --out results/stage0_screen.<tag>.jsonl
python src/measure_neutral_access.py --model <model> --screen results/stage0_screen.<tag>.jsonl --out results/neutral_access.<tag>.json
python src/verify_injectable.py --screen results/stage0_screen.<tag>.jsonl --model <model> --batch-size 24 --out results/injectable.<tag>.json
python src/build_g4_routes.py --injectable results/injectable.<tag>.json --screen results/stage0_screen.<tag>.jsonl --neutral-access results/neutral_access.<tag>.json --out data/tasks/<fam>/g4_routes.json
python src/build_g4_split.py --arm data/tasks/<fam>/g4_routes.json --outdir data/tasks/<fam> --seed 4
python src/build_preserve_sets.py --measure data/tasks/v2/g4_split_X_measure.json --model Qwen/Qwen2.5-3B-Instruct --control-pool data/tasks/v2/control_pool.json --outdir data/tasks/v2
python src/build_preserve_sets.py --measure data/tasks/falcon/g4_split_X_measure.json --model tiiuae/Falcon3-3B-Instruct --outdir data/tasks/falcon
python src/build_transfer_sets.py --measure data/tasks/<fam>/g4_split_X_measure.json --model <model> --outdir data/tasks/<fam>_transfer
python src/build_bmask_sets.py --measure data/tasks/<fam>/g4_split_X_measure.json --coherent data/tasks/<fam>_transfer/transfer_coherent.json --out data/tasks/<fam>_transfer/transfer_coherent_bmask.json
python src/build_answeronly_sets.py --measure data/tasks/<fam>/g4_split_X_measure.json --coherent data/tasks/<fam>_transfer/transfer_coherent.json --out data/tasks/<fam>_transfer/transfer_coherent_answeronly.json
```

`--per-type 100000` screens every chain (the default samples 150 per composition type). The
rehearsal sets are matched on content tokens by subsampling rows, not routes: on Qwen the budget
is that of a control set of unrelated Wikidata facts (`data/tasks/v2/control_pool.json`, used
only as a token budget), so 21 of the 229 comparison routes have one of their two facts
rehearsed under the atomic condition; on Falcon, run without a pool, the budget is the atomic
set's own and every route has both.
`--no-audit-gate` skips the screen's pre-run reproduction check, whose full form compares
against a dataset from an earlier study that is not part of this deposit.

Model outputs in `results/` are verbatim except that email-like strings a model generated were
replaced by `<email-redacted>` (5 occurrences across three files; no task file contains an email
address, so none of them was copied from an input).

## Layout

| | |
|---|---|
| `paper/` | manuscript source, PDF, build script |
| `src/` | every script the manuscript rests on |
| `data/tasks/` | route sets, counterbalanced halves, rehearsal and transfer sets, per family |
| `results/` | per-item measurement files, patching results, per-adapter training evals |
| `results/tables/` | the manuscript's tables T1–T12, regenerated from `results/` |
| `PREDICTIONS.md` | declared predictions and outcomes |

## Data provenance

Routes are drawn from TwoHopFact (Yang et al., 2024; CC BY 4.0; Hub revision
`59a84cd883f71641a03fb6fa50da92a9603e7845`). Injected facts are Wikidata statements (CC0),
retrieved live through the Wikidata API. The retention corpus is the WikiText-2 test split. See
`LICENSE`.

## Next steps

The formats that preserve composition leave the first hop unanswerable on its own, and the
formats that make it answerable cost composition. The next study asks whether rehearsal can do
both: make a known fact directly statable without breaking its latent compositional use, by
mixing answer-only composition rows with fact rows, or by adding a loss against emitting the
intermediate entity to fact rehearsal.

## Citation

See [`CITATION.md`](CITATION.md). License: CC BY 4.0 for the manuscript, documentation and
derived data; MIT for the code ([`LICENSE`](LICENSE)).

## Disclosure of LLM use

Design, code, runs, analyses and drafting were carried out with Claude (Anthropic) under the
author's direction, and a second, independent model reviewer examined the design and results at
three points. The author directed the work and is responsible for every claim.
