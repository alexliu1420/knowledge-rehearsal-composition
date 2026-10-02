# Declared predictions and their outcomes

Every prediction in this study was written down, with its threshold, in a dated design document
before the run that could adjudicate it. All are listed here with their outcomes, including the
ones that failed, in the form they were declared rather than the form the result would have
suggested. The design documents themselves are internal working records; their predictions,
thresholds and declaration dates are reproduced here in full.

Predictions were **not** registered with a third party. What is claimed is the declare-then-test
order, not independent timestamping. Three seed extensions (Falcon and Qwen in the primary
comparison, and both families in G18, from three seeds to six) were decided after the three-seed intervals were seen and are labelled as such wherever
they appear.

Estimator throughout: the training seed is the unit of replication; contrasts are paired over
seeds with t(k−1) intervals. Primary checkpoint: the first epoch at which a 40-item probe of the
injected facts reaches at least 99% strict recall. All values regenerate from `results/` via
`src/make_tables.py`.

## P — Preservation: route vs atomic rehearsal (§4; declared 2026-09-12)

Declared before seeds 1–2 and before any acquisition-matched measurement existed.

| prediction | declared criterion | outcome |
|---|---|---|
| P1 held-out compositional access is higher under route rehearsal than atomic rehearsal | route − atomic > 0, interval excluding 0 at three seeds | **Held on Qwen** (+0.119 [+0.035, +0.203]). **Not met on Falcon at three seeds** (+0.232 [−0.143, +0.607]; every seed positive, interval not excluding 0). Post hoc extension to six seeds: Falcon +0.204 [+0.097, +0.311], Qwen +0.166 [+0.032, +0.301] |
| P2 atomic rehearsal preserves the atoms | hop-1 access atomic ≫ route | **Held.** 0.95–0.98 vs 0.02–0.31 |
| P3 atomic rehearsal costs composition relative to the update alone | atomic < none | **Not held.** atomic − none +0.055 [−0.108, +0.217] (Qwen), +0.101 [−0.301, +0.504] (Falcon): no protection detected; equivalence not established |
| recall criterion | injected-fact recall matched across conditions at the primary checkpoint | **Held** on the 40-item probe by construction; full-set recall 0.979 / 0.972 (Qwen) and 0.976 / 0.978 (Falcon), minimum 0.93 |

## N — No-rehearsal baseline (§4; declared 2026-09-17)

| prediction | criterion | outcome |
|---|---|---|
| N-a route − none > 0, both families, both checkpoints | interval excluding 0 | **Held.** +0.173 [+0.080, +0.267] Qwen; +0.334 [+0.207, +0.461] Falcon |
| N-b atomic − none, two-sided; wording follows the result | — | intervals span 0 in both families → "no protection detected" |
| N-c bridge emission is near zero without rehearsal | descriptive | **Held.** 0.001–0.012 (none) vs 0.078–0.129 (atomic, six seeds) |

## F — Failure-matched causal test (§6; declared 2026-09-16)

Patching the held-out phrasings the atomic-rehearsal adapter answers wrongly. Declared to
distinguish two accounts: *the second hop is damaged* vs *the intermediate entity is unavailable
at the subject position*. Thresholds fixed before the run.

| prediction | criterion | outcome |
|---|---|---|
| second-hop-damaged account | own-bridge rescue < 15% of failures | **Rejected** in every seed of both families |
| entity-unavailable account | own-bridge rescue ≥ 30% | **Held.** Qwen 0.857 [0.733, 0.980]; Falcon 0.610 [0.310, 0.911]; context control 0.04 / 0.10 |
| wrong bridge redirects failures as it redirects successes | descriptive | failures − matched successes: Qwen −0.025 [−0.078, +0.028]; Falcon −0.057 [−0.243, +0.129] |
| bridge-emission failures: a wrong bridge yields the *donor's bridge* | descriptive | **Not held.** donor's bridge emitted 0.0–2.3%; donor's *answer* 0.31–0.53 |

The first account had been suggested by screens of computations that succeed (S1); the
declared test decided against it.

## A — Natural-activation transfer (S3; declared 2026-09-17)

| prediction | criterion | outcome |
|---|---|---|
| natural-state account | base-model state at the subject position rescues ≥ 30%; wrong-subject state < 10% | **Partly.** base 0.252 [0.183, 0.321] Qwen (declared mixed band), 0.363 [0.235, 0.490] Falcon; wrong subject 0.117 / 0.204 (above threshold) |
| injection-only account | base state rescues < 15% | **Rejected** in both families |
| position specificity | base state at the last token rescues less than at the subject position | Qwen: yes (0.063 vs 0.252); Falcon: no (0.278 vs 0.363) |
| route-adapter state vs base state | descriptive | +0.051 [−0.059, +0.162] Qwen; +0.016 [−0.071, +0.103] Falcon |

## T — Transfer and format comparison, Qwen (§7; declared 2026-09-17)

340 routes eligible for rehearsal, 320 never rehearsed and entity-disjoint; four formats at
matched content tokens; three seeds.

| prediction | criterion | outcome |
|---|---|---|
| T1 transfer: route − atomic on never-rehearsed routes | > 0, interval excluding 0 | **Sign held in every seed (3/3), interval not excluding 0**: +0.206 [−0.365, +0.776]; one atomic seed collapsed |
| T5 floor: route − none on never-rehearsed routes | two-sided | **+0.129 [+0.036, +0.222]**, 3/3 |
| T5 floor: bridge-as-context − none | two-sided | **+0.128 [+0.052, +0.204]**, 3/3; route − bridge-as-context +0.001 [−0.025, +0.027] |
| T5 floor: coherent − none; atomic − none | two-sided | +0.039 [−0.060, +0.138]; −0.076 [−0.696, +0.543] |
| T2 replication on rehearsed routes: route − atomic > 0 | sign | 3/3, +0.337 [−0.402, +1.075] |
| T3 format: coherent − atomic on rehearsed routes | two-sided | +0.184 [−0.426, +0.793], 3/3; bridge-as-context − coherent +0.139 [−0.005, +0.283] (rehearsed), +0.089 [−0.003, +0.181] (never-rehearsed), 3/3 each |
| T4 mechanism: bridge emission under bridge-as-context < half of atomic, with higher held-out access | declared | **Held in every seed.** 0.005 vs 0.287; 0.970 vs 0.648 |
| shortcut check (declared 2026-09-20): identity of the bridge representation on never-rehearsed routes under route / bridge-as-context within ±0.10 of base = genuine composition; a fall > 0.15 = shortcut | declared | **No shortcut.** Route +0.597 (+0.094 vs base +0.503, inside the band); bridge-as-context +0.682 (+0.179, *above* the band, in the direction opposite to a shortcut) |

## T′ — Replication of the transfer experiment on the second family (§7; declared 2026-09-20)

Same predictions on Falcon3-3B (571 rehearsed-eligible / 541 never-rehearsed, entity-disjoint;
route, atomic, bridge-as-context; three seeds).

| prediction | criterion | outcome |
|---|---|---|
| route − none on never-rehearsed routes > 0 | interval excluding 0 | **Held.** +0.269 [+0.078, +0.460], 3/3 |
| bridge-as-context − none > 0; route − bridge-as-context ≈ 0 | declared | **Held.** +0.251 [+0.111, +0.391]; +0.018 [−0.036, +0.073] |
| bridge emission under bridge-as-context ≪ atomic on rehearsed routes; held-out access higher | declared | **Held in every seed.** 0.000 vs 0.048; +0.098 [−0.073, +0.269], 3/3 |
| route − atomic on rehearsed routes > 0 | sign | 3/3, +0.122 [−0.063, +0.308] |
| atomic − none on never-rehearsed routes | not declared directional | +0.238 [+0.041, +0.436]: fact rehearsal of other routes also transfers general protection on this family |
| full-set recall at the primary checkpoint | reported | route 0.990, atomic 0.972, bridge-as-context 0.960 (one seed 0.888) |

## G16 — Bridge-token loss-mask ablation (§7; declared 2026-09-25, v0.2.0)

The coherent rows with the leading intermediate-entity tokens (the answer to the first-hop
prompt) given no loss; a copied mention of the entity later in the supervised continuation keeps
its loss; text, chat boundary, rows and schedule unchanged; Qwen and Falcon, three seeds each (Falcon's coherent condition was run
alongside). Declared consequence if B1 failed: supervising the bridge is not the operative
property for composition, and the mechanistic reading of v0.1.0 is withdrawn.

| prediction | criterion | outcome |
|---|---|---|
| B1 masked − coherent, composition on rehearsed routes | > 0 in every seed, each family | **Not met.** Qwen +0.053 [−0.093, +0.198] (2/3); Falcon −0.017 [−0.156, +0.122] (1/3) |
| B2 bridge emission, masked vs coherent | masked below half of coherent in every seed | **Met.** Qwen 0.049 → 0.001; Falcon 0.007 → 0.000 |
| B3 bridge-as-context − masked; first-hop access under masking | descriptive | +0.086 [+0.064, +0.109] Qwen, +0.157 [+0.032, +0.282] Falcon, every seed; first-hop access falls to the no-rehearsal level (−0.670, −0.819) |

The v0.1.0 reading — that the formats which fail do so because they make the intermediate entity
a supervised output — is withdrawn as the account of the composition loss, as declared. B1's
failure is not evidence of no effect: the intervals allow a partial contribution, and at the fixed
schedule masking closes most of Qwen's gap and widens Falcon's. Supervision of the copied mention,
bundled with the supervised continuation, is not separated here; G17 separates it. Supervising the intermediate
entity is supported as the source of the bridge-as-answer failure and of standalone first-hop
access.

## G17 — Answer-only loss on the coherent format (§7; declared 2026-09-27, v0.2.0)

Same rows and text as the coherent format, loss on the final answer only. It differs from
bridge-as-context only in where the chat boundary falls, and from the masked format only in the
loss on the restated facts. f is the share of the masked-to-bridge-as-context gap on rehearsed
routes that answer-only closes.

| prediction | criterion | outcome |
|---|---|---|
| Supervision account | f ≥ 2/3 in both families, answer-only above masked in every seed | **Not met.** f = −0.21 Qwen, 0.26 Falcon |
| Placement account | f ≤ 1/3 in both families, bridge-as-context above answer-only in every seed | **Met.** +0.105 [−0.027, +0.236] Qwen (3/3), +0.117 [−0.039, +0.272] Falcon (3/3) |
| Never-rehearsed routes | descriptive | same direction, every seed: +0.057 [−0.082, +0.196], +0.107 [−0.106, +0.319] |
| Fixed-schedule checkpoint | descriptive | weaker: +0.042 [−0.114, +0.199] Qwen (2/3), +0.077 [−0.027, +0.181] Falcon (3/3) |

Per-family three-seed intervals include zero; the rule was declared on direction in every seed.

## G18 — Seeds 3–5 for the placement contrast (§7; declared 2026-09-28, post hoc)

Decided after G17's result was seen, and labelled post hoc. G17's three-seed outcome stands as
declared. The rule was fixed before seeds 3–5 ran, but extending after seeing a result raises the
chance of a positive finding, and the added seeds alone do not exclude zero: the six-seed result
supports the chat-format effect rather than establishing it independently.

| prediction | criterion | outcome |
|---|---|---|
| Placement effect established, per family | bridge-as-context − answer-only on rehearsed routes, seeds 0–5, t(5) interval excludes zero with positive mean | **Met in both.** Qwen +0.099 [+0.045, +0.152] (6/6); Falcon +0.091 [+0.019, +0.163] (5/6) |
| Seeds 3–5 alone | descriptive | Qwen +0.093 [−0.056, +0.242] (3/3); Falcon +0.066 [−0.125, +0.258] (2/3, one seed at zero) |
| Never-rehearsed routes | descriptive | Qwen +0.049 [−0.000, +0.099] (6/6); Falcon +0.089 [+0.006, +0.172] (6/6) |
| Fixed-schedule checkpoint | descriptive | Qwen +0.082 [+0.014, +0.151] (5/6); Falcon +0.108 [+0.029, +0.186] (6/6) |

## Exploratory analyses not declared in advance

- Restricting the primary contrast to comparison routes whose second hop the base model answers
  under all six direct phrasings (second-hop knowledge was measured but was not a selection
  criterion): +0.148 [+0.007, +0.289] Qwen (179 of 229 routes), +0.175 [+0.047, +0.302] Falcon
  (231 of 414), six seeds (`results/tables/T11_hop2_sensitivity.md`).
- Restricting to comparison routes with both facts rehearsed under the atomic condition (token
  matching subsamples rows, leaving 21 of 229 Qwen routes with one fact; Falcon all paired):
  +0.171 [+0.036, +0.307] Qwen (208 routes); with the all-six second-hop restriction as well,
  +0.153 [+0.010, +0.296] (162 routes).
- The shortcut check covers Qwen, seed 0, layer 12 only.
- Pooling the six G17 seed pairs across families: bridge-as-context − answer-only +0.111
  [+0.056, +0.166] on rehearsed routes and +0.082 [+0.008, +0.155] on never-rehearsed routes,
  6/6 (`results/tables/T12_decisions.md`).
- The collapse threshold (held-out access < 0.7) was set after the atomic adapters were seen;
  the non-collapsed atomic adapters lie at 0.705–0.928 (`T9_collapse.md`).

## What was expected and did not appear

- P1 at three seeds on Falcon: not met; met at six seeds, an extension decided after the fact.
- Atomic rehearsal harming composition relative to no rehearsal (P3): not supported.
- A wrong bridge producing the donor's *bridge* on bridge-emission failures: not supported.
- A clean natural-state account: partly supported only; the wrong-subject control exceeded its
  threshold in both families.
- Transfer as a route − atomic contrast (T1): sign-consistent on Qwen, near zero on Falcon;
  established against no rehearsal instead.
