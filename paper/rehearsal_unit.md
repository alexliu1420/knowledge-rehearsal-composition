# What You Rehearse Is What You Keep: Rehearsal Format Decides Whether Known Facts Stay Composable

**Preprint v0.6, 2026-09-25.** All runs are in. Every number is emitted by
`src/make_tables.py` into `results/tables/` and checked for presence there by
`src/audit_numbers_s4.py` before the PDF is built.

**Author:** Alex Liu
**License:** CC BY 4.0
**Keywords:** continual learning · knowledge injection · rehearsal · replay · multi-hop composition · latent reasoning · catastrophic forgetting

---

## Abstract

Fine-tuning a language model on new facts can leave the facts it already knew intact as
answers while removing its ability to use them together. Rehearsal is the standard protection,
and the usual unit of rehearsal is the individual fact. We ask what rehearsal must supervise to
preserve composition.

On two-hop routes that two 3B-parameter models (Qwen2.5-3B-Instruct, Falcon3-3B-Instruct)
compose robustly under twelve phrasings before any update, an update with no rehearsal costs
18 and 34 percentage points of compositional access on phrasings the model never saw.
Rehearsing the composition in a single phrasing restores it to 0.99. Rehearsing the two
constituent facts as answers, with about twice the supervised tokens, leaves it near the
no-rehearsal level (composition − facts: +0.166 [+0.032, +0.301] on Qwen, +0.204 [+0.097,
+0.311] on Falcon, six seeds each). Fact rehearsal produces a specific failure — the model
answers the composite question with the intermediate entity — on 8–13% of held-out phrasings
against about 1% without rehearsal, and it collapses composition (held-out access below 0.7)
in 3 of 18 adapters at each checkpoint, with loss curves and memorisation epochs like those of
the adapters that did not collapse. On the failing
phrasings the second inferential step is usable: supplying the intermediate entity's
representation restores the answer on 86% (Qwen) and 61% (Falcon) of them.

A controlled transfer experiment shows the effect is one of rehearsal format and that the
protection generalises. Rehearsing 340 routes protects 320 never-rehearsed routes that share no
subject or intermediate entity with them (+0.129 [+0.036, +0.222]); on a second family, +0.269
[+0.078, +0.460]. Among formats containing the same two facts per route, the one in which the
intermediate entity is only ever context preserves composition (0.970) and transfers it
(0.947) as well as rehearsing the composition; the format that supervises it inside a
continuation is lower (0.831, 0.858); the format that supervises it as an isolated answer is
lowest (0.648, 0.743, with 29% of answers being the entity itself). **Rehearsal format decides
whether known facts stay composable; the formats that fail are the ones that make the
intermediate entity a supervised output**, which we put forward as the operative property, not
as an isolated cause: the two closest formats also differ in chat boundary and supervised span.

---

## 1 · Introduction

Continual updating of a deployed model has a familiar cost: what it learns next can degrade
what it knew before. Rehearsal — mixing previously learned material into the update — is the
oldest and most reliable mitigation, and its unit is almost always the individual example or
fact. That choice is rarely examined. If a model's knowledge of *A* and of *B* is preserved,
its ability to use *A* and *B* together is assumed to follow.

**The question.** What must rehearsal supervise to preserve a model's ability to compose facts
it already knows?

We answer it in the cleanest setting we could construct. The task is factual two-hop
composition: *the country where the author of novel X was born* requires resolving an
intermediate entity (the author) and then a fact about it (birth country). We take routes the
base model already composes reliably across six phrasings and two personas, inject unrelated
new facts about the intermediate entities by LoRA fine-tuning, and compare rehearsal
conditions at matched content-token budgets and matched acquisition of the injected facts. The
outcome is compositional access on phrasings and a persona the model never saw during the
update.

**The answer, in four parts.** (§4) An update with no rehearsal costs 18–34 points of held-out
compositional access; rehearsing the composition restores it; rehearsing the constituent facts
as answers does not, despite roughly double the supervised tokens. (§5) Fact rehearsal induces
a specific failure — the intermediate entity emitted as the final answer — that is absent
without rehearsal, and it collapses composition in a minority of training seeds. (§6) On the
failing phrasings the second step is usable when the intermediate entity is supplied, which is
what makes the next result more than a correlation between format and outcome. (§7) The
protection transfers to routes never rehearsed, in both families, and a comparison of formats
with identical facts shows that the formats which fail are the ones that make the intermediate
entity a supervised output.

The practical rule is short. **Do not train the model to answer with an entity that its
existing reasoning is supposed to derive.** In our data, rehearsal formats that keep the
intermediate entity out of the supervised output preserve composition, and formats that make it
an answer degrade it — most as an isolated answer, less inside a longer continuation.

### Contributions

1. A controlled comparison of rehearsal units — nothing, constituent facts, the composition —
   at matched content tokens and matched acquisition, on routes the model composes before the
   update, across two model families and six seeds each, with the exposure asymmetry
   reported in full (§4).
2. A reproducible failure signature of fact rehearsal, the intermediate entity emitted as the
   answer, shown to be induced by the rehearsal rather than by the update, with an
   operational definition of the collapse it produces and its frequency (§5).
3. A transfer experiment on 320 (Qwen) and 541 (Falcon) routes disjoint in subject and
   intermediate entity from every rehearsed route, with a format comparison at identical facts
   that ranks formats by whether the intermediate entity is supervised (§7), supported by a
   failure-matched causal test with thresholds declared in advance (§6).

### What this paper does not claim

The routes in the comparison set are composed *latently*: the base model names the intermediate
entity on request for 30% of Qwen's and 2% of Falcon's comparison routes. Fact rehearsal
therefore mostly *teaches* standalone access rather than preserving it, and the claim is that
fact rehearsal does not ensure compositional retention, not that preserving already accessible
facts is insufficient (§8). The format comparison ranks whole formats; the two formats that
differ least (§7) also differ in chat boundary and in how many tokens are supervised, so it
isolates the entity's presence in the supervised output as a property of the format, not the
entity token alone. §6 shows that failing computations are recoverable under intervention; it
does not show that every failing computation has an undamaged second hop. We make no claim
about the timing of the model's own resolution of the intermediate entity. Two 3B models with
LoRA adapters, one dataset, two-hop routes.

---

## 2 · Related work

**Rehearsal and replay.** Interleaving old material into new training is the classic remedy
for catastrophic forgetting [Robins 1995; Rolnick et al. 2019]; the unit rehearsed is the
training example. We hold the method fixed and vary what is supervised.

**Forgetting as lost access rather than lost knowledge.** Zheng et al. [2025] show that
performance drops in continual learning of language models often reflect lost task alignment
rather than lost knowledge. Our no-rehearsal baseline is a case: the facts remain recallable
and the composition does not. Our contribution is which rehearsal format restores it and why
the obvious one does not.

**Atomic knowledge without composition.** Yu et al. [2026] show that post-training recipes
with indistinguishable atomic knowledge differ by over 40 points in multi-hop composition. We
observe the same dissociation with the recipe fixed and only the rehearsed material varied, and
identify the format property that tracks it.

**Supervising intermediates.** Two lines bear directly on our claim. In algorithmic reasoning
with depth-recurrent transformers, supervising every intermediate step is harmful relative to
a silent objective [Chen 2026]; internalised chain-of-thought removes intermediate supervision
progressively [Deng et al. 2024]. Our finding is the natural-fact,
pretrained-model, continual-update instance of that principle, with transfer and a causal
test. Conversely, Lin et al. [2025] add a zero-hop identity supervision on the bridge entity
("identity bridge") to *enable* out-of-distribution two-hop generalisation, and report that
correct two-hop predictions coincide with a direct subject-to-answer association. That raises
the question whether protection in our setting is genuine composition or a shortcut; §7 tests
it causally on the never-rehearsed routes.

**Latent multi-hop reasoning.** Yang et al. [2024] and Biran et al. [2024] establish that
two-hop questions are answered by resolving the intermediate entity at the subject position in
early-to-middle layers, and that patching that representation can repair failures. The
instrument in §6 is theirs; we use it to test availability, not timing. Balesni et al. [2024]
find that models fine-tuned on two synthetic facts fail to compose them latently unless the
facts co-occur in training or the prompt, but can compose one synthetic and one natural fact;
our routes are pretrained knowledge on both hops, and our manipulation is supervision format at
fixed content.

**Circuit preservation and knowledge editing.** Comparisons of training methods by how much of
the base circuitry survives [Laitinen-Fredriksson Lundstrom-Imanov 2026; Rojas Nunez et al.
2026] and editing methods that make a new fact reach its multi-hop uses [Zhao et al. 2026; Yang
et al. 2025; He et al. 2026] are adjacent; we compare
rehearsal targets and ask whether an unrelated update leaves existing composition intact.

---

## 3 · Setup

### 3.1 Routes the model already composes

Routes are TwoHopFact chains [Yang et al. 2024]: subject → intermediate entity ("bridge") →
answer. A route enters the study only if the base model answers the composite question under
all six phrasings (canonical, named, question, imperative, "known as", "refers to") under a
domain persona and under a neutral persona, and is not answered when its first entity is
replaced by a different entity in the same question frame. That control removes routes answered
from the frame or from an answer prior; it does not rule out a memorised association from the
correct subject to the answer, which §7 addresses with a causal test. The
screen covers all 45,595 TwoHopFact chains; 4,475 (Qwen) and 3,176 (Falcon) pass the
domain-persona criterion, 1,605 and 2,254 also pass the neutral persona, and after requiring a
well-formed injectable fact about the bridge 753 and 1,639 routes enter the design, over 655
and 1,427 distinct bridges, of which 660 and 1,112 are measurable in the half used here.
Knowledge of the second hop when asked directly is measured but is **not** a selection
criterion: on the comparison set, the base model answers all six direct second-hop phrasings for
179 of 229 Qwen routes and 231 of 414 Falcon routes, and none of them for 21 and 117. An
exploratory restriction to routes whose second hop is known on all six phrasings leaves the
primary result unchanged in direction and excluding zero (Table S11; §4).

### 3.2 The update and the within-model split

For each route, an unrelated fact about the bridge entity, drawn from Wikidata and verified
absent from the base model, is injected by LoRA fine-tuning [Hu et al. 2022] (rank 16, α 32,
dropout 0.05, all attention and MLP projections, AdamW lr 2e-4, weight decay 0.01, micro-batch
1 with gradient accumulation over 8 rows, 12 epochs, answer-only loss, chat template with the
domain persona). The incomplete last accumulation group of each epoch is discarded. Routes are
split into two counterbalanced halves so that one adapter injects facts about half the bridges
and every route is measured in the same adapter as its own control group; the analyses here
use half X. The split makes injected and non-injected routes comparable within one adapter,
where the update's general effect is shared. The rehearsal contrasts in this paper are different:
they compare different adapters, paired only by training seed, and their general effects differ
(§7), which is why every one of them is reported against a no-rehearsal adapter as well.

### 3.3 Rehearsal conditions

Rows are appended to the injection set; the update is otherwise identical. **none**: the
injected facts only. **atomic**: hop-1 prompt → bridge and hop-2 prompt → answer. **route**:
the composite question in its canonical phrasing → answer. Rehearsal sets are matched on content
tokens by subsampling rows, not routes, to a token budget (on Qwen, the budget of a control
set of unrelated Wikidata facts; on Falcon, the atomic set's own). On Qwen this leaves 21 of
the 229 comparison routes with only one of their two facts rehearsed under the atomic condition;
on Falcon all 414 have both. Restricting to the 208 Qwen routes with both facts gives +0.171
[+0.036, +0.307] at six seeds against +0.166 on all 229 (exploratory; Table S11). Because atomic
rows are shorter, the atomic condition receives 1.7–2.1× the supervised tokens and 17–21% more
optimizer steps per epoch (Table S4). The comparison set is the routes rehearsed in both
conditions: 229 on Qwen, 414 on Falcon. §7 adds two formats with identical facts.

### 3.4 Outcome, checkpoints and estimator

**Held-out compositional access** is the fraction correct over five non-canonical phrasings
under the domain persona and six phrasings under the neutral persona; the canonical phrasing
under the domain persona is the rehearsed string and is excluded. Scoring is answer-or-alias
containment after lower-casing and stripping punctuation; word-boundary and exact-match
variants applied to the five domain-persona predictions change the primary contrast by at most
0.004 (Table S1). **Bridge emission** is the fraction of the five domain-persona predictions
whose wrong answer contains the bridge entity.

The **primary checkpoint** is the first epoch at which a fixed 40-item probe of the injected
facts reaches at least 99% strict recall ("acquisition-matched"); full-set recall over all
injected routes at that checkpoint is reported per condition (0.97–0.98 in both families for
the primary comparison, minimum 0.93; Table 1 and Table S10). The **secondary** checkpoint is
the fixed 12-epoch schedule, reported alongside. The unit of replication is the training seed;
contrasts are paired over seeds with t(k−1) intervals.

### 3.5 Declared before the run

Each experiment's design, predictions and thresholds were written and dated before its runs;
all are listed with their outcomes in `PREDICTIONS.md`. Nothing was deposited with a
third-party registry. Two seed extensions, from three seeds to six, were decided after the
three-seed intervals were seen and are reported as such (§4).

---

## 4 · The unit of rehearsal determines what is preserved

**Table 1.** Held-out compositional access on the comparison set, acquisition-matched
checkpoint. Route and atomic: six seeds; none: seeds 0–2, and contrasts against it pair those
three seeds (`results/tables/T1_preservation.md`, `T2_no_rehearsal.md`).

| | none | atomic | route | route − atomic (6 seeds) | route − none (3 seeds) | atomic − none (3 seeds) |
|---|---:|---:|---:|---|---|---|
| Qwen2.5-3B | 0.817 | 0.825 | 0.991 | **+0.166 [+0.032, +0.301]** | +0.173 [+0.080, +0.267] | +0.055 [−0.108, +0.217] |
| Falcon3-3B | 0.659 | 0.786 | 0.990 | **+0.204 [+0.097, +0.311]** | +0.334 [+0.207, +0.461] | +0.101 [−0.301, +0.504] |

Per-seed route − atomic: Qwen +0.157 / +0.107 / +0.092 / +0.062 / +0.164 / +0.416; Falcon
+0.176 / +0.403 / +0.118 / +0.147 / +0.184 / +0.195. The declared three-seed results were
+0.119 [+0.035, +0.203] (Qwen) and +0.232 [−0.143, +0.607] (Falcon; every seed positive,
interval not excluding zero); both extensions to six seeds were decided after those intervals
were seen, and Falcon seeds 3–5 alone give +0.175 [+0.113, +0.238]. At the fixed 12-epoch
schedule: +0.126 [−0.006, +0.258] and +0.183 [+0.098, +0.267]. Full-set injected recall at the
primary checkpoint: route 0.979 / atomic 0.972 (Qwen), 0.976 / 0.978 (Falcon).

An update with no rehearsal costs 18 (Qwen) to 34 (Falcon) points of compositional access on
phrasings the model never saw. Rehearsing the composition in one phrasing returns it to 0.99,
and the advantage is present on every held-out phrasing and tightest on the least similar one
(Jaccard 0.60 to the rehearsed string), so it is not memorisation of the string. Rehearsing the
constituent facts preserves the facts — first-hop access 0.95–0.98 under atomic rehearsal
against 0.02–0.31 under route rehearsal — and leaves composition near the no-rehearsal level:
the atomic − none intervals span zero in both families at three seeds, which does not
establish equivalence but does show that no protection was detected.

Exposure is not matched beyond content tokens: atomic rehearsal supervises 1.7–2.1× as many
tokens and takes more optimizer steps both per epoch and to the acquisition-matched checkpoint
(Table S4). The route advantage therefore does not come from more rehearsal signal, but extra
supervised updates need not help monotonically, so the asymmetry is not excluded as a source of
additional drift under atomic rehearsal. Retention-perplexity changes overlap between conditions
(every adapter's value is in Table S9). On the comparison set the base model names the bridge on request for 30% of Qwen
routes and 2% of Falcon routes, so atomic rehearsal largely *teaches* standalone access; on the
51 Qwen routes where the bridge is already at ceiling, the effect is the same size and, at six
seeds, excludes zero (+0.138 [+0.008, +0.268]). Restricting to routes whose second hop the base
model answers under all six direct phrasings (exploratory; Table S11) gives +0.148 [+0.007,
+0.289] on Qwen (179 routes) and +0.175 [+0.047, +0.302] on Falcon (231 routes).

---

## 5 · Fact rehearsal induces a specific failure that sometimes collapses composition

**Table 2.** Bridge emission on the five held-out domain-persona phrasings, comparison set,
acquisition-matched checkpoint; none three seeds, route and atomic six.

| | none | route | atomic (per seed) |
|---|---:|---:|---|
| Qwen2.5-3B | 0.012 | 0.004 | **0.129** (0.112 / 0.072 / 0.074 / 0.028 / 0.145 / 0.341) |
| Falcon3-3B | 0.001 | 0.000 | **0.078** (0.042 / 0.191 / 0.027 / 0.036 / 0.105 / 0.069) |

Under atomic rehearsal the dominant wrong answer to the composite question is the intermediate
entity itself: *the country where the author of X was born* → the author. It accounts for over
half of all misses on Qwen in every seed. The rate is about 1% or less with no rehearsal and
under route rehearsal; the failure is induced by rehearsing the bridge as an answer, not by the
update. (Falcon adds a smaller second mode, a city offered where a country was asked, absent on
Qwen; Table S2.)

The failure sometimes collapses composition. Defining collapse as held-out access below 0.7 on
the comparison set, **3 of the 18 atomic-rehearsal adapters in this paper collapse at the
acquisition-matched checkpoint and 3 of 18 at the fixed schedule** (0.32–0.59 and 0.38–0.67),
against 0 of 18 route adapters at either (Table S9). The remaining atomic adapters lie between
0.705 and 0.928, so the count depends on a threshold placed in a narrow gap. Collapsed adapters'
memorisation epochs lie within their siblings' range; their retention-perplexity change lies
within it for three of the six and outside it for three, above in two cases and below in one,
so neither is an indicator of collapse in these data. The instability is a property of the
condition, and it is why every contrast against atomic rehearsal has a wider interval than its
consistent sign suggests.

---

## 6 · On the failing phrasings the second step is usable

If the format comparison in §7 is to be more than a correlation between format and outcome,
the failures under fact rehearsal must be shown to be of the kind that supervising the
intermediate entity could produce: a missing intermediate entity, not a broken second step. We
tested this on the computations that actually fail, with thresholds declared before the run
(below 15% rescue would support a broken second step; at or above 30%, a missing entity).

For every held-out phrasing the atomic-rehearsal adapter answers wrongly (Qwen 194 / 153 / 110,
Falcon 375 / 904 / 279, seeds 0–2), the residual stream at the subject's final token at a
mid-depth layer (Qwen 12 of 36, Falcon 11 of 22) was overwritten during prefill with the bridge
entity's representation read from a prompt that names it; with a different route's bridge; or
with the same position read from the bridge-prefixed prompt as a context control. The
unpatched pass reproduced the stored miss on 99.6–99.8% of prompts.

**Table 3.** Fraction of failing phrasings answered correctly after the patch; seed means with
t(2) intervals (`T5_rescue.md`).

| donor | Qwen2.5-3B | Falcon3-3B |
|---|---|---|
| own bridge | **0.857 [0.733, 0.980]** | **0.610 [0.310, 0.911]** |
| context control | 0.037 | 0.103 |
| wrong bridge → that route's answer | 0.564 (on successes: 0.58) | 0.458 (on successes: 0.53) |

Every seed in both families clears the upper threshold. Handed the bridge, the model answers
on most failing phrasings; handed a wrong bridge, it follows it to the wrong route's answer
about as often as on succeeding phrasings. The second step is usable when the bridge is
supplied at the subject position; on the remaining 14% (Qwen) and 39% (Falcon) of failures it
is not, and those are not characterised further here. (The competing account — a damaged
second step, suggested by screens of computations that succeed — is the one the declared
thresholds were set to distinguish; both accounts, the screens, and a further experiment on
whether the base model's own state at that site is sufficient are in Supplementary S1–S3.)

---

## 7 · The protection transfers, and the formats that break composition supervise the entity

Two objections remain after §4–§6. Rehearsing the composite question and testing its
paraphrases might be ordinary rehearsal of the tested association; and isolated question–answer
rows might be an unfair way to rehearse facts. One experiment, run on both families, answers
both.

**Design.** The 660 measurable Qwen routes were split once into 340 eligible for rehearsal (R)
and 320 never rehearsed in any condition (L), with routes sharing a subject or a bridge
clustered so that no entity appears on both sides. Four rehearsal sets were drawn from R at
matched content tokens (within 0.3%): **route** (340 routes); **atomic** (238 routes, two rows
each); **coherent** — hop-1 prompt → "*bridge*. *hop-2 statement* *answer*", both facts
supervised in one continuation (232 routes); and **bridge-as-context** — "*hop-1 prompt*
*bridge*. *hop-2 prompt*" → answer, the same two facts with the bridge only ever in the prompt
(232 routes). Coherent and bridge-as-context contain identical facts; they differ in where the
chat boundary falls and therefore in whether the bridge and the second-hop statement are
inside the supervised output. Three seeds; predictions declared in advance. The Falcon
replication uses the same builder (571 rehearsed-eligible, 541 never-rehearsed) with route,
atomic and bridge-as-context.

**Table 4.** Held-out compositional access and bridge emission, acquisition-matched
(`T6_transfer.md`).

| Qwen2.5-3B | none | atomic | coherent | bridge-as-context | route |
|---|---:|---:|---:|---:|---:|
| L: 320 never-rehearsed routes | 0.819 | 0.743 | 0.858 | **0.947** | **0.948** |
| R∩: 232 routes rehearsed in all four | 0.762 | 0.648 | 0.831 | **0.970** | **0.985** |
| bridge emission on R∩ | 0.011 | **0.287** | 0.049 | 0.005 | 0.003 |

| Falcon3-3B | none | atomic | bridge-as-context | route |
|---|---:|---:|---:|---:|
| L: 541 never-rehearsed routes | 0.654 | 0.893 | **0.905** | **0.923** |
| R∩: 375 routes rehearsed in all three | 0.681 | 0.850 | **0.948** | **0.972** |
| bridge emission on R∩ | 0.003 | **0.048** | 0.000 | 0.000 |

| contrast on the never-rehearsed routes | Qwen | Falcon |
|---|---|---|
| route − none | **+0.129 [+0.036, +0.222]**, 3/3 | **+0.269 [+0.078, +0.460]**, 3/3 |
| bridge-as-context − none | **+0.128 [+0.052, +0.204]**, 3/3 | **+0.251 [+0.111, +0.391]**, 3/3 |
| route − bridge-as-context | +0.001 [−0.025, +0.027] | +0.018 [−0.036, +0.073] |
| atomic − none | −0.076 [−0.696, +0.543] | +0.238 [+0.041, +0.436], 3/3 |
| bridge-as-context − coherent | +0.089 [−0.003, +0.181], 3/3 | — |
| route − atomic | +0.206 [−0.365, +0.776], 3/3 | +0.031 [−0.091, +0.152] |

**Transfer.** Rehearsing routes raises compositional access on routes that share no subject or
bridge with them, in every seed of both families, with intervals excluding zero. Most of the
benefit is general, not confined to the rehearsed association. The second family adds a
nuance: on Falcon, fact rehearsal of *other* routes also transfers general protection (0.893
against 0.923 under route rehearsal), so the damage the atomic format does is local to the
routes whose intermediate entity it supervises, while the general drift of the update is
repaired by any on-distribution rehearsal. On Qwen, in the two atomic seeds that did not
collapse, atomic rehearsal also exceeds no rehearsal on the never-rehearsed routes (0.878 and
0.873 against 0.829 and 0.787).

**Format.** On the rehearsed routes, bridge-as-context exceeds atomic rehearsal in every seed
of both families (+0.323 on Qwen; +0.098 [−0.073, +0.269] on Falcon) and exceeds coherent
rehearsal in every Qwen seed; coherent exceeds atomic in every Qwen seed. Bridge emission falls
0.287 → 0.049 → 0.005 (Qwen) and 0.048 → 0.000 (Falcon) as the bridge leaves the supervised
output. Atomic rehearsal's position relative to no rehearsal differs by family: below it on Qwen
(0.648 against 0.762, driven by a collapsed seed) and above it on Falcon (0.850 against 0.681).
The two formats with identical facts, coherent and bridge-as-context, differ by +0.139 [−0.005,
+0.283] on rehearsed routes and +0.089 [−0.003, +0.181] on never-rehearsed ones, positive in
every seed with intervals touching zero. The declared mechanism prediction — emission under
bridge-as-context below half of atomic, with higher held-out access — was met in every seed of
both families. Taken together: **rehearsal format decides the outcome, and the formats that fail
are the ones that make the intermediate entity a supervised output.** We read this as support for
supervision of the intermediate entity as the operative property; how much of the effect is the
entity token itself, as against the chat boundary and the supervised span that move with it,
this design does not separate.

**No sign of a shortcut.** On the never-rehearsed routes, transplanting a wrong bridge at the
subject position changes the answer under route and bridge-as-context rehearsal at least as much
as in the base model (identity +0.597 and +0.682 against +0.503; the declared band for "genuine
composition" was ±0.10 of base, and bridge-as-context lies above it, in the direction opposite
to a shortcut; Table S7). This test covers Qwen, one seed and one layer. Within that scope the
protected answers still depend on the bridge representation; it does not exclude a
subject-to-answer association carried elsewhere in the network or in other seeds.

---

## 8 · Limitations

Two 3B models with LoRA adapters; one dataset; two-hop routes only. The comparison-set routes
are composed latently (§4), so "preserving known facts" would misdescribe the atomic condition;
nor was knowledge of the second hop a selection criterion (§3.1), though restricting to routes
where it is fully known leaves the primary result in place (Table S11).
The acquisition-matched checkpoint is defined on a 40-item probe; full-set recall at that
checkpoint is 0.97–0.98 for the primary comparison but 0.89 for one Falcon transfer adapter,
and the fixed-schedule results are reported beside every primary contrast for that reason.
Exposure is matched on content tokens only (Table S4); the asymmetry disfavours the route
condition. Two seed extensions were decided post hoc and are labelled; one route seed dipped at
the fixed schedule only (0.81) and is retained. The four-format comparison is on one family;
transfer and the atomic-versus-context contrast replicate on the second. Never-rehearsed routes
share relation types with rehearsed ones by design, so transfer is across entities within
relation types. §6 establishes recoverability under intervention at one position and layer,
not the timing or the full extent of the damage.

**A prediction for larger models.** The effect operates through the training objective —
supervising a token as a target strengthens the prompt-to-token mapping at the output — and
through the latent pathway that resolves the intermediate entity at the subject position, which
is documented from 7B to 70B. Both are present at every scale, so we expect the *direction* to
hold: fact rehearsal that supervises the intermediate entity should still raise bridge emission
and protect composition less than formats that keep the entity in context. We expect the
*magnitude* and the collapse frequency to shrink with scale if pathway redundancy grows with
depth: the shallower family here (Falcon, 22 layers) showed more distributed damage and a lower
explicit-bridge rescue than the deeper one (Qwen, 36 layers; 61% vs 86%), consistent with more
places for the same interference to spread across. A larger-model replication would decide
between the two readings; an effect of undiminished size at 70B would be the more surprising
result, because it would mean the interference is not a capacity phenomenon.

---

## 9 · Conclusion

Rehearsal preserves what it supervises. On routes a model already composes, rehearsing the
constituent facts as answers preserves the facts and not the composition, produces a specific
failure in which the intermediate entity is emitted as the answer, collapses composition in a
minority of seeds, and does so despite more supervised signal than rehearsing the composition.
On the failing phrasings the second step is usable once the intermediate entity is supplied.
Rehearsal format decides whether known facts stay composable: across formats with identical
facts, the ones that fail make the intermediate entity a supervised output, while the composite
question and a bridge-as-context statement of the same two facts preserve composition and
transfer the protection to routes never rehearsed, in two model families. The rule that follows: do not train a model to answer
with an entity its existing computation is meant to derive.

---

## Supplementary material

**S1 · Causal screens on successful computations.** Before the failure-matched test, the
transplant instrument was run on the canonical prompt of every comparison-set route each adapter
still answers correctly, at three layers per family, with the bridge representation read from a
prompt that names it and a wrong bridge from a distinct-answer route of the same template
(Table S8). Identity of the bridge representation (own − wrong) under atomic rehearsal sits
above base at every layer in both families (Qwen +0.576 / +0.668 / +0.659 against base +0.449 /
+0.498 / +0.626 at layers 6 / 12 / 24; Falcon +0.497–0.547 against +0.393–0.396 at layers 7 and
11, three seeds), and route rehearsal at or below base on Falcon (route − atomic −0.176
[−0.296, −0.056] and −0.212 [−0.393, −0.031] at layers 7 and 11). The "answer follows the
patch" statistic was first computed loosely (any same-template answer accepted) and is stored
in the result files both loosely and strictly (the actual donor's answer); strict following
under atomic rehearsal exceeds route rehearsal at every layer on both families.

**S2 · The two accounts the failure-matched test was declared to distinguish.** The screens in
S1 show atomic rehearsal *raising* the answer's dependence on the bridge representation on
computations that succeed. One reading is that the bridge is resolved and the second hop is
damaged; the other that the bridge is unavailable on the phrasings that fail. Because S1 only
examines successes, it cannot decide. The test in §6 was declared with thresholds (rescue below
15% for the first account, at or above 30% for the second) and decided for the second in every
seed of both families.

**S3 · Natural-activation transfer.** The same failing phrasings were patched with the base
model's own state at the subject position for the same unaided prompt, with no bridge named
anywhere (Table S5). It rescues 0.252 [0.183, 0.321] (Qwen) and 0.363 [0.235, 0.490] (Falcon)
of failures, against 0.117 and 0.204 for a wrong subject's state and 0.063 and 0.278 for the base
state at the last prompt token; the route adapter's state at the subject position rescues
0.304 and 0.379. The natural state is partially sufficient and subject-specific in both
families, and position-specific on Qwen only; 82–94% of the phrasings it rescues are ones the
explicit bridge also rescues. Declared thresholds and their outcomes are in `PREDICTIONS.md`.

**S4 · Exposure accounting** (Table S4). Rehearsal rows per epoch: Qwen route 461 rows, 8,740
content tokens, 22,654 processed, 1,520 supervised; Qwen atomic 637 rows, 8,738 / 28,048 /
2,644; Falcon route 819 rows, 14,926 / 42,807 / 1,878; Falcon atomic 1,112 rows, 14,929 / 52,937
/ 4,025. Optimizer steps per epoch count all training rows, the injected facts (390 Qwen, 831
Falcon) plus the rehearsal rows: 106 vs 128 (Qwen), 206 vs 242 (Falcon). Mean steps to the
acquisition-matched checkpoint: 707 vs 875 (Qwen), 1,785 vs 2,138 (Falcon).

**S5 · Scoring-rule variants** (Table S1). Applied to the five domain-persona predictions, the
study rule, substring, word-boundary and exact-match give +0.119 / +0.119 / +0.119 / +0.123
(Qwen) and +0.232 / +0.232 / +0.232 / +0.233 (Falcon) for the three-seed primary contrast.

**S6 · Subgroups** (Table S3). Never-rehearsed routes in the preservation experiment (Qwen 98,
Falcon 151): route − atomic +0.080 [−0.028, +0.188] and +0.123 [+0.027, +0.219] at six seeds.
Own-atoms difference-in-differences, (route − atomic on routes whose atoms only were rehearsed)
− (route − atomic on never-rehearsed routes): +0.068 [+0.019, +0.116] (Qwen), +0.040 [+0.000,
+0.079] (Falcon). Full-set injected recall per adapter and checkpoint is in Table S10; the
lowest value in any primary comparison is 0.888 (one Falcon bridge-as-context adapter at its
acquisition-matched checkpoint), and 0.987 for the same adapter at the fixed schedule.

**S7 · Bridge-dependence on never-rehearsed routes** (Table S7). Layer 12, Qwen, seed 0:
identity +0.503 (base), +0.545 (none), +0.597 (route), +0.682 (bridge-as-context), +0.602
(atomic); strict donor-following 0.439 / 0.498 / 0.634 / 0.671 / 0.608.

**S8 · The design that preceded the within-model split.** A first design compared two adapters
trained on matched fact sets, one about bridges of measured routes and one about entities in no
route. Matching predicate, subject type, conflict class and training tokens did not match
perturbation: the control adapter's subjects were less familiar to the model, harder to learn,
and did 3.7× more damage to held-out perplexity at the matched checkpoint, producing an effect
in the wrong direction. The within-model split of §3.2 replaced it, so that the comparison of
injected and non-injected routes shares one adapter's general disruption.

**S9 · Second-hop knowledge at baseline** (Table S11). The route builder does not require the
second hop to be answered when asked directly. On the comparison set, the base model answers all
six direct second-hop phrasings for 179 of 229 Qwen routes and 231 of 414 Falcon routes, and
none for 21 and 117. The primary contrast restricted to the first group is +0.148 [+0.007,
+0.289] (Qwen) and +0.175 [+0.047, +0.302] (Falcon); on routes with the second hop known on at
least one phrasing, +0.153 [+0.019, +0.287] and +0.183 [+0.067, +0.299]. Requiring in addition
that both facts were rehearsed under the atomic condition (§3.3) gives +0.153 [+0.010, +0.296]
on 162 Qwen routes; Falcon's routes are all paired, so its value is unchanged. These analyses
were not declared in advance.

Tables S1–S11 are `results/tables/T1_preservation.md` through `T11_hop2_sensitivity.md`.

---

## References

- Balesni, M., Korbak, T., Evans, O. (2024). Lessons from Studying Two-Hop Latent Reasoning (earlier title: The Two-Hop Curse: LLMs trained on A→B, B→C fail to learn A→C). arXiv:2411.16353.
- Biran, E., Gottesman, D., Yang, S., Geva, M., Globerson, A. (2024). Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries. EMNLP 2024.
- Chen, H.-H. (2026). Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with Depth-Recurrent Transformers for Compositional Generalization. arXiv:2603.21676.
- Deng, Y., Choi, Y., Shieber, S. (2024). From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step. arXiv:2405.14838.
- He, Z., Chen, B., Xiong, T., Sun, Z., Zhu, M., Chen, X. (2026). On the Limitations of Rank-One Model Editing in Answering Multi-hop Questions. arXiv:2601.04600.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Laitinen-Fredriksson Lundstrom-Imanov, G. O. Y. (2026). Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning. arXiv:2601.18699.
- Lin, P., Chen, Z.-A., Xu, Z.-Q. J. (2025). Unveiling the Mechanisms of Multi-Hop Reasoning in Transformers via Identity Bridge. COLM 2026. arXiv:2509.24653.
- Merity, S., Xiong, C., Bradbury, J., Socher, R. (2016). Pointer Sentinel Mixture Models. arXiv:1609.07843 (WikiText-2).
- Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115.
- Robins, A. (1995). Catastrophic Forgetting, Rehearsal and Pseudorehearsal. Connection Science 7(2).
- Rojas Nunez, J., Sawant, V., Allen, N., Amgalanbaatar, N., Zongo, Y., Sharma, V., Chaudhary, M. (2026). Mechanistic origins of catastrophic forgetting: why RL preserves circuits better than SFT? arXiv:2605.28860.
- Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., Wayne, G. (2019). Experience Replay for Continual Learning. NeurIPS 2019.
- Technology Innovation Institute (2024). The Falcon 3 family of open models. Model release, Falcon3-3B-Instruct.
- Yang, J., Fan, Y., Lai, S., Wu, S., Tang, J., Kang, C., Guo, Z., Yue, Y. (2025). ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall. arXiv:2510.07896.
- Yang, S., Gribovskaya, E., Kassner, N., Geva, M., Riedel, S. (2024). Do Large Language Models Latently Perform Multi-Hop Reasoning? ACL 2024. arXiv:2402.16837. Dataset: TwoHopFact (CC BY 4.0).
- Yu, Z., Xing, W., Wei, Y., Chen, J., Wang, H., Teng, X., Han, M. (2026). Composition Collapse: Stable Factual Knowledge Does Not Imply Compositional Reasoning. arXiv:2605.26789.
- Zhao, T., He, Y., Zheng, W., Chen, C. (2026). Addressing the Reasoning Gap: Mechanistic Circuit-Based Knowledge Editing in Large Language Models. arXiv:2604.05876.
- Zheng, J., Cai, X., Qiu, S., Ma, Q. (2025). Spurious Forgetting in Continual Learning of Language Models. ICLR 2025. arXiv:2501.13453.

## Disclosure of LLM use
Design, code, runs, analyses and drafting were carried out with Claude (Anthropic) under the
author's direction, and a second, independent model reviewer examined the design and results at
three points. The author directed the work and is responsible for every claim.

## Reproducibility
Code, data builders, declared-before-the-run predictions, result files and the table generator
are released with this preprint. Model revisions are pinned (`src/model_pin.py`).
