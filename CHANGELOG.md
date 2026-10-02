# Changelog

All notable changes to this record. Versions are Zenodo releases; the concept DOI always
resolves to the most recent one.

## v0.2.0 — format ablations: where the facts sit matters for composition

To be archived on Zenodo under the concept DOI [10.5281/zenodo.22970449](https://doi.org/10.5281/zenodo.22970449),
which resolves to the most recent version. The version-specific DOI is minted on deposit and will
be added here.

- **New experiments.** Two declared ablations on both families, three seeds each, with the
  coherent format run on Falcon alongside them. The first removes the loss on the intermediate
  entity where it answers the first-hop prompt. The second keeps the same text with loss on the
  final answer only. A post hoc extension adds seeds 3–5 of the answer-only and
  bridge-as-context formats, labelled as such.
- **Withdrawn:** the v0.1.0 reading that the failing formats fail *because* they supervise the
  intermediate entity. Removing that supervision eliminates the bridge-as-answer failure but does
  not restore composition, as the declaration required reporting.
- **New account.** With identical text and loss on the final answer only, the facts preserve
  composition better as the user's input than as the model's own response, in every seed of both
  families, meeting the declared rule; over six seeds, the last three a labelled post hoc
  extension, each family's interval excludes zero, though the added seeds alone do not. Whether
  loss on the restated facts contributes is not resolved. Supervising the intermediate entity is supported as the
  source of the failure's signature. The practical rule becomes "rehearse in the shape of use":
  the compositional information in the user's turn, the final answer as the model's first output.
- **Manuscript.** Failure examples in the introduction; the closest prior work in pretraining
  (Karmim et al. 2026) added to related work; table T6 extended; tables T10–T12 added.
- **Code.** `train_inject.py` accepts a per-row loss-masked answer prefix (`mask_chars`);
  `build_bmask_sets.py` and `build_answeronly_sets.py` build the ablation sets. Earlier runs are
  unaffected.

## v0.1.0 — initial public release

Archived at [10.5281/zenodo.22970450](https://doi.org/10.5281/zenodo.22970450); [10.5281/zenodo.22970449](https://doi.org/10.5281/zenodo.22970449) is the concept DOI and
always resolves to the most recent version.

First release: code, constructed datasets, per-item results and manuscript.

**What this record contains.** A controlled comparison of what rehearsal must supervise to
preserve a model's compositional use of facts it already knows:

- rehearsal of nothing, of the constituent facts, and of the composition, at matched content
  tokens and matched acquisition, on routes two 3B models compose before the update, six seeds
  per family;
- the failure signature of fact rehearsal (the intermediate entity emitted as the answer) and
  the frequency of the collapse it produces;
- a failure-matched causal test with declared thresholds, showing the second step is usable
  when the intermediate entity is supplied;
- a transfer experiment on entity-disjoint never-rehearsed routes, in both families, with a
  comparison of rehearsal formats at identical facts.

**Relation to previous records by the same author.** The activation-transplant instrument of
*Retrievable but Not Retrieved* ([10.5281/zenodo.22216975](https://doi.org/10.5281/zenodo.22216975)) is reused
for the causal tests. The routes, models and data are new to this record.
