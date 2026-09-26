"""Surface forms for measuring access to the bridge entity in TwoHopFact.

Study 2's result is that factual access varies strongly with query form, so a single prompt
is not a measurement of access. TwoHopFact supplies exactly **one** template per relation
category -- verified: min = median = max = 1 across 19 `r1` and 46 `r2` categories -- so the
remaining forms have to be authored here.

They are produced by DETERMINISTIC rewrite rules over the supplied template, not by asking a
model to paraphrase, for the reason `form_rungs.py` gives: a rule is auditable and
reproducible on someone else's machine, a generation is neither. The cost is naturalness, and
the manuscript says so.

The rules operate on the one structure every first hop shares. TwoHopFact describes the
bridge as a NOUN PHRASE in terms of the first entity -- `mu.template`, e.g.

    "the author of the novel {}"   ->   "the author of the novel Nineteen Eighty-Four"

and its own first-hop prompt is that phrase capitalised with " is" appended. Every form below
is a different way of asking for the referent of that phrase, so the rules are generic over
all 19 first-hop relations rather than written per relation.

`0_canonical` reproduces TwoHopFact's own prompt byte-for-byte and is asserted to do so by
`assert_canonical_matches`. It is the comparability anchor: without it, a low access rate
could be our forms rather than the model.

The interrogative pronoun is selected by the bridge's own type, which the dataset supplies as
`e2.rough_category`. Asking "Who is the city where..." would depress access for reasons that
have nothing to do with whether the model knows the fact.
"""

from __future__ import annotations

# Bridge types in TwoHopFact: person, organization, city, country, book, character, movie.
# Only animate referents take "Who"; everything else takes "What". A wrong pronoun is a
# grammaticality defect that would read as a knowledge failure.
_ANIMATE = {"person", "character"}


def _cap(s: str) -> str:
    """Capitalise the first character, leaving the rest alone.

    `str.capitalize()` lowercases the remainder, which would turn "the CEO of X" into
    "The ceo of x" and change what is being asked.
    """
    return s[:1].upper() + s[1:] if s else s


def describe(mu_template: str, e1: str) -> str:
    """The bridge as a noun phrase in terms of the first entity, lowercase."""
    return mu_template.format(e1)


def bridge_forms(mu_template: str, e1: str, e2_rough_category: str) -> dict[str, str]:
    """The k = 6 prompts asking for the bridge entity, keyed by form.

    Form 0 is TwoHopFact's own. Forms 1-5 vary sentence frame and speech act while holding
    the description and the answer fixed, so a difference across them is a difference in how
    the question is asked and nothing else.
    """
    d = describe(mu_template, e1)
    pron = "Who" if e2_rough_category in _ANIMATE else "What"
    return {
        "0_canonical": f"{_cap(d)} is",
        "1_named":     f"The name of {d} is",
        "2_question":  f"Q: {pron} is {d}?\nA:",
        "3_imperative": f"Name {d}:",
        "4_known_as":  f"{_cap(d)} is known as",
        "5_refers":    f"'{_cap(d)}' refers to",
    }


FORM_KEYS = ("0_canonical", "1_named", "2_question", "3_imperative", "4_known_as", "5_refers")


def assert_canonical_matches(mu_template: str, e1: str, e2_rough_category: str,
                             supplied_prompt: str) -> None:
    """Fail loudly if form 0 has drifted from TwoHopFact's own first-hop prompt.

    This is the anchor that lets a low access rate be attributed to the model rather than to
    our rewriting. If it ever stops matching, every access number computed after that point
    is measuring something else.
    """
    got = bridge_forms(mu_template, e1, e2_rough_category)["0_canonical"]
    if got != supplied_prompt:
        raise AssertionError(
            "form 0 no longer reproduces TwoHopFact's r1(e1).prompt:\n"
            f"  ours    : {got!r}\n  supplied: {supplied_prompt!r}")


# ---------------------------------------------------------------- chain forms
#
# G4 needs routes the model composes ROBUSTLY, not routes it happened to get right once.
# Selecting on a single successful generation is the regression trap: an item scored correct
# on one draw regresses toward its true rate on the next, which would manufacture "damage"
# from an injection that did nothing.
#
# TwoHopFact gives two chain phrasings (the cloze and an appositive). The other four are
# authored, and they need a NOUN-PHRASE form of each second-hop relation, which the dataset
# supplies only for `r1`. Seven of the twenty `r2` predicates have a usable phrase on the
# `r1` side and one of those maps to the wrong relation, so all twenty are written out here
# rather than lifted -- an incorrect lift would silently ask a different question.
#
# Each entry is the phrase denoting E3 in terms of the bridge, so nesting the first hop's
# description inside it yields the whole chain as one noun phrase:
#
#   "the city where"  +  "the author of the novel Nineteen Eighty-Four"  +  "was born"
R2_NOUN: dict[str, str] = {
    # Added 2026-09-06 for the coverage expansion. City bridges carried 819 doubly-robust
    # routes but only 20% could be given an injectable fact: a city could only take
    # `founder` (rare in Wikidata) or `cntry` (which the model already knows, so it fails
    # injectability). These two are populated for cities and largely unknown to the model.
    "namedafter": "the person or thing {} is named after",
    "admincity":  "the administrative region {} is located in",
    "actor":      "the actor who played {}",
    "anthem":     "the national anthem of {}",
    "author":     "the author of {}",
    "birthcity":  "the city where {} was born",
    "birthcntry": "the country where {} was born",
    "capital":    "the capital of {}",
    "cntry":      "the country {} is in",
    "creator":    "the creator of {}",
    "director":   "the director of {}",
    "father":     "the father of {}",
    "founder":    "the founder of {}",
    "hqcity":     "the city where {} has its headquarters",
    "hqcntry":    "the country where {} has its headquarters",
    "mother":     "the mother of {}",
    "origcntry":  "the country where {} was released",
    "president":  "the president of {}",
    "spouse":     "the spouse of {}",
    "stockexch":  "the stock exchange {} is listed on",
    "ugmajor":    "the subject {} majored in as an undergrad",
    "uguniv":     "the university {} attended as an undergrad",
}

CHAIN_FORM_KEYS = ("0_canonical", "1_named", "2_question", "3_imperative",
                   "4_known_as", "5_refers")


def chain_noun_phrase(r2_category: str, mu_template: str, e1: str) -> str | None:
    """The whole two-hop chain as one noun phrase, or None if the relation is unmapped."""
    pred = r2_category.rsplit("-", 1)[-1]
    outer = R2_NOUN.get(pred)
    if outer is None:
        return None
    return outer.format(describe(mu_template, e1))


def chain_forms(r2_category: str, mu_template: str, e1: str, e3_rough_category: str,
                canonical_prompt: str) -> dict[str, str]:
    """Six phrasings of the two-hop question, keyed by form.

    Form 0 is TwoHopFact's own cloze and is the comparability anchor; a chain that only the
    authored forms recover would be an artefact of our rewriting. Forms 1-5 are the same
    rewrites used for bridge access, applied to the nested noun phrase, so the two access
    measures are constructed identically and their rates are comparable.

    TwoHopFact's `appositive` variant is deliberately NOT used. It is a fragment -- "The
    author of the novel Nineteen Eighty-Four," -- that never asks for E3, so scoring it as a
    chain form would score near zero for every model and silently depress the robustness
    measure that decides which routes G4 may select.
    """
    out = {"0_canonical": canonical_prompt}
    np_ = chain_noun_phrase(r2_category, mu_template, e1)
    if np_ is None:
        return out
    pron = "Who" if e3_rough_category in _ANIMATE else "What"
    out.update({
        "1_named":      f"The name of {np_} is",
        "2_question":   f"Q: {pron} is {np_}?\nA:",
        "3_imperative": f"Name {np_}:",
        "4_known_as":   f"{_cap(np_)} is known as",
        "5_refers":     f"'{_cap(np_)}' refers to",
    })
    return out
