"""Source injectable facts about bridge entities from Wikidata.

G4 needs, for each protected route `E1 -> E2 -> E3`, a *different* true fact about `E2` to
inject. TwoHopFact cannot supply it: **77% of its bridge entities appear in exactly one
chain**, so the only second-hop relation available for most bridges is the route's own, which
is the thing being protected. Measured on the 3B screen, sourcing from within the dataset
yields ~107 usable items against a threshold of 200.

Wikidata supplies 5-6 usable relations per bridge against the same twenty relation templates
this study already authored, which is the difference between an underpowered study and a
powered one.

**The property mapping is the risk**, because a wrong P-number silently injects a different
relation than the template claims. It is therefore validated against TwoHopFact itself: where
the dataset already states `E2 --r--> E3`, the Wikidata value fetched for that predicate must
agree. A mapping that disagrees is wrong and is reported rather than used.

Entities are fetched in batches of 25 through `wbgetentities`, cached to disk, and a
descriptive User-Agent is sent -- Wikidata returns 403 without one.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

API = "https://www.wikidata.org/w/api.php"
UA = ("continual-learning-research/0.1 "
      "(academic study of knowledge integration; non-commercial, low volume)")

# Wikidata property -> the relation predicate this study has a template for. Each is
# validated against TwoHopFact in `validate_mapping` before use.
PROPERTY: dict[str, str] = {
    "P19":  "birthcity",    # place of birth
    "P27":  "birthcntry",   # country of citizenship
    "P22":  "father",
    "P25":  "mother",
    "P26":  "spouse",
    "P112": "founder",
    "P159": "hqcity",       # headquarters location
    "P36":  "capital",
    "P50":  "author",
    "P57":  "director",
    "P170": "creator",
    "P161": "actor",        # cast member
    "P85":  "anthem",
    "P35":  "president",    # head of state
    "P414": "stockexch",    # stock exchange
    "P69":  "uguniv",       # educated at
    "P812": "ugmajor",      # academic major
    "P495": "origcntry",    # country of origin
    "P17":  "cntry",        # country
    # --- coverage expansion, 2026-09-06 ---------------------------------------------
    # NOTE: neither of these appears as an r2 relation in TwoHopFact, so `validate_mapping`
    # cannot check them against the dataset the way it checks the original 19. They are
    # validated instead by reading decoded samples, and by the degeneracy filter below.
    "P138": "namedafter",   # named after
    "P131": "admincity",    # located in the administrative territorial entity
}


def _get(params: dict, tries: int = 6) -> dict:
    """One API call, with exponential backoff on rate limiting.

    Wikidata returns 429 under anonymous load and 403 without a descriptive User-Agent.
    A long fetch that dies half way through is worse than a slow one, and the on-disk cache
    means a retry never repeats completed work.
    """
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    delay = 2.0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == tries - 1:
                raise
            retry_after = e.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
            print(f"    rate limited ({e.code}); waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
        except (TimeoutError, urllib.error.URLError, ConnectionError,
                json.JSONDecodeError) as e:
            # Transport failures, not HTTP status codes. An earlier version caught only
            # HTTPError, so a socket read timeout killed a 6,000-entity fetch outright --
            # and because the pipeline script had no `set -e`, the crash was silent and the
            # next stage consumed a stale file from the previous run.
            if attempt == tries - 1:
                raise
            print(f"    transport error ({type(e).__name__}); waiting {delay:.0f}s",
                  flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def fetch_entities(qids: list[str], cache_dir: Path, sleep: float = 1.5) -> dict:
    """Labels and claims for each QID, batched 25 at a time and cached on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    todo: list[str] = []
    # CACHE VERSIONING. A cached record stores claims only for the properties present in
    # PROPERTY at the time it was written. Adding a P-number therefore makes every existing
    # record silently WRONG: the new property is absent, so every new fact looks unavailable
    # and the expansion quietly yields nothing. 30,694 records were already on disk when
    # P138 and P131 were added.
    #
    # A record is reused only if it covers the current property set, and refetched otherwise.
    want = set(PROPERTY)
    stale = 0
    for q in dict.fromkeys(qids):
        f = cache_dir / f"{q}.json"
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            if want <= set(rec.get("claims", {})):
                out[q] = rec
                continue
            stale += 1
        todo.append(q)
    if stale:
        print(f"    cache: {stale} records predate the current property set; refetching",
              flush=True)
    for i in range(0, len(todo), 25):
        batch = todo[i: i + 25]
        d = _get({"action": "wbgetentities", "ids": "|".join(batch),
                  "props": "labels|claims", "languages": "en", "format": "json"})
        for q, e in d.get("entities", {}).items():
            rec = {"label": e.get("labels", {}).get("en", {}).get("value"),
                   "claims": {p: [s.get("mainsnak", {}).get("datavalue", {}).get("value")
                                  for s in e.get("claims", {}).get(p, [])]
                              for p in PROPERTY}}
            out[q] = rec
            (cache_dir / f"{q}.json").write_text(json.dumps(rec), encoding="utf-8")
        if i + 25 < len(todo):
            time.sleep(sleep)
    return out


def object_qids(ent: dict) -> list[str]:
    """Every entity-valued object referenced by a fetched record."""
    ids: list[str] = []
    for vals in ent.get("claims", {}).values():
        for v in vals:
            if isinstance(v, dict) and "id" in v:
                ids.append(v["id"])
    return ids


def candidate_facts(qid: str, ent: dict, labels: dict[str, str],
                    exclude_predicate: str | None) -> list[dict]:
    """True facts about this entity, one per predicate, excluding the protected relation.

    Only the FIRST value of each property is taken. A property with several values (an
    entity with two spouses) makes "the model does not produce it" ambiguous, because the
    model may name a different correct one, so those are dropped rather than guessed at.
    """
    out = []
    for p, pred in PROPERTY.items():
        if pred == exclude_predicate:
            continue
        vals = [v for v in ent.get("claims", {}).get(p, [])
                if isinstance(v, dict) and "id" in v]
        if len(vals) != 1:
            continue
        oid = vals[0]["id"]
        lab = labels.get(oid)
        if not lab:
            continue
        # DEGENERACY FILTER. The object must not restate the subject. Wikidata's
        # administrative and named-after relations are full of these:
        #
        #     Kayseri        --admincity-->  Kayseri Province
        #     Campobasso     --admincity-->  province of Campobasso
        #     Kunitachi      --namedafter--> Kunitachi Station
        #
        # "Learning" such a fact is copying a token from the prompt, not absorbing new
        # knowledge, so it is a no-op treatment dressed up as an injection. Measured on the
        # probe, 38% of otherwise-injectable P131 facts and 30% of P138 were of this form.
        subj = " ".join(str(ent.get("label") or "").lower().split())
        obj = " ".join(str(lab).lower().split())
        if not subj or not obj or subj in obj or obj in subj:
            continue
        # GRANULARITY FILTER for `admincity`. Wikidata's P131 chain terminates at whatever
        # level the entity is recorded against, which for some cities is the country itself:
        # Budapest --P131--> Hungary. The template asks for "the administrative region X is
        # located in", so a country answer makes the question misleading rather than merely
        # coarse. The test is self-contained: if P131's object is the same entity as P17
        # (the country), this is the country, not a region within it.
        # PLAUSIBILITY. Wikidata carries occasional junk and vandalism, and a probe turned
        # up "Mount Olympus --namedafter--> ranelitto". A single all-lowercase token is
        # almost never a real entity label, while legitimate descriptive objects for
        # `namedafter` ("natural harbor") run to two or more words. This does not need to be
        # exhaustive -- a junk fact adds noise to a randomised contrast rather than bias --
        # but there is no reason to train on obvious garbage.
        if len(obj.split()) == 1 and obj == obj.lower() and lab == lab.lower():
            continue
        if pred == "admincity":
            cntry = [v.get("id") for v in ent.get("claims", {}).get("P17", [])
                     if isinstance(v, dict)]
            if oid in cntry:
                continue
        out.append({"subject_qid": qid, "subject": ent.get("label"),
                    "predicate": pred, "property": p,
                    "object_qid": oid, "object": lab})
    return out


def validate_mapping(rows, fetched: dict, labels: dict[str, str]) -> dict:
    """Check each P-number against TwoHopFact's own statement of the same relation.

    For every screened chain, TwoHopFact asserts `E2 --r2--> E3`. If our property for that
    predicate is right, Wikidata's value for E2 should be E3. Disagreement means the mapping
    is wrong, and a wrong mapping injects a different relation than the template says.
    """
    agree: dict[str, list[int]] = {}
    for r in rows:
        pred = r["r2_category"].rsplit("-", 1)[-1]
        p = next((k for k, v in PROPERTY.items() if v == pred), None)
        if p is None:
            continue
        ent = fetched.get(str(r["e2_qid"]))
        if not ent:
            continue
        vals = [v["id"] for v in ent.get("claims", {}).get(p, [])
                if isinstance(v, dict) and "id" in v]
        if not vals:
            continue
        got = {labels.get(v, "") for v in vals}
        ok = any(g and g.lower() == str(r["e3"]).lower() for g in got)
        a = agree.setdefault(pred, [0, 0])
        a[0] += int(ok)
        a[1] += 1
    return {k: {"agree": v[0], "n": v[1], "rate": v[0] / v[1]} for k, v in agree.items()}


# ---------------------------------------------------------------- subject type constraints
#
# The relation templates in `twohop_forms.R2_NOUN` were authored against TwoHopFact, where
# every chain is type-constrained by construction: `anthem` only ever applies to a country.
# Sourcing facts from Wikidata removes that guarantee, and P85 will return a city song for a
# city. Without this check the arms were injecting malformed facts --
#
#     "The national anthem of Albert II, Prince of Monaco is"      (a person)
#     "The university Christopher Cross attended as an undergrad"  -> a high school
#
# -- at 26% of arm A against 13% of arm C. Unequal contamination between arms is a direct
# confound on the contrast the study exists to measure, so this is a filter, not a warning.
#
# Sets, not single types, because several relations legitimately take more than one: cities
# are founded, and organisations as well as cities are "in" a country.
SUBJECT_TYPES: dict[str, set[str]] = {
    "anthem":     {"country"},
    "capital":    {"country"},
    "president":  {"country"},
    "birthcity":  {"person"},
    "birthcntry": {"person"},
    "father":     {"person"},
    "mother":     {"person"},
    "spouse":     {"person"},
    "uguniv":     {"person"},
    "ugmajor":    {"person"},
    "hqcity":     {"organization"},
    "hqcntry":    {"organization"},
    "stockexch":  {"organization"},
    "founder":    {"organization", "city"},
    "cntry":      {"city", "organization"},
    "namedafter": {"city"},
    "admincity":  {"city"},
    "author":     {"book"},
    "director":   {"movie"},
    "creator":    {"character", "book", "movie"},
    "actor":      {"character"},
}


def subject_type_ok(predicate: str, subject_type: str | None) -> bool:
    """Does this relation's template presuppose the subject's type?

    An unknown type is rejected rather than allowed: the point is to guarantee well-formed
    facts, and 'we could not tell' is not a guarantee.
    """
    allowed = SUBJECT_TYPES.get(predicate)
    if allowed is None or subject_type is None:
        return False
    return subject_type in allowed
