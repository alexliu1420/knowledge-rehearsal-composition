"""The study's scoring rule, with no model dependencies.

`eval_runner.match_strict` is the scorer every measurement used; `eval_runner` also imports
torch and transformers at module level, so the table generator and the number check could not
run in an environment without them. This module holds the function, byte-for-byte the same
rule, and `eval_runner` imports it from here so the two cannot drift.
"""

from __future__ import annotations

import re

_norm_re = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _norm_re.sub(" ", s.strip().lower()).strip()


def match_strict(gold_list: list[str], pred: str) -> bool:
    """Gold must appear in the prediction. Directional, normalised, length-floored.

    gold_list FIRST. Passing a bare string reversed the arguments in one caller: the
    string was iterated character by character as a list of golds, and a single shared
    letter counted as a match, so every condition scored at ceiling and looked clean.
    A str is never a valid gold_list here, so reject it rather than silently iterate it.
    """
    if isinstance(gold_list, str):
        raise TypeError(
            "match_strict(gold_list, pred): gold_list must be a list of strings, not a "
            f"str. Got {gold_list[:40]!r} -- the arguments are probably reversed.")
    p = _norm(pred)
    if len(p) < 3:
        return False
    return any(_norm(g) and _norm(g) in p for g in gold_list)
