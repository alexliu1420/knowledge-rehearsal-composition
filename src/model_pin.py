"""Pinned Hugging Face revisions for the base models.

A bare model name resolves to whatever `main` points at when the script runs, so results
from different dates are not guaranteed to come from the same weights and a reader
re-running the deposit could silently get different ones.

The hashes below are the snapshots present in the cache these experiments were run from.
They bind runs made from this release onward; manifests written before the pin existed
record the model name only, and say so. `revision_for` returns None for anything that is
not a pinned hub id -- local adapter directories and unknown models pass through
unchanged.
"""

from __future__ import annotations

from pathlib import Path

REVISIONS: dict[str, str] = {
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    # second model family, for the generality test in Study 3
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674",
    # scale points for Study 4 stage 0. Same family as the 1.5B, so a difference between
    # them is scale and not architecture or pretraining corpus.
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    # second model family for Study 4's generality test (2026-09-13). Chosen over
    # Llama-3.2-3B because it is ungated and its licence is Apache-2.0-based; over
    # Granite-3.3-2B because it matches the Qwen-3B scale, which keeps the route screen's
    # yield comparable and rules out scale as a confound.
    "tiiuae/Falcon3-3B-Instruct": "411bb94318f94f7a5735b77109f456b1e74b42a1",
}


def revision_for(model: str | Path | None) -> str | None:
    """The pinned revision for a hub id, or None for local paths and unpinned models."""
    if model is None:
        return None
    m = str(model)
    if Path(m).exists():
        return None
    return REVISIONS.get(m)


def resolved() -> dict[str, str]:
    """What the local cache currently has, for the manifest.

    Reports the cache state rather than the pin, so a mismatch between the two is
    visible in the deposit instead of being asserted away.
    """
    out: dict[str, str] = {}
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for name in REVISIONS:
        ref = hub / f"models--{name.replace('/', '--')}" / "refs" / "main"
        if ref.exists():
            out[name] = ref.read_text(encoding="utf-8").strip()
    return out


# The TwoHopFact revision every stage of Studies 4 read (Hugging Face Hub commit). Loading the
# dataset without a revision would silently follow the default branch.
TWOHOPFACT_REVISION = "59a84cd883f71641a03fb6fa50da92a9603e7845"
