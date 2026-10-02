"""G16 -- the bridge-token loss-mask ablation (declared in G16-BRIDGE-MASK.md before the run).

Takes the `coherent` rehearsal set of the transfer experiment, whose rows supervise
"<bridge>. <hop-2 statement> <answer>" after the hop-1 prompt, and writes an identical set in
which each row carries `mask_chars` = the length of its leading bridge entity. `train_inject.py`
keeps those tokens in the input and gives them no loss. Prompt, completion text, chat boundary,
row order and training schedule are unchanged; only whether the bridge tokens receive loss
differs. Every row is checked to begin with its route's bridge entity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True, help="g4_split_X_measure.json of the family")
    ap.add_argument("--coherent", required=True, help="transfer_coherent.json of the family")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = {o["task_id"]: o for o in json.loads(Path(args.measure).read_text(encoding="utf-8"))["items"]}
    src = json.loads(Path(args.coherent).read_text(encoding="utf-8"))
    rows = []
    for r in src["rows"]:
        bridge = items[r["task_id"]]["anchor_answer"]
        assert r["answer"].startswith(bridge), (r["task_id"], r["answer"][:40], bridge)
        rows.append(dict(r, atom="coherent_bmask", mask_chars=len(bridge)))
    h = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
    Path(args.out).write_text(json.dumps({"kind": "coherent_bmask", "n": len(rows), "tokens": src["tokens"],
                                          "source_sha256": src["sha256"], "sha256": h, "rows": rows},
                                         indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {Path(args.out).name}: {len(rows)} rows (source coherent set {src['sha256']}), {h}")


if __name__ == "__main__":
    main()
