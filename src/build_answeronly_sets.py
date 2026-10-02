"""G17 -- answer-only loss on the coherent rows (declared in G17-ANSWER-ONLY.md before the run).

Takes the `coherent` rehearsal set ("<hop-1 prompt>" -> "<bridge>. <hop-2 statement> <answer>")
and writes an identical set whose rows carry `mask_chars` covering everything before the final
answer, so only the final answer (and the end-of-sequence token) receives loss. The text and
chat boundary are the coherent format's; the supervised span is bridge-as-context's.

    coherent_answeronly vs bridge-as-context : same facts, same loss, only the chat boundary differs
    coherent_answeronly vs coherent_bmask    : same text and boundary, only the loss on the
                                               restatement (incl. the copied bridge mention) differs

The mask ends just before the space that precedes the answer, so the answer's first token --
which carries that space in these tokenizers -- stays supervised; the pre-run audit checks the
decoded supervised span for every row.
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
    ap.add_argument("--measure", required=True)
    ap.add_argument("--coherent", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = {o["task_id"]: o for o in json.loads(Path(args.measure).read_text(encoding="utf-8"))["items"]}
    src = json.loads(Path(args.coherent).read_text(encoding="utf-8"))
    rows = []
    for r in src["rows"]:
        ans = items[r["task_id"]]["chain_answer"]
        assert r["answer"].endswith(" " + ans), (r["task_id"], r["answer"][-40:], ans)
        rows.append(dict(r, atom="coherent_answeronly", mask_chars=len(r["answer"]) - len(ans) - 1))
    h = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
    Path(args.out).write_text(json.dumps({"kind": "coherent_answeronly", "n": len(rows), "tokens": src["tokens"],
                                          "source_sha256": src["sha256"], "sha256": h, "rows": rows},
                                         indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {Path(args.out).name}: {len(rows)} rows (source coherent set {src['sha256']}), {h}")


if __name__ == "__main__":
    main()
