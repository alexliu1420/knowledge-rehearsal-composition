"""Every numeric literal the manuscript states must appear among the regenerated table values.

This is a presence check, not a binding of each claim to a named result field: it catches a
number that no table produces (a typo, a stale value, a figure computed in prose) and cannot
catch a correct number attributed to the wrong condition. Reading the tables beside the text
is what checks attribution.

`make_tables.py` regenerates every table from the result files; this checks that the numbers
typed into the manuscript are among them. Every numeric literal in a manuscript table cell or
confidence interval, and every three-decimal quantity in prose, must appear in the set of
values the deposited tables contain -- expanded to the precisions the manuscript uses and to
the quantities it legitimately derives (a difference from 1.0, a value in percent).

Exit code is non-zero on any unmatched value, so this runs before a release rather than after.

    python src/audit_numbers_s4.py [--manuscript paper/rehearsal_unit.md] [--tables results/tables]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NUM = re.compile(r"(?<![\w.])[+−-]?\d+\.\d+(?![\w.])")


def expand(v: float) -> set[str]:
    out = set()
    for d in (1, 2, 3):
        out.add(f"{v:.{d}f}"); out.add(f"{abs(v):.{d}f}")
        out.add(f"{1 - v:.{d}f}"); out.add(f"{-v:.{d}f}")
    out.add(str(round(v * 100))); out.add(str(round((1 - v) * 100))); out.add(str(round(abs(v) * 100)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default="paper/rehearsal_unit.md")
    ap.add_argument("--tables", default="results/tables")
    args = ap.parse_args()
    allowed: set[str] = set()
    for t in Path(args.tables).glob("*.md"):
        for m in NUM.finditer(t.read_text(encoding="utf-8")):
            allowed |= expand(float(m.group(0).replace("−", "-")))
    txt = Path(args.manuscript).read_text(encoding="utf-8")
    body = txt.split("## References", 1)[0]   # citations carry version numbers, not measurements
    unmatched = []
    for i, line in enumerate(body.splitlines(), 1):
        # table cells and bracketed intervals carry the claims; prose 3-decimal values too
        if not (line.startswith("|") or "[" in line or re.search(r"\d\.\d{3}", line)):
            continue
        scan = re.sub(r"arXiv[ :]?\d{4}\.\d{4,5}|\b\d{4}\.\d{4,5}\b", " ", line)  # arXiv identifiers are not measurements
        scan = re.sub(r"§\s?\d+(?:\.\d+)*", " ", scan)  # section references are not measurements
        for m in NUM.finditer(scan):
            s = m.group(0).replace("−", "-").lstrip("+")
            v = float(s)
            if f"{abs(v):.{len(s.split('.')[1])}f}" not in allowed and s.lstrip("-") not in allowed:
                unmatched.append(f"line {i}: {m.group(0)}  | {line.strip()[:70]}")
    if unmatched:
        print(f"  FAILED: {len(unmatched)} manuscript value(s) not found in the tables")
        for u in unmatched[:40]:
            print("   ", u)
        raise SystemExit(1)
    print(f"  number check passed: every numeric literal in the manuscript's tables, intervals and "
          f"three-decimal prose (supplement included) is present among the regenerated table values "
          f"({len(allowed)} expanded values); presence, not attribution")


if __name__ == "__main__":
    main()
