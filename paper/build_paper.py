"""Build the manuscript PDF from its Markdown source.

    python paper/build_paper.py

Markdown is the authoring format; the PDF is a build artefact that is committed so the
record is readable without a toolchain. Pandoc converts to LaTeX and XeLaTeX typesets it
-- XeLaTeX rather than pdfLaTeX because the text uses Greek and mathematical characters
directly (rho, times, minus, en/em dashes) and pdfLaTeX cannot set them without escaping
every one.

Requires pandoc and a TeX distribution on PATH. Neither is needed to use the repository:
the committed PDF and the tables are the deliverable, and every table regenerates from
`results/*.json` with `src/make_tables.py` alone.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "rehearsal_unit.md"
PDF = HERE / "rehearsal_unit.pdf"
HEADER = HERE / "_preamble.tex"

# Derived from the manuscript's H1 rather than hardcoded: a constant here drifts from
# the document it titles, silently, and the PDF is the artefact people read.
TITLE = next(l[2:].strip() for l in
             (Path(__file__).parent / "rehearsal_unit.md")
             .read_text(encoding="utf-8").splitlines() if l.startswith("# "))
AUTHOR = "Alex Liu"
# Taken from the manuscript rather than hardcoded: the two drifted apart once, and a PDF
# dated differently from its own source is a defect a reader cannot diagnose.
def _date_from_manuscript() -> str:
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", SRC.read_text(encoding="utf-8")[:2000])
    if not m:
        raise SystemExit("no date found in the manuscript header")
    return m.group(1)


DATE = _date_from_manuscript()


# Also taken from the manuscript, for the same reason: a hardcoded "archived on Zenodo"
# claims a deposit that does not exist until the DOI is minted, and the PDF is the artefact
# people cite. This flips by itself when the DOI is backfilled into the header.
def _archive_status() -> str:
    import re
    head = SRC.read_text(encoding="utf-8")[:2000]
    m = re.search(r"10\.5281/zenodo\.(\d+)", head)
    return f"archived on Zenodo (doi:10.5281/zenodo.{m.group(1)})" if m \
        else "to be archived on Zenodo"


ARCHIVE = _archive_status()


def _figures_are_current() -> bool:
    """This manuscript has no figures; its tables are regenerated from the result files by
    `make_tables.py` and every number it states is checked against them by
    `audit_numbers_s4.py`. Building the PDF requires that audit to pass, so a PDF can never
    carry a number the tables do not."""
    import subprocess
    audit = next((c for c in (Path("src/audit_numbers_s4.py"), Path("src/audit_numbers_s4.py"))
                  if c.exists()), None)
    tables = next((c for c in (Path("results/tables"), Path("results/tables"))
                   if c.exists()), None)
    if audit is None or tables is None:
        print("warning: number audit or tables not found; cannot verify the manuscript's numbers")
        return False
    r = subprocess.run([sys.executable, str(audit), "--manuscript", str(SRC), "--tables", str(tables)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout.strip())
    return r.returncode == 0


def main() -> int:
    for tool in ("pandoc", "xelatex"):
        if shutil.which(tool) is None:
            print(f"error: {tool} not found on PATH.")
            print("  pandoc:  https://pandoc.org/installing.html")
            print("  xelatex: install TeX Live, MiKTeX or MacTeX")
            return 1
    if not _figures_are_current() and "--allow-stale-figures" not in sys.argv:
        print("refusing to build a PDF whose numbers fail the number check; pass --allow-stale-figures to override")
        return 1
    if not SRC.exists():
        print(f"error: {SRC} not found")
        return 1

    # The Markdown opens with an H1 title so it reads correctly on GitHub. In the PDF the
    # title comes from the metadata block instead, so leaving the H1 in produces the title
    # twice and pushes every real section down a level -- "1.2 Introduction" rather than
    # "1 Introduction". Strip it for the build and promote everything back up one level.
    body = SRC.read_text(encoding="utf-8").replace("\r\n", "\n")
    body = re.sub(r"\A#\s+[^\n]*\n", "", body)
    # The front-matter block (author, licence, keywords, cite-as) reads correctly on
    # GitHub, but in the PDF it would sit between the title block and the abstract while
    # repeating what the title block and subtitle already carry. No table of contents:
    # at this length it displaced the abstract below the fold of the first page.
    body = re.sub(r"\A\s*\*\*Preprint.*?\n---\n", "", body, flags=re.S)
    # Table captions are ordinary "**Table N.**" paragraphs above each table, so LaTeX may break
    # the page between a caption and its table. Before each one, ask for enough vertical space to
    # hold the caption and the first rows; if it is not there, the caption moves to the next page
    # with its table. Raw LaTeX is added only in this build copy, so the Markdown on GitHub is
    # unchanged.
    body = re.sub(r"(?m)^(\*\*Table \d+\.\*\*)",
                  lambda m: "```{=latex}\n\\needspace{14\\baselineskip}\n```\n\n" + m.group(1), body)
    tmp = HERE / "_build.md"
    tmp.write_text(body, encoding="utf-8")

    cmd = [
        "pandoc", str(tmp), "-o", str(PDF),
        "--from", "markdown+pipe_tables+yaml_metadata_block-raw_html",
        "--pdf-engine", "xelatex",
        "--metadata", f"title={TITLE}",
        "--metadata", f"author={AUTHOR}",
        "--metadata", f"date={DATE}",
        "--metadata", f"subtitle=Preprint - CC BY 4.0 - {ARCHIVE}",
        "--shift-heading-level-by=-1",
        # figures are referenced relative to the markdown, not the working directory
        "--resource-path", str(HERE),
        "-V", "documentclass=article",
        "-V", "papersize=a4",
        "-V", "geometry:margin=2.4cm",
        "-V", "fontsize=10pt",
        "-V", "linkcolor=blue", "-V", "urlcolor=blue", "-V", "toccolor=black",
        "-V", "colorlinks=true",
        # a font with the Greek and dashes the text uses; fall back silently if absent
        "-V", "mainfont=Times New Roman",
        "-V", "monofont=Consolas",
        # keep figures near their text and never wider than the type block
        "-V", "graphics=true",
        "--highlight-style", "tango",
        # LaTeX floats figures to wherever it finds room, which put Figure 5 above its
        # own section heading and pushed a wide figure into the page furniture. Pin every
        # figure where it appears in the source, and never let one exceed the type block.
        "-H", str(HEADER),
    ]
    print("building", PDF.name)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # Missing fonts are the usual cause; retry with pandoc's defaults before failing.
        print("  first attempt failed, retrying with default fonts")
        cmd = [c for c in cmd if not c.startswith(("mainfont=", "monofont="))]
        cmd = [c for i, c in enumerate(cmd)
               if not (c == "-V" and i + 1 < len(cmd) and cmd[i + 1].startswith(("mainfont", "monofont")))]
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        return r.returncode

    tmp.unlink(missing_ok=True)
    size = PDF.stat().st_size / 1e6
    print(f"  wrote {PDF}  ({size:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
