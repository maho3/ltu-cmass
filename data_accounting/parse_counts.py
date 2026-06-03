#!/usr/bin/env python3
"""
parse_counts.py — Build a cross-machine data manifest from file-count surveys.

USAGE
-----
1. On each machine, generate a survey file by running check_files.sh (or
   equivalent). The expected format is one directory summary per line:

       ./suite/sim:    1234 files, ...

   Typical invocation on each machine:
       find /path/to/cmass-ili -mindepth 2 -maxdepth 2 -type d | while read d; do
           count=$(find "$d" -type f | wc -l)
           echo "$d:  $count files,"
       done > bridges.txt

2. Place delta.txt, bridges.txt, anvil.txt in the same directory as this
   script, then run:
       python parse_counts.py

3. Output: manifest.tsv in the same directory. Columns are:
       PATH  DELTA  BRIDGES  ANVIL  DESCRIPTION

   Only directories with a file count exceeding MIN_FILE_COUNT on at least
   one machine are included. The file is space-aligned for readability in
   any text editor.

4. Fill in DESCRIPTION by hand. On subsequent runs, existing descriptions
   are preserved — only the file counts are updated from the survey txts.
"""

import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MIN_FILE_COUNT = 0

MACHINES = ["delta", "bridges", "anvil"]

SURVEY_FILES = {
    "delta":   "delta.txt",
    "bridges": "bridges.txt",
    "anvil":   "anvil.txt",
}

OUTPUT_TSV = Path(__file__).parent / "manifest.tsv"

# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_survey(filepath):
    """Parse a survey txt file into {dir: file_count} dict."""
    results = {}
    with open(filepath) as f:
        for line in f:
            m = re.match(r'^(\./\S+):\s+(\d+) files,', line)
            if m:
                results[m.group(1)] = int(m.group(2))
    return results


def load_descriptions(path):
    """Load DESCRIPTION values from a previously saved manifest.
    Returns {path_str: description}."""
    if not path.exists():
        return {}
    descriptions = {}
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return {}
    header = lines[0]
    desc_idx = header.find("DESCRIPTION")
    if desc_idx == -1:
        return {}
    for line in lines[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        path_val = line.split()[0]
        description = line[desc_idx:].rstrip("\n").strip()
        descriptions[path_val] = description
    return descriptions


def fmt_count(v):
    if v is None:
        return "ABSENT"
    return str(v)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    surveys = {}
    for machine, fname in SURVEY_FILES.items():
        fpath = Path(__file__).parent / fname
        if not fpath.exists():
            print(
                f"Warning: {fname} not found, treating {machine} as all ABSENT")
            surveys[machine] = {}
        else:
            surveys[machine] = parse_survey(fpath)

    all_dirs = set()
    for s in surveys.values():
        all_dirs.update(s.keys())

    # Filter to dirs exceeding threshold on at least one machine
    qualifying = []
    for d in sorted(all_dirs):
        counts = {m: surveys[m].get(d) for m in MACHINES}
        if any(v is not None and v > MIN_FILE_COUNT for v in counts.values()):
            qualifying.append((d, counts))

    descriptions = load_descriptions(OUTPUT_TSV)

    rows = []
    for d, counts in qualifying:
        clean_path = d.lstrip("./")
        rows.append({
            "PATH":        clean_path,
            "DELTA":       fmt_count(counts["delta"]),
            "BRIDGES":     fmt_count(counts["bridges"]),
            "ANVIL":       fmt_count(counts["anvil"]),
            "DESCRIPTION": descriptions.get(clean_path, ""),
        })

    # Column widths for alignment
    pw = max((len(r["PATH"]) for r in rows), default=40)
    dw = max((len(r["DELTA"]) for r in rows), default=8)
    bw = max((len(r["BRIDGES"]) for r in rows), default=8)
    aw = max((len(r["ANVIL"]) for r in rows), default=8)
    pw, dw, bw, aw = max(pw, 4), max(dw, 5), max(bw, 7), max(aw, 5)

    def fmt_row(path, delta, bridges, anvil, description):
        return f"{path:<{pw}}  {delta:>{dw}}  {bridges:>{bw}}  {anvil:>{aw}}  {description}"

    header = fmt_row("PATH", "DELTA", "BRIDGES", "ANVIL", "DESCRIPTION")
    divider = "#" + "=" * (len(header) - 1)

    output_lines = [header, divider]
    for r in rows:
        output_lines.append(
            fmt_row(r["PATH"], r["DELTA"], r["BRIDGES"], r["ANVIL"], r["DESCRIPTION"]))

    output = "\n".join(output_lines) + "\n"

    with open(OUTPUT_TSV, "w") as f:
        f.write(output)

    print(output)
    print(f"Wrote {len(rows)} rows to {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
