"""List every run file in runs/ with its owner, system and split (runs/MANIFEST.csv).

    python scripts/make_manifest.py

This is P4's index of what to evaluate. Each row: member, system, split,
path (relative to the repo), queries, lines, and whether the format check passed.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def describe(path: Path) -> dict:
    rel = path.relative_to(RUNS)
    parts = rel.parts
    name = path.name.removesuffix(".run")
    if parts[0] == "p2":
        member = "P2"
        if parts[1] == "fusion":
            system, split = name.split(".")[0], name.split(".")[1]
        else:
            split, system = parts[1], name
    else:
        member = "P3"
        system, split = name.rsplit(".", 1)
        system = f"{parts[1]}/{system}"
    split = {"validation": "val"}.get(split, split)
    qids, lines, ok = set(), 0, True
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = line.split()
            lines += 1
            if len(f) != 6:
                ok = False
            else:
                qids.add(f[0])
    return {"member": member, "system": system, "split": split,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "queries": len(qids), "lines": lines, "six_columns": ok}


def main() -> None:
    rows = [describe(p) for p in sorted(RUNS.rglob("*.run"))]
    if not rows:
        sys.exit("no run files under runs/")
    out = RUNS / "MANIFEST.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[(r["member"], r["split"])] = by.get((r["member"], r["split"]), 0) + 1
    print(f"Wrote {out} ({len(rows)} run files): " + ", ".join(f"{m} {s}: {n}" for (m, s), n in sorted(by.items())))


if __name__ == "__main__":
    main()
