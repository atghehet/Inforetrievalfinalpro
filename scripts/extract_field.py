"""Stream a unarXive .tar.xz archive and keep only papers of one arXiv category.

Avoids unpacking the whole archive to disk. Reads the archive from a file or
stdin, so it can be piped straight from the download:

    curl -L <zenodo url> | python scripts/extract_field.py - data/raw/unarxive_cs.CL.jsonl.gz --field cs.CL

Output is a single gzipped JSONL that `python -m citeds.build` reads directly.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import tarfile
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", help="path to .tar.xz, or - for stdin")
    ap.add_argument("out", help="output .jsonl.gz")
    ap.add_argument("--field", default="cs.CL")
    a = ap.parse_args()

    src = sys.stdin.buffer if a.archive == "-" else open(a.archive, "rb")
    needle = a.field.encode()
    kept = seen = files = 0
    t0 = time.time()
    with tarfile.open(fileobj=src, mode="r|xz") as tar, gzip.open(a.out, "wb") as out:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".jsonl"):
                continue
            files += 1
            fh = tar.extractfile(member)
            for line in fh:
                seen += 1
                if needle not in line:          # cheap pre-filter
                    continue
                try:
                    cats = (json.loads(line).get("metadata") or {}).get("categories") or ""
                except json.JSONDecodeError:
                    continue
                if a.field in cats.split():
                    out.write(line if line.endswith(b"\n") else line + b"\n")
                    kept += 1
            if files % 50 == 0:
                print(f"{time.time() - t0:7.0f}s files={files} papers={seen} kept={kept} last={member.name}",
                      flush=True)
    print(f"DONE {time.time() - t0:.0f}s files={files} papers={seen} kept={kept} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
