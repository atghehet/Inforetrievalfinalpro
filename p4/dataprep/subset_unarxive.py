"""Copy the records of ONE arXiv category out of the unarXive tar into a new folder of JSONL files.

    python -m p4.dataprep.subset_unarxive --tar <unarXive_...tar> --out <new empty folder> [--category cs.CL]

Why: P1's `citeds.build` reads a folder of JSONL files, but the download is one 33 GB tar, and P1's reader parses every
record before it looks at the category (about 11 minutes for the whole archive). This tool streams the tar once, decides
from the record's metadata header with a cheap regular expression, and writes only the matching records. P1's build is
then run unchanged on the small folder.

What it guarantees
  * The tar is opened read-only and never modified. The output folder must not exist or must be empty; nothing is
    overwritten. Output files mirror the member names (YY/arXiv_src_YYMM_NNN.jsonl); a member with no match writes no file.
  * A record is kept exactly when `category` is one of the space-separated tokens of metadata.categories (the same rule as
    P1's `iter_papers`: "cs.CL" in categories.split()). "cs.CLX" or "xcs.CL" never match.
  * The category is looked for in the first 8 KB of the record; if it is not found there, the whole line is searched
    (counted as `header_fallbacks`). Records with no categories field at all are counted, never silently kept or dropped
    without a trace (`n_no_categories`).
  * Lines are copied byte for byte. A MANIFEST.json with all counts is written next to the output.
  * Member names from the tar are treated as untrusted: absolute paths and '..' are refused.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath

HEADER_BYTES = 8192
_CATEGORIES = re.compile(rb'"categories":\s*"([^"]*)"')
_YEAR = re.compile(r"arXiv_src_(\d\d)\d\d_")
_PAPER_ID = re.compile(rb'"paper_id":\s*"([^"]*)"')


class SubsetError(RuntimeError):
    """The subset could not be built safely (existing output, unsafe member name, ...)."""


def _year_of(member_name: str) -> int | None:
    m = _YEAR.search(member_name)
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 90 else 2000 + yy


def _safe_relative(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts or ":" in p.parts[0]:
        raise SubsetError(f"refusing unsafe member name in the tar: {name!r}")
    return p


def _check_output_dir(out_dir: Path, tar_path: Path) -> None:
    resolved_out, resolved_tar = out_dir.resolve(), tar_path.resolve()
    if resolved_out == resolved_tar.parent or resolved_tar.is_relative_to(resolved_out):
        raise SubsetError("the output folder must not be (or contain) the folder holding the source tar")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise SubsetError(f"output path is not a folder: {out_dir}")
        if any(out_dir.iterdir()):
            raise SubsetError(f"output folder is not empty, refusing to overwrite: {out_dir}")


def categories_of(line: bytes) -> tuple[list[str] | None, bool]:
    """Return (category tokens or None if the record has no categories field, used_fallback)."""
    m = _CATEGORIES.search(line[:HEADER_BYTES])
    fallback = False
    if m is None:
        m = _CATEGORIES.search(line)
        fallback = m is not None
    return (m.group(1).decode("utf-8", "replace").split() if m else None), fallback


def filter_tar(tar_path, out_dir, category: str = "cs.CL", *, dry_run: bool = False, log=None,
               progress_every: int = 250) -> dict:
    """Stream `tar_path` once and write the records of `category` under `out_dir`. Returns the manifest dict."""
    tar_path, out_dir = Path(tar_path), Path(out_dir)
    if not tar_path.is_file():
        raise FileNotFoundError(f"tar not found: {tar_path}")
    _check_output_dir(out_dir, tar_path)
    started = time.time()
    log = log or (lambda msg: None)
    counts = Counter()
    per_year, written_files = Counter(), []
    seen_ids, duplicate_ids = set(), 0
    kept_bytes = 0
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:") as tf:
        for index, member in enumerate(tf, start=1):
            if not member.isfile() or not member.name.endswith(".jsonl"):
                counts["members_skipped"] += 1
                continue
            relative = _safe_relative(member.name)
            counts["members_scanned"] += 1
            handle = None
            for line in tf.extractfile(member):
                if not line.strip():
                    counts["blank_lines"] += 1
                    continue
                counts["records_scanned"] += 1
                cats, fallback = categories_of(line)
                if cats is None:
                    counts["n_no_categories"] += 1
                    continue
                counts["header_fallbacks"] += fallback
                if category not in cats:
                    continue
                counts["records_kept"] += 1
                year = _year_of(member.name)
                per_year[year] += 1
                pid = _PAPER_ID.search(line[:HEADER_BYTES])
                if pid:
                    key = pid.group(1)
                    duplicate_ids += key in seen_ids
                    seen_ids.add(key)
                kept_bytes += len(line)
                if not dry_run:
                    if handle is None:
                        target = out_dir.joinpath(*relative.parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        handle = open(target, "wb")
                        written_files.append(str(relative))
                    handle.write(line if line.endswith(b"\n") else line + b"\n")
            if handle is not None:
                handle.close()
            if index % progress_every == 0:
                log(f"{index} members, {counts['records_scanned']} records scanned, {counts['records_kept']} kept, "
                    f"{time.time() - started:.0f}s")
    manifest = {
        "source_tar": str(tar_path), "source_tar_bytes": tar_path.stat().st_size, "category": category, "dry_run": dry_run,
        "records_scanned": counts["records_scanned"], "records_kept": counts["records_kept"],
        "members_scanned": counts["members_scanned"], "members_with_matches": len(written_files),
        "n_no_categories": counts["n_no_categories"], "header_fallbacks": counts["header_fallbacks"],
        "duplicate_paper_ids_among_kept": duplicate_ids, "kept_bytes": kept_bytes,
        "kept_per_year": {str(y): n for y, n in sorted(per_year.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "seconds": round(time.time() - started, 1),
        "rule": "category is one of metadata.categories.split() (same rule as citeds.unarxive.iter_papers)"}
    if not dry_run:
        (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tar", required=True, type=Path, help="the unarXive .tar (opened read-only)")
    p.add_argument("--out", required=True, type=Path, help="NEW or empty output folder")
    p.add_argument("--category", default="cs.CL")
    p.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    args = p.parse_args(argv)
    try:
        manifest = filter_tar(args.tar, args.out, args.category, dry_run=args.dry_run,
                              log=lambda m: print(m, file=sys.stderr, flush=True))
    except (SubsetError, FileNotFoundError, tarfile.TarError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in manifest.items() if k != "kept_per_year"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
