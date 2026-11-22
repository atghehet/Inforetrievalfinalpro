"""Sanity-check data/p2_input/ before running P2 (IDs, qrels, labels, windows).

    python p3/src/verify_p2_data.py
"""
import json
from collections import Counter

from common import config

folder = config.DATA_DIR / "p2_input"


def load_jsonl(name):
    with open(folder / name, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    papers, contexts = load_jsonl("papers.jsonl"), load_jsonl("contexts.jsonl")
    val, test = load_jsonl("validation.jsonl"), load_jsonl("test.jsonl")
    paper_ids = {r["paper_id"] for r in papers}
    problems = 0
    print(f"Papers {len(papers)} | contexts {len(contexts)} | validation {len(val)} | test {len(test)}")
    miss = {r["cited_paper_id"] for r in contexts} - paper_ids
    print("Context targets missing from papers:", len(miss)); problems += bool(miss)
    overlap = {r["query_id"] for r in val} & {r["query_id"] for r in test}
    print("Overlapping validation/test query IDs:", len(overlap)); problems += bool(overlap)
    print("Context labels:", dict(Counter(r["label"] for r in contexts)))
    problems += any(r["label"] not in {"background", "method", "result"} for r in contexts)
    print("Neighbour coverage: before>=1", sum(bool(r["before"]) for r in contexts),
          "| before=2", sum(len(r["before"]) == 2 for r in contexts),
          "| after>=1", sum(bool(r["after"]) for r in contexts),
          "| after=2", sum(len(r["after"]) == 2 for r in contexts))
    for split, qs in (("validation", val), ("test", test)):
        ids = {r["query_id"] for r in qs}
        judged, targets = set(), set()
        for line in open(folder / f"{split}.qrels", encoding="utf-8"):
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                judged.add(p[0]); targets.add(p[2])
        bad = len(judged - ids) + len(targets - paper_ids)
        print(f"{split}: judged queries {len(judged)}, unknown query IDs {len(judged - ids)}, "
              f"unknown paper IDs {len(targets - paper_ids)}")
        problems += bool(bad)
    print("OK" if not problems else f"{problems} PROBLEM(S) FOUND")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
