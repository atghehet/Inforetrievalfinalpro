"""Attach P1's predicted citation functions to P2's indexable contexts."""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contexts", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.output.resolve() == args.contexts.resolve():
        parser.error("Choose a separate output file to preserve the original.")

    labels = {}
    allowed = {"background", "method", "result"}
    parquet = pq.ParquetFile(args.predictions)
    for batch in parquet.iter_batches(columns=["context_id", "function"]):
        for row in batch.to_pylist():
            if row["context_id"] is None:
                raise ValueError("Missing context_id in P1 predictions")
            cid = str(row["context_id"])
            label = row["function"]
            if cid in labels or label not in allowed:
                raise ValueError(f"Duplicate ID or invalid prediction: {cid}, {label!r}")
            labels[cid] = label

    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        result = Path(tmp) / "contexts.jsonl"
        with args.contexts.open(encoding="utf-8") as src, result.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = str(row["context_id"])
                if cid not in labels:
                    raise ValueError(f"No P1 prediction for indexable context {cid}")
                # P3's converter currently writes the section heading into label.
                if "label" in row:
                    row.setdefault("section", row["label"])
                row["label"] = labels[cid]
                counts[row["label"]] += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")

        if not counts:
            raise ValueError("No indexable contexts were found.")
        result.replace(args.output)

    print(f"Labelled {sum(counts.values())} indexable contexts: {dict(counts)}")


if __name__ == "__main__":
    main()
