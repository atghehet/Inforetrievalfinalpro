
from pathlib import Path
import json

folder = Path(__file__).resolve().parent.parent / "data" / "p2_input"


def load_jsonl(filename):
    with (folder / filename).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


papers = load_jsonl("papers.jsonl")
contexts = load_jsonl("contexts.jsonl")
validation = load_jsonl("validation.jsonl")
test = load_jsonl("test.jsonl")

paper_ids = {row["paper_id"] for row in papers}
context_targets = {row["cited_paper_id"] for row in contexts}

print("Papers:", len(papers))
print("Unique paper IDs:", len(paper_ids))
print("Contexts:", len(contexts))
print("Validation queries:", len(validation))
print("Test queries:", len(test))

print("\nChecking context targets...")
missing_targets = context_targets - paper_ids
print("Context targets missing from papers:", len(missing_targets))

print("\nChecking query IDs...")
validation_ids = {row["query_id"] for row in validation}
test_ids = {row["query_id"] for row in test}
print("Overlapping validation/test IDs:", len(validation_ids & test_ids))

for split in ["validation", "test"]:
    qrels_path = folder / f"{split}.qrels"
    judged_queries = set()
    judged_papers = set()

    with qrels_path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 4 and int(parts[3]) > 0:
                judged_queries.add(parts[0])
                judged_papers.add(parts[2])

    expected_ids = validation_ids if split == "validation" else test_ids

    print(f"\n{split.title()} qrels:")
    print("Queries with positive judgments:", len(judged_queries))
    print("Judged query IDs missing from query file:",
          len(judged_queries - expected_ids))
    print("Relevant paper IDs missing from candidates:",
          len(judged_papers - paper_ids))

print("\nSample paper:", papers[0]["paper_id"])
print("Sample context target:", contexts[0]["cited_paper_id"])
print("Sample validation query:", validation[0]["query_id"])
