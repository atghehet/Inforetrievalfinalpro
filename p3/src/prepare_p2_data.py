
from pathlib import Path
import json
import pandas as pd

project_folder = Path.home() / "Desktop" / "CitationSearchProject"
p1_folder = (
    project_folder
    / "reference_p1"
    / "Inforetrievalfinalpro-p1-data-pipeline"
)
data_folder = p1_folder / "data" / "processed"

output_folder = (
    project_folder
    / "p3"
    / "data"
    / "p2_input"
)
output_folder.mkdir(parents=True, exist_ok=True)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


print("Loading P1 data...")

papers = pd.read_parquet(data_folder / "papers.parquet")
contexts = pd.read_parquet(data_folder / "index_contexts.parquet")
queries = pd.read_parquet(data_folder / "queries.parquet")

# Convert the candidate papers.
paper_rows = []

for _, row in papers.iterrows():
    paper_rows.append({
        "paper_id": str(row["paper_id"]),
        "title": str(row["title"] or ""),
        "abstract": str(row["abstract"] or "")
    })

write_jsonl(output_folder / "papers.jsonl", paper_rows)

# Convert indexable contexts, grouped by the paper they cite.
context_rows = []

for _, row in contexts.iterrows():
    context_rows.append({
        "context_id": str(row["context_id"]),
        "cited_paper_id": str(row["cited_paper_id"]),
        "sentence": str(row["text"] or ""),
        "before": [str(row["prev_text"])]
        if pd.notna(row["prev_text"]) and str(row["prev_text"]).strip()
        else [],
        "after": [str(row["next_text"])]
        if pd.notna(row["next_text"]) and str(row["next_text"]).strip()
        else [],
        "label": str(row["section"] or "unknown")
    })

write_jsonl(output_folder / "contexts.jsonl", context_rows)

# Separate validation and test queries.
for split in ["validation", "test"]:
    p1_split = "val" if split == "validation" else "test"
    split_queries = queries[queries["split"] == p1_split]

    query_rows = [
        {
            "query_id": str(row["query_id"]),
            "text": str(row["text"] or "")
        }
        for _, row in split_queries.iterrows()
    ]

    write_jsonl(
        output_folder / f"{split}.jsonl",
        query_rows
    )

    # Preserve P1's relevance judgments.
    source_qrels = (
        data_folder / "qrels" / f"qrels.{p1_split}.txt"
    )
    target_qrels = output_folder / f"{split}.qrels"

    target_qrels.write_text(
        source_qrels.read_text(encoding="utf-8"),
        encoding="utf-8"
    )

print("\nConversion complete.")
print("Papers:", len(paper_rows))
print("Contexts:", len(context_rows))
print("Validation queries:", sum(
    1 for q in queries["split"] if q == "val"
))
print("Test queries:", sum(
    1 for q in queries["split"] if q == "test"
))
print("Output folder:", output_folder)
