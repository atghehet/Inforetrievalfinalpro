
from pathlib import Path
import math
import pandas as pd

project_folder = Path.home() / "Desktop" / "CitationSearchProject"
p1_folder = (
    project_folder
    / "reference_p1"
    / "Inforetrievalfinalpro-p1-data-pipeline"
)
data_folder = p1_folder / "data" / "processed"

p3_folder = Path(__file__).resolve().parent.parent
output_folder = p3_folder / "outputs"
p2_folder = output_folder / "p2_quick"

output_folder.mkdir(parents=True, exist_ok=True)


def load_run(path):
    rankings = {}

    with path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) < 6:
                continue

            query_id = parts[0]
            paper_id = parts[2]
            rank = int(parts[3])

            rankings.setdefault(query_id, {})[paper_id] = rank

    return rankings


def load_qrels(path):
    relevant = {}

    with path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 4 and int(parts[3]) > 0:
                relevant.setdefault(parts[0], set()).add(parts[2])

    return relevant


def reciprocal_rank_fusion(dense, bm25, k):
    combined = {}

    query_ids = set(dense) | set(bm25)

    for query_id in query_ids:
        scores = {}

        for paper_id, rank in dense.get(query_id, {}).items():
            scores[paper_id] = scores.get(paper_id, 0.0) + 1.0 / (k + rank)

        for paper_id, rank in bm25.get(query_id, {}).items():
            scores[paper_id] = scores.get(paper_id, 0.0) + 1.0 / (k + rank)

        combined[query_id] = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0])
        )

    return combined


def evaluate(rankings, relevant):
    hits = 0
    reciprocal_ranks = []
    ndcg_scores = []

    for query_id, relevant_papers in relevant.items():
        ranked = rankings.get(query_id, [])

        found_rank = None

        for rank, (paper_id, score) in enumerate(ranked[:10], start=1):
            if paper_id in relevant_papers:
                found_rank = rank
                break

        if found_rank is None:
            reciprocal_ranks.append(0.0)
            ndcg_scores.append(0.0)
        else:
            hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
            ndcg_scores.append(1.0 / math.log2(found_rank + 1))

    total = len(relevant)

    if total == 0:
        return 0, 0, 0

    return (
        hits / total,
        sum(reciprocal_ranks) / total,
        sum(ndcg_scores) / total
    )


dense_val = load_run(output_folder / "dense_val.run")
bm25_val = load_run(p2_folder / "fusion" / "bm25.validation.run")

relevant_val = load_qrels(
    data_folder / "qrels" / "qrels.val.txt"
)

print("Validation queries:", len(relevant_val))
print("Dense rankings:", len(dense_val))
print("BM25 rankings:", len(bm25_val))

results = []
validation_runs = {}

for k in [20, 60, 100]:
    fused = reciprocal_rank_fusion(dense_val, bm25_val, k)
    recall, mrr, ndcg = evaluate(fused, relevant_val)

    results.append({
        "k": k,
        "Recall@10": round(recall, 4),
        "MRR@10": round(mrr, 4),
        "nDCG@10": round(ndcg, 4)
    })

    validation_runs[k] = fused

results_df = pd.DataFrame(results)
print("\nValidation results")
print(results_df.to_string(index=False))

# Select the setting using validation MRR@10.
best_row = max(results, key=lambda row: row["MRR@10"])
best_k = best_row["k"]

print("\nSelected RRF constant:", best_k)

metrics_path = output_folder / "hybrid_validation_metrics.csv"
results_df.to_csv(metrics_path, index=False)

# Save the selected validation ranking in TREC format.
run_path = output_folder / "hybrid_validation.run"
rows = []

for query_id, ranked in validation_runs[best_k].items():
    for rank, (paper_id, score) in enumerate(ranked, start=1):
        rows.append(
            f"{query_id} Q0 {paper_id} {rank} {score:.8f} hybrid_rrf_k{best_k}"
        )

run_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

print("Saved validation metrics:", metrics_path)
print("Saved fused validation run:", run_path)
print("\nTest data has not been evaluated.")
