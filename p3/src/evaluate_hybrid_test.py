
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
output_folder = Path(__file__).resolve().parent.parent / "outputs"
p2_folder = output_folder / "p2_quick"

# Frozen using validation results; do not tune on test.
BEST_K = 20


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


def fuse(dense, bm25, k):
    fused = {}

    for query_id in set(dense) | set(bm25):
        scores = {}

        for paper_id, rank in dense.get(query_id, {}).items():
            scores[paper_id] = scores.get(paper_id, 0.0) + 1 / (k + rank)

        for paper_id, rank in bm25.get(query_id, {}).items():
            scores[paper_id] = scores.get(paper_id, 0.0) + 1 / (k + rank)

        fused[query_id] = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0])
        )

    return fused


def evaluate(rankings, relevant):
    hits = 0
    reciprocal_ranks = []
    ndcg_scores = []

    for query_id, relevant_papers in relevant.items():
        ranked = rankings.get(query_id, {})
        ranked_ids = sorted(ranked, key=ranked.get)

        found_rank = None

        for rank, paper_id in enumerate(ranked_ids[:10], start=1):
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

    return {
        "queries": total,
        "Recall@10": hits / total if total else 0,
        "MRR@10": sum(reciprocal_ranks) / total if total else 0,
        "nDCG@10": sum(ndcg_scores) / total if total else 0
    }


dense_test = load_run(output_folder / "dense_test.run")
bm25_test = load_run(p2_folder / "fusion" / "bm25.test.run")
relevant_test = load_qrels(data_folder / "qrels" / "qrels.test.txt")

hybrid_test = fuse(dense_test, bm25_test, BEST_K)

results = []

for name, rankings in [
    ("Dense", dense_test),
    ("BM25", bm25_test),
]:
    metrics = evaluate(rankings, relevant_test)
    results.append({"system": name, **metrics})

# Convert the fused score ranking to a rank dictionary.
hybrid_rank_dict = {
    query_id: {
        paper_id: rank
        for rank, (paper_id, score) in enumerate(ranked, start=1)
    }
    for query_id, ranked in hybrid_test.items()
}

metrics = evaluate(hybrid_rank_dict, relevant_test)
results.append({"system": f"Hybrid RRF k={BEST_K}", **metrics})

results_df = pd.DataFrame(results)
numeric_columns = ["Recall@10", "MRR@10", "nDCG@10"]
results_df[numeric_columns] = results_df[numeric_columns].round(4)

print("Held-out test evaluation")
print("Frozen RRF constant:", BEST_K)
print(results_df.to_string(index=False))

metrics_path = output_folder / "hybrid_test_metrics.csv"
results_df.to_csv(metrics_path, index=False)

run_path = output_folder / "hybrid_test.run"
rows = []

for query_id, ranked in hybrid_test.items():
    for rank, (paper_id, score) in enumerate(ranked, start=1):
        rows.append(
            f"{query_id} Q0 {paper_id} {rank} {score:.8f} hybrid_rrf_k{BEST_K}"
        )

run_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

print("\nSaved test metrics:", metrics_path)
print("Saved hybrid test run:", run_path)
