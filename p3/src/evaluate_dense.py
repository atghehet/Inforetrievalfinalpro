
from pathlib import Path
import math
import pandas as pd

project_folder = Path.home() / "Desktop" / "CitationSearchProject"
p1_folder = project_folder / "reference_p1" / "Inforetrievalfinalpro-p1-data-pipeline"
data_folder = p1_folder / "data" / "processed"
output_folder = Path(__file__).resolve().parent.parent / "outputs"

results = []

for split in ["train", "val", "test"]:
    qrels_path = data_folder / "qrels" / f"qrels.{split}.txt"
    run_path = output_folder / f"dense_{split}.run"

    relevant = {}

    with qrels_path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 4 and int(parts[3]) > 0:
                relevant.setdefault(parts[0], set()).add(parts[2])

    rankings = {}

    with run_path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 6:
                qid = parts[0]
                doc_id = parts[2]
                rank = int(parts[3])
                rankings.setdefault(qid, []).append((rank, doc_id))

    hits = 0
    reciprocal_ranks = []
    ndcg_scores = []

    for qid, relevant_docs in relevant.items():
        ranked_docs = sorted(rankings.get(qid, []))
        ranked_ids = [doc_id for rank, doc_id in ranked_docs]

        found_rank = None
        for rank, doc_id in ranked_docs:
            if rank <= 10 and doc_id in relevant_docs:
                found_rank = rank
                break

        if found_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
            ndcg_scores.append(1.0 / math.log2(found_rank + 1))
        else:
            reciprocal_ranks.append(0.0)
            ndcg_scores.append(0.0)

    total = len(relevant)

    if total == 0:
        print(f"WARNING: No relevant queries found for {split}")
        continue

    recall = hits / total
    mrr = sum(reciprocal_ranks) / total
    ndcg = sum(ndcg_scores) / total

    results.append({
        "split": split,
        "queries_evaluated": total,
        "Recall@10": round(recall, 4),
        "MRR@10": round(mrr, 4),
        "nDCG@10": round(ndcg, 4)
    })

print("\nDense retrieval evaluation")
print("=" * 60)
print(pd.DataFrame(results).to_string(index=False))

results_path = output_folder / "dense_metrics.csv"
pd.DataFrame(results).to_csv(results_path, index=False)
print(f"\nSaved metrics to: {results_path}")
