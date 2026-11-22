
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

project_folder = Path.home() / "Desktop" / "CitationSearchProject"
p1_folder = project_folder / "reference_p1" / "Inforetrievalfinalpro-p1-data-pipeline"
data_folder = p1_folder / "data" / "processed"
output_folder = Path(__file__).resolve().parent.parent / "outputs"
output_folder.mkdir(parents=True, exist_ok=True)

print("Loading papers and queries...")
papers = pd.read_parquet(data_folder / "papers.parquet")
queries = pd.read_parquet(data_folder / "queries.parquet")

papers["title"] = papers["title"].fillna("")
papers["abstract"] = papers["abstract"].fillna("")
queries["text"] = queries["text"].fillna("")

doc_ids = papers["paper_id"].astype(str).tolist()
documents = (
    papers["title"].astype(str) + ". " + papers["abstract"].astype(str)
).tolist()

print("Papers:", len(documents))
print("Queries:", len(queries))

print("\nLoading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("\nEncoding papers...")
doc_embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nEncoding queries...")
query_embeddings = model.encode(
    queries["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nCalculating similarity scores...")
similarities = cosine_similarity(query_embeddings, doc_embeddings)

for split in ["train", "val", "test"]:
    split_queries = queries[queries["split"] == split]
    rows = []

    for query_index in split_queries.index:
        query = queries.loc[query_index]
        scores = similarities[query_index]
        ranked_indices = np.argsort(-scores)

        for rank, doc_index in enumerate(ranked_indices, start=1):
            rows.append(
                f"{query['query_id']} Q0 {doc_ids[doc_index]} "
                f"{rank} {scores[doc_index]:.6f} dense_minilm"
            )

    output_path = output_folder / f"dense_{split}.run"
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Saved {split} run: {output_path} ({len(rows)} rankings)")

print("\nDense retrieval baseline complete.")
