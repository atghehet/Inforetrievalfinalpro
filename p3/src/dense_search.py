"""Interactive dense search over the corpus (for the demo).

    python p3/src/dense_search.py "a method that adapts pretrained models with small adapter layers"
    python p3/src/dense_search.py "..." --variant abstract --model sentence-transformers/allenai-specter

Uses the embeddings cached by dense_baseline.py (run that first).
"""
import argparse

import numpy as np

from common import CACHE_DIR, io
from dense_baseline import model_tag, search_maxp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--variant", default="expanded", choices=["abstract", "contexts", "expanded"])
    ap.add_argument("-k", type=int, default=10)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    cache = CACHE_DIR / model_tag(a.model)
    papers = io.load_papers(["paper_id", "title"])
    ctx = io.load_index_contexts(["cited_paper_id"])
    pe = np.load(cache / "papers.npy")
    if a.variant == "abstract":
        emb, owner = pe, papers["paper_id"].to_numpy()
    else:
        ce = np.load(cache / "index_contexts.npy")
        emb = ce if a.variant == "contexts" else np.vstack([pe, ce])
        owner = (ctx["cited_paper_id"].to_numpy() if a.variant == "contexts"
                 else np.concatenate([papers["paper_id"].to_numpy(), ctx["cited_paper_id"].to_numpy()]))
    q = SentenceTransformer(a.model).encode([a.query], normalize_embeddings=True).astype("float32")
    titles = papers.set_index("paper_id")["title"]
    for rank, (pid, score) in enumerate(search_maxp(emb, owner, q, a.k, a.k * 50)[0], 1):
        print(f"{rank:2d}. {score:.3f}  {pid:12s} {titles[pid][:90]}")


if __name__ == "__main__":
    main()
