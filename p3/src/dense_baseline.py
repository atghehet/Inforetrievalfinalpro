"""Dense retrieval with Sentence Transformers + FAISS, for three representations.

    python p3/src/dense_baseline.py                                  # MiniLM, all variants, val+test
    python p3/src/dense_baseline.py --model sentence-transformers/allenai-specter
    python p3/src/dense_baseline.py --variants abstract --splits val

Representations (same candidates and queries for all):
  abstract  one vector per paper: "title. abstract"
  contexts  one vector per leakage-controlled citing sentence (index_contexts);
            a paper's score = max over its context vectors (MaxP). Papers that
            are never cited cannot be retrieved by this variant.
  expanded  abstract vector + context vectors as passages; score = max (MaxP).
            Dense analogue of document expansion that avoids truncating long
            concatenations at the model's max sequence length.

Search uses an exact inner-product FAISS index on L2-normalised vectors
(= cosine similarity). Embeddings are cached in cache/p3/<model>/. Runs keep
the top 1000 papers per query (runs/p3/dense/<model>_<variant>.<split>.run).
"""
from __future__ import annotations

import argparse
import time

import faiss
import numpy as np
import pandas as pd

from common import CACHE_DIR, DENSE_DIR, SPLITS, TOP_K, io, write_run

VARIANTS = ("abstract", "contexts", "expanded")
MODEL_TAGS = {"sentence-transformers/all-MiniLM-L6-v2": "minilm",
              "sentence-transformers/allenai-specter": "specter"}


def model_tag(name: str) -> str:
    return MODEL_TAGS.get(name, name.split("/")[-1].lower())


def encode_cached(model, texts: list[str], path, batch_size: int) -> np.ndarray:
    if path.exists():
        emb = np.load(path)
        if len(emb) == len(texts):
            return emb
    t0 = time.time()
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    print(f"  encoded {len(texts)} texts in {time.time() - t0:.0f}s -> {path.name}")
    return emb


def search_maxp(passage_emb: np.ndarray, passage_owner: np.ndarray, query_emb: np.ndarray,
                top_k: int, depth: int) -> list[list[tuple[str, float]]]:
    """Search passages, keep each owner's best passage score, return top_k owners per query."""
    index = faiss.IndexFlatIP(passage_emb.shape[1])
    index.add(passage_emb)
    scores, idx = index.search(query_emb, min(depth, len(passage_emb)))
    out = []
    for s_row, i_row in zip(scores, idx):
        best: dict[str, float] = {}
        for s, i in zip(s_row, i_row):
            if i < 0:
                continue
            o = passage_owner[i]
            if o not in best:          # results are sorted, first hit is the max
                best[o] = float(s)
        out.append(sorted(best.items(), key=lambda x: (-x[1], x[0]))[:top_k])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=["train", "val", "test"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args(argv)

    from sentence_transformers import SentenceTransformer
    tag = model_tag(a.model)
    cache = CACHE_DIR / tag
    print(f"Loading data and model {a.model} ...")
    papers = io.load_papers(["paper_id", "title", "abstract"])
    ctx = io.load_index_contexts(["context_id", "cited_paper_id", "text"])
    queries = io.load_queries()
    queries = queries[queries["split"].isin(a.splits)].reset_index(drop=True)
    model = SentenceTransformer(a.model)

    # SPECTER was trained on "title [SEP] abstract"; other models get "title. abstract".
    sep = model.tokenizer.sep_token if "specter" in a.model.lower() and model.tokenizer.sep_token else "."
    paper_text = (papers["title"] + f"{sep} " + papers["abstract"]).tolist()
    paper_emb = encode_cached(model, paper_text, cache / "papers.npy", a.batch_size)
    ctx_emb = (encode_cached(model, ctx["text"].tolist(), cache / "index_contexts.npy", a.batch_size)
               if {"contexts", "expanded"} & set(a.variants) else None)
    query_emb = encode_cached(model, queries["text"].tolist(),
                              cache / f"queries_{'_'.join(sorted(a.splits))}.npy", a.batch_size)

    paper_ids = papers["paper_id"].to_numpy()
    for variant in a.variants:
        t0 = time.time()
        if variant == "abstract":
            emb, owner, depth = paper_emb, paper_ids, a.top_k
        elif variant == "contexts":
            emb, owner, depth = ctx_emb, ctx["cited_paper_id"].to_numpy(), a.top_k * 20
        else:
            emb = np.vstack([paper_emb, ctx_emb])
            owner = np.concatenate([paper_ids, ctx["cited_paper_id"].to_numpy()])
            depth = a.top_k * 20
        results = search_maxp(emb, owner, query_emb, a.top_k, depth)
        for split in a.splits:
            mask = (queries["split"] == split).to_numpy()
            ranked = {qid: results[i] for i, qid in enumerate(queries["query_id"]) if mask[i]}
            path = DENSE_DIR / f"{tag}_{variant}.{split}.run"
            write_run(path, ranked, f"dense_{tag}_{variant}", a.top_k)
            print(f"  {variant:9s} {split:5s} -> {path.relative_to(DENSE_DIR.parents[2])} "
                  f"({len(ranked)} queries, {time.time() - t0:.1f}s)")
    print("Dense retrieval complete. Next: python p3/src/evaluate_dense.py")


if __name__ == "__main__":
    main()
