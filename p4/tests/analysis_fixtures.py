"""A small hand-designed dataset for the analysis tests (NOT a test file). Everything here is invented.

40 queries q01..q40, each with its own relevant paper T01..T40, plus 10 distractor papers.

Popularity (papers.citation_count of the relevant paper)
    q01-q20: 3..7 (bucket "3-10")      q21-q40: 61..80 (bucket ">50")
    distractors D01-D05: 0 citations, D06-D10: 1 citation
Year of the relevant paper: 2019 for odd query numbers, 2020 for even ones (20 queries each)
Citation function of the query sentence: ["background", "method", "result"][i % 3]   (13 / 14 / 13 queries)

Systems
    good   the relevant paper at rank 1 for every query; ranks 2-10 are D06..D10 then D01..D04
    weak   hits (rank 1) for q01-q05 and q21-q35. Otherwise D06 is at rank 1 and the relevant paper is at rank 13
           (a miss at k=10, category "rank 11 or deeper") except q18-q20, where it is absent from the run.
So for `weak` the mean MRR@10 is 5/20 = 0.25 in the 3-10 group and 15/20 = 0.75 in the >50 group; `good` scores 1.0.
"""
import numpy as np
import pandas as pd

N = 40
QIDS = [f"q{i:02d}" for i in range(1, N + 1)]


def target(i: int) -> str:
    return f"T{i:02d}"


def citation_count(i: int) -> int:
    return 3 + (i - 1) % 5 if i <= 20 else 60 + (i - 20)


def make_papers() -> pd.DataFrame:
    rows = [{"paper_id": target(i), "title": f"Paper {target(i)}", "year": 2019 if i % 2 else 2020,
             "citation_count": citation_count(i)} for i in range(1, N + 1)]
    rows += [{"paper_id": f"D{j:02d}", "title": f"Distractor D{j:02d}", "year": 2018, "citation_count": 0 if j <= 5 else 1}
             for j in range(1, 11)]
    return pd.DataFrame(rows)


def make_queries() -> pd.DataFrame:
    return pd.DataFrame({
        "query_id": QIDS, "split": "test", "context_id": [f"ctx{i:02d}" for i in range(1, N + 1)],
        "citing_paper_id": [f"C{i:02d}" for i in range(1, N + 1)], "cited_paper_id": [target(i) for i in range(1, N + 1)],
        "text": [f"query sentence {i}" for i in range(1, N + 1)],
        "cited_citation_count": [citation_count(i) for i in range(1, N + 1)]})


def make_functions() -> pd.DataFrame:
    labels = ["background", "method", "result"]
    return pd.DataFrame({"context_id": [f"ctx{i:02d}" for i in range(1, N + 1)],
                         "function": [labels[i % 3] for i in range(1, N + 1)],
                         "confidence": [0.9 if i % 2 else 0.5 for i in range(1, N + 1)]})


def make_qrels() -> dict:
    return {q: {target(i): 1} for i, q in enumerate(QIDS, start=1)}


FILLER = [f"D{j:02d}" for j in (6, 7, 8, 9, 10, 1, 2, 3, 4)]                 # ranks 2-10 of the `good` run


def make_runs() -> dict:
    good, weak = {}, {}
    for i, q in enumerate(QIDS, start=1):
        good[q] = [(target(i), 100.0)] + [(d, 99.0 - n) for n, d in enumerate(FILLER)]
        hit = i <= 5 or 21 <= i <= 35
        if hit:
            weak[q] = list(good[q])
        else:
            docs = ["D06"] + [f"X{n:02d}" for n in range(11)]                  # 12 wrong papers: D06 at rank 1, then X00..X10
            if not 18 <= i <= 20:
                docs = docs + [target(i)] + [f"Y{n:02d}" for n in range(3)]     # relevant paper at rank 13
            weak[q] = [(d, 50.0 - n) for n, d in enumerate(docs)]
    return {"good": good, "weak": weak}


def make_index_counts() -> pd.Series:
    """Indexable contexts per relevant paper: 1 for q01-q10, 8 for q11-q20, 30 for q21-q40."""
    counts = {target(i): (1 if i <= 10 else 8 if i <= 20 else 30) for i in range(1, N + 1)}
    return pd.Series(counts, name="n_index_contexts")


def write_files(folder) -> dict:
    """Write the dataset to disk in P1's formats and return the paths."""
    from pathlib import Path
    folder = Path(folder)
    qrels = make_qrels()
    paths = {"qrels": folder / "qrels.txt", "queries": folder / "queries.parquet", "papers": folder / "papers.parquet",
             "functions": folder / "context_functions.parquet", "index_contexts": folder / "index_contexts.parquet"}
    paths["qrels"].write_text("".join(f"{q} 0 {next(iter(d))} 1\n" for q, d in qrels.items()))
    make_queries().to_parquet(paths["queries"], index=False)
    make_papers().to_parquet(paths["papers"], index=False)
    make_functions().to_parquet(paths["functions"], index=False)
    counts = make_index_counts()
    pd.DataFrame({"context_id": [f"c{n}" for n in range(int(counts.sum()))],
                  "cited_paper_id": np.repeat(counts.index.to_numpy(), counts.to_numpy())}).to_parquet(paths["index_contexts"], index=False)
    for name, run in make_runs().items():
        paths[f"run_{name}"] = folder / f"{name}.run"
        paths[f"run_{name}"].write_text("".join(f"{q} Q0 {d} {r} {s} {name}\n" for q, docs in run.items()
                                                 for r, (d, s) in enumerate(docs, start=1)))
    return paths
