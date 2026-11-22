"""The pre-specified comparisons of the study, with paired tests, bootstrap confidence intervals and Holm correction.

    python -m p4.experiments.key_comparisons --per-query <run_main>/eval_fusion_test/per_query.csv --out <NEW folder> [--n-boot 10000]

The list of comparisons below was fixed from the research questions BEFORE the test results were looked at. One family per metric
(Holm correction over all comparisons in the list), so a reported "significant" difference has survived the correction. The per-query scores
come from the evaluation tool (per_query.csv), where a query with no results already counts as 0.

RQ1  does expanding a paper with citation contexts beat the abstract alone?   RQ2  lexical vs dense vs hybrid
RQ3  does the leakage-safe authority prior help?                              (lexical model choice within P2 is included for context)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from p4.eval.significance import DEFAULT_N_BOOT, compare_systems

METRICS = {"MRR@10": "rr@10", "Recall@10": "recall@10", "nDCG@10": "ndcg@10"}
# (question, system, baseline)
COMPARISONS = [
    ("RQ1", "p2_expanded_bm25", "p2_abstract_bm25"),
    ("RQ1", "p2_context_bm25", "p2_abstract_bm25"),
    ("RQ1", "p2_expanded_bm25", "p2_context_bm25"),
    ("RQ1", "p3_minilm_expanded", "p3_minilm_abstract"),
    ("RQ1", "p3_minilm_contexts", "p3_minilm_abstract"),
    ("RQ2", "p3_minilm_abstract", "p2_abstract_bm25"),
    ("RQ2", "p3_minilm_expanded", "p2_expanded_bm25"),
    ("RQ2", "p3_hybrid_rrf", "p2_expanded_bm25"),
    ("RQ2", "p3_hybrid_rrf", "p3_minilm_expanded"),
    ("P2", "p2_expanded_bm25", "p2_expanded_rm3"),
    ("P2", "p2_expanded_bm25", "p2_expanded_ql"),
    ("P2", "p2_expanded_bm25", "p2_fielded_bm25f"),
    ("RQ3", "p2_abstract_bm25+authority", "p2_abstract_bm25"),
    ("RQ3", "p2_expanded_bm25+authority", "p2_expanded_bm25"),
    ("RQ3", "p3_minilm_expanded+authority", "p3_minilm_expanded"),
    ("RQ3", "p3_hybrid_rrf+authority", "p3_hybrid_rrf"),
]


def key_comparisons(per_query: pd.DataFrame, *, n_boot: int = DEFAULT_N_BOOT, comparisons=COMPARISONS) -> pd.DataFrame:
    frames = []
    for metric, column in METRICS.items():
        wide = per_query.pivot(index="query_id", columns="system", values=column)
        table = compare_systems(wide, pairs=[(s, b) for _, s, b in comparisons], metric=metric, n_boot=n_boot)
        table.insert(0, "question", [q for q, _, _ in comparisons])
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| Q | Metric | System | vs | mean sys | mean base | diff | 95% CI | p (Holm) | verdict |", "|---|---|---|---|---:|---:|---:|---|---:|---|"]
    for r in table.itertuples():
        verdict = ("significant, " + ("better" if r.direction == "system_better" else "worse")) if r.significant_holm else "not significant"
        lines.append(f"| {r.question} | {r.metric} | {r.system} | {r.baseline} | {r.mean_system:.4f} | {r.mean_baseline:.4f} | {r.mean_diff:+.4f} | "
                     f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.p_holm:.3g} | {verdict} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-query", required=True, type=Path, help="per_query.csv from the TEST evaluation")
    ap.add_argument("--out", required=True, type=Path, help="NEW folder")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--note", default="")
    ap.add_argument("--only-queries", type=Path, help="text file with one query_id per line: compare on these queries only (a subset analysis)")
    args = ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"error: {args.out} is not empty", file=sys.stderr)
        return 2
    per_query = pd.read_csv(args.per_query)
    if args.only_queries:
        keep = {line.strip() for line in args.only_queries.read_text(encoding="utf-8").splitlines() if line.strip()}
        unknown = keep - set(per_query["query_id"])
        if unknown or not keep:
            print(f"error: --only-queries must list queries present in {args.per_query.name} ({len(unknown)} unknown, {len(keep)} listed)", file=sys.stderr)
            return 2
        per_query = per_query[per_query["query_id"].isin(keep)]
        args.note = f"{args.note} Subset: {per_query['query_id'].nunique()} queries from {args.only_queries.name}.".strip()
    missing = sorted({n for _, s, b in COMPARISONS for n in (s, b)} - set(per_query["system"]))
    if missing:
        print(f"error: systems missing from {args.per_query.name}: {missing}", file=sys.stderr)
        return 2
    table = key_comparisons(per_query, n_boot=args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "key_comparisons.csv", index=False)
    header = f"# Pre-specified comparisons (paired Wilcoxon, bootstrap 95% CI, Holm over {len(COMPARISONS)} comparisons per metric)\n\n{args.note}\n\n"
    (args.out / "key_comparisons.md").write_text(header + to_markdown(table), encoding="utf-8")
    print(f"wrote {len(table)} comparisons to {args.out}; significant after Holm: {int(table['significant_holm'].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
