"""Assemble the P4 report: fill `{{table:NAME}}` placeholders in the narrative sources with tables generated from the output files.

    python -m p4.experiments.make_report --real-dir p4/outputs/real --src p4/report/src --out p4/report/final

Every number in a table comes from a file under --real-dir (nothing is typed by hand); the narrative sources only refer to tables by name.
A placeholder with no table, or a table that cannot be built from the files, stops the build (no silent gaps). Existing output is not overwritten.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PLACEHOLDER = re.compile(r"\{\{table:([a-z0-9_]+)\}\}")
MAIN_METRICS = ["Recall@5", "Recall@10", "Recall@100", "MRR@10", "nDCG@10", "MAP"]


class ReportError(RuntimeError):
    """A placeholder without a table, or an input file that is missing."""


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.4f}", index: bool = False) -> str:
    if index:
        df = df.reset_index()
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) and pd.notna(v) else ("" if pd.isna(v) else str(v)))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows]) + "\n"


def fill(template: str, tables: dict[str, str]) -> str:
    """Replace every {{table:NAME}}; unknown names raise ReportError."""
    missing = sorted({m for m in PLACEHOLDER.findall(template) if m not in tables})
    if missing:
        raise ReportError(f"no table for placeholder(s) {missing}; available: {sorted(tables)}")
    return PLACEHOLDER.sub(lambda m: tables[m.group(1)].rstrip("\n") + "\n", template)


# ---------------------------------------------------------------------------------------------- tables from the output files
def _read(path: Path, kind: str = "csv"):
    if not path.is_file():
        raise ReportError(f"missing input file {path}")
    return pd.read_csv(path) if kind == "csv" else json.loads(path.read_text(encoding="utf-8"))


def main_metrics_table(summary: pd.DataFrame, systems: list[str] | None = None) -> str:
    cols = ["system", "n_queries", "n_no_results", *MAIN_METRICS]
    table = summary[cols]
    if systems is not None:
        table = table[table["system"].isin(systems)]
    return df_to_md(table)


def fusion_table(real: Path, systems: list[str], test_summary: pd.DataFrame) -> str:
    rows = []
    summary = test_summary.set_index("system")
    for name in systems:
        best = _read(real / "run_main" / "fusion" / name / "fusion_best.json", "json")
        fused = f"{name}+authority"
        rows.append({"system": name, "prior chosen on val": best["prior"], "lambda": best["lambda"], "val MRR@10 base": best["baseline_validation_value"],
                     "val MRR@10 fused": best["validation_value"], "test MRR@10 base": summary.loc[name, "MRR@10"],
                     "test MRR@10 fused": summary.loc[fused, "MRR@10"], "test nDCG@10 base": summary.loc[name, "nDCG@10"],
                     "test nDCG@10 fused": summary.loc[fused, "nDCG@10"], "test Recall@10 base": summary.loc[name, "Recall@10"],
                     "test Recall@10 fused": summary.loc[fused, "Recall@10"]})
    return df_to_md(pd.DataFrame(rows))


def fusion_tuning_table(real: Path, systems: list[str]) -> str:
    """Best validation MRR@10 reached by each prior column (over lambda), and the lambda that reached it."""
    rows = []
    for name in systems:
        tuning = _read(real / "run_main" / "fusion" / name / "fusion_tuning.csv")
        base = tuning[tuning["lambda"] == 0.0]["MRR@10"].iloc[0]
        for prior, part in tuning.groupby("prior", sort=False):
            top = part.sort_values(["MRR@10", "lambda"], ascending=[False, True]).iloc[0]
            rows.append({"system": name, "prior": prior, "val MRR@10 (lambda 0)": base, "best val MRR@10": top["MRR@10"], "at lambda": top["lambda"]})
    return df_to_md(pd.DataFrame(rows))


def graph_table(facts: dict) -> str:
    pr = facts["pagerank"]
    rows = [("papers (nodes)", f"{facts['nodes']:,}"), ("citation edges, full graph", f"{facts['edges_full']:,}"),
            ("edges removed (citing papers of val/test queries held out)", f"{facts['edge_removal']['edges_removed']:,} ({facts['edge_removal']['share_removed']:.1%})"),
            ("edges kept in the evaluation graph", f"{facts['edge_removal']['edges_kept']:,}"),
            ("held-out citing papers", f"{facts['edge_removal']['held_out_citing_papers']:,}"),
            ("edges whose citing paper is not older than the cited paper", f"{facts['direction_check']['citing_year_>=_cited_year']:.2%}"),
            ("papers never cited, full graph", f"{facts['papers_never_cited_full_graph']:.1%}"),
            ("papers never cited, evaluation graph", f"{facts['papers_never_cited_safe_graph']:.1%}"),
            ("test targets with 0 citations, full graph / evaluation graph",
             f"{facts['test_targets']['full_count_zero']:.1%} / {facts['test_targets']['safe_count_zero']:.1%}"),
            ("mean citations of a test target, full / evaluation graph", f"{facts['test_targets']['full_count_mean']:.2f} / {facts['test_targets']['safe_count_mean']:.2f}"),
            ("Spearman: PageRank (d=0.85) vs evaluation citation count", f"{facts['spearman_pagerank_d085_vs_safe_count']:.3f}")]
    for key, label in (("pagerank_d050", "0.50"), ("pagerank_d085", "0.85"), ("pagerank_d095", "0.95")):
        rows.append((f"PageRank damping {label}: iterations / converged / final change",
                     f"{pr[key]['iterations']} / {pr[key]['converged']} / {pr[key]['final_delta']:.1e}"))
    return df_to_md(pd.DataFrame(rows, columns=["quantity", "value"]))


def efficiency_tables(real: Path) -> dict[str, str]:
    index = _read(real / "run_main" / "eval_test" / "efficiency_index.csv")
    runs = _read(real / "run_main" / "eval_test" / "efficiency_runs.csv")
    best = runs.sort_values("validation_mrr_at_10", ascending=False).groupby(["variant", "model"], sort=False).head(1)
    best = best[["variant", "model", "validation_mrr_at_10", "median_ms", "mean_ms", "size_mib"]].sort_values(["variant", "model"])
    return {"efficiency_index": df_to_md(index[["index", "build_seconds", "size_mib"]], "{:.2f}"),
            "efficiency_query": df_to_md(best.rename(columns={"validation_mrr_at_10": "val MRR@10", "median_ms": "median ms/query", "mean_ms": "mean ms/query",
                                                              "size_mib": "index MiB"}), "{:.2f}")}


def build_tables(real: Path) -> dict[str, str]:
    from p4.experiments import key_comparisons as KC
    test = _read(real / "run_main" / "eval_fusion_test" / "summary.csv")
    val = _read(real / "eval_val" / "summary.csv")
    base_systems = [s for s in test["system"] if not s.endswith("+authority")]
    comps = _read(real / "key_comparisons" / "key_comparisons.csv")
    fusion_systems = _read(real / "run_main" / "EXPERIMENT_MANIFEST.json", "json")["settings"]["fusion_systems"]
    tables = {"test_main": main_metrics_table(test, base_systems), "val_main": main_metrics_table(val),
              "test_fused": main_metrics_table(test, [s for s in test["system"] if s.endswith("+authority") or s in fusion_systems]),
              "fusion_summary": fusion_table(real, fusion_systems, test), "fusion_tuning": fusion_tuning_table(real, fusion_systems),
              "graph_facts": graph_table(_read(real / "graph_facts.json", "json")),
              "key_comparisons_mrr": KC.to_markdown(comps[comps["metric"] == "MRR@10"]),
              "key_comparisons_recall": KC.to_markdown(comps[comps["metric"] == "Recall@10"]),
              "key_comparisons_ndcg": KC.to_markdown(comps[comps["metric"] == "nDCG@10"]),
              **efficiency_tables(real)}
    return tables


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-dir", required=True, type=Path)
    ap.add_argument("--src", required=True, type=Path, help="folder with the narrative .md sources containing {{table:NAME}} placeholders")
    ap.add_argument("--out", required=True, type=Path, help="NEW folder for the assembled report")
    args = ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"error: {args.out} is not empty", file=sys.stderr)
        return 2
    try:
        tables = build_tables(args.real_dir)
        extra = _extra_tables(args.real_dir)
        tables.update(extra)
        sources = sorted(args.src.glob("*.md"))
        if not sources:
            raise ReportError(f"no .md sources in {args.src}")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "tables").mkdir()
        for name, text in tables.items():
            (args.out / "tables" / f"{name}.md").write_text(text, encoding="utf-8")
        for src in sources:
            (args.out / src.name).write_text(fill(src.read_text(encoding="utf-8"), tables), encoding="utf-8")
    except ReportError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"wrote {len(sources)} report file(s) and {len(tables)} table(s) to {args.out}")
    return 0


def _extra_tables(real: Path) -> dict[str, str]:
    """Analysis tables (popularity, cold start, year, function). Defined in p4.experiments.analysis_tables when present."""
    try:
        from p4.experiments import analysis_tables
    except ImportError:
        return {}
    return analysis_tables.build(real)


if __name__ == "__main__":
    sys.exit(main())
