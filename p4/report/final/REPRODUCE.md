# Reproducing the P4 experiments

All commands are run from the repository root with the project's virtual environment (`.venv`), on Windows, in the order below. `PYTHONUTF8=1` avoids a Windows
encoding problem in the team's code. Every output folder is **new** (the tools refuse to overwrite). Paths in the saved configuration files
(`p4/outputs/real/config/*.json`) are absolute paths of the machine that produced them; edit them if the repository is elsewhere.

**Paths.** `p4/outputs/real/...` below is the folder in the repository. In the submission archive `P4_Real_Outputs.zip` the same files are at the archive root (for example `run_main/EXPERIMENT_MANIFEST.json` instead of `p4/outputs/real/run_main/EXPERIMENT_MANIFEST.json`).

Inputs (not part of P4): `data/processed/` (P1, 8,843 papers, 3,742 queries) and `runs/` with `MANIFEST.csv` (P2 and P3). They are read, never modified.

```powershell
$env:PYTHONUTF8 = "1"

# 0. P2/P3 index input regenerated with the team's own export code (needs the team's `integration` checkout; used only for the leakage gate)
#    CITEDS_DATA=<repo>/data  python -m citeds.export_p2 --out p4/outputs/real/stage1/p2_input_regenerated      (run inside the integration checkout)

# 1. Discover and pair the runs (validation <-> test) from the manifest, checking metadata, tags, splits and queries
python -m p4.experiments.discover_runs --manifest runs/MANIFEST.csv --processed data/processed `
    --out p4/outputs/real/config/runs_discovered.json --report p4/outputs/real/config/discovery_report.json
#    The discovery output was not saved as a separate file: p4/outputs/real/config/runs.json (provided) is that output plus the leakage-gate and P2-reproduction entries (context_files, p2_check)
#    --out below may therefore be any new path; use the provided runs.json for the later steps
#    p4/outputs/real/config/experiment_config.json additionally limits fusion and the analyses to a subset of systems

# 2. Stage 1 checks (read-only)
python -m p4.experiments.verify_integration --processed data/processed --config p4/outputs/real/config/runs.json --out p4/outputs/real/stage1/verify_integration.json
python -m p4.experiments.crosscheck_check_run --config p4/outputs/real/config/runs.json --processed data/processed `
    --check-run <integration checkout>/scripts/check_run.py --integration-src <integration checkout> --split val --out p4/outputs/real/stage1/crosscheck_check_run_val.csv

# 3. Validation table for all 17 systems (analysis only; nothing is tuned on test)
python -m p4.experiments.eval_config --config p4/outputs/real/config/runs.json --processed data/processed --split val --out p4/outputs/real/eval_val --n-boot 2000

# 4. THE test run: leakage gate, evaluation graph, frozen test evaluation, fusion (tuned on validation, applied once to test), analyses
python -m p4.experiments.run_all --data-label real --processed data/processed --config p4/outputs/real/config/experiment_config.json `
    --out p4/outputs/real/run_main --confirm-integrated-project --fusion-mode raw --n-boot 5000 `
    --analyses popularity coldstart year function errors

# 5. Pre-specified significance tests (Holm over 16 comparisons per metric)
python -m p4.experiments.key_comparisons --per-query p4/outputs/real/run_main/eval_fusion_test/per_query.csv --out p4/outputs/real/key_comparisons

# 6. Graph facts, leak-B diagnostic and the exploratory clean-subset comparisons (defined after the main test results were known)
python -m p4.experiments.graph_facts --processed data/processed --graph-dir p4/outputs/real/run_main/eval_graph --out p4/outputs/real/graph_facts.json
python -m p4.experiments.leak_b_diagnostic --processed data/processed --per-query p4/outputs/real/run_main/eval_fusion_test/per_query.csv --split test --out p4/outputs/real/leak_b_diagnostic
python -m p4.experiments.key_comparisons --per-query p4/outputs/real/run_main/eval_fusion_test/per_query.csv `
    --only-queries p4/outputs/real/leak_b_diagnostic/queries_without_same_paper_same_target.txt --out p4/outputs/real/key_comparisons_unexposed

# 7. Validation-only sensitivity of the authority-prior weight (wider lambda grid)
bash p4/outputs/real/sensitivity_extended_lambda_val/run.sh

# 8. Assemble the report sections (tables are generated from the files above)
python -m p4.experiments.make_report --real-dir p4/outputs/real --src p4/report/src --out p4/report/final
```

## Tests

```powershell
python -m pytest p4 -q              # P4: all tests use toy, DEMO or SYNTHETIC fixtures
python -m pytest tests -q           # P1
cd lexical; python -m unittest discover   # P2
```

## Settings that matter

* Bootstrap: 10,000 resamples for the pre-specified comparisons, 5,000 for the evaluation tool and analyses, 2,000 for the validation table; seed 42. Paired Wilcoxon signed-rank test (zero differences dropped), Holm correction, alpha 0.05.
* Fusion: `score + lambda * log(prior)` (raw scores, prior value 0 replaced by 1e-12); prior in {evaluation citation count, PageRank damping 0.5 / 0.85 / 0.95}; lambda grid 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3; choice by validation MRR@10, ties to the smaller lambda.
* Evaluation graph: edges of the citing papers of validation and test queries removed (protocol `leakage-safe-v1`); unweighted PageRank, 100 iterations maximum.
* Analyses: popularity buckets 0 / 1-2 / 3-10 / 11-50 / >50 citations (evaluation counts), groups with fewer than 10 queries shown but not compared.
