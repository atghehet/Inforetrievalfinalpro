"""Citation-function classifier: Naive Bayes vs logistic regression (TF-IDF).

Classes: background | method | result   (citeds.schema.CITATION_FUNCTIONS)

Training data (download manually into data/external/, see README):
  * SciCite  (Cohan et al., 2019): train.jsonl / dev.jsonl / test.jsonl,
    fields "string" and "label" in {background, method, result}.
  * ACL-ARC  (Jurgens et al., 2018; Cohan et al. jsonl release): fields
    "text" and "intent" in 6 classes, mapped to the 3 classes below.

    python -m citeds.citation_function train   --data data/external/scicite
    python -m citeds.citation_function predict  # labels data/processed/contexts.parquet

Hyper-parameters (NB alpha, LR C) are chosen on the dev split by macro-F1;
the test split is used once for the reported numbers.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

from . import config, schema
from .clean import clean_text

log = logging.getLogger(__name__)
MODEL_DIR = config.ROOT / "models"
RESULTS_MD = config.ROOT / "docs" / "citation_function_results.md"
LABELS = list(schema.CITATION_FUNCTIONS)

ACL_ARC_MAP = {
    "background": "background", "motivation": "background", "future": "background",
    "uses": "method", "extends": "method",
    "compareorcontrast": "result",
}
SCICITE_MAP = {"background": "background", "method": "method", "result": "result"}

# Training sentences still contain author-year / numeric citations; our
# contexts have markers stripped, so strip them here too (avoids domain shift).
_PAREN_CITE = re.compile(r"\((?:[^()]*?(?:et al\.?|\b(?:19|20)\d{2}[a-z]?\b)[^()]*?)\)")
_NUM_CITE = re.compile(r"\[\s*\d+(?:\s*[,–\-]\s*\d+)*\s*\]")
_AUTHOR_INLINE = re.compile(r"\b[A-Z][A-Za-z\-]+(?: et al\.?| and [A-Z][A-Za-z\-]+)?\s*\(\s*(?:19|20)\d{2}[a-z]?\s*\)")


def normalise_training_text(text: str) -> str:
    t = _PAREN_CITE.sub(" ", text)
    t = _NUM_CITE.sub(" ", t)
    t = _AUTHOR_INLINE.sub(" ", t)
    return clean_text(t)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_labelled(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Return {"train","dev","test"} -> DataFrame(text, label). Detects SciCite vs ACL-ARC."""
    out = {}
    for split in ("train", "dev", "test"):
        p = Path(data_dir) / f"{split}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (expected train/dev/test.jsonl)")
        rows = _read_jsonl(p)
        if "string" in rows[0]:
            texts = [r["string"] for r in rows]
            labels = [SCICITE_MAP.get(str(r["label"]).lower()) for r in rows]
        else:
            texts = [r.get("text") or r.get("cleaned_cite_text") for r in rows]
            labels = [ACL_ARC_MAP.get(str(r.get("intent", "")).lower().replace("_", "")) for r in rows]
        df = pd.DataFrame({"text": [normalise_training_text(t or "") for t in texts], "label": labels})
        df = df.dropna().loc[lambda d: d["text"].str.len() > 0].drop_duplicates().reset_index(drop=True)
        out[split] = df
    return out


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True,
                           strip_accents="unicode")


def candidates() -> dict[str, list[tuple[str, Pipeline]]]:
    nb = [(f"MultinomialNB(alpha={a})", Pipeline([("tfidf", _vectorizer()), ("clf", MultinomialNB(alpha=a))]))
          for a in (0.01, 0.05, 0.1, 0.3, 1.0)]
    nb += [(f"ComplementNB(alpha={a})", Pipeline([("tfidf", _vectorizer()), ("clf", ComplementNB(alpha=a))]))
           for a in (0.1, 0.3, 1.0)]
    lr = [(f"LogisticRegression(C={c})", Pipeline([("tfidf", _vectorizer()),
          ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=config.SEED))]))
          for c in (0.3, 1.0, 3.0, 10.0)]
    return {"naive_bayes": nb, "logistic_regression": lr}


def _scores(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
            "per_class_f1": dict(zip(LABELS, map(float, f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0))))}


def train(data_dir: Path, model_dir: Path = MODEL_DIR, results_md: Path = RESULTS_MD,
          dataset_name: str | None = None) -> dict:
    data = load_labelled(data_dir)
    tr, dv, te = data["train"], data["dev"], data["test"]
    dataset_name = dataset_name or Path(data_dir).name
    results = {"dataset": dataset_name, "sizes": {k: len(v) for k, v in data.items()},
               "label_distribution_train": tr["label"].value_counts().to_dict(), "models": {}}
    # Majority-class baseline for context.
    maj = tr["label"].mode()[0]
    results["models"]["majority_class"] = {"chosen": f"always '{maj}'", "dev": _scores(dv["label"], [maj] * len(dv)),
                                           "test": _scores(te["label"], [maj] * len(te))}
    model_dir.mkdir(parents=True, exist_ok=True)
    for family, cands in candidates().items():
        best = None
        tuning = []
        for name, pipe in cands:
            pipe.fit(tr["text"], tr["label"])
            s = _scores(dv["label"], pipe.predict(dv["text"]))
            tuning.append({"model": name, **{k: s[k] for k in ("accuracy", "macro_f1")}})
            if best is None or s["macro_f1"] > best[2]["macro_f1"]:
                best = (name, pipe, s)
        name, pipe, dev_s = best
        pred = pipe.predict(te["text"])
        results["models"][family] = {
            "chosen": name, "dev": dev_s, "test": _scores(te["label"], pred), "tuning": tuning,
            "confusion_matrix_test": {"labels": LABELS,
                                      "matrix": confusion_matrix(te["label"], pred, labels=LABELS).tolist()},
            "report_test": classification_report(te["label"], pred, labels=LABELS, zero_division=0, output_dict=True),
        }
        joblib.dump(pipe, model_dir / f"citation_function_{family}.joblib")
        log.info("%s: chose %s, test macro-F1 %.3f", family, name, results["models"][family]["test"]["macro_f1"])
    (model_dir / "citation_function_results.json").write_text(json.dumps(results, indent=2))
    results_md.parent.mkdir(parents=True, exist_ok=True)
    results_md.write_text(results_markdown(results))
    return results


def results_markdown(r: dict) -> str:
    L = [f"# Citation-function classification results ({r['dataset']})", "",
         "Generated by `python -m citeds.citation_function train`. TF-IDF (word 1-2 grams, sublinear tf).",
         "Hyper-parameters chosen on dev by macro-F1; test evaluated once.", "",
         f"Split sizes: {r['sizes']}. Train label distribution: {r['label_distribution_train']}.", "",
         "| Model | Chosen setting | Dev macro-F1 | Test accuracy | Test macro-F1 | F1 background | F1 method | F1 result |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for fam, m in r["models"].items():
        t = m["test"]
        pc = t["per_class_f1"]
        L.append(f"| {fam} | {m['chosen']} | {m['dev']['macro_f1']:.3f} | {t['accuracy']:.3f} | {t['macro_f1']:.3f} | "
                 f"{pc['background']:.3f} | {pc['method']:.3f} | {pc['result']:.3f} |")
    for fam in ("naive_bayes", "logistic_regression"):
        m = r["models"].get(fam)
        if not m:
            continue
        L += ["", f"## {fam}: dev tuning", "", "| Setting | Dev accuracy | Dev macro-F1 |", "|---|---:|---:|"]
        L += [f"| {t['model']} | {t['accuracy']:.3f} | {t['macro_f1']:.3f} |" for t in m["tuning"]]
        cm = m["confusion_matrix_test"]
        L += ["", f"Test confusion matrix (rows = true, columns = predicted: {', '.join(cm['labels'])}):", ""]
        L += ["| true \\ pred | " + " | ".join(cm["labels"]) + " |", "|---|" + "---:|" * len(cm["labels"])]
        L += [f"| {lab} | " + " | ".join(str(x) for x in row) + " |" for lab, row in zip(cm["labels"], cm["matrix"])]
    return "\n".join(L) + "\n"


def predict_contexts(model_path: Path, contexts_path: Path = config.CONTEXTS_PATH,
                     out_path: Path = config.CONTEXT_FUNCTIONS_PATH) -> pd.DataFrame:
    """Label every cleaned context (all contexts, so query sentences get labels
    too — Member 4 needs them for the per-function error analysis)."""
    pipe = joblib.load(model_path)
    ctx = pd.read_parquet(contexts_path, columns=["context_id", "text"])
    proba = pipe.predict_proba(ctx["text"])
    classes = np.asarray(pipe.classes_)
    out = pd.DataFrame({"context_id": ctx["context_id"],
                        "function": classes[proba.argmax(1)],
                        "confidence": proba.max(1).astype("float32")})
    import pyarrow.parquet as pq
    pq.write_table(schema.conform(out, schema.CONTEXT_FUNCTIONS), out_path)
    log.info("wrote %s (%d rows): %s", out_path, len(out), out["function"].value_counts().to_dict())
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=config.EXTERNAL_DIR / "scicite")
    p = sub.add_parser("predict")
    p.add_argument("--model", choices=["naive_bayes", "logistic_regression"], default="naive_bayes",
                   help="which trained model labels the contexts (default: naive_bayes, per the plan)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "train":
        train(a.data)
    else:
        predict_contexts(MODEL_DIR / f"citation_function_{a.model}.joblib")


if __name__ == "__main__":
    main()
