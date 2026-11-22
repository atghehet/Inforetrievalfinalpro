import json
import random

import pandas as pd
import pytest

from citeds import citation_function as cf
from citeds import crawler


def _write_fake_scicite(d, n=300, seed=0):
    rng = random.Random(seed)
    cues = {"background": "has been widely studied in prior work on",
            "method": "we use the implementation and follow the training setup of",
            "result": "our results outperform and are consistent with the scores of"}
    d.mkdir(parents=True, exist_ok=True)
    for split, m in (("train", n), ("dev", n // 3), ("test", n // 3)):
        with open(d / f"{split}.jsonl", "w") as fh:
            for _ in range(m):
                lab = rng.choice(list(cues))
                s = f"{cues[lab]} topic {rng.randint(0, 50)} (Smith et al., 2019) [3]."
                fh.write(json.dumps({"string": s, "label": lab}) + "\n")


def test_training_text_normalisation():
    t = cf.normalise_training_text("As shown by Devlin et al. (2019) and others (Peters et al., 2018; Liu 2019) [4, 5].")
    assert "2019" not in t and "2018" not in t and "[" not in t


def test_acl_arc_label_mapping(tmp_path):
    for split in ("train", "dev", "test"):
        with open(tmp_path / f"{split}.jsonl", "w") as fh:
            for intent in ("Uses", "Motivation", "CompareOrContrast", "Extends", "Future", "Background"):
                fh.write(json.dumps({"text": f"some citing text for {intent}", "intent": intent}) + "\n")
    data = cf.load_labelled(tmp_path)
    assert data["train"]["label"].tolist() == ["method", "background", "result", "method", "background", "background"]


def test_train_and_predict(tmp_path):
    _write_fake_scicite(tmp_path / "scicite")
    res = cf.train(tmp_path / "scicite", model_dir=tmp_path / "models", results_md=tmp_path / "r.md")
    assert res["models"]["naive_bayes"]["test"]["macro_f1"] > 0.9
    assert res["models"]["logistic_regression"]["test"]["macro_f1"] > 0.9
    assert "naive_bayes" in (tmp_path / "r.md").read_text()
    ctx = pd.DataFrame({"context_id": ["a", "b"],
                        "text": ["We use the implementation of", "This has been widely studied in prior work"]})
    ctx.to_parquet(tmp_path / "ctx.parquet")
    out = cf.predict_contexts(tmp_path / "models" / "citation_function_naive_bayes.joblib",
                              tmp_path / "ctx.parquet", tmp_path / "out.parquet")
    assert out["function"].tolist() == ["method", "background"]


def test_crawler_refuses_placeholder_email():
    with pytest.raises(SystemExit):
        crawler.PoliteClient(crawler.PLACEHOLDER)


def test_inverted_index_and_row():
    inv = {"Adapters": [0], "are": [1], "small": [2]}
    assert crawler.abstract_from_inverted_index(inv) == "Adapters are small"
    w = {"id": "https://openalex.org/W1", "title": "T", "publication_year": 2024,
         "abstract_inverted_index": {f"w{i}": [i] for i in range(30)},
         "locations": [{"landing_page_url": "https://arxiv.org/abs/2401.01234v1"}],
         "cited_by_count": 3, "referenced_works": ["W2", "W3"]}
    r = crawler.work_to_row(w)
    assert r["paper_id"] == "2401.01234" and r["referenced_works"] == "W2|W3"


def test_crawler_caches_responses(tmp_path, monkeypatch):
    calls = []

    class Resp:
        status_code = 200
        text = ""
        headers = {}

        def json(self):
            return {"ok": True}

    client = crawler.PoliteClient("student@uni.edu", cache_dir=tmp_path, min_interval=0)
    monkeypatch.setattr(client, "_allowed", lambda url: True)
    monkeypatch.setattr(client.session, "get", lambda url, timeout: calls.append(url) or Resp())
    assert client.get_json("https://api.openalex.org/works?x=1") == {"ok": True}
    assert client.get_json("https://api.openalex.org/works?x=1") == {"ok": True}
    assert len(calls) == 1
    assert "mailto:student@uni.edu" in client.session.headers["User-Agent"]
