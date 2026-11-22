"""Generate a SYNTHETIC unarXive-format sample for tests and dry runs.

    python scripts/make_sample.py --out data/raw/sample --papers 400

The text is random but follows the real JSONL structure (metadata, abstract,
body_text with {{cite:..}} markers, bib_entries with arxiv_id) and includes the
messy cases the cleaner must handle: LaTeX, missing abstracts, other-field
papers, self-citations, multi-citation sentences, repeated and near-duplicate
sentences. Results on it mean nothing — use it only to check the code runs.
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

TOPICS = {
    "adapters": "adapter layers small bottleneck modules inserted into pretrained transformers parameter efficient fine tuning",
    "translation": "neural machine translation encoder decoder attention bilingual parallel corpus bleu",
    "parsing": "dependency parsing syntactic trees graph based transition parser treebank",
    "sentiment": "sentiment analysis polarity reviews opinion classification aspect",
    "qa": "question answering reading comprehension span extraction passages squad",
    "retrieval": "dense passage retrieval bm25 ranking queries documents index",
    "summarization": "abstractive summarization news articles rouge summaries extractive",
    "ner": "named entity recognition sequence labeling crf tagging entities",
}
GENERIC = "we propose model method results show approach task data training evaluation improves performance".split()
FUNC_TEMPLATES = {
    "background": ["Prior work has studied {t} extensively {c}.", "{T} has been a long-standing problem {c}.",
                   "Several studies addressed {t} {c}, motivating our work."],
    "method": ["We follow {c} and use {t} in our model.", "Our implementation builds on {t} as described in {c}.",
               "We adopt the {t} approach of {c} for training."],
    "result": ["Our results outperform {t} reported by {c}.", "Compared with {c}, our {t} system improves accuracy.",
               "This finding is consistent with {t} results from {c}."],
}


def words(topic: str, n: int, rng: random.Random) -> str:
    vocab = TOPICS[topic].split()
    return " ".join(rng.choice(vocab if rng.random() < 0.7 else GENERIC) for _ in range(n))


def make(n_papers: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ids = [f"{18 + i % 6:02d}{1 + i % 12:02d}.{10000 + i:05d}" for i in range(n_papers)]
    topic_of = {pid: rng.choice(list(TOPICS)) for pid in ids}
    # Popularity: a few papers get most citations (Zipf-like), some get none.
    weights = [1.0 / (r + 1) ** 0.9 for r in range(n_papers)]
    recs = []
    for i, pid in enumerate(ids):
        topic = topic_of[pid]
        in_field = rng.random() > 0.1
        cats = "cs.CL cs.LG" if in_field else "cs.CV"
        abstract = "" if rng.random() < 0.05 else (
            f"We study {words(topic, 4, rng)}. " + words(topic, 40, rng).capitalize() +
            r". Our \textbf{method} achieves $F_1$ gains on " + words(topic, 6, rng) + ".")
        bib, body = {}, []
        n_refs = rng.randint(3, 15)
        cited = set()
        while len(cited) < n_refs:
            j = rng.choices(range(n_papers), weights=weights)[0]
            if j != i or rng.random() < 0.05:  # occasional self-citation
                cited.add(ids[j])
        cited = sorted(cited)
        ref_of = {}
        for c in cited:
            ref = str(uuid.UUID(int=rng.getrandbits(128)))
            ref_of[c] = ref
            bib[ref] = {"bib_entry_raw": f"Author. Title {c}.", "contained_arXiv_ids": [],
                        "ids": {"arxiv_id": f"{c}v{rng.randint(1, 3)}", "open_alex_id": ""}}
        # an unresolvable reference
        bib["ffffffff-0000-0000-0000-000000000000"] = {"bib_entry_raw": "A book.", "ids": {"arxiv_id": ""}}
        sections = ["Introduction", "Related Work", "Method", "Experiments", "Conclusion"]
        for sec in sections:
            for _ in range(rng.randint(1, 3)):
                sents = []
                for _ in range(rng.randint(3, 6)):
                    r = rng.random()
                    if r < 0.45 and cited:
                        k = 1 if rng.random() < 0.8 else rng.randint(2, 3)
                        cs = rng.sample(cited, min(k, len(cited)))
                        marks = " ".join("{{cite:%s}}" % ref_of[c] for c in cs)
                        func = rng.choice(list(FUNC_TEMPLATES))
                        t = words(topic_of[cs[0]], 5, rng)
                        sents.append(rng.choice(FUNC_TEMPLATES[func]).format(t=t, T=t.capitalize(), c=marks))
                    elif r < 0.5:
                        sents.append("See {{cite:ffffffff-0000-0000-0000-000000000000}} for details on " + words(topic, 6, rng) + ".")
                    elif r < 0.52:
                        sents.append("{{cite:%s}}." % ref_of[cited[0]])  # marker-only sentence
                    else:
                        sents.append(words(topic, rng.randint(8, 20), rng).capitalize() + r" as shown in {{figure:abc}} and Eq.~\ref{eq1}.")
                text = " ".join(sents)
                body.append({"section": sec, "sec_number": "1", "sec_type": "section",
                             "content_type": "paragraph", "text": text, "cite_spans": [], "ref_spans": []})
        if body and rng.random() < 0.1:
            body.append(dict(body[0]))  # repeated paragraph -> duplicate contexts
        recs.append({
            "paper_id": pid,
            "metadata": {"id": pid, "title": f"On {words(topic, 4, rng)}", "categories": cats,
                         "abstract": abstract, "versions": [{"version": "v1", "created": f"Mon, 1 Jan 20{pid[:2]} 00:00:00 GMT"}]},
            "discipline": "Computer Science",
            "abstract": {"section": "Abstract", "text": abstract, "cite_spans": [], "ref_spans": []},
            "body_text": body, "bib_entries": bib, "ref_entries": {},
        })
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/sample"))
    ap.add_argument("--papers", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    recs = make(a.papers, a.seed)
    path = a.out / "synthetic_sample.jsonl"
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} synthetic papers to {path}")


if __name__ == "__main__":
    main()
