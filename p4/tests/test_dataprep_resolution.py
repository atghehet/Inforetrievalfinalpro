"""Tests for p4.dataprep.resolution_audit: the proposed raw-text fallback and the policy comparison."""
import json

import pytest

from p4.dataprep import resolution_audit as R


# ---- ID shape validation -----------------------------------------------------------------------------
@pytest.mark.parametrize("identifier, valid", [
    ("2104.06921", True), ("0704.0001", True), ("1412.6980", True), ("1501.00001", True),
    ("1501.0001", False),            # 4-digit sequence after 2015 does not exist
    ("1412.69801", False),           # 5-digit sequence before 2015 does not exist
    ("0703.0001", False),            # new-style numbering starts in 0704
    ("2113.00001", False),           # month 13
    ("2100.00001", False),           # month 00
    ("hep-th/9901001", True), ("cs/0112017", True), ("math/0309136", True), ("hep-th/9913001", False),
    ("2019.1234", False),            # a year.number, not an arXiv ID
    ("abc", False)])
def test_valid_arxiv_id(identifier, valid):
    assert R.valid_arxiv_id(identifier) is valid


# ---- the proposed raw-text fallback --------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("K. Author. A title. arXiv preprint arXiv:2104.06921, 2021.", ["2104.06921"]),
    ("A title. arXiv:2104.06921v3 [cs.CL]", ["2104.06921"]),                               # version suffix dropped
    ("A title. CoRR, abs/1706.03762, 2017. arXiv:1706.03762", ["1706.03762"]),
    ("https://arxiv.org/abs/2010.11929", ["2010.11929"]),
    ("https://arxiv.org/pdf/hep-th/9901001", ["hep-th/9901001"]),
    ("arXiv:hep-th/9901001v2", ["hep-th/9901001"]),
    ("K. Baumgaertl and D. Grundler 10.48550/arxiv.2208.10923 (2022).", ["2208.10923"]),     # arXiv DOI
    ("ARXIV: 1810.04805", ["1810.04805"]),
    ("arXiv:2104.06921 and also arXiv:2010.11929", ["2104.06921", "2010.11929"]),            # two IDs: ambiguous downstream
    ("arXiv:2104.06921 arXiv:2104.06921", ["2104.06921"]),                                  # repeated, one distinct ID
])
def test_extract_raw_ids_finds_the_ids_people_write(text, expected):
    assert R.extract_raw_ids(text) == expected


@pytest.mark.parametrize("text", [
    "pp. 1234.5678 in Proceedings of ACL, 2019",                                     # page numbers, no 'arXiv' marker
    "Journal of X, 2019.1234", "Volume 2104.06921 of Lecture Notes",                 # bare number without the marker
    "arXiv:1501.0001",                                                                # impossible shape
    "arXiv:2113.00001", "", None])
def test_extract_raw_ids_rejects_look_alikes_and_empty_input(text):
    assert R.extract_raw_ids(text) == []


def test_title_consistency():
    ref = "A. Vaswani et al. Attention is all you need. In NeurIPS, 2017. arXiv:1706.03762"
    assert R.title_consistent(ref, "Attention Is All You Need")
    assert not R.title_consistent(ref, "BERT: Pre-training of Deep Bidirectional Transformers")
    assert not R.title_consistent(ref, "")


def test_yymm_ordering_helper():
    assert R._yymm("2104.06921") == 2104 and R._yymm("hep-th/9901001") == 0 and R._yymm("garbage") is None


# ---- policies on a tiny hand-made corpus ---------------------------------------------------------------
def paper(pid, title, refs, sentences, cats="cs.CL"):
    """refs: {key: bib entry dict}; sentences: list of sentence strings containing {{cite:key}} markers."""
    return {"paper_id": pid, "metadata": {"id": pid, "title": title, "categories": cats},
            "bib_entries": refs, "body_text": [{"content_type": "paragraph", "section": "intro", "text": " ".join(sentences)}]}


def write_corpus(folder):
    folder.mkdir()
    long = "We build on this earlier approach to the task and adapt it for our setting here {{cite:%s}}."
    records = [
        paper("2001.00001", "Alpha method for tagging", {}, []),
        paper("2002.00002", "Beta model for parsing", {}, []),
        paper("2003.00003", "Gamma study of translation", {}, []),
        paper("2101.00004", "Delta citing paper", {
            "k_ids": {"bib_entry_raw": "Alpha method for tagging. 2020.", "ids": {"arxiv_id": "2001.00001"}},
            "k_cont": {"bib_entry_raw": "Beta model for parsing. 2020.", "ids": {}, "contained_arXiv_ids": [{"id": "2002.00002"}]},
            "k_raw": {"bib_entry_raw": "Gamma study of translation. arXiv preprint arXiv:2003.00003, 2020.", "ids": {}},
            "k_amb": {"bib_entry_raw": "See arXiv:2001.00001 and arXiv:2002.00002.", "ids": {}},
            "k_none": {"bib_entry_raw": "A book. Publisher, 2018.", "ids": {}},
        }, [long % "k_ids", long % "k_cont", long % "k_raw", long % "k_amb", long % "k_none"]),
        paper("2101.00005", "Not in the field", {}, [], cats="math.PR"),
    ]
    (folder / "recs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return folder


def test_policies_recover_progressively_more_references(tmp_path):
    report = R.audit(write_corpus(tmp_path / "corpus"))
    assert report["papers"] == 4 and report["references"] == 5                          # the math.PR paper is excluded
    p = report["policies"]
    assert p["ids"]["resolved_references"] == 1                                         # k_ids
    assert p["current"]["resolved_references"] == 2                                     # + k_cont
    assert p["proposed"]["resolved_references"] == 3                                    # + k_raw; k_amb (two IDs) stays unresolved
    assert [p[x]["in_corpus_edges"] for x in ("ids", "current", "proposed")] == [1, 2, 3]
    assert [p[x]["usable_contexts"] for x in ("ids", "current", "proposed")] == [1, 2, 3]
    assert p["proposed"]["distinct_cited_ids"] == 3 and p["proposed"]["distinct_cited_outside_corpus"] == 0


def test_cited_papers_outside_the_corpus_are_counted_by_year(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    records = [paper("2101.00001", "Citing", {
        "a": {"bib_entry_raw": "x", "ids": {"arxiv_id": "1810.04805"}},               # outside the corpus, 2018
        "b": {"bib_entry_raw": "y", "ids": {"arxiv_id": "hep-th/9901001"}},          # outside, old-style
        "c": {"bib_entry_raw": "z arXiv:1810.04805", "ids": {}}}, [])]               # the same paper again, found via raw text
    (folder / "r.jsonl").write_text(json.dumps(records[0]) + chr(10))
    p = R.audit(folder)["policies"]
    assert p["current"]["distinct_cited_outside_corpus"] == 2 and p["proposed"]["distinct_cited_outside_corpus"] == 2   # distinct, not repeated
    assert p["proposed"]["outside_corpus_by_id_year"] == {"2018": 1, "pre-2007": 1}
    assert p["proposed"]["distinct_cited_in_corpus"] == 0 and p["current"]["in_corpus_edges"] == 0


def test_raw_text_quality_counts(tmp_path):
    q = R.audit(write_corpus(tmp_path / "corpus"))["raw_text_quality"]
    assert q["references_resolved_only_by_raw_text"] == 1 and q["ambiguous_references_several_ids"] == 1
    assert q["in_corpus_hits_raw_only"] == 1 and q["title_consistent_share_raw_only"] == 1.0   # Gamma's title is in the reference
    assert q["in_corpus_hits_control"] == 2 and q["title_consistent_share_control"] == 1.0
    assert q["implausible_cited_newer_than_citing"] == 0


def test_a_wrong_raw_match_is_flagged_by_the_title_check_and_the_date_check(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    records = [paper("2001.00001", "Alpha method for tagging", {}, []),
               paper("2002.00002", "Citing paper", {
                   "k": {"bib_entry_raw": "Something entirely different. arXiv:2001.00001, 2020.", "ids": {}},          # wrong title
                   "future": {"bib_entry_raw": "Later work. arXiv:2301.00001.", "ids": {}}}, [])]
    (folder / "r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    q = R.audit(folder)["raw_text_quality"]
    assert q["in_corpus_hits_raw_only"] == 1 and q["title_consistent_share_raw_only"] == 0.0
    assert q["title_inconsistent_raw_examples"][0]["resolved_to"] == "2001.00001"
    assert q["implausible_cited_newer_than_citing"] == 1                                # 2301 is newer than the citing 2002 paper


def test_agreement_with_other_sources(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    records = [paper("2002.00002", "Citing", {
        "same": {"bib_entry_raw": "x arXiv:2001.00001", "ids": {"arxiv_id": "2001.00001"}},
        "diff": {"bib_entry_raw": "y arXiv:2001.00009", "ids": {"arxiv_id": "2001.00001"}}}, [])]
    (folder / "r.jsonl").write_text(json.dumps(records[0]) + "\n")
    q = R.audit(folder)["raw_text_quality"]
    assert (q["agree_with_other_sources"], q["disagree_with_other_sources"]) == (1, 1)


def test_cli_writes_a_report_and_refuses_to_overwrite(tmp_path, capsys):
    corpus = write_corpus(tmp_path / "corpus")
    out = tmp_path / "out" / "report.json"
    assert R.main(["--subset", str(corpus), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["policies"]["proposed"]["resolved_references"] == 3
    assert R.main(["--subset", str(corpus), "--out", str(out)]) == 2 and "already exists" in capsys.readouterr().err
    assert R.main(["--subset", str(tmp_path / "nope"), "--out", str(tmp_path / "x.json")]) == 2
