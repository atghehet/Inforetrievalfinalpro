from citeds.clean import clean_text, find_cite_ids, normalise_arxiv_id, split_sentences
from citeds.unarxive import parse_record


def test_citation_markers_removed():
    t = "Adapters {{cite:a1b2-c3}} {{cite:d4e5}} work well, see {{figure:x}}."
    assert find_cite_ids(t) == ["a1b2-c3", "d4e5"]
    assert clean_text(t) == "Adapters work well, see."


def test_marker_runs_leave_no_comma_runs():
    t = "Spam filters fail, {{cite:aa}}, {{cite:bb}}, {{cite:cc}}. Next."
    assert clean_text(t) == "Spam filters fail. Next."
    assert clean_text("Attacks {{cite:aa}}, {{cite:bb}}; and more, {{cite:cc}}.") == "Attacks, and more."


def test_urls_removed():
    assert clean_text("We use the architecture we described in.https://github.com/UKPLab/x-y and more.") \
        == "We use the architecture we described in. and more."
    assert clean_text("Words from ICON 16 http://ltrc.iiit.ac.in/icon2016/ were collected.") \
        == "Words from ICON 16 were collected."
    assert clean_text("See www.example.org/page for data.") == "See for data."


def test_latex_removed():
    t = r"We use \textbf{BERT}~\cite{x} with $F_1$ and \emph{small} {adapters}."
    assert clean_text(t) == "We use BERT with and small adapters."


def test_legacy_markers():
    assert clean_text("As shown in [CITATION] and [3, 4].") == "As shown in and."


def test_sentence_split_keeps_abbreviations_and_markers():
    p = "Houlsby et al. {{cite:aa}} add adapters, e.g. small layers. Results improve. {{cite:bb}}."
    s = split_sentences(p)
    assert len(s) == 2
    assert s[0].startswith("Houlsby et al. {{cite:aa}}")
    assert s[1].endswith("{{cite:bb}}.")  # marker-only tail merged into previous sentence


def test_arxiv_id_normalisation():
    assert normalise_arxiv_id("arXiv:2106.09685v2") == "2106.09685"
    assert normalise_arxiv_id("https://arxiv.org/abs/1902.00751") == "1902.00751"
    assert normalise_arxiv_id("cs.CL/0112017v1") == "cs/0112017"
    assert normalise_arxiv_id("hep-th/9901001") == "hep-th/9901001"
    assert normalise_arxiv_id("not an id") is None
    assert normalise_arxiv_id("") is None


def test_parse_record_resolves_citations():
    rec = {
        "paper_id": "2101.00001",
        "metadata": {"title": "T", "categories": "cs.CL", "abstract": "a",
                     "versions": [{"created": "Fri, 1 Jan 2021 00:00:00 GMT"}]},
        "abstract": {"text": "An abstract."},
        "body_text": [{"section": "Intro", "content_type": "paragraph",
                       "text": "First sentence here. Adapters help {{cite:r1}} {{cite:r2}} {{cite:r3}}. Last one."}],
        "bib_entries": {"r1": {"ids": {"arxiv_id": "1902.00751v2"}},
                        "r2": {"ids": {"arxiv_id": ""}, "contained_arXiv_ids": [{"id": "1810.04805"}]},
                        "r3": {"ids": {"arxiv_id": ""}}},
    }
    p = parse_record(rec)
    assert p.paper_id == "2101.00001" and p.year == 2021
    sents = p.paragraphs[0]
    assert [s.cited_arxiv_ids for s in sents] == [[], ["1902.00751", "1810.04805"], []]
    assert sents[1].section == "intro"
