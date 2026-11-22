"""Text cleaning: citation markers, LaTeX artefacts, sentence splitting."""
from __future__ import annotations

import re
import unicodedata

# unarXive 2022 marks citations as {{cite:<uuid>}} and other refs as {{figure:..}},
# {{table:..}}, {{formula:..}}. Older dumps use [CITATION] or <cit .../>.
CITE_MARKER = re.compile(r"\{\{cite:([^{}\s]+)\}\}")
OTHER_MARKER = re.compile(r"\{\{(figure|table|formula|algorithm|ref|footnote):[^}]*\}\}")
LEGACY_CITE = re.compile(r"\[(?:CITATION|cite|\d+(?:\s*[,\u2013\-]\s*\d+)*)\]|<cit[^>]*/>")

_LATEX_DROP_WITH_ARG = re.compile(r"\\(?:cite[a-z]*|ref|eqref|autoref|cref|Cref|label)\*?(?:\[[^\]]*\])*\{[^{}]*\}")
_LATEX_CMD_WITH_ARG = re.compile(r"\\(?:textbf|textit|emph|texttt|mathrm|mathbf|text|url|footnote)\{([^{}]*)\}")
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?")
_URL = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)
_INLINE_MATH = re.compile(r"\$[^$]{0,200}\$")
_BRACES = re.compile(r"[{}]")
_EMPTY_BRACKETS = re.compile(r"\(\s*[;,]?\s*\)|\[\s*[;,]?\s*\]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)\]])")
_PUNCT_RUN = re.compile(r"([,;])(?:\s*[,;])+")          # ",,," -> ","
_PUNCT_BEFORE_END = re.compile(r"\s*[,;]+\s*([.!?])")  # ",." -> "."
_WS = re.compile(r"\s+")

# Abbreviations that should not end a sentence.
_ABBREV = ("e.g", "i.e", "et al", "etc", "vs", "cf", "Fig", "Figs", "Eq", "Eqs",
           "Sec", "Tab", "No", "resp", "approx", "al", "Dr", "Mr", "Ms")
_HAS_WORD = re.compile(r"[A-Za-z0-9]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[{])")


def find_cite_ids(text: str) -> list[str]:
    """Citation marker IDs in raw text, in order of appearance."""
    return CITE_MARKER.findall(text)


def clean_text(text: str) -> str:
    """Remove citation/ref markers and LaTeX artefacts; normalise whitespace."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = CITE_MARKER.sub(" ", t)
    t = OTHER_MARKER.sub(" ", t)
    t = LEGACY_CITE.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _INLINE_MATH.sub(" ", t)
    t = _LATEX_DROP_WITH_ARG.sub(" ", t)
    for _ in range(3):  # nested commands
        t = _LATEX_CMD_WITH_ARG.sub(r"\1", t)
    t = _LATEX_CMD.sub(" ", t)
    t = _BRACES.sub("", t)
    t = t.replace("~", " ")
    t = _WS.sub(" ", t)
    t = _EMPTY_BRACKETS.sub(" ", t)
    t = _WS.sub(" ", t)
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = _PUNCT_RUN.sub(r"\1", t)
    t = _PUNCT_BEFORE_END.sub(r"\1", t)
    return t.strip(" ,;")


def split_sentences(paragraph: str) -> list[str]:
    """Rule-based sentence splitter that keeps {{cite:..}} markers intact.

    Splits on . ! ? followed by whitespace and an upper-case/digit start, but
    not after common abbreviations ("et al.", "e.g.") and not inside markers.
    """
    if not paragraph:
        return []
    # Protect markers and abbreviations from splitting.
    protected = CITE_MARKER.sub(lambda m: m.group(0).replace(".", "\x00"), paragraph)
    for ab in _ABBREV:
        protected = re.sub(rf"\b{re.escape(ab)}\.", ab.replace(".", "\x00") + "\x00", protected)
    parts = _SENT_SPLIT.split(protected)
    out = []
    for p in parts:
        p = p.replace("\x00", ".").strip()
        if p:
            out.append(p)
    # A sentence that is only citation markers belongs to the previous one.
    merged: list[str] = []
    for s in out:
        if merged and not _HAS_WORD.search(clean_text(s)):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def num_tokens(text: str) -> int:
    return len(text.split())


_NORM = re.compile(r"[^a-z0-9 ]+")


def normalise_for_match(text: str) -> str:
    """Lower-case, strip punctuation: used for exact-duplicate detection."""
    return _WS.sub(" ", _NORM.sub(" ", text.lower())).strip()


_ARXIV_NEW = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?$")
_ARXIV_OLD = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$")


def normalise_arxiv_id(raw: str | None) -> str | None:
    """'arXiv:2106.09685v2' -> '2106.09685'; 'cs.CL/0112017' -> 'cs/0112017'."""
    if not raw:
        return None
    s = str(raw).strip()
    s = re.sub(r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:)", "", s, flags=re.I)
    s = s.removesuffix(".pdf")
    m = _ARXIV_NEW.search(s)
    if m:
        return m.group(1)
    m = _ARXIV_OLD.search(s)
    if m:
        archive, num = m.group(1).split("/")
        return f"{archive.split('.')[0]}/{num}"
    return None
