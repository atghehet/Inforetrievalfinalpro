"""SQLite-backed field postings, with the same FTS5 analyzer for documents and queries."""
import json
import math
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

STOP = set("a an and are as at be been by for from in into is it of on or that the their these this those to was were with".split())
WORDS = re.compile(r"[^\W_]+", re.UNICODE)
FIELDS = ("title", "abstract", "context")
VARIANTS = ("abstract", "context", "expanded", "fielded")


def clean(text):
    return " ".join(w for w in WORDS.findall(text.lower()) if w not in STOP)


def records(path):
    with path.open(encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def table(db, name, stem):
    analyzer = "porter unicode61" if stem else "unicode61"
    db.execute(f"CREATE VIRTUAL TABLE {name} USING fts5(title,abstract,context,tokenize='{analyzer}')")
    db.execute(f"CREATE VIRTUAL TABLE {name}_terms USING fts5vocab({name},instance)")


def window_text(row, width):
    before = row.get("before", [])
    after = row.get("after", [])
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("before and after must be arrays in reading order")
    return " ".join([*(before[-width:] if width else []), row["sentence"], *after[:width]])


def build(data, path, variant, width=0, stem=False, method_only=False):
    if variant not in VARIANTS or width not in (0, 1, 2):
        raise ValueError("Invalid representation or context window")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        db = sqlite3.connect(Path(tmp) / "index.db")
        try:
            db.executescript("""
                CREATE TABLE papers(id TEXT PRIMARY KEY, title TEXT, abstract TEXT);
                CREATE TABLE contexts(id TEXT PRIMARY KEY, target TEXT, passage TEXT);
                CREATE INDEX targets ON contexts(target);
                CREATE TABLE docs(id TEXT UNIQUE, lt INTEGER, la INTEGER, lc INTEGER);
                CREATE TABLE config(stem INTEGER);
            """)
            db.execute("INSERT INTO config VALUES(?)", (int(stem),))
            db.executemany("INSERT INTO papers VALUES(?,?,?)",
                           ((str(r["paper_id"]), r["title"], r["abstract"])
                            for r in records(data / "papers.jsonl")))
            if variant != "abstract":
                for r in records(data / "contexts.jsonl"):
                    if method_only and r["label"].lower() != "method":
                        continue
                    db.execute("INSERT INTO contexts VALUES(?,?,?)",
                               (r["context_id"], str(r["cited_paper_id"]), window_text(r, width)))
                if db.execute("SELECT 1 FROM contexts c LEFT JOIN papers p ON c.target=p.id WHERE p.id IS NULL LIMIT 1").fetchone():
                    raise ValueError("Context target missing from papers.jsonl")
            table(db, "idx", stem)
            for pid, title, abstract in db.execute("SELECT * FROM papers ORDER BY id"):
                passages = db.execute("SELECT passage FROM contexts WHERE target=? ORDER BY id", (pid,))
                fields = [title if variant == "fielded" else "",
                          abstract if variant != "context" else "",
                          " ".join(p[0] for p in passages)]
                fields = [clean(f) for f in fields]
                rid = db.execute("INSERT INTO docs VALUES(?,0,0,0)", (pid,)).lastrowid
                db.execute("INSERT INTO idx(rowid,title,abstract,context) VALUES(?,?,?,?)", (rid,*fields))
            db.executescript("""
                CREATE TABLE postings(term TEXT, doc INTEGER, field TEXT, tf INTEGER,
                                      PRIMARY KEY(term,doc,field)) WITHOUT ROWID;
                INSERT INTO postings SELECT term,doc,col,count(*) FROM idx_terms GROUP BY term,doc,col;
                CREATE INDEX forward ON postings(doc);
                CREATE TABLE stats(term TEXT PRIMARY KEY, df INTEGER, cf INTEGER);
                INSERT INTO stats SELECT term,count(DISTINCT doc),sum(tf) FROM postings GROUP BY term;
                UPDATE docs SET (lt,la,lc)=(SELECT coalesce(sum(tf) FILTER(WHERE field='title'),0),
                    coalesce(sum(tf) FILTER(WHERE field='abstract'),0),
                    coalesce(sum(tf) FILTER(WHERE field='context'),0) FROM postings WHERE doc=docs.rowid);
                DROP TABLE idx_terms; DROP TABLE idx; DROP TABLE papers; DROP TABLE contexts;
            """)
            if not db.execute("SELECT 1 FROM stats LIMIT 1").fetchone():
                raise ValueError("Index has no searchable terms")
            db.commit()
            db.execute("VACUUM")
        finally:
            db.close()
        (Path(tmp) / "index.db").replace(path)


class Search:
    def __init__(self, path):
        if not path.is_file():
            raise FileNotFoundError(path)
        self.db = sqlite3.connect(path)
        self.tokenizer = sqlite3.connect(":memory:")
        table(self.tokenizer, "q", self.db.execute("SELECT stem FROM config").fetchone()[0])
        self.ids = {}
        self.length = {}
        for doc, pid, *lengths in self.db.execute("SELECT rowid,* FROM docs"):
            self.ids[doc] = pid
            self.length[doc] = lengths
        self.n = len(self.ids)
        self.avg = [sum(x[i] for x in self.length.values()) / self.n for i in range(3)]
        self.total = sum(sum(x) for x in self.length.values())

    def close(self):
        self.db.close()
        self.tokenizer.close()

    def terms(self, query):
        self.tokenizer.execute("DELETE FROM q")
        self.tokenizer.execute("INSERT INTO q(title) VALUES(?)", (clean(query),))
        return Counter(dict(self.tokenizer.execute("SELECT term,count(*) FROM q_terms GROUP BY term")))

    def rank(self, terms, model, k=100, k1=1.2, b=.75, mu=1000, weights=(1,1,1)):
        score = defaultdict(float)
        known = 0.0
        for term, qw in terms.items():
            stat = self.db.execute("SELECT df,cf FROM stats WHERE term=?", (term,)).fetchone()
            if not stat:
                continue
            df, cf = stat
            known += qw
            idf = math.log1p((self.n-df+.5)/(df+.5))
            by_doc = defaultdict(dict)
            for doc, field, tf in self.db.execute("SELECT doc,field,tf FROM postings WHERE term=?", (term,)):
                by_doc[doc][FIELDS.index(field)] = tf
            for doc, counts in by_doc.items():
                tf = sum(counts.values())
                lengths = self.length[doc]
                if model == "ql":
                    value = math.log1p(tf/(mu*cf/self.total))
                elif model == "tfidf":
                    value = (1+math.log(tf))*(math.log((1+self.n)/(1+df))+1)
                else:
                    if model == "bm25f":
                        norm = sum(weights[i]*freq/(1-b+b*lengths[i]/self.avg[i])
                                   for i,freq in counts.items())
                    elif model == "bm25":
                        norm = tf/(1-b+b*sum(lengths)/sum(self.avg))
                    else:
                        raise ValueError(model)
                    value = idf*(k1+1)*norm/(k1+norm)
                score[doc] += qw*value
        if model == "ql" and known:
            # Keep the document length prior even when all query terms miss the document.
            for doc, lengths in self.length.items():
                score[doc] -= known*math.log1p(sum(lengths)/mu)
        import heapq
        return heapq.nsmallest(k, score.items(), key=lambda pair: (-pair[1], self.ids[pair[0]]))

    def search(self, query, model, k=100, feedback_docs=5, feedback_terms=10, original=.5, **params):
        terms = self.terms(query)
        if model == "rm3" and terms:
            # QL log scores differ from log P(query|doc) by a query-wide constant.
            top = self.rank(terms, "ql", k=feedback_docs, mu=params.get("mu",1000))
            feedback = Counter()
            for doc, score in top:
                length = sum(self.length[doc])
                if length:
                    posterior = math.exp(score-top[0][1])
                    for term, count in self.db.execute("SELECT term,sum(tf) FROM postings WHERE doc=? GROUP BY term", (doc,)):
                        feedback[term] += posterior*count/length
            chosen = dict(feedback.most_common(feedback_terms))
            if chosen:
                mixed = Counter({t:original*c/sum(terms.values()) for t,c in terms.items()})
                mass = sum(chosen.values())
                for t,c in chosen.items():
                    mixed[t] += (1-original)*c/mass
                terms = mixed
        result = self.rank(terms, "bm25" if model == "rm3" else model, k=k, **params)
        return [(self.ids[d], s) for d,s in result]
