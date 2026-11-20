import json
import math
import tempfile
import unittest
from pathlib import Path

from p2.core import Search, build, window_text

class LexicalTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        def put(name, rows):
            (self.root/name).write_text("".join(json.dumps(r)+"\n" for r in rows))
        put("papers.jsonl", [dict(paper_id="A",title="quasar",abstract="running running"),
                             dict(paper_id="B",title="",abstract="star")])
        put("contexts.jsonl",[dict(context_id="C",cited_paper_id="B",sentence="quasar",label="background")])

    def engine(self,variant,**kw):
        path = self.root/(variant+".db")
        build(self.root,path,variant,**kw)
        db = Search(path)
        self.addCleanup(db.close)
        return db

    def test_stem(self):
        self.assertEqual(self.engine("abstract").search("run","bm25"),[])
        self.assertEqual(self.engine("expanded",stem=True).search("run","bm25")[0][0],"A")

    def test_bm25_and_ql(self):
        db = self.engine("abstract")
        d,s = db.search("running","bm25")[0]
        self.assertEqual(d,"A")
        self.assertAlmostEqual(s, math.log(2)*2*2.2/(2+1.2*(.25+.75*2/1.5)))
        hits = dict(db.search("running","ql",mu=3))
        self.assertAlmostEqual(hits["A"], math.log((2+2)/(2+3))-math.log(2/3))
        self.assertAlmostEqual(hits["B"], math.log(2/(1+3))-math.log(2/3))

    def test_expansion_and_field_weights(self):
        self.assertEqual(self.engine("abstract").search("quasar","bm25"),[])
        self.assertEqual(self.engine("expanded").search("quasar","bm25")[0][0],"B")
        db = self.engine("fielded")
        self.assertEqual(db.search("quasar","bm25f",weights=(4,1,1))[0][0],"A")
        self.assertEqual(db.search("quasar","bm25f",weights=(1,1,4))[0][0],"B")

    def test_filter_window_and_empty_query(self):
        db = self.engine("expanded",method_only=True)
        self.assertEqual(db.search("quasar","rm3"),[])
        self.assertEqual(db.search("and the","ql"),[])
        row = dict(sentence="c",before=["a","b"],after=["d","e"])
        self.assertEqual([window_text(row,i) for i in range(3)],["c","b c d","a b c d e"])

if __name__=="__main__":
    unittest.main()
