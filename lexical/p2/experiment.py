"""Run from the repository root: python3 -m p2.experiment --help"""
import argparse
import csv
import itertools
import json
import statistics
import time
from pathlib import Path

from .core import VARIANTS, Search, build, records


def grid(model):
    if model == "bm25":
        return [dict(k1=k, b=b) for k,b in itertools.product((.8,1.2,1.6),(.4,.75))]
    if model == "bm25f":
        return [dict(weights=w, k1=1.2, b=.75) for w in [(1,1,1),(2,1,1),(2,1,2)]]
    if model == "ql":
        return [dict(mu=m) for m in (300,1000,2000)]
    if model == "rm3":
        return [dict(feedback_docs=d, feedback_terms=10, original=a)
                for d,a in itertools.product((3,5),(.5,.8))]
    return [{}]


def load_queries(root, split):
    result = {}
    for row in records(root / f"{split}.jsonl"):
        key = str(row["query_id"])
        if key in result or not key or any(c.isspace() for c in key):
            raise ValueError(f"Duplicate or invalid query ID: {key}")
        result[key] = row["text"]
    if not result:
        raise ValueError(f"No {split} queries")
    return result


def run(engine, queries, model, params, path, tag, qrels=None):
    ranks, latency = [], []
    with path.open("w", encoding="utf8") as out:
        for qid, query in queries.items():
            start = time.perf_counter()
            hits = engine.search(query, model, k=100, **params)
            latency.append(1000*(time.perf_counter()-start))
            rr = 0
            for i,(pid,score) in enumerate(hits,1):
                if any(c.isspace() for c in pid) or not pid:
                    raise ValueError(f"Invalid TREC paper ID: {pid!r}")
                out.write(f"{qid} Q0 {pid} {i} {score:.15g} {tag}\n")
                if qrels and i <= 10 and not rr and pid in qrels[qid]:
                    rr = 1/i
            ranks.append(rr)
    return statistics.mean(ranks), statistics.median(latency), statistics.mean(latency)


def main():
    p = argparse.ArgumentParser(description="Build P2 indexes, tune on validation and export frozen test runs.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--method-filter", action="store_true", help="Also test method-only contexts; requires P1 labels")
    p.add_argument("--quick", action="store_true", help="One parameter setting, sentence only, no stemming")
    args = p.parse_args()
    out, data = args.out, args.data
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error("Use a new or empty output directory to avoid mixing experiment versions.")
    val = load_queries(data,"validation")
    qrels = {qid:set() for qid in val}
    for line in (data/"validation.qrels").read_text().splitlines():
        qid, _, pid, rel = line.split()
        if int(rel)>0:
            qrels[qid].add(pid)
    if not all(qrels.values()):
        raise ValueError("Every validation query needs a positive judgment")
    for folder in ("indexes","validation","test","fusion"):
        (out/folder).mkdir()
    best = {}
    with (out/"tuning.csv").open("w",newline="") as log, (out/"efficiency.csv").open("w",newline="") as perf:
        writer, sizes = csv.writer(log), csv.writer(perf)
        writer.writerow(["run","variant","model","parameters","validation_mrr_at_10","median_ms","mean_ms"])
        sizes.writerow(["index","build_seconds","bytes"])
        for variant in VARIANTS:
            widths = [0] if args.quick or variant=="abstract" else [0,1,2]
            filters = [False,True] if args.method_filter and variant!="abstract" else [False]
            for stem,width,filtered in itertools.product([False] if args.quick else [False,True],widths,filters):
                name = f"{variant}-w{width}-{'porter' if stem else 'plain'}-{'method' if filtered else 'all'}"
                path = out/"indexes"/f"{name}.db"
                start = time.perf_counter()
                build(data,path,variant,width,stem,filtered)
                sizes.writerow([name,time.perf_counter()-start,path.stat().st_size])
                print(f"Built {name}",flush=True)
                engine = Search(path)
                try:
                    paper_ids = set(engine.ids.values())
                    if any(pid not in paper_ids for relevant in qrels.values() for pid in relevant):
                        raise ValueError("A validation target is missing from papers.jsonl")
                    for model in (["bm25f"] if variant=="fielded" else ["tfidf","bm25","ql","rm3"]):
                        configs = grid(model)
                        for index,params in enumerate(configs[:1] if args.quick else configs):
                            tag = f"{name}-{model}-{index}"
                            score,median,mean = run(engine,val,model,params,out/"validation"/f"{tag}.run",tag,qrels)
                            writer.writerow([tag,variant,model,json.dumps(params),score,median,mean])
                            key = f"{variant}_{model}"
                            if key not in best or score>best[key]["validation_mrr_at_10"]:
                                best[key] = dict(index=name,model=model,params=params,tag=tag,validation_mrr_at_10=score)
                finally:
                    engine.close()
                log.flush()
                perf.flush()
    (out/"best.json").write_text(json.dumps(best,indent=2)+"\n")
    test = load_queries(data,"test")
    if val.keys() & test.keys():
        raise ValueError("Validation and test query IDs overlap")
    for key,config in best.items():
        engine = Search(out/"indexes"/(config["index"]+".db"))
        try:
            run(engine,test,config["model"],config["params"],out/"test"/f"{key}.run",config["tag"])
        finally:
            engine.close()
    bmkey = max((k for k in best if best[k]["model"]=="bm25"),key=lambda k:best[k]["validation_mrr_at_10"])
    import shutil
    shutil.copyfile(out/"validation"/(best[bmkey]["tag"]+".run"),out/"fusion"/"bm25.validation.run")
    shutil.copyfile(out/"test"/f"{bmkey}.run",out/"fusion"/"bm25.test.run")
    (out/"fusion"/"selection.txt").write_text(best[bmkey]["tag"]+"\n")
    print(f"Finished. Give {out/'fusion'} to P3 and runs plus logs to P4.")


if __name__ == "__main__":
    main()
