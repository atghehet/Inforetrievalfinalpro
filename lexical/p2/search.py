import argparse
import json
from pathlib import Path
from .core import Search

def main():
    cli = argparse.ArgumentParser(description="Search a frozen P2 configuration.")
    cli.add_argument("out", type=Path)
    cli.add_argument("query")
    cli.add_argument("--system", default="expanded_bm25")
    args = cli.parse_args()
    config = json.loads((args.out/"best.json").read_text())[args.system]
    engine = Search(args.out/"indexes"/(config["index"]+".db"))
    try:
        for rank,(pid,score) in enumerate(engine.search(args.query,config["model"],k=10,**config["params"]),1):
            print(f"{rank:2d}  {pid:12s}  {score:.5f}")
    finally:
        engine.close()

if __name__ == "__main__":
    main()
