"""Convert P1's processed data into P2's JSONL input (data/p2_input/).

    python p3/src/prepare_p2_data.py

Kept for backwards compatibility: the conversion now lives in P1's package
(`python -m citeds.export_p2`), which writes citation-function labels and up to
two neighbour sentences per side, as P2's runner expects.
"""
from common import REPO_ROOT  # noqa: F401  (puts the repo on sys.path)
from citeds.export_p2 import main

if __name__ == "__main__":
    main()
