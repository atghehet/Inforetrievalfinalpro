import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def sample_tables():
    """Core tables built from a small synthetic unarXive sample (in memory)."""
    import make_sample
    from citeds.tables import build_tables
    from citeds.unarxive import parse_record

    recs = make_sample.make(250, seed=1)
    parsed = [parse_record(r) for r in recs if "cs.CL" in r["metadata"]["categories"].split()]
    return build_tables(parsed)
