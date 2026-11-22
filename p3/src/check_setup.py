"""Check Python, packages and data needed by P3.

    python p3/src/check_setup.py
"""
import importlib
import sys

from common import config

print("Python:", sys.version.split()[0], "|", sys.executable)
for pkg in ("numpy", "pandas", "pyarrow", "sklearn", "torch", "sentence_transformers", "faiss"):
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg:22s} {getattr(m, '__version__', 'ok')}")
    except ImportError:
        print(f"  {pkg:22s} MISSING  -> pip install -r p3/requirements.txt")
print("Processed data:", config.PROCESSED_DIR, "found" if config.PAPERS_PATH.exists() else "MISSING")
