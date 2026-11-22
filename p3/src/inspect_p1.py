"""Print the shape, columns and first rows of P1's processed tables.

    python p3/src/inspect_p1.py
"""
from common import config, io

for name, loader in (("papers", io.load_papers), ("index_contexts", io.load_index_contexts),
                     ("queries", io.load_queries)):
    df = loader()
    print("\n" + "=" * 60 + f"\n{name}  shape={df.shape}\ncolumns={list(df.columns)}\n" + "=" * 60)
    print(df.head(2).to_string(index=False)[:1500])
print("\nQuery splits:", io.load_queries()["split"].value_counts().to_dict())
print("Data folder:", config.PROCESSED_DIR)
