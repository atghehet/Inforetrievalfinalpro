
from pathlib import Path
import pandas as pd

project_folder = Path.home() / "Desktop" / "CitationSearchProject"
p1_folder = project_folder / "reference_p1" / "Inforetrievalfinalpro-p1-data-pipeline"
data_folder = p1_folder / "data" / "processed"

for filename in ["papers.parquet", "queries.parquet", "index_contexts.parquet"]:
    file_path = data_folder / filename
    df = pd.read_parquet(file_path)

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 2 rows:")
    print(df.head(2).to_string(index=False))

print("\n" + "=" * 60)
print("QUERY SPLITS")
print("=" * 60)

queries = pd.read_parquet(data_folder / "queries.parquet")
print(queries["split"].value_counts().sort_index())
