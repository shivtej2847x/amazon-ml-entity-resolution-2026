from pathlib import Path
import pandas as pd


BASE = Path(__file__).parent
DATASET = BASE / "dataset"


FILES = {
    "train_source1": DATASET / "train" / "train_source1.tsv",
    "train_source2": DATASET / "train" / "train_source2.tsv",
    "train_source3": DATASET / "train" / "train_source3.tsv",
    "train_ground_truth": DATASET / "train" / "train_ground_truth.tsv",
    "test_source1": DATASET / "test" / "test_source1.tsv",
    "test_source2": DATASET / "test" / "test_source2.tsv",
    "test_source3": DATASET / "test" / "test_source3.tsv",
}


def inspect_file(name, path):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        return

    print("Path:", path)
    print("Size:", round(path.stat().st_size / (1024 ** 2), 2), "MB")

    # Read only a sample first
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=10000,
        dtype=str,
        keep_default_na=False
    )

    print("\nColumns:")
    print(list(df.columns))

    print("\nSample rows:")
    print(df.head(5).to_string(index=False))

    print("\nSample shape:")
    print(df.shape)

    print("\nMissing/empty values in sample:")

    for col in df.columns:
        empty = (df[col].astype(str).str.strip() == "").sum()
        print(f"  {col}: {empty}")

    print("\nUnique values in sample:")

    for col in df.columns:
        print(f"  {col}: {df[col].nunique()}")


for name, path in FILES.items():
    inspect_file(name, path)


print("\n\nInspection complete.")