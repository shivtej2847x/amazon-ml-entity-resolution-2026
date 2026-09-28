from pathlib import Path
from collections import Counter
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


def count_rows(path):
    """Count TSV data rows without loading the whole file."""
    count = 0

    with open(path, "rb") as f:
        for _ in f:
            count += 1

    # subtract header
    return count - 1


def country_distribution(path):
    counts = Counter()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["country"],
        dtype=str,
        chunksize=100000,
        keep_default_na=False
    ):
        counts.update(chunk["country"].str.strip())

    return counts


def ground_truth_distribution(path):
    match_counts = Counter()
    total = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100000,
        keep_default_na=False
    ):
        for value in chunk["matched_entity_ids"]:
            value = str(value).strip()

            if value == "":
                n = 0
            else:
                n = len([
                    x for x in value.split(",")
                    if x.strip()
                ])

            match_counts[n] += 1
            total += 1

    return total, match_counts


print("\n" + "=" * 80)
print("FULL DATASET ROW COUNTS")
print("=" * 80)

for name, path in FILES.items():

    if not path.exists():
        print(f"{name}: FILE NOT FOUND")
        continue

    print(f"{name:20s}: counting...", end="", flush=True)

    rows = count_rows(path)

    print(f" {rows:,} rows")


print("\n" + "=" * 80)
print("COUNTRY DISTRIBUTION")
print("=" * 80)

for name in [
    "train_source1",
    "train_source2",
    "train_source3",
    "test_source1",
    "test_source2",
    "test_source3"
]:

    path = FILES[name]

    print(f"\n{name}:")

    counts = country_distribution(path)

    total = sum(counts.values())

    for country, count in counts.most_common():
        percentage = count / total * 100
        print(f"  {country:15s}: {count:10,} ({percentage:.2f}%)")


print("\n" + "=" * 80)
print("GROUND TRUTH MATCH DISTRIBUTION")
print("=" * 80)

gt_total, match_counts = ground_truth_distribution(
    FILES["train_ground_truth"]
)

print(f"\nTotal Source 1 training entities: {gt_total:,}")

for n in sorted(match_counts):

    count = match_counts[n]
    percentage = count / gt_total * 100

    if n == 0:
        label = "0 matches (singleton)"
    elif n == 1:
        label = "1 match"
    elif n == 2:
        label = "2 matches"
    elif n == 3:
        label = "3 matches"
    else:
        label = f"{n} matches"

    print(f"  {label:25s}: {count:10,} ({percentage:.2f}%)")


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)