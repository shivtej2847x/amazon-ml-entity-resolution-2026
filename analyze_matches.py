from pathlib import Path
import random
import pandas as pd
from rapidfuzz.fuzz import ratio


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).parent
DATASET = BASE / "dataset"

GT_FILE = DATASET / "train" / "train_ground_truth.tsv"
S1_FILE = DATASET / "train" / "train_source1.tsv"
S2_FILE = DATASET / "train" / "train_source2.tsv"
S3_FILE = DATASET / "train" / "train_source3.tsv"


# ============================================================
# SETTINGS
# ============================================================

NUMBER_OF_EXAMPLES = 100
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# STEP 1: SAMPLE GROUND TRUTH
# ============================================================

print("=" * 80)
print("STEP 1: READING GROUND TRUTH")
print("=" * 80)

sample_rows = []

for chunk in pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    chunksize=100000,
    keep_default_na=False
):
    # Keep only entities that actually have matches
    chunk = chunk[
        chunk["matched_entity_ids"].str.strip() != ""
    ]

    if len(chunk) > 0:
        sample_rows.append(chunk)


gt = pd.concat(sample_rows, ignore_index=True)

print("Ground-truth entities with at least one match:", len(gt))

# Randomly select examples
if len(gt) > NUMBER_OF_EXAMPLES:
    gt_sample = gt.sample(
        n=NUMBER_OF_EXAMPLES,
        random_state=RANDOM_SEED
    )
else:
    gt_sample = gt


# ============================================================
# STEP 2: EXTRACT REQUIRED IDs
# ============================================================

s1_ids = set(gt_sample["source1_entity_id"])

s2_ids = set()
s3_ids = set()

for value in gt_sample["matched_entity_ids"]:

    for entity_id in value.split(","):

        entity_id = entity_id.strip()

        if not entity_id:
            continue

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


print("\nSelected Source 1 entities:", len(s1_ids))
print("Required Source 2 records:", len(s2_ids))
print("Required Source 3 records:", len(s3_ids))


# ============================================================
# STEP 3: FIND SOURCE 1 RECORDS
# ============================================================

print("\n" + "=" * 80)
print("STEP 2: LOADING MATCHED RECORDS")
print("=" * 80)

s1_records = []

for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    chunksize=100000,
    keep_default_na=False
):
    matched = chunk[
        chunk["entity_id"].isin(s1_ids)
    ]

    if len(matched) > 0:
        s1_records.append(matched)


s1_df = pd.concat(s1_records, ignore_index=True)

print("Source 1 records found:", len(s1_df))


# ============================================================
# STEP 4: FIND SOURCE 2 RECORDS
# ============================================================

print("Searching Source 2...")

s2_records = []

for chunk in pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    chunksize=100000,
    keep_default_na=False
):
    matched = chunk[
        chunk["entity_id"].isin(s2_ids)
    ]

    if len(matched) > 0:
        s2_records.append(matched)


if s2_records:
    s2_df = pd.concat(s2_records, ignore_index=True)
else:
    s2_df = pd.DataFrame(
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )


print("Source 2 records found:", len(s2_df))


# ============================================================
# STEP 5: FIND SOURCE 3 RECORDS
# ============================================================

print("Searching Source 3...")

s3_records = []

for chunk in pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str,
    chunksize=100000,
    keep_default_na=False
):
    matched = chunk[
        chunk["entity_id"].isin(s3_ids)
    ]

    if len(matched) > 0:
        s3_records.append(matched)


if s3_records:
    s3_df = pd.concat(s3_records, ignore_index=True)
else:
    s3_df = pd.DataFrame(
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )


print("Source 3 records found:", len(s3_df))


# ============================================================
# STEP 6: CREATE LOOKUP DICTIONARIES
# ============================================================

s1_lookup = s1_df.set_index("entity_id").to_dict("index")
s2_lookup = s2_df.set_index("entity_id").to_dict("index")
s3_lookup = s3_df.set_index("entity_id").to_dict("index")


# ============================================================
# STEP 7: PRINT MATCH EXAMPLES
# ============================================================

print("\n" + "=" * 80)
print("ACTUAL MATCH EXAMPLES")
print("=" * 80)


for i, (_, row) in enumerate(gt_sample.iterrows(), start=1):

    s1_id = row["source1_entity_id"]

    s1 = s1_lookup.get(s1_id)

    if not s1:
        continue

    print("\n" + "-" * 80)
    print(f"EXAMPLE {i}")
    print("-" * 80)

    print("\nSOURCE 1")
    print("ID      :", s1_id)
    print("Name    :", s1["business_name"])
    print("Address :", s1["business_address"])
    print("Country :", s1["country"])

    matched_ids = [
        x.strip()
        for x in row["matched_entity_ids"].split(",")
        if x.strip()
    ]

    for matched_id in matched_ids:

        if matched_id.startswith("S2-"):
            record = s2_lookup.get(matched_id)
        else:
            record = s3_lookup.get(matched_id)

        if not record:
            continue

        name_similarity = ratio(
            s1["business_name"].lower(),
            record["business_name"].lower()
        )

        address_similarity = ratio(
            s1["business_address"].lower(),
            record["business_address"].lower()
        )

        print("\nMATCH:", matched_id)

        print("Name    :", record["business_name"])
        print("Address :", record["business_address"])
        print("Country :", record["country"])

        print(
            f"Name similarity    : {name_similarity:.1f}"
        )

        print(
            f"Address similarity : {address_similarity:.1f}"
        )


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)