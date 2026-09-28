from pathlib import Path
import re
import unicodedata
import pandas as pd


BASE = Path(__file__).parent
DATASET = BASE / "dataset"

S1_FILE = DATASET / "train" / "train_source1.tsv"
S2_FILE = DATASET / "train" / "train_source2.tsv"
S3_FILE = DATASET / "train" / "train_source3.tsv"
GT_FILE = DATASET / "train" / "train_ground_truth.tsv"


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

SAMPLE_SIZE = 10000
RANDOM_STATE = 42


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def normalize_text(text):
    if not text:
        return ""

    text = str(text).lower().strip()

    # Remove accents where possible
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def token_signature(text):
    text = normalize_text(text)

    if not text:
        return ""

    tokens = text.split()

    # Remove extremely short tokens
    tokens = [
        token for token in tokens
        if len(token) >= 2
    ]

    return " ".join(sorted(set(tokens)))


def address_tokens(text):
    text = normalize_text(text)

    if not text:
        return []

    tokens = text.split()

    return [
        token for token in set(tokens)
        if len(token) >= 4
    ]


# ------------------------------------------------------------
# READ DATA
# ------------------------------------------------------------

print("=" * 80)
print("LOADING SAMPLE")
print("=" * 80)

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

# Only entities with matches
gt = gt[
    gt["matched_entity_ids"].str.strip() != ""
]

gt = gt.sample(
    n=min(SAMPLE_SIZE, len(gt)),
    random_state=RANDOM_STATE
)

s1_ids = set(gt["source1_entity_id"])

s2_ids = set()
s3_ids = set()

for value in gt["matched_entity_ids"]:

    for entity_id in value.split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


print("S1 sampled:", len(s1_ids))
print("S2 true matches:", len(s2_ids))
print("S3 true matches:", len(s3_ids))


# ------------------------------------------------------------
# LOAD ONLY REQUIRED S1
# ------------------------------------------------------------

def load_required(path, ids):

    pieces = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100000,
        keep_default_na=False
    ):

        result = chunk[
            chunk["entity_id"].isin(ids)
        ]

        if len(result):
            pieces.append(result)

    return pd.concat(pieces, ignore_index=True)


print("\nLoading Source 1...")
s1 = load_required(S1_FILE, s1_ids)

print("Loading Source 2...")
s2 = load_required(S2_FILE, s2_ids)

print("Loading Source 3...")
s3 = load_required(S3_FILE, s3_ids)


# ------------------------------------------------------------
# CREATE NORMALIZED COLUMNS
# ------------------------------------------------------------

for df in [s1, s2, s3]:

    df["name_norm"] = df["business_name"].map(
        normalize_text
    )

    df["name_signature"] = df["business_name"].map(
        token_signature
    )

    df["address_norm"] = df["business_address"].map(
        normalize_text
    )

    df["address_signature"] = df["business_address"].map(
        token_signature
    )


# ------------------------------------------------------------
# CREATE BLOCKING DICTIONARIES
# ------------------------------------------------------------

print("\nCreating blocking indexes...")


def make_index(df, column):

    index = {}

    for _, row in df.iterrows():

        value = row[column]

        if not value:
            continue

        index.setdefault(value, []).append(
            row["entity_id"]
        )

    return index


indexes = {}

for source_name, df in [
    ("s2", s2),
    ("s3", s3)
]:

    indexes[source_name] = {

        "name_norm":
            make_index(df, "name_norm"),

        "name_signature":
            make_index(df, "name_signature"),

        "address_norm":
            make_index(df, "address_norm"),

        "address_signature":
            make_index(df, "address_signature"),
    }


# ------------------------------------------------------------
# EVALUATE BLOCKING RECALL
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("BLOCKING RECALL")
print("=" * 80)


total_true_matches = 0

found_name = 0
found_name_signature = 0
found_address = 0
found_address_signature = 0
found_any = 0


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    gt_row = gt[
        gt["source1_entity_id"] == s1_id
    ]

    if gt_row.empty:
        continue

    true_ids = set(
        x.strip()
        for x in gt_row.iloc[0]["matched_entity_ids"].split(",")
        if x.strip()
    )

    total_true_matches += len(true_ids)

    candidates_name = set()
    candidates_name_sig = set()
    candidates_address = set()
    candidates_address_sig = set()

    for source_name in ["s2", "s3"]:

        idx = indexes[source_name]

        candidates_name.update(
            idx["name_norm"].get(
                row["name_norm"], []
            )
        )

        candidates_name_sig.update(
            idx["name_signature"].get(
                row["name_signature"], []
            )
        )

        candidates_address.update(
            idx["address_norm"].get(
                row["address_norm"], []
            )
        )

        candidates_address_sig.update(
            idx["address_signature"].get(
                row["address_signature"], []
            )
        )

    all_candidates = (
        candidates_name |
        candidates_name_sig |
        candidates_address |
        candidates_address_sig
    )

    found_name += len(
        true_ids & candidates_name
    )

    found_name_signature += len(
        true_ids & candidates_name_sig
    )

    found_address += len(
        true_ids & candidates_address
    )

    found_address_signature += len(
        true_ids & candidates_address_sig
    )

    found_any += len(
        true_ids & all_candidates
    )


print(
    f"\nTotal true matches examined: "
    f"{total_true_matches:,}"
)


def show(name, value):

    recall = (
        value / total_true_matches * 100
        if total_true_matches
        else 0
    )

    print(
        f"{name:30s}: "
        f"{value:,} "
        f"({recall:.2f}% recall)"
    )


show(
    "Exact normalized name",
    found_name
)

show(
    "Sorted name tokens",
    found_name_signature
)

show(
    "Exact normalized address",
    found_address
)

show(
    "Sorted address tokens",
    found_address_signature
)

show(
    "UNION of all blockers",
    found_any
)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)