from pathlib import Path
import re
import unicodedata
import pandas as pd


BASE = Path(__file__).parent
DATASET = BASE / "dataset"
TEST = DATASET / "test"
OUTPUT = BASE / "output"

OUTPUT.mkdir(exist_ok=True)

S1_FILE = TEST / "test_source1.tsv"
S2_FILE = TEST / "test_source2.tsv"
S3_FILE = TEST / "test_source3.tsv"

OUT_FILE = OUTPUT / "matching_results.tsv"

CHUNK_SIZE = 100_000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = unicodedata.normalize(
        "NFKD",
        text
    )

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def token_signature(text):

    text = normalize_text(text)

    if not text:
        return ""

    tokens = [
        t for t in text.split()
        if len(t) >= 2
    ]

    return " ".join(sorted(set(tokens)))


# ============================================================
# INDEX
# ============================================================

indexes = {
    "name": {},
    "name_sig": {},
    "address": {},
    "address_sig": {},
}


def add(index, key, entity_id):

    if not key:
        return

    index.setdefault(key, []).append(entity_id)


def build_index(path, label):

    print(f"\nIndexing {label}...")

    total = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            country = normalize_text(
                row.country
            )

            name = normalize_text(
                row.business_name
            )

            name_sig = token_signature(
                row.business_name
            )

            address = normalize_text(
                row.business_address
            )

            address_sig = token_signature(
                row.business_address
            )

            # Country is part of every key.
            # This prevents US/India/France collisions.

            add(
                indexes["name"],
                (country, name),
                entity_id
            )

            add(
                indexes["name_sig"],
                (country, name_sig),
                entity_id
            )

            add(
                indexes["address"],
                (country, address),
                entity_id
            )

            add(
                indexes["address_sig"],
                (country, address_sig),
                entity_id
            )

        total += len(chunk)

        print(
            f"\r{label}: {total:,}",
            end="",
            flush=True
        )

    print()


# ============================================================
# BUILD SOURCE 2 + SOURCE 3 INDEX
# ============================================================

print("=" * 70)
print("EMERGENCY SUBMISSION GENERATOR")
print("=" * 70)

build_index(
    S2_FILE,
    "Source 2"
)

build_index(
    S3_FILE,
    "Source 3"
)


# ============================================================
# PROCESS TEST SOURCE 1
# ============================================================

print("\nProcessing Source 1...")
print(f"Output: {OUT_FILE}")

# Delete old incomplete file
if OUT_FILE.exists():
    OUT_FILE.unlink()


first_write = True
processed = 0
matched_entities = 0


for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE,
    keep_default_na=False
):

    output_rows = []

    for row in chunk.itertuples(index=False):

        s1_id = row.entity_id

        country = normalize_text(
            row.country
        )

        name = normalize_text(
            row.business_name
        )

        name_sig = token_signature(
            row.business_name
        )

        address = normalize_text(
            row.business_address
        )

        address_sig = token_signature(
            row.business_address
        )

        candidates = set()

        # ----------------------------------------------------
        # Exact normalized name
        # ----------------------------------------------------

        candidates.update(
            indexes["name"].get(
                (country, name),
                []
            )
        )

        # ----------------------------------------------------
        # Sorted name tokens
        # ----------------------------------------------------

        candidates.update(
            indexes["name_sig"].get(
                (country, name_sig),
                []
            )
        )

        # ----------------------------------------------------
        # Exact normalized address
        # ----------------------------------------------------

        candidates.update(
            indexes["address"].get(
                (country, address),
                []
            )
        )

        # ----------------------------------------------------
        # Sorted address tokens
        # ----------------------------------------------------

        candidates.update(
            indexes["address_sig"].get(
                (country, address_sig),
                []
            )
        )

        # Make deterministic
        candidates = sorted(candidates)

        if candidates:
            matched_entities += 1

        output_rows.append(
            {
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(
                    candidates
                )
            }
        )

    output_df = pd.DataFrame(
        output_rows,
        columns=[
            "source1_entity_id",
            "matched_entity_ids"
        ]
    )

    output_df.to_csv(
        OUT_FILE,
        sep="\t",
        index=False,
        mode="w" if first_write else "a",
        header=first_write
    )

    first_write = False

    processed += len(chunk)

    print(
        f"\rSource 1 processed: "
        f"{processed:,}",
        end="",
        flush=True
    )


print("\n")
print("=" * 70)
print("SUBMISSION FILE CREATED")
print("=" * 70)

print(
    f"Source 1 records: {processed:,}"
)

print(
    f"Entities with matches: "
    f"{matched_entities:,}"
)

print(
    f"File: {OUT_FILE}"
)

print("=" * 70)