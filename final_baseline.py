from pathlib import Path
import re
import unicodedata
import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).parent
DATASET = BASE / "dataset"
TEST = DATASET / "test"
OUTPUT = BASE / "output"

OUTPUT.mkdir(exist_ok=True)

S1_FILE = TEST / "test_source1.tsv"
S2_FILE = TEST / "test_source2.tsv"
S3_FILE = TEST / "test_source3.tsv"


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000

# Conservative thresholds because F0.5 is precision-heavy.
NAME_HIGH = 92
ADDRESS_HIGH = 92

NAME_MED = 82
ADDRESS_MED = 82

# Prevent absurdly large candidate lists.
MAX_CANDIDATES_PER_S1 = 500


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
        x for x in text.split()
        if len(x) >= 2
    ]

    return " ".join(
        sorted(set(tokens))
    )


def number_tokens(text):

    text = normalize_text(text)

    return set(
        re.findall(
            r"\d+",
            text
        )
    )


# ============================================================
# LOAD TEST SOURCE 1
# ============================================================

print("=" * 70)
print("LOADING TEST SOURCE 1")
print("=" * 70)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(
    f"Source 1 records: {len(s1):,}"
)


# ============================================================
# CREATE INDEX
# ============================================================

indexes = {
    "name": {},
    "name_sig": {},
    "address": {},
    "address_sig": {},
    "number": {}
}


def add_index(index, key, entity_id):

    if not key:
        return

    index.setdefault(
        key,
        []
    ).append(entity_id)


def index_source(path, source_name):

    print()
    print(
        f"Indexing {source_name}: "
        f"{path.name}"
    )

    total = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = row.entity_id

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

            nums = number_tokens(
                row.business_address
            )

            add_index(
                indexes["name"],
                name,
                entity_id
            )

            add_index(
                indexes["name_sig"],
                name_sig,
                entity_id
            )

            add_index(
                indexes["address"],
                address,
                entity_id
            )

            add_index(
                indexes["address_sig"],
                address_sig,
                entity_id
            )

            # Only use reasonably informative
            # numeric tokens.
            for num in nums:

                if len(num) >= 3:

                    add_index(
                        indexes["number"],
                        num,
                        entity_id
                    )

        total += len(chunk)

        print(
            f"\r  {total:,} records",
            end="",
            flush=True
        )

    print()


index_source(
    S2_FILE,
    "Source 2"
)

index_source(
    S3_FILE,
    "Source 3"
)


# ============================================================
# REMOVE VERY COMMON KEYS
# ============================================================

def prune_index(index, maximum=5000):

    return {
        key: values
        for key, values in index.items()
        if len(values) <= maximum
    }


print("\nPruning extremely common blocking keys...")

indexes["name"] = prune_index(
    indexes["name"]
)

indexes["name_sig"] = prune_index(
    indexes["name_sig"]
)

indexes["address"] = prune_index(
    indexes["address"]
)

indexes["address_sig"] = prune_index(
    indexes["address_sig"]
)

indexes["number"] = prune_index(
    indexes["number"],
    maximum=2000
)


# ============================================================
# CACHE TEST SOURCE RECORDS
# ============================================================

print("\nLoading candidate records...")

record_cache = {}


def cache_source(path):

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = row.entity_id

            record_cache[entity_id] = (
                normalize_text(
                    row.business_name
                ),
                normalize_text(
                    row.business_address
                ),
                normalize_text(
                    row.country
                )
            )


cache_source(S2_FILE)
cache_source(S3_FILE)

print(
    f"Cached records: "
    f"{len(record_cache):,}"
)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(row):

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

    nums = number_tokens(
        row.business_address
    )

    candidates = set()

    # Exact normalized name
    candidates.update(
        indexes["name"].get(
            name,
            []
        )
    )

    # Sorted name tokens
    candidates.update(
        indexes["name_sig"].get(
            name_sig,
            []
        )
    )

    # Exact address
    candidates.update(
        indexes["address"].get(
            address,
            []
        )
    )

    # Sorted address tokens
    candidates.update(
        indexes["address_sig"].get(
            address_sig,
            []
        )
    )

    # Address numbers
    for num in nums:

        if len(num) >= 3:

            candidates.update(
                indexes["number"].get(
                    num,
                    []
                )
            )

    return candidates


# ============================================================
# SCORE CANDIDATES
# ============================================================

def score_candidate(
    s1_name,
    s1_address,
    s1_country,
    candidate_id
):

    if candidate_id not in record_cache:
        return 0, 0, 0

    c_name, c_address, c_country = (
        record_cache[candidate_id]
    )

    if s1_country != c_country:
        return 0, 0, 0

    if not s1_name or not c_name:
        name_score = 0
    else:
        name_score = ratio(
            s1_name,
            c_name
        )

    if not s1_address or not c_address:
        address_score = 0
    else:
        address_score = ratio(
            s1_address,
            c_address
        )

    token_name_score = token_set_ratio(
        s1_name,
        c_name
    )

    token_address_score = token_set_ratio(
        s1_address,
        c_address
    )

    # Combined score.
    #
    # Address is important because the dataset
    # contains multilingual/name variations.

    combined = (
        0.45 * max(
            name_score,
            token_name_score
        )
        +
        0.55 * max(
            address_score,
            token_address_score
        )
    )

    return (
        combined,
        name_score,
        address_score
    )


# ============================================================
# PROCESS SOURCE 1
# ============================================================

print("\n")
print("=" * 70)
print("GENERATING FINAL CANDIDATES + MATCHES")
print("=" * 70)

matching_rows = []
candidate_rows = []

total = len(s1)

for i, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    s1_id = row.entity_id

    s1_name = normalize_text(
        row.business_name
    )

    s1_address = normalize_text(
        row.business_address
    )

    s1_country = normalize_text(
        row.country
    )

    candidates = generate_candidates(
        row
    )

    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    scored = []

    for candidate_id in candidates:

        combined, name_score, address_score = (
            score_candidate(
                s1_name,
                s1_address,
                s1_country,
                candidate_id
            )
        )

        if combined > 0:

            scored.append(
                (
                    combined,
                    name_score,
                    address_score,
                    candidate_id
                )
            )

    # Highest scores first
    scored.sort(
        reverse=True
    )

    # Keep manageable candidate list
    scored = scored[
        :MAX_CANDIDATES_PER_S1
    ]

    final_candidates = [
        x[3]
        for x in scored
    ]

    # --------------------------------------------------------
    # Match decision
    # --------------------------------------------------------

    matches = []

    for combined, name_score, address_score, candidate_id in scored:

        # Very strong name
        strong_name = (
            name_score >= NAME_HIGH
        )

        # Very strong address
        strong_address = (
            address_score >= ADDRESS_HIGH
        )

        # Both moderately strong
        balanced = (
            name_score >= NAME_MED
            and
            address_score >= ADDRESS_MED
            and
            combined >= 84
        )

        # Extremely strong combined score
        very_strong = (
            combined >= 94
        )

        if (
            strong_name
            or strong_address
            or balanced
            or very_strong
        ):

            matches.append(
                candidate_id
            )

    # Remove duplicates
    matches = list(
        dict.fromkeys(matches)
    )

    # Candidate list must contain every final match
    candidate_set = set(
        final_candidates
    )

    matches = [
        x for x in matches
        if x in candidate_set
    ]

    matching_rows.append(
        {
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(
                matches
            )
        }
    )

    candidate_rows.append(
        {
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(
                final_candidates
            )
        }
    )

    if (
        i % 5000 == 0
        or i == total
    ):

        print(
            f"\rProcessed "
            f"{i:,}/{total:,} "
            f"({i / total:.1%})",
            end="",
            flush=True
        )


print("\n")


# ============================================================
# WRITE OUTPUT
# ============================================================

matching_path = (
    OUTPUT /
    "matching_results.tsv"
)

candidate_path = (
    OUTPUT /
    "candidate_pairs.tsv"
)

print("Writing matching_results.tsv...")

pd.DataFrame(
    matching_rows
).to_csv(
    matching_path,
    sep="\t",
    index=False
)

print("Writing candidate_pairs.tsv...")

pd.DataFrame(
    candidate_rows
).to_csv(
    candidate_path,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

nonempty_matches = sum(
    bool(x["matched_entity_ids"])
    for x in matching_rows
)

avg_candidates = (
    sum(
        len(
            x["candidate_entity_ids"].split(",")
        )
        if x["candidate_entity_ids"]
        else 0
        for x in candidate_rows
    )
    /
    len(candidate_rows)
)

print("\n")
print("=" * 70)
print("FINAL OUTPUT CREATED")
print("=" * 70)

print(
    f"Source 1 records:       {len(s1):,}"
)

print(
    f"Entities with matches:  {nonempty_matches:,}"
)

print(
    f"Entities with no match: "
    f"{len(s1) - nonempty_matches:,}"
)

print(
    f"Average candidates/S1:  "
    f"{avg_candidates:.1f}"
)

print(
    f"\n{matching_path}"
)

print(
    f"{candidate_path}"
)

print("=" * 70)