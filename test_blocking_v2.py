import os
import re
import sqlite3
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE = r"C:\Users\Shivtej\Downloads\6ab10eb3b23ba_student_resource\student_resource"

TRAIN_DIR = os.path.join(BASE, "dataset", "train")

S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
GT_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

DB_PATH = os.path.join(BASE, "blocking_v2.sqlite")

# Same evaluation size as your previous blocker
SAMPLE_SIZE = 10_000

# Number of FTS results retrieved from each field
NAME_TOP_K = 300
ADDRESS_TOP_K = 300

# Read large files in chunks
CHUNK_SIZE = 50_000


# ============================================================
# CHECK FILES
# ============================================================

def check_files():

    paths = {
        "Source 1": S1_PATH,
        "Source 2": S2_PATH,
        "Source 3": S3_PATH,
        "Ground Truth": GT_PATH,
    }

    print("\nChecking dataset files...")
    print("-" * 60)

    missing = False

    for name, path in paths.items():

        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024 ** 3)

            print(
                f"[OK] {name:<15} "
                f"{size_gb:.2f} GB"
            )

        else:
            print(
                f"[MISSING] {name}: {path}"
            )
            missing = True

    print("-" * 60)

    if missing:
        raise FileNotFoundError(
            "\nOne or more dataset files could not be found."
        )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    # Replace punctuation with spaces.
    # Unicode letters/numbers are preserved.
    value = re.sub(
        r"[^\w]+",
        " ",
        value,
        flags=re.UNICODE
    )

    return " ".join(value.split())


def get_tokens(value):

    text = normalize_text(value)

    tokens = text.split()

    # Ignore very short tokens
    return [
        token
        for token in tokens
        if len(token) >= 3
    ]


# ============================================================
# FTS5 HELPERS
# ============================================================

def escape_fts_token(token):

    token = token.replace('"', '""')

    # Prefix search
    return f'"{token}"*'


def make_fts_query(tokens):

    if not tokens:
        return None

    # Remove duplicates while preserving order
    tokens = list(dict.fromkeys(tokens))

    return " OR ".join(
        escape_fts_token(token)
        for token in tokens
    )


# ============================================================
# CREATE FTS5 DATABASE
# ============================================================

def create_database():

    # Remove incomplete previous database
    for path in [
        DB_PATH,
        DB_PATH + "-wal",
        DB_PATH + "-shm"
    ]:

        if os.path.exists(path):

            try:
                os.remove(path)
            except PermissionError:
                pass

    print("\n")
    print("=" * 60)
    print("CREATING BLOCKING V2 INDEX")
    print("=" * 60)

    print("\nSource 2 + Source 3 will be indexed.")
    print("This may take some time.\n")

    con = sqlite3.connect(DB_PATH)

    # Performance settings
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-200000")

    # --------------------------------------------------------
    # FTS TABLE
    # --------------------------------------------------------

    con.execute("""
        CREATE VIRTUAL TABLE records USING fts5(

            entity_id UNINDEXED,

            source UNINDEXED,

            country,

            name,

            address

        )
    """)

    con.commit()

    insert_sql = """
        INSERT INTO records
        (
            entity_id,
            source,
            country,
            name,
            address
        )
        VALUES (?, ?, ?, ?, ?)
    """

    # ========================================================
    # SOURCE 2
    # ========================================================

    print("\nIndexing Source 2...")

    total = 0

    reader = pd.read_csv(
        S2_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,

        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    for chunk in reader:

        rows = []

        for row in chunk.itertuples(index=False):

            entity_id = (
                "" if pd.isna(row.entity_id)
                else str(row.entity_id)
            )

            name = normalize_text(
                row.business_name
            )

            address = normalize_text(
                row.business_address
            )

            country = normalize_text(
                row.country
            )

            rows.append(
                (
                    entity_id,
                    "source2",
                    country,
                    name,
                    address
                )
            )

        con.executemany(
            insert_sql,
            rows
        )

        con.commit()

        total += len(rows)

        print(
            f"\rSource 2 indexed: {total:,}",
            end="",
            flush=True
        )

    print("\nSource 2 complete.")

    # ========================================================
    # SOURCE 3
    # ========================================================

    print("\nIndexing Source 3...")

    total = 0

    reader = pd.read_csv(
        S3_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,

        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    for chunk in reader:

        rows = []

        for row in chunk.itertuples(index=False):

            entity_id = (
                "" if pd.isna(row.entity_id)
                else str(row.entity_id)
            )

            name = normalize_text(
                row.business_name
            )

            address = normalize_text(
                row.business_address
            )

            country = normalize_text(
                row.country
            )

            rows.append(
                (
                    entity_id,
                    "source3",
                    country,
                    name,
                    address
                )
            )

        con.executemany(
            insert_sql,
            rows
        )

        con.commit()

        total += len(rows)

        print(
            f"\rSource 3 indexed: {total:,}",
            end="",
            flush=True
        )

    print("\nSource 3 complete.")

    con.close()

    print("\n")
    print("=" * 60)
    print("FTS5 INDEX CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Database: {DB_PATH}")


# ============================================================
# LOAD 10K TRAINING SAMPLE
# ============================================================

def load_training_sample():

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str
    )

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .fillna("")
        .astype(str)
    )

    # Count matches
    gt["match_count"] = gt[
        "matched_entity_ids"
    ].apply(
        lambda x:
            0
            if not x.strip()
            else len(
                [
                    v for v in x.split(",")
                    if v.strip()
                ]
            )
    )

    # We want entities with actual matches
    matched_gt = gt[
        gt["match_count"] > 0
    ]

    sample_size = min(
        SAMPLE_SIZE,
        len(matched_gt)
    )

    sample = matched_gt.sample(
        n=sample_size,
        random_state=42
    )

    wanted_ids = set(
        sample["source1_entity_id"]
    )

    print(
        f"Selected {len(wanted_ids):,} "
        f"Source 1 entities."
    )

    # --------------------------------------------------------
    # Read Source 1
    # --------------------------------------------------------

    s1_parts = []

    reader = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        part = chunk[
            chunk["entity_id"].isin(wanted_ids)
        ]

        if len(part):
            s1_parts.append(part)

    s1 = pd.concat(
        s1_parts,
        ignore_index=True
    )

    # Ground truth lookup
    gt_map = dict(
        zip(
            sample["source1_entity_id"],
            sample["matched_entity_ids"]
        )
    )

    print(
        f"Loaded {len(s1):,} Source 1 records."
    )

    return s1, gt_map


# ============================================================
# SEARCH NAME
# ============================================================

def search_name(
    con,
    country,
    name_tokens
):

    query_text = make_fts_query(
        name_tokens
    )

    if not query_text:
        return set()

    # country is an indexed FTS column
    query = (
        f"country:{escape_fts_token(country)} "
        f"AND name:({query_text})"
    )

    try:

        results = con.execute(
            """
            SELECT entity_id
            FROM records
            WHERE records MATCH ?
            LIMIT ?
            """,
            (
                query,
                NAME_TOP_K
            )
        ).fetchall()

        return {
            row[0]
            for row in results
            if row[0]
        }

    except sqlite3.OperationalError:

        return set()


# ============================================================
# SEARCH ADDRESS
# ============================================================

def search_address(
    con,
    country,
    address_tokens
):

    query_text = make_fts_query(
        address_tokens
    )

    if not query_text:
        return set()

    query = (
        f"country:{escape_fts_token(country)} "
        f"AND address:({query_text})"
    )

    try:

        results = con.execute(
            """
            SELECT entity_id
            FROM records
            WHERE records MATCH ?
            LIMIT ?
            """,
            (
                query,
                ADDRESS_TOP_K
            )
        ).fetchall()

        return {
            row[0]
            for row in results
            if row[0]
        }

    except sqlite3.OperationalError:

        return set()


# ============================================================
# RETRIEVE CANDIDATES
# ============================================================

def get_candidates(
    con,
    business_name,
    business_address,
    country
):

    country = normalize_text(country)

    name_tokens = get_tokens(
        business_name
    )

    address_tokens = get_tokens(
        business_address
    )

    candidates = set()

    # Name candidates
    candidates.update(
        search_name(
            con,
            country,
            name_tokens
        )
    )

    # Address candidates
    candidates.update(
        search_address(
            con,
            country,
            address_tokens
        )
    )

    return candidates


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    s1, gt_map = load_training_sample()

    print("\nOpening FTS5 database...")

    con = sqlite3.connect(
        DB_PATH
    )

    total_true = 0
    total_found = 0

    candidate_counts = []

    perfect_entities = 0

    total_entities = len(s1)

    print("\n")
    print("=" * 60)
    print("RUNNING BLOCKING V2")
    print("=" * 60)

    for i, row in enumerate(
        s1.itertuples(index=False),
        start=1
    ):

        candidates = get_candidates(
            con=con,

            business_name=row.business_name,

            business_address=row.business_address,

            country=row.country
        )

        candidate_counts.append(
            len(candidates)
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        true_ids = {
            entity_id.strip()

            for entity_id
            in gt_map[row.entity_id].split(",")

            if entity_id.strip()
        }

        found = (
            true_ids
            & candidates
        )

        total_true += len(
            true_ids
        )

        total_found += len(
            found
        )

        # Perfect recall for this entity
        if found == true_ids:

            perfect_entities += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            i % 250 == 0
            or i == total_entities
        ):

            recall = (
                total_found / total_true
                if total_true
                else 0
            )

            avg_candidates = (
                sum(candidate_counts)
                / len(candidate_counts)
            )

            perfect_recall = (
                perfect_entities
                / i
            )

            print(
                f"{i:>6,}/{total_entities:,} | "
                f"Recall: {recall:7.2%} | "
                f"Perfect: {perfect_recall:7.2%} | "
                f"Avg candidates: {avg_candidates:7.1f}"
            )

    con.close()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    recall = (
        total_found / total_true
        if total_true
        else 0
    )

    perfect_entity_recall = (
        perfect_entities / total_entities
        if total_entities
        else 0
    )

    average_candidates = (
        sum(candidate_counts)
        / len(candidate_counts)
        if candidate_counts
        else 0
    )

    maximum_candidates = (
        max(candidate_counts)
        if candidate_counts
        else 0
    )

    median_candidates = (
        sorted(candidate_counts)
        [len(candidate_counts) // 2]
        if candidate_counts
        else 0
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("BLOCKING V2 RESULTS")
    print("=" * 60)

    print(
        f"Sample S1 entities:        "
        f"{total_entities:,}"
    )

    print(
        f"True match IDs:             "
        f"{total_true:,}"
    )

    print(
        f"True matches retrieved:     "
        f"{total_found:,}"
    )

    print(
        f"Candidate recall:           "
        f"{recall:.2%}"
    )

    print(
        f"Perfect entity recall:      "
        f"{perfect_entity_recall:.2%}"
    )

    print(
        f"Average candidates / S1:   "
        f"{average_candidates:.1f}"
    )

    print(
        f"Median candidates / S1:    "
        f"{median_candidates}"
    )

    print(
        f"Maximum candidates / S1:   "
        f"{maximum_candidates:,}"
    )

    print("=" * 60)

    # ========================================================
    # DECISION
    # ========================================================

    if recall >= 0.95:

        print(
            "\nSTATUS: EXCELLENT"
        )

        print(
            "Proceed to candidate scoring/matcher."
        )

    elif recall >= 0.90:

        print(
            "\nSTATUS: GOOD"
        )

        print(
            "Candidate generation is usable."
        )

    elif recall >= 0.75:

        print(
            "\nSTATUS: NEEDS IMPROVEMENT"
        )

        print(
            "Improve blocking before building the matcher."
        )

    else:

        print(
            "\nSTATUS: TOO LOW"
        )

        print(
            "Do NOT build the final matcher yet."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION")
    print("BLOCKING V2")
    print("=" * 60)

    check_files()

    # --------------------------------------------------------
    # Build database if it doesn't exist
    # --------------------------------------------------------

    if not os.path.exists(DB_PATH):

        create_database()

    else:

        print(
            f"\nExisting database found:"
            f"\n{DB_PATH}"
        )

        print(
            "\nUsing existing database."
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate()