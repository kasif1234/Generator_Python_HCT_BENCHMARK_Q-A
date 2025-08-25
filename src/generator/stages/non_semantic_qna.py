# src/generator/stages/non_semantic_tables_and_qna.py
# STAGE 4: Generate non-semantic versions of tables and Q&A
# - input: semantic Q&A and tables
from __future__ import annotations

import random
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

# --- robust imports: work as module or script ---------------------------------
try:
    # when run via "python -m src.generator..."
    from ..config import (
        SEMANTIC_QANDA_FOLDER,
        SEMANTIC_TABLES_FOLDER,
        NON_SEMANTIC_QANDA_FOLDER,
        NON_SEMANTIC_TABLES_FOLDER,
    )
except Exception:
    # when run directly
    from generator.config import (
        SEMANTIC_QANDA_FOLDER,
        SEMANTIC_TABLES_FOLDER,
        NON_SEMANTIC_QANDA_FOLDER,
        NON_SEMANTIC_TABLES_FOLDER,
    )

import pandas as pd


# ==============================================================================
# Helpers
# ==============================================================================

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def replace_semantics(text: str, mapping: List[Tuple[str, str]]) -> str:
    """
    Replace each semantic token with its non-semantic code.
    We use word-boundary regex and escape the key to avoid over-matching.
    """
    out = text
    # Longest keys first to avoid partial overlaps.
    for sem, non in sorted(mapping, key=lambda kv: len(kv[0]), reverse=True):
        # \b doesn't work well around non-word chars; allow boundaries at start/end or non-alnum.
        pattern = r"(?<![A-Za-z0-9])" + re.escape(sem) + r"(?![A-Za-z0-9])"
        out = re.sub(pattern, non, out)
    # normalize CRLF like the R code's gsub("\r","",...)
    out = out.replace("\r", "")
    return out

def gen_nonsemantic_codes(num_codes: int) -> List[str]:
    """
    Generate many consonant-only codes: all 2-letter combs, then 3-letter, etc.,
    shuffle and take as many as needed.
    """
    consonants = list("bcdfghjklmnpqrstvwx yz".replace(" ", ""))
    bag: List[str] = []

    k = 2
    while len(bag) < num_codes and k <= 6:  # practical cap
        for tup in combinations(consonants, k):
            bag.append("".join(tup))
        k += 1

    random.seed(0)  # deterministic like set.seed(0)
    random.shuffle(bag)
    if len(bag) < num_codes:
        # fallback (shouldn’t happen often)
        bag = (bag * ((num_codes // max(1, len(bag))) + 1))[:num_codes]
    return bag[:num_codes]

def dict_from_db_csv(db_csv: Path) -> List[Tuple[str, str]]:
    """
    Build (semantic -> nonsemantic) mapping from the *_DB.csv of a series.

    Assumed CSV layout (as produced by Stage 1/2 in this repo):
      - Header row: human-readable column names (HCT)
      - First data row: SQL-safe names (underscored)
      - Subsequent rows: label values + Value (numeric)
    """
    df = pd.read_csv(db_csv, dtype=str).fillna("")
    if df.empty:
        return []

    # Drop the Value column entirely
    no_val = df.iloc[:, :-1].copy()
    n_rows, n_cols = no_val.shape

    # We need (n_rows + 1) * n_cols codes (as in the R logic)
    # (+1 because we also map the two feature names per column)
    num_codes = (n_rows + 1) * n_cols
    pool = gen_nonsemantic_codes(num_codes)

    # Matrix of codes (row-major), then a simple set of feature prefixes
    code_matrix = [pool[i : i + n_cols] for i in range(0, len(pool), n_cols)]
    feat_prefixes = [
        "AA","BB","CC","DD","EE","FF","GG","HH","JJ","KK","LL","MM",
        "NN","OO","PP","QQ","RR","SS","TT","UU","VV","WW","XX","YY","ZZ",
    ]

    mapping: List[Tuple[str, str]] = []

    for i in range(n_cols):
        col = no_val.iloc[:, i]

        # First data row is SQL-safe feature name (e.g., "Type_of_pollution")
        feat_name_db = str(col.iloc[0])
        # HCT display version (spaces instead of underscores)
        feat_name_hct = feat_name_db.replace("_", " ")

        # Unique semantic values for this column (skip the first data row)
        semantic_vals = [str(v) for v in pd.unique(col.iloc[1:]) if str(v) != ""]

        # Build non-semantic labels for this column
        prefix = feat_prefixes[i % len(feat_prefixes)]
        col_codes = [prefix + code_matrix[r][i] for r in range(min(len(semantic_vals), len(code_matrix)))]

        # Map the two feature header names to the prefix (as in the R: AA/BB/..)
        mapping.append((feat_name_hct, prefix))
        mapping.append((feat_name_db,  prefix))

        # Map each semantic value to its prefixed code
        for sem_val, code in zip(semantic_vals, col_codes):
            mapping.append((sem_val, code))

    return mapping


# ==============================================================================
# Stage 4 main
# ==============================================================================
def main() -> None:
    print("[Stage 4] Generate non-semantic versions of tables and Q&A")

    # Ensure output dirs exist
    Path(NON_SEMANTIC_QANDA_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(NON_SEMANTIC_TABLES_FOLDER).mkdir(parents=True, exist_ok=True)

    # Find all series roots via semantic Q&A files: <root>_QandA.json
    qanda_files = sorted(Path(SEMANTIC_QANDA_FOLDER).glob("*_QandA.json"))
    print(f"[DISCOVER] Found {len(qanda_files)} semantic Q&A files")

    for idx, qf in enumerate(qanda_files, 1):
        root = qf.name[:-len("_QandA.json")]
        print(f"[{idx}/{len(qanda_files)}] {root}")

        # --- Build semantic->nonsemantic dictionary from *_DB.csv
        db_csv = Path(SEMANTIC_TABLES_FOLDER) / f"{root}_DB.csv"
        if not db_csv.exists():
            print(f"  [WARN] Missing DB csv for {root}, skipping")
            continue
        mapping = dict_from_db_csv(db_csv)
        if not mapping:
            print(f"  [WARN] Empty mapping for {root}, skipping")
            continue

        # -------- Files to rewrite (semantic -> non-semantic text)
        tasks = [
            # Q&A (json)
            (Path(SEMANTIC_QANDA_FOLDER)  / f"{root}_QandA.json",
             Path(NON_SEMANTIC_QANDA_FOLDER) / f"{root}_QandA_NONSEM.json"),

            # DB (html, csv)
            (Path(SEMANTIC_TABLES_FOLDER) / f"{root}_DB.html",
             Path(NON_SEMANTIC_TABLES_FOLDER) / f"{root}_DB_NONSEM.html"),
            (Path(SEMANTIC_TABLES_FOLDER) / f"{root}_DB.csv",
             Path(NON_SEMANTIC_TABLES_FOLDER) / f"{root}_DB_NONSEM.csv"),

            # HCT (html, csv, json)
            (Path(SEMANTIC_TABLES_FOLDER) / f"{root}_HCT.html",
             Path(NON_SEMANTIC_TABLES_FOLDER) / f"{root}_HCT_NONSEM.html"),
            (Path(SEMANTIC_TABLES_FOLDER) / f"{root}_HCT.csv",
             Path(NON_SEMANTIC_TABLES_FOLDER) / f"{root}_HCT_NONSEM.csv"),
            (Path(SEMANTIC_TABLES_FOLDER) / f"{root}_HCT.json",
             Path(NON_SEMANTIC_TABLES_FOLDER) / f"{root}_HCT_NONSEM.json"),
        ]

        # -------- Apply replacements per file
        for fin, fout in tasks:
            if not fin.exists():
                # Some series may not have every artifact (e.g., HCT.csv if not generated)
                continue
            text_in = read_text(fin)
            text_out = replace_semantics(text_in, mapping)
            write_text(fout, text_out)
            print(f"  [WRITE] {fout}")

    print("[DONE] Stage 4 complete")


if __name__ == "__main__":
    main()
