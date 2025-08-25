# src/generator/stages/cleanup_and_benchmark.py
# STAGE 5: Cleanup and collect complete files into Benchmark folder
# - input: all previous outputs

from __future__ import annotations
from pathlib import Path
import shutil

try:
    from ..config import (
        SEMANTIC_TABLES_FOLDER,
        SEMANTIC_QANDA_FOLDER,
        NON_SEMANTIC_TABLES_FOLDER,
        NON_SEMANTIC_QANDA_FOLDER,
        BENCHMARK_FOLDER,
    )
except Exception:
    from generator.config import (
        SEMANTIC_TABLES_FOLDER,
        SEMANTIC_QANDA_FOLDER,
        NON_SEMANTIC_TABLES_FOLDER,
        NON_SEMANTIC_QANDA_FOLDER,
        BENCHMARK_FOLDER,
    )


def main() -> None:
    print("[Stage 5] Cleanup and collect complete files into Benchmark folder")

    sem_tables = Path(SEMANTIC_TABLES_FOLDER)
    sem_qanda = Path(SEMANTIC_QANDA_FOLDER)
    nonsem_tables = Path(NON_SEMANTIC_TABLES_FOLDER)
    nonsem_qanda = Path(NON_SEMANTIC_QANDA_FOLDER)
    benchmark = Path(BENCHMARK_FOLDER)

    benchmark.mkdir(parents=True, exist_ok=True)

    suffix = "_QandA.json"
    qanda_files = sorted(sem_qanda.glob(f"*{suffix}"))
    if not qanda_files:
        print(f"[WARN] No files matching *{suffix} in {sem_qanda}")
        return

    # CORRECT root extraction
    roots = [f.name[:-len(suffix)] for f in qanda_files]

    roots_to_keep: list[str] = []

    for idx, root in enumerate(roots, 1):
        print(f"[CHECK] {idx}/{len(roots)} {root}")

        required_files = [
            nonsem_qanda / f"{root}_QandA_NONSEM.json",
            sem_tables / f"{root}_HCT.html",
            sem_tables / f"{root}_HCT.csv",
            sem_tables / f"{root}_DB.html",
            sem_tables / f"{root}_DB.csv",
            nonsem_tables / f"{root}_HCT_NONSEM.html",
            nonsem_tables / f"{root}_HCT_NONSEM.csv",
            nonsem_tables / f"{root}_DB_NONSEM.html",
            nonsem_tables / f"{root}_DB_NONSEM.csv",
        ]

        missing = [str(p) for p in required_files if not p.exists()]
        if missing:
            print(f"  [SKIP] Missing {len(missing)} files (showing first 2): {missing[:2]}")
            continue

        roots_to_keep.append(root)

    if not roots_to_keep:
        print("[INFO] No complete roots found. Nothing to copy.")
        return

    for idx, root in enumerate(roots_to_keep, 1):
        print(f"[COPY] {idx}/{len(roots_to_keep)} {root}")

        files_to_copy = [
            sem_qanda / f"{root}_QandA.json",
            nonsem_qanda / f"{root}_QandA_NONSEM.json",
            sem_tables / f"{root}_HCT.html",
            sem_tables / f"{root}_HCT.csv",
            sem_tables / f"{root}_DB.html",
            sem_tables / f"{root}_DB.csv",
            nonsem_tables / f"{root}_HCT_NONSEM.html",
            nonsem_tables / f"{root}_HCT_NONSEM.csv",
            nonsem_tables / f"{root}_DB_NONSEM.html",
            nonsem_tables / f"{root}_DB_NONSEM.csv",
        ]

        for src in files_to_copy:
            if src.exists():
                shutil.copy(src, benchmark)
            else:
                print(f"  [WARN] Expected but missing at copy time: {src}")

    print(f"[DONE] Copied {len(roots_to_keep)} complete sets to {benchmark.resolve()}")


if __name__ == "__main__":
    main()
