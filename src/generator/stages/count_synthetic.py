# src/generator/stages/count_benchmark.py

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

try:
    from ..config import BENCHMARK_FOLDER
except Exception:
    from generator.config import BENCHMARK_FOLDER


def main() -> None:
    print("[Stage 6] Counting synthetic benchmark HCT and Q&A")

    qanda_folder = Path(BENCHMARK_FOLDER)

    if not qanda_folder.exists():
        print(f"[ERROR] Benchmark folder not found: {qanda_folder}")
        return

    suffix = "_QandA.json"
    qanda_files = sorted([f for f in qanda_folder.glob(f"*{suffix}")])

    if not qanda_files:
        print(f"[WARN] No Q&A files matching *{suffix} in {qanda_folder}")
        return

    base_table_names = []
    set_nums = []
    num_questions = []

    # loop through each QandA file
    for idx, f in enumerate(qanda_files, 1):
        print(f"[READ] {idx}/{len(qanda_files)} -- {100*idx/len(qanda_files):.1f}% {f.name}")

        # parse filename: BaseTable_setX_QandA.json
        parts = f.name.split("_set")
        if len(parts) < 2:
            print(f"  [SKIP] Unexpected filename format: {f.name}")
            continue

        base_table = parts[0]
        set_num = parts[1].replace(suffix, "")

        # open json and count questions
        with f.open(encoding="utf-8") as jf:
            qanda = json.load(jf)

        n_questions = len(qanda.get("questions", []))

        base_table_names.append(base_table)
        set_nums.append(set_num)
        num_questions.append(n_questions)

    # build dataframe -> justt for statistics
    df = pd.DataFrame({
        "baseTableName": base_table_names,
        "setNum": set_nums,
        "numQuestions": num_questions,
    })

    # summarise per base table
    results = (
        df.groupby("baseTableName")
          .agg(NumTables=("setNum", "count"),
               NumQuestions=("numQuestions", "sum"))
          .reset_index()
          .rename(columns={"baseTableName": "NameTables"})
    )

    # add total row
    total_row = pd.DataFrame({
        "NameTables": ["Total"],
        "NumTables": [results["NumTables"].sum()],
        "NumQuestions": [results["NumQuestions"].sum()],
    })

    df_results = pd.concat([results, total_row], ignore_index=True)

    print("\n=== Benchmark Summary ===")
    print(df_results.to_string(index=False))

    # optional: save summary CSV in benchmark folder
    out_file = qanda_folder / "benchmark_summary.csv"
    df_results.to_csv(out_file, index=False)
    print(f"[DONE] Summary saved to {out_file}")


if __name__ == "__main__":
    main()
