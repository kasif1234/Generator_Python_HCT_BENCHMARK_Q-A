# src/generator/pipeline.py
from __future__ import annotations

import argparse
import importlib
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


# ---- Helpers -----------------------------------------------------------------
@dataclass
class Stage:
    num: int
    key: str
    desc: str
    importer: Tuple[str, str]  # (module_path, function_name = "main")

    def run(self) -> None:
        mod_path, fn = self.importer
        mod = importlib.import_module(mod_path)
        getattr(mod, fn)()  # call main()


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _fmt_dur(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(sec, 60)
    return f"{int(m)}m{int(s)}s"


# ---- Registry (adjust module names if yours differ) --------------------------
# Make sure each stage module has a `main()` function.
STAGES: List[Stage] = [
    Stage(1, "tables_from_patterns", "Generate JSON tables from patterns",
          ("src.generator.stages.tables_from_patterns", "main")),
    Stage(2, "hct_from_json_tables", "Build HCT/DB artifacts from JSON tables",
          ("src.generator.stages.hct_from_json_tables", "main")),
    Stage(3, "sql_and_nlq_from_templates", "Generate SQL + NLQ from templates",
          ("src.generator.stages.sql_and_nlq_from_templates", "main")),
    Stage(4, "non_semantic_qna", "Make non-semantic (masked) copies",
          ("src.generator.stages.non_semantic_qna", "main")),
    Stage(5, "clean_benchmark", "Filter complete roots and copy to benchmark",
          ("src.generator.stages.clean_benchmark", "main")),
    Stage(6, "count_synthetic", "Count sets & questions; emit summary",
          ("src.generator.stages.count_synthetic", "main")),
]
STAGE_BY_NUM: Dict[int, Stage] = {s.num: s for s in STAGES}
STAGE_BY_KEY: Dict[str, Stage] = {s.key: s for s in STAGES}


# ---- Orchestration -----------------------------------------------------------
def run_pipeline(
    start: int,
    end: int,
    skip: Optional[List[int]] = None,
    only: Optional[List[int]] = None,
) -> None:
    skip = set(skip or [])
    plan: List[Stage]

    if only:
        plan = [STAGE_BY_NUM[n] for n in only]
    else:
        plan = [s for s in STAGES if start <= s.num <= end]

    print(f"[{_ts()}] Pipeline plan:")
    for s in plan:
        flag = " (SKIP)" if s.num in skip else ""
        print(f"  - {s.num}. {s.key}: {s.desc}{flag}")

    total_start = time.time()
    completed: List[int] = []

    for s in plan:
        if s.num in skip:
            print(f"[{_ts()}] Skip stage {s.num}: {s.key}")
            continue

        print(f"[{_ts()}] ▶ Start {s.num}: {s.key} — {s.desc}")
        t0 = time.time()
        try:
            s.run()
        except Exception as e:
            dur = _fmt_dur(time.time() - t0)
            print(f"[{_ts()}] ✖ Failed at stage {s.num} after {dur}")
            raise
        else:
            dur = _fmt_dur(time.time() - t0)
            print(f"[{_ts()}] ✓ Done {s.num}: {s.key} in {dur}")
            completed.append(s.num)

    total_dur = _fmt_dur(time.time() - total_start)
    print(f"[{_ts()}] Pipeline finished in {total_dur}. Completed: {completed}")


# ---- CLI ---------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="generator-pipeline",
        description="Run the HCT/DB/SQL/NLQ generation pipeline."
    )
    p.add_argument("--start", type=int, default=1, help="First stage number (default: 1)")
    p.add_argument("--end", type=int, default=6, help="Last stage number (default: 6)")
    p.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated stage numbers to skip (e.g., '3,5')"
    )
    p.add_argument(
        "--only", type=str, default="",
        help="Run only these comma-separated stage numbers (e.g., '2,4')"
    )
    p.add_argument(
        "--list", action="store_true",
        help="List stages and exit"
    )
    return p.parse_args()


def _parse_csv_nums(s: str) -> List[int]:
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    args = _parse_args()

    if args.list:
        print("Available stages:")
        for s in STAGES:
            print(f"  {s.num}. {s.key:>22}  — {s.desc}")
        return

    start, end = int(args.start), int(args.end)
    skip = _parse_csv_nums(args.skip)
    only = _parse_csv_nums(args.only)

    # Validate
    valid_nums = {s.num for s in STAGES}
    for lst, name in [(skip, "skip"), (only, "only")]:
        bad = [n for n in lst if n not in valid_nums]
        if bad:
            raise SystemExit(f"Unknown stage number(s) in --{name}: {bad}")

    if only and (start != 1 or end != STAGES[-1].num):
        print("[INFO] --only provided; ignoring --start/--end.")

    run_pipeline(start=start, end=end, skip=skip, only=only)


if __name__ == "__main__":
    main()

#==============================================================================
# COMMANDS YOU CAN USE FOR BETTER CONTROL
#==============================================================================
# (1) Full Pipeline (all stages): python -m src.generator.pipeline
# (2) List Stages: python -m src.generator.pipeline --list
# (3) Run a subset (e.g., only stages 3 and 4): python -m src.generator.pipeline --only 3,4
# (4) Run 1→6 but skip stage 3: python -m src.generator.pipeline --skip 3
# (5) Run 2→5: python -m src.generator.pipeline --start 2 --end 5


