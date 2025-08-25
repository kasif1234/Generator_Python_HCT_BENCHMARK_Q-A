# src/generator/stages/tables_from_patterns.py 
# STAGE 1: Generate table specifications from patterns and parameters
# - input: semantics.json, table_templates.json

from __future__ import annotations

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from pathlib import Path
import json, itertools, random
from datetime import datetime
from collections import Counter

from generator.config import (
    CONFIGS,
    PARAM_SEMANTICS_JSON, PARAM_TABLE_TEMPLATES_JSON, PARAM_TABLE_TO_GEN_JSON,
    S1_OUT,
)

# ---------- small I/O helpers ----------
def pread(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def pwrite(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

# ---------- Stage 1 generator ----------
class TableGenerator:
    def __init__(
        self,
        semantics_file: Path = PARAM_SEMANTICS_JSON,
        table_templates_file: Path = PARAM_TABLE_TEMPLATES_JSON,
        output_file: Path = PARAM_TABLE_TO_GEN_JSON,
    ):
        self.semantics_file = Path(semantics_file)
        self.table_templates_file = Path(table_templates_file)
        self.output_file = Path(output_file)
        self.generated_tables = []

        cfg = pread(CONFIGS / "config.json") if (CONFIGS / "config.json").exists() else {}
        self.seed = cfg.get("seed", 1)
        self.feature_flags = cfg.get("feature_flags", {})
        random.seed(self.seed)

        # metadata collectors
        self.meta = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "seed": self.seed,
            "replica": None,
            "templates_total": 0,
            "combinations_total": 0,
            "generated_total": 0,
            "skipped_total": 0,
            "generated_per_template": {},   # e.g., {"1/7": 42, ...}
            "by_shuffle": Counter(),
            "by_row_format": Counter(),
            "by_levels": Counter(),         # key = "cols_rows" (e.g., "3_2")
        }

    # 1) load inputs
    def load_data(self):
        self.allSemanticAttributes = pread(self.semantics_file)
        self.tablePatterns = pread(self.table_templates_file)
        self.replica = self.tablePatterns.get("replica")
        self.meta["replica"] = self.replica

    # 2) parameter combinations
    def _param_combinations(self):
        combos = {
            "shuffle":            self.tablePatterns.get("shuffle", []),
            "col_row_name_pos":   self.tablePatterns.get("col_row_name_pos", []),
            "col_row_agg_pos":    self.tablePatterns.get("col_row_agg_pos", []),
            "col_row_levels":     self.tablePatterns.get("col_row_levels", []),
            "row_format":         self.tablePatterns.get("row_format", []),
        }
        rev = [
            combos["row_format"],
            combos["col_row_levels"],
            combos["col_row_agg_pos"],
            combos["col_row_name_pos"],
            combos["shuffle"],
        ]
        prod = itertools.product(*rev)
        return [(sh, nm, ag, lv, rf) for rf, lv, ag, nm, sh in prod]

    # 3) expand templates × combinations
    def generate_tables(self):
        param_combi = self._param_combinations()
        table_templates = self.tablePatterns.get("tables", [])
        self.meta["templates_total"] = len(table_templates)
        self.meta["combinations_total"] = len(param_combi)

        count = 0
        for ti, cur in enumerate(table_templates, 1):
            sval = cur.get("values")
            values_type = sval if not (isinstance(sval, list) and len(sval) == 1) else sval[0]

            valueName = cur.get("valueName", "table")
            rowCodes  = cur.get("rowCodes", [])
            rowSamples= cur.get("rowSamples", [])
            colCodes  = cur.get("colCodes", [])
            colSamples= cur.get("colSamples", [])
            agg_name1 = cur.get("agg_name1")
            agg_fun1  = cur.get("agg_fun1")

            key_t = f"{ti}/{len(table_templates)}"
            gen_for_this_template = 0

            for ci, (shuffle, name_pos, agg_pos, levels, row_format) in enumerate(param_combi, 1):
                print(f"{ti}/{len(table_templates)} -- {ci}/{len(param_combi)}")
                try:
                    col_levels_str, row_levels_str = levels.split("_")
                    col_levels, row_levels = int(col_levels_str), int(row_levels_str)
                    col_name_pos, row_name_pos = name_pos.split("_")
                    col_agg_pos,  row_agg_pos  = agg_pos.split("_")
                except Exception as e:
                    print("Bad pattern combo:", e)
                    self.meta["skipped_total"] += 1
                    continue

                if row_format == "new" and row_agg_pos == "top":
                    print("NOT GENERATED")
                    self.meta["skipped_total"] += 1
                    continue

                tableName = f"{valueName.replace(' ', '_')}_set{ci}"

                def mk_attrs(codes, samples, name_pos, agg_pos, nmax):
                    out = []
                    n = min(nmax, len(codes))
                    for i in range(n):
                        samp = samples[i] if i < len(samples) else []
                        samp = samp if isinstance(samp, list) and len(samp) == 2 else [0]
                        out.append({"code": codes[i], "pos": name_pos, "sample": samp, "agg_pos1": agg_pos})
                    return out

                colAttribs = mk_attrs(colCodes, colSamples, col_name_pos, col_agg_pos, col_levels)
                rowAttribs = mk_attrs(rowCodes, rowSamples, row_name_pos, row_agg_pos, row_levels)

                self.generated_tables.append({
                    "name": tableName,
                    "replica": self.replica,
                    "shuffle": shuffle,
                    "agg_fun1": agg_fun1,
                    "agg_name1": agg_name1,
                    "values": values_type,
                    "valueName": valueName,
                    "row_format": row_format,
                    "columns": {"groups": [{"attributes": colAttribs}]},
                    "rows":    {"groups": [{"attributes": rowAttribs}]},
                })

                # metadata counters
                count += 1
                gen_for_this_template += 1
                self.meta["generated_total"] = count
                self.meta["by_shuffle"][shuffle] += 1
                self.meta["by_row_format"][row_format] += 1
                self.meta["by_levels"][f"{col_levels}_{row_levels}"] += 1

            self.meta["generated_per_template"][key_t] = gen_for_this_template

        print("NUMBER OF TABLES:", count)

    # 4) write outputs (canonical + cache + metadata)
    def write_outputs(self):
        # canonical param file (used by Stage 2)
        pwrite(Path(PARAM_TABLE_TO_GEN_JSON), self.generated_tables)
        # stage artifact in cache/s1_tables
        pwrite(Path(S1_OUT) / "tables_to_generate.json", self.generated_tables)

        # convert Counters to dicts for JSON
        meta_out = dict(self.meta)
        meta_out["by_shuffle"] = dict(self.meta["by_shuffle"])
        meta_out["by_row_format"] = dict(self.meta["by_row_format"])
        meta_out["by_levels"] = dict(self.meta["by_levels"])

        # metadata file
        pwrite(Path(S1_OUT) / "tables_to_generate_metadata.json", meta_out)

        print(f"Wrote: {PARAM_TABLE_TO_GEN_JSON}")
        print(f"Wrote: {Path(S1_OUT) / 'tables_to_generate.json'}")
        print(f"Wrote: {Path(S1_OUT) / 'tables_to_generate_metadata.json'}")

    def run(self):
        self.load_data()
        self.generate_tables()
        self.write_outputs()

def run(_cfg=None):
    TableGenerator().run()

if __name__ == "__main__":
    run()
