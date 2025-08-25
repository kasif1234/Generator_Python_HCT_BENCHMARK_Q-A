# src/generator/stages/hct_from_json_tables.py
# STAGE 2: Generate HCT HTML/CSV and DB CSV/HTML from table specifications
# - input: semantics.json, table_to_gen.json

from __future__ import annotations

import os, json, csv, random, itertools
from pathlib import Path
import numpy as np
import pandas as pd

# --- robust imports for both "python -m src.generator..." and direct runs
try:
    from src.generator.config import (
        CONFIGS, PARAM_SEMANTICS_JSON, PARAM_TABLE_TO_GEN_JSON,
        SEMANTIC_TABLES_FOLDER, STR_REAL_VAL_FORMAT, STR_INT_VAL_FORMAT, NAME_SEP
    )
    from src.generator.utils.toolbox.tables import getSQLattrNames  # aliased in module
except Exception:
    from generator.config import (
        CONFIGS, PARAM_SEMANTICS_JSON, PARAM_TABLE_TO_GEN_JSON,
        SEMANTIC_TABLES_FOLDER, STR_REAL_VAL_FORMAT, STR_INT_VAL_FORMAT, NAME_SEP
    )
    from generator.utils.toolbox.tables import getSQLattrNames

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def read_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_unique_floats(n: int, low: float, high: float, decimals: int = 2) -> list[float]:
    """Generate n unique rounded floats in [low, high]."""
    seen, attempts, cap = set(), 0, n * 20
    while len(seen) < n and attempts < cap:
        attempts += 1
        seen.add(round(random.uniform(low, high), decimals))
    if len(seen) < n:
        # fallback: allow duplicates if range is too tight
        return [round(random.uniform(low, high), decimals) for _ in range(n)]
    return list(seen)

def sql_safe_row_headers(headers: list[str]) -> list[str]:
    """Second CSV row: SQL-safe field names via repo helper."""
    return getSQLattrNames(headers, name_sep=NAME_SEP)

def write_db_csv_html_sig(
    out_dir: Path, base: str, headers_cols: list[str], headers_rows: list[str], rows_with_vals: list[list]
):
    # Build DB dataframe (human-readable headers)
    cols = headers_cols + headers_rows + ["Value"]
    df = pd.DataFrame(rows_with_vals, columns=cols)

    # ---- CSV (header row = human names, second row = SQL-safe)
    out_csv = out_dir / f"{base}_DB.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    sql_row = pd.DataFrame([sql_safe_row_headers(cols)], columns=cols)
    # IMPORTANT: Put SQL row as the FIRST data row, to match the R behavior.
    full = pd.concat([sql_row, df.astype(str)], ignore_index=True)
    full.to_csv(out_csv, index=False)

    # ---- HTML (no SQL row)
    out_html = out_dir / f"{base}_DB.html"
    df.to_html(out_html, index=False)

    # ---- _SIG_DB.json (signature without aggregation info)
    # Build a simple signature: values used + column/row headers + a fixed style
    def uniq_in_order(seq):
        s, out = set(), []
        for x in seq:
            if x not in s:
                s.add(x); out.append(x)
        return out

    # collect unique label values (flatten everything except numeric "Value")
    used_vals = []
    for h in headers_cols + headers_rows:
        used_vals.extend(df[h].astype(str).tolist())
    all_vals = "&&&".join(uniq_in_order(used_vals))

    col_names_str = "&&&".join(headers_cols)
    row_names_str = "&&&".join(headers_rows)
    cur_sig_db = (
        f"ALL&&&{all_vals}"
        f"&&&COLS&&&{col_names_str}"
        f"&&&ROWS&&&{row_names_str}"
        f"&&&STYLE&&&WithoutBorderLines"
    )
    signature = cur_sig_db + "&&&AGG_NAME&&&&&&AGG_FUN&&&&&&AGG_COLS&&&&&&AGG_ROWS&&&"

    # pick display format from actual values
    try:
        as_float = pd.to_numeric(df["Value"], errors="coerce")
        all_ints = bool(np.isfinite(as_float).all() and np.all(np.mod(as_float, 1) == 0))
    except Exception:
        all_ints = False
    fmt = STR_INT_VAL_FORMAT if all_ints else STR_REAL_VAL_FORMAT

    sig_obj = {
        "id": base,
        "formatValue": fmt,
        "seedValue": str(SEED_VALUE_USED),  # set below per replica
        "signature": signature
    }
    out_sig = out_dir / f"{base}_SIG_DB.json"
    with open(out_sig, "w", encoding="utf-8") as f:
        json.dump(sig_obj, f, ensure_ascii=False)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    # ---- seed from repo config/config.json (fallback 0)
    cfg = read_json(CONFIGS / "config.json") if (CONFIGS / "config.json").exists() else {}
    seed = int(cfg.get("seed", 0))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(SEMANTIC_TABLES_FOLDER)
    semantics = read_json(Path(PARAM_SEMANTICS_JSON))
    tables = read_json(Path(PARAM_TABLE_TO_GEN_JSON))

    # we only need the labels/value range from the table spec;
    # the sampling here mirrors your a2 script logic
    
    for itab, t in enumerate(tables, 1):
        name = t["name"]
        num_rep = int(t.get("replica", 1))

        # extract column/row attribute lists (first group)
        col_attrs = (t.get("columns", {}).get("groups") or [{}])[0].get("attributes", [])  # type: ignore
        row_attrs = (t.get("rows", {}).get("groups") or [{}])[0].get("attributes", [])     # type: ignore

        # helper: sample from the semantics JSON like your original script
        def sample_axis(attrs):
            sampled = []
            for idx, a in enumerate(attrs):
                code = a.get("code")
                sample = a.get("sample", [1, 1])
                if not isinstance(sample, (list, tuple)) or len(sample) < 2:
                    sample = [1, 1]
                # find semantic entry with matching code
                for entry in semantics["data"]:
                    if entry.get("code") != code:
                        continue
                    values = entry.get("values", [])
                    names  = entry.get("names", [f"level{idx+1}"])
                    lvl1   = names[0] if names else f"level{idx+1}"
                    lvl2   = names[1] if len(names) > 1 else f"{lvl1}_sub"
                    mn, mx = int(sample[0]), int(sample[1])

                    if values and isinstance(values[0], dict):
                        # hierarchical: [{'A':[...]} , {'B':[...]}]
                        pairs = []
                        for d in values:
                            for k, v in d.items():
                                pairs.append((k, v))
                        take1 = min(len(pairs), random.randint(mn, mx))
                        for l1, lst in random.sample(pairs, take1):
                            take2 = min(len(lst), random.randint(mn, mx))
                            sampled.append({lvl1: l1, lvl2: random.sample(lst, take2) if take2 else []})
                    else:
                        # flat list of strings
                        take = min(len(values), random.randint(mn, mx))
                        for item in random.sample(values, take):
                            sampled.append({lvl1: item})
                    break
            return sampled

        # headers order
        def headers_in_order(sampled):
            seen, out = set(), []
            for it in sampled:
                for k in it.keys():
                    if k not in seen:
                        seen.add(k); out.append(k)
            return out

        # collect values list for a header across samples; pad to desired length
        def collect_for(header, sampled, target_len):
            vals = []
            for it in sampled:
                if header in it:
                    v = it[header]
                    if isinstance(v, list):
                        vals.extend(v)
                    else:
                        vals.append(v)
            if not vals:
                return [""] * target_len
            # repeat/crop deterministically to match target_len
            rep = (target_len // len(vals)) + 1
            vals = (vals * rep)[:target_len]
            return vals

        for rep in range(1, num_rep + 1):
            # per-replica seed as in R
            global SEED_VALUE_USED
            SEED_VALUE_USED = (itab - 1) * num_rep + rep
            random.seed(SEED_VALUE_USED)
            np.random.seed(SEED_VALUE_USED)

            # sample columns & rows
            sampled_cols = sample_axis(col_attrs)
            sampled_rows = sample_axis(row_attrs)

            col_headers = headers_in_order(sampled_cols)
            row_headers = headers_in_order(sampled_rows)

            rowAttr = row_attrs
            colAttr = col_attrs
            values_spec = t.get("values", [0, 100])
            agg_fun  = t.get("agg_fun1", "sum")
            agg_name = t.get("agg_name1", "Total")

            # ---- New variables with defaults ----
            row_format = str(t.get("row_format", "new"))

            # Collect agg_pos1 values as strings (not lists)
            col_agg_pos1 = str(col_attrs[0].get("agg_pos1", "none")) if col_attrs else "none"
            row_agg_pos1 = str(row_attrs[0].get("agg_pos1", "none")) if row_attrs else "none"


                        # ---- Safeguard: reset unsupported combo "indent-right-top" ----
            if row_format == "indent" and col_agg_pos1 == "right" and row_agg_pos1 == "top":
                row_format = "new"
                col_agg_pos1 = "none"
                row_agg_pos1 = "none"

            print('====================')
            print(row_format)
            print(col_agg_pos1)
            print(row_agg_pos1)
            print('====================')



            print({
                "rowAttr": rowAttr,
                "colAttr": colAttr,
                "values": values_spec,
                "agg": {"func": agg_fun, "name": agg_name},
                "row_format": row_format,
                "col_agg_pos1": col_agg_pos1,
                "row_agg_pos1": row_agg_pos1
            })



            # determine column/row counts (same strategy as your a2 file)
            col_count = 0
            for c in sampled_cols:
                for v in c.values():
                    if isinstance(v, list):
                        col_count += len(v)
            if col_count == 0:
                for c in sampled_cols:
                    col_count += len(c.values())

            if sampled_rows:
                first_key = next(iter(sampled_rows[0].keys()))
                first_key_count = sum(1 for r in sampled_rows if next(iter(r.keys())) == first_key)
                max_val_len = 0
                for r in sampled_rows:
                    for v in r.values():
                        max_val_len = max(max_val_len, len(v) if isinstance(v, list) else 1)
                row_count = first_key_count * max_val_len
            else:
                row_count = 0

            total_cells = col_count * row_count
            if col_count == 0 or row_count == 0 or total_cells == 0:
                # nothing to write for this replica
                continue

            # build unique combinations for columns/rows (row-wise)
            col_matrix = [[collect_for(h, sampled_cols, col_count)[i] for h in col_headers] for i in range(col_count)]
            row_matrix = [[collect_for(h, sampled_rows, row_count)[i] for h in row_headers] for i in range(row_count)]

            # cartesian combine (columns × rows), then append numeric Value
            combined = []
            for col_vec in col_matrix:
                for row_vec in row_matrix:
                    combined.append(col_vec + row_vec)

            # values range from spec (fallback if invalid)
            low, high = 0.0, 100.0
            try:
                rng = t.get("values", [0, 100])
                low, high = float(rng[0]), float(rng[1])
            except Exception:
                pass
            vals = generate_unique_floats(len(combined), low, high, decimals=2)

            rows_with_vals = [r + [v] for r, v in zip(combined, vals)]


            # ---- Create a DataFrame and inspect it ----
            cols = col_headers + row_headers + ["Value"]
            df = pd.DataFrame(rows_with_vals, columns=cols)

            # print all rows
            print("\n=== Full DataFrame ===")
            print(df.to_string(index=False))   # ensures all rows & no truncated display
            print(f"Shape: {df.shape}")
            print("======================\n")

            # ---- Turn DataFrame into a pivot table and print all of it ----
            agg_fun  = t.get("agg_fun1", "sum")

            def _agg_reducer(name: str):
                n = (name or "").lower()
                return {
                    "avg": np.nanmean, "mean": np.nanmean, "average": np.nanmean,
                    "min": np.nanmin, "max": np.nanmax, "sum": np.nansum
                }.get(n, np.nansum)

            pivot = pd.pivot_table(
                df,
                index=row_headers,          # rows
                columns=col_headers,        # columns
                values="Value",
                aggfunc=_agg_reducer(agg_fun),
                observed=False,
                dropna=True,
            )

            # sort for readability
            try:
                pivot = pivot.sort_index().sort_index(axis=1)
            except Exception:
                pass

            # print entire pivot (no truncation)
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 2000):
                print("\n=== Pivot Table ===")
                print(pivot)
                print("===================\n")

            #Reshaping
            # ================= Reshape into 1 of 12 aggregate placements =================
            # ---------- DEBUG KNOBS ----------
            CASE_ROW_FORMAT = row_format   # "indent" or "new"
            CASE_COL_AGG_POS = col_agg_pos1  # "none" or "right"
            CASE_ROW_AGG_POS = row_agg_pos1  # "top", "bottom", "none"

            AGG_FUNC_NAME = t.get("agg_fun1", "sum")
            AGG_LABEL     = t.get("agg_name1", "Total")

            # ---------- helpers (unchanged) ----------
            def _agg_reducer_from_name(nm: str):
                nm = (nm or "").lower()
                return {
                    "avg": np.nanmean, "mean": np.nanmean, "average": np.nanmean,
                    "min": np.nanmin, "max": np.nanmax, "sum": np.nansum
                }.get(nm, np.nansum)

            def _add_col_total(df: pd.DataFrame, label: str, agg_name: str) -> pd.DataFrame:
                reducer = _agg_reducer_from_name(agg_name)
                col_total = df.apply(reducer, axis=1)
                if isinstance(df.columns, pd.MultiIndex):
                    new_key = tuple([label] + [""]*(df.columns.nlevels-1))
                else:
                    new_key = label
                out = df.copy()
                out[new_key] = col_total
                return out

            def _add_row_total(df: pd.DataFrame, label: str, agg_name: str, position: str) -> pd.DataFrame:
                reducer = _agg_reducer_from_name(agg_name)
                row_total = df.apply(lambda s: reducer(s.values), axis=0).to_frame().T
                if isinstance(df.index, pd.MultiIndex):
                    idx_label = tuple([label] + [""]*(df.index.nlevels-1))
                    row_total.index = pd.MultiIndex.from_tuples([idx_label], names=df.index.names)
                else:
                    row_total.index = pd.Index([label], name=df.index.name)
                return pd.concat([row_total, df]) if position == "top" else pd.concat([df, row_total])

            # ---------- NEW helpers to match your screenshots ----------
            def _format_indent_console(df: pd.DataFrame) -> pd.DataFrame:
                """
                Screenshot-style 'Indent':
                • Insert a group subtotal row (first-level label) BEFORE its children.
                • Children appear as indented rows under a blank first-level cell.
                """
                if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels < 2:
                    return df.reset_index()

                reducer = _agg_reducer_from_name(AGG_FUNC_NAME)
                lvl0_name, lvl1_name = df.index.names[:2]
                blocks = []

                for g in df.index.get_level_values(0).unique():
                    sub = df.xs(g, level=0, drop_level=False)

                    # subtotal row for group 'g' (aggregate over its children)
                    subtotal_vals = sub.apply(lambda s: reducer(s.values), axis=0)
                    subtotal_row = pd.DataFrame([subtotal_vals])
                    subtotal_row.insert(0, lvl0_name, g)
                    subtotal_row.insert(1, lvl1_name, "")  # blank second-level
                    blocks.append(subtotal_row)

                    # children rows, indented
                    det = sub.copy().reset_index()
                    det[lvl0_name] = ""  # blank first-level cell
                    det[lvl1_name] = det[lvl1_name].map(lambda x: f"  {x}")
                    blocks.append(det)

                disp = pd.concat(blocks, ignore_index=True)
                return disp  # print with index=False

            def _format_new_console(df: pd.DataFrame) -> pd.DataFrame:
                """
                Screenshot-style 'New':
                • Two index columns.
                • Show the first-level label ONCE per block; blanks for subsequent rows.
                """
                if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels < 2:
                    return df.reset_index()

                disp = df.reset_index()
                lvl0_name = disp.columns[0]
                last = object()
                for i in range(len(disp)):
                    cur = disp.at[i, lvl0_name]
                    if cur == last:
                        disp.at[i, lvl0_name] = ""
                    else:
                        last = cur
                return disp  # print with index=False

            # ---------- apply chosen case ----------
            reshaped = pivot.copy()
            if CASE_COL_AGG_POS == "right":
                reshaped = _add_col_total(reshaped, AGG_LABEL, AGG_FUNC_NAME)
            if CASE_ROW_AGG_POS in ("top", "bottom"):
                reshaped = _add_row_total(reshaped, AGG_LABEL, AGG_FUNC_NAME, CASE_ROW_AGG_POS)

            if CASE_ROW_FORMAT == "indent":
                to_print = _format_indent_console(reshaped)
            elif CASE_ROW_FORMAT == "new":
                to_print = _format_new_console(reshaped)
            else:
                to_print = reshaped  # fallback

            
                


            #Save the html in tables
            # ---- Save reshaped pivot as HTML: <name>_<rep>_HCT.html ----
            # ---- Save the reshaped table exactly as printed ----
            to_save = to_print.copy()             # same structure as console output

            # format numeric cells to 2 decimals without changing layout
            def _fmt(x):
                try:
                    if pd.notna(x) and isinstance(x, (int, float, np.floating)):
                        return f"{float(x):.2f}"
                except Exception:
                    pass
                return x

            for c in to_save.select_dtypes(include=[np.number]).columns:
                to_save[c] = to_save[c].map(_fmt)

            # write <name>_<rep>_HCT.html under SEMANTIC_TABLES_FOLDER
            Path(SEMANTIC_TABLES_FOLDER).mkdir(parents=True, exist_ok=True)
            hct_html_path = Path(SEMANTIC_TABLES_FOLDER) / f"{t['name']}_{rep}_HCT.html"

            # keep exact shape: multiindex column headers preserved; no row numbers
            to_save.to_html(hct_html_path, index=False)
            print(f"Saved HCT HTML → {hct_html_path}")

            # ---- ALSO save a CSV mirror of the HCT view ----
            df_hct_csv = to_save.copy()

            # If columns are MultiIndex, flatten them for CSV using your NAME_SEP
            if isinstance(df_hct_csv.columns, pd.MultiIndex):
                df_hct_csv.columns = [
                    NAME_SEP.join([str(p) for p in tup if str(p).strip()])
                    for tup in df_hct_csv.columns.to_list()
                ]

            hct_csv_path = Path(SEMANTIC_TABLES_FOLDER) / f"{t['name']}_{rep}_HCT.csv"
            df_hct_csv.to_csv(hct_csv_path, index=False, encoding="utf-8")
            print(f"Saved HCT CSV  → {hct_csv_path}")


            
            # ---- Build & save _SIG_HCT.json and _HCT.json ----
            base_id = f"{t['name']}_{rep}"

            def _uniq(seq):
                s, out = set(), []
                for x in seq:
                    if x not in s:
                        s.add(x); out.append(x)
                return out

            # DB-like part of signature
            used = []
            for h in col_headers + row_headers:
                used.extend(df[h].astype(str).tolist())
            all_vals = "&&&".join(_uniq(used))
            col_names_str = "&&&".join(col_headers)
            row_names_str = "&&&".join(row_headers)
            cur_sig_db = (
                f"ALL&&&{all_vals}"
                f"&&&COLS&&&{col_names_str}"
                f"&&&ROWS&&&{row_names_str}"
                f"&&&STYLE&&&WithoutBorderLines"
            )

            # Aggregation info reflected in the reshaped table
            agg_col_names = col_headers if CASE_COL_AGG_POS == "right" else []
            agg_row_names = row_headers if CASE_ROW_AGG_POS in ("top", "bottom") else []

            sig_hct = (
                cur_sig_db +
                "&&&AGG_NAME&&&" + str(AGG_LABEL) +
                "&&&AGG_FUN&&&" + str(AGG_FUNC_NAME) +
                "&&&AGG_COLS&&&" + "&&&".join(agg_col_names) +
                "&&&AGG_ROWS&&&" + "&&&".join(agg_row_names)
            )

            # formatValue
            try:
                as_float = pd.to_numeric(df["Value"], errors="coerce")
                all_ints = bool(np.isfinite(as_float).all() and np.all(np.mod(as_float, 1) == 0))
            except Exception:
                all_ints = False
            fmt_hct = STR_INT_VAL_FORMAT if (all_ints and AGG_FUNC_NAME.lower() in ("sum", "count")) else STR_REAL_VAL_FORMAT

            # <name>_<rep>_SIG_HCT.json
            sig_hct_obj = {"id": base_id, "formatValue": fmt_hct, "seedValue": str(SEED_VALUE_USED), "signature": sig_hct}
            with open(Path(SEMANTIC_TABLES_FOLDER) / f"{base_id}_SIG_HCT.json", "w", encoding="utf-8") as f:
                json.dump(sig_hct_obj, f, ensure_ascii=False)

            # <name>_<rep>_HCT.json
            props = {
                "Standard Relational Table": False,
                "Multi Level Column": len(col_headers) > 1,
                "Balanced Multi Level Column": True,
                "Symmetric Multi Level Column": True,
                "Unbalanced Multi Level Column": False,
                "Asymmetric Multi Level Column": False,
                "Column Aggregation": bool(agg_col_names),
                "Global Column Aggregation": bool(agg_col_names),
                "Local Column-Group Aggregation": False,
                "Explicit Column Aggregation Terms": True,
                "Implicit Column Aggregation Terms": False,
                "Row Nesting": len(row_headers) > 1,
                "Balanced Row Nesting": True,
                "Symmetric Row Nesting": True,
                "Unbalanced Row Nesting": False,
                "Asymmetric Row Nesting": False,
                "Row Aggregation": bool(agg_row_names),
                "Global Row Aggregation": bool(agg_row_names),
                "Local Row-Group Aggregation": False,
                "Explicit Row Aggregation Terms": True,
                "Implicit Row Aggregation Terms": False,
                "Split Header Cell": False,
                "Row Group Label": (CASE_ROW_FORMAT == "indent"),
            }
            hct_obj = {
                "id": base_id,
                "formatValue": fmt_hct,
                "seedValue": str(SEED_VALUE_USED),
                "signature": sig_hct,
                "image_source": str(hct_html_path),  # we saved HTML; adjust if you later save a PDF/PNG
                "state": "labelled",
                "concern": False,
                "notes": "",
                "properties": props,
                "themes": [],
            }
            with open(Path(SEMANTIC_TABLES_FOLDER) / f"{base_id}_HCT.json", "w", encoding="utf-8") as f:
                json.dump(hct_obj, f, ensure_ascii=False)




            base = f'{t["name"]}_{rep}'
            write_db_csv_html_sig(
                out_dir=output_dir,
                base=base,
                headers_cols=col_headers,
                headers_rows=row_headers,
                rows_with_vals=rows_with_vals
            )

            print(f"Saved DB artifacts for {base} → {output_dir}")

            

if __name__ == "__main__":
    main()