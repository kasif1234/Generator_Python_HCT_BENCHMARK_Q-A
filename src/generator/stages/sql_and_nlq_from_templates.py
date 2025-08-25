# src/generator/stages/sql_and_nlq_from_templates.py
# STAGE 3: Generate SQL queries and NL questions from table instances

from __future__ import annotations

import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# --- Project config paths -----------------------------------------------------
# Uses the config.py you provided (unchanged)
try:
    # when used as module
    from ..config import (
        PARAM_SEMANTICS_JSON,
        PARAM_TABLE_TEMPLATES_JSON,
        PARAM_NLQ_TEMPLATES_JSON,
        PARAM_TABLE_TO_GEN_JSON,
        SEMANTIC_TABLES_FOLDER,
        SEMANTIC_QANDA_FOLDER,
        S3_OUT,
        get_sql_attr_names,
    )
except Exception:
    # allow running as a script too
    from pathlib import Path
    import importlib.util

    cfg_path = Path(__file__).resolve().parents[1] / "config.py"
    spec = importlib.util.spec_from_file_location("config", cfg_path)
    cfg = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(cfg)  # type: ignore

    PARAM_SEMANTICS_JSON = cfg.PARAM_SEMANTICS_JSON
    PARAM_TABLE_TEMPLATES_JSON = cfg.PARAM_TABLE_TEMPLATES_JSON
    PARAM_NLQ_TEMPLATES_JSON = cfg.PARAM_NLQ_TEMPLATES_JSON
    PARAM_TABLE_TO_GEN_JSON = cfg.PARAM_TABLE_TO_GEN_JSON
    SEMANTIC_TABLES_FOLDER = cfg.SEMANTIC_TABLES_FOLDER
    SEMANTIC_QANDA_FOLDER = cfg.SEMANTIC_QANDA_FOLDER
    S3_OUT = cfg.S3_OUT
    get_sql_attr_names = cfg.get_sql_attr_names

# --- General settings ----------------------------------------------------------
random.seed(1)
SEMANTIC_QANDA_FOLDER.mkdir(parents=True, exist_ok=True)
S3_OUT.mkdir(parents=True, exist_ok=True)

# filenames we expect per table instance
SIG_FILE_EXT = "_SIG_HCT.json"
FORM_FILE_EXT = "_HCT.json"
DB_FILE_EXT = "_DB.csv"

# list of families to process (same as the R script)
LIST_DATA_PREFIX = [
    "Evolution_of_pollution_in_percent",
    "Food_import-export_in_tons",
    "Number_of_accidents",
    "Number_of_constructions",
    "Number_of_students",
    "Number_of_graduations",
    "Weather_statistics",
]

# --- Utilities ----------------------------------------------------------------
def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_sqlite(df: pd.DataFrame) -> Tuple[sqlite3.Connection, str]:
    """Load a DataFrame into an in-memory SQLite DB and return (conn, table_name)."""
    conn = sqlite3.connect(":memory:")
    df_sql = df.copy()
    df_sql.columns = get_sql_attr_names(df_sql.columns.tolist())
    table = "DBdata"
    df_sql.to_sql(table, conn, index=False)
    return conn, table


def format_results(df: pd.DataFrame) -> str:
    """A compact, stable string representation used in the outputs."""
    if df is None or df.empty:
        return ""
    return "; ".join(
        [", ".join([f"{c}={v}" for c, v in row.items()]) for row in df.to_dict(orient="records")]
    )


def parse_signature(sig: str) -> Dict[str, List[str]]:
    """
    Parse the HCT signature string (same tokens as in R).
    Returns dict with keys: colNames, rowNames, aggColNames, aggRowNames, aggFun, aggName.
    """
    parts = [p.strip() for p in sig.split("&&&")]
    idx = {k: i for i, k in enumerate(parts)}
    col_i = idx.get("COLS")
    row_i = idx.get("ROWS")
    style_i = idx.get("STYLE")
    agg_name_i = idx.get("AGG_NAME")
    agg_fun_i = idx.get("AGG_FUN")
    agg_cols_i = idx.get("AGG_COLS")
    agg_rows_i = idx.get("AGG_ROWS")

    def slice_between(a: Optional[int], b: Optional[int]) -> List[str]:
        if a is None or b is None:
            return []
        if b - a <= 1:
            return []
        return [p.strip() for p in parts[a + 1 : b]]

    col_names = slice_between(col_i, row_i)
    row_names = slice_between(row_i, style_i)
    agg_col_names = slice_between(agg_cols_i, agg_rows_i)
    agg_row_names = [] if agg_rows_i is None else [p.strip() for p in parts[agg_rows_i + 1 :]]

    agg_fun = parts[agg_fun_i + 1] if agg_fun_i is not None else ""
    agg_name = parts[agg_name_i + 1] if agg_name_i is not None else ""

    # normalize to SQL-safe names
    col_names = get_sql_attr_names(col_names)
    row_names = get_sql_attr_names(row_names)
    agg_col_names = get_sql_attr_names(agg_col_names)
    agg_row_names = get_sql_attr_names(agg_row_names)

    return dict(
        colNames=col_names,
        rowNames=row_names,
        aggColNames=agg_col_names or [""],
        aggRowNames=agg_row_names or [""],
        aggFun=agg_fun,
        aggName=agg_name,
    )


def get_headers(df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    if not names:
        return pd.DataFrame()
    names = [n for n in names if n in df.columns]
    if not names:
        return pd.DataFrame()
    return df.loc[:, names].drop_duplicates().reset_index(drop=True)


def pick_indices(df: pd.DataFrame, nmin: int = 1, nmax: int = 1) -> List[int]:
    """
    Safe sampler:
    - clamps nmin/nmax to the population
    - never asks random.sample() for more than available
    - if the frame is empty, returns []
    """
    n = len(df)
    if n == 0:
        return []
    nmin = max(1, min(nmin, n))
    nmax = max(1, min(nmax, n))
    if nmin > nmax:
        nmin = nmax
    k = random.randint(nmin, nmax)
    if k > n:
        k = n
    return sorted(random.sample(range(n), k))


def clause_for_selection(df: pd.DataFrame, idxs: List[int]) -> str:
    if not idxs:
        return "TRUE"
    rows = []
    for i in idxs:
        conds = [f"{col} = '{str(df.iloc[i][col])}'" for col in df.columns]
        rows.append("(" + " AND ".join(conds) + ")")
    return "(" + " OR ".join(rows) + ")"


def expr_clause(exprs: Sequence[str]) -> str:
    if not exprs:
        return "Value"
    parts = [f"{e}(Value) AS {e}_Value" for e in exprs]
    return ", ".join(parts)


def run_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn)
    except Exception:
        return pd.DataFrame()


def nlq_fill(pattern: str, colH: Optional[pd.DataFrame], rowH: Optional[pd.DataFrame]) -> str:
    """
    Very light NLQ substitution: replaces tokens like $Year, $Gender with chosen values.
    """
    s = pattern

    def to_text(h: Optional[pd.DataFrame]) -> Dict[str, str]:
        if h is None or h.empty:
            return {}
        out: Dict[str, str] = {}
        for col in h.columns:
            vals = h[col].astype(str).tolist()
            seen, uniq = set(), []
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    uniq.append(v)
            out[col] = " or ".join(uniq)
        return out

    repl = {}
    repl.update(to_text(colH))
    repl.update(to_text(rowH))

    tokens = set(re.findall(r"\$[A-Za-z0-9_]+", s))
    for t in tokens:
        key = t[1:]
        key2 = key.replace("__", "_").replace("_", " ")
        val = (repl.get(key) or repl.get(key2) or repl.get(key.replace("_", " ")) or "").strip()
        val = (val + " ").strip() + " "
        s = s.replace(t, "$" + key.replace(" ", "_"))
        s = s.replace("$" + key.replace(" ", "_"), val)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\s*\?", " ?", s)
    return s


# --- JSON helpers to match R output shapes ------------------------------------
def sql_json_template(tnum: int, sql: str, gt: str, nlq_list: Optional[List[str]]) -> dict:
    return {
        "name": f"template_{tnum}",
        "sql": sql,
        "GTresult": gt,
        "NLquestions": nlq_list if nlq_list is not None else "NA",
    }


def qna_json_line(
    tnum: int,
    rowH: pd.DataFrame,
    colH: pd.DataFrame,
    indRow: Optional[List[int]],
    indCol: Optional[List[int]],
    result: pd.DataFrame,
    exprs: Optional[Sequence[str]] = None,
    tableAggFun: Optional[str] = None,
    topk: Optional[int] = None,
) -> dict:
    return {
        "template_id": tnum,
        "row_filter": bool(indRow),
        "col_filter": bool(indCol),
        "has_expr": bool(exprs),
        "table_agg": tableAggFun or "",
        "topk": int(topk or 0),
        "row_selected": [] if indRow is None else indRow,
        "col_selected": [] if indCol is None else indCol,
        "n_returned": 0 if result is None else int(len(result)),
    }


def aggregate_nlq(qna_list: List[dict]) -> str:
    return json.dumps(qna_list, ensure_ascii=False)


# --- Main pipeline -------------------------------------------------------------
def main(generate_nlq: bool = True) -> None:
    # Load params
    print("[START] Stage 3: SQL & NLQ generation")
    table_instances = read_json(PARAM_TABLE_TO_GEN_JSON)
    table_patterns = read_json(PARAM_TABLE_TEMPLATES_JSON)
    semantics = read_json(PARAM_SEMANTICS_JSON)
    nlq_patterns = read_json(PARAM_NLQ_TEMPLATES_JSON)

    # Map valueName -> all human attr names
    value_to_all_attrnames: Dict[str, List[str]] = {}
    for t in table_patterns["tables"]:
        all_codes = list(t["rowCodes"]) + list(t["colCodes"])
        names: List[str] = []
        for code in all_codes:
            for d in semantics["data"]:
                if d["code"] == code:
                    names.extend(d["names"])
        value_to_all_attrnames[t["valueName"]] = names

    annotator_chunks: List[str] = []
    dump_all_records: List[dict] = []

    # process each prefix family
    for data_prefix in LIST_DATA_PREFIX:
        family_roots = []
        for p in SEMANTIC_TABLES_FOLDER.glob(f"{data_prefix}_set*{SIG_FILE_EXT}"):
            family_roots.append(p.name.replace(SIG_FILE_EXT, ""))

        if not family_roots:
            print(f"[SKIP] No tables found for prefix '{data_prefix}' in {SEMANTIC_TABLES_FOLDER}")
            continue

        print(f"[FAMILY] {data_prefix}: {len(family_roots)} table(s) to process")
        dump_rows: List[dict] = []

        for root in family_roots:
            csv_file = SEMANTIC_TABLES_FOLDER / f"{root}{DB_FILE_EXT}"
            sig_file = SEMANTIC_TABLES_FOLDER / f"{root}{SIG_FILE_EXT}"
            annot_file = SEMANTIC_TABLES_FOLDER / f"{root}{FORM_FILE_EXT}"

            if not csv_file.exists() or not sig_file.exists() or not annot_file.exists():
                print(f"[WARN] Missing one of {{CSV,SIG,FORM}} for {root}; skipping")
                continue

            print(f"  [TABLE] {root}")
            df0 = pd.read_csv(csv_file)

            if len(df0) >= 2 and isinstance(df0.iloc[0, 0], str) and df0.columns[0] != "Value":
                df = df0.iloc[1:].copy()
            else:
                df = df0.copy()

            if "Value" in df.columns:
                df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
            df = df.reset_index(drop=True)

            sig = read_json(sig_file)
            annot = read_json(annot_file)
            sig_meta = parse_signature(sig.get("signature", ""))

            col_names = sig_meta["colNames"]
            row_names = sig_meta["rowNames"]
            agg_col_names = [a for a in sig_meta["aggColNames"] if a]
            agg_fun = sig_meta["aggFun"]

            df.columns = get_sql_attr_names(df.columns.tolist())

            col_headers = get_headers(df, col_names)
            row_headers = get_headers(df, row_names)

            COLnum = len(col_headers)
            ROWnum = len(row_headers)
            COLdepth = len(col_headers.columns)
            ROWdepth = len(row_headers.columns)
            COLagg = len(agg_col_names)
            ROWagg = 0

            conn, table = to_sqlite(df)

            table_name_human = re.sub(r"_", " ", data_prefix)
            nlq_block = None
            for block in nlq_patterns:
                if block.get("tableName") == table_name_human:
                    nlq_block = block
                    break
            if not nlq_block:
                nlq_block = {
                    "template_report": ["Please,_report_the_corresponding_$REPORTATTR"],
                    "valueMeaning": [""],
                    "simplifyNested": [],
                }

            value_meaning = (nlq_block.get("valueMeaning") or [""])[0]

            sql_nlq_entries: List[dict] = []
            json_qna_items: List[dict] = []
            nlq_firsts: Dict[int, str] = {}
            ans_firsts: Dict[int, str] = {}

            def _record(
                tnum: int,
                nlq_list: List[str],
                sql_str: str,
                res_df: pd.DataFrame,
                ind_row: Optional[List[int]],
                ind_col: Optional[List[int]],
                exprs: Optional[Sequence[str]] = None,
                tableAggFun: Optional[str] = None,
                topk: Optional[int] = None,
            ) -> None:
                nonlocal dump_rows, dump_all_records, sql_nlq_entries, nlq_firsts, ans_firsts, json_qna_items

                gt = format_results(res_df)
                sql_nlq_entries.append(sql_json_template(tnum, sql_str, gt, nlq_list if generate_nlq else None))
                json_qna_items.append(
                    qna_json_line(
                        tnum,
                        row_headers,
                        col_headers,
                        ind_row,
                        ind_col,
                        res_df,
                        exprs=exprs,
                        tableAggFun=tableAggFun,
                        topk=topk,
                    )
                )
                if nlq_list:
                    nlq_firsts[tnum] = nlq_list[0]
                ans_firsts[tnum] = gt

                for q in (nlq_list or [""]):
                    dump_rows.append(
                        dict(
                            tableFile=root,
                            questionTemplate=tnum,
                            NLQ=q,
                            SQLR=gt,
                            SQLQ=sql_str,
                            indData=1,
                            COLdepth=COLdepth,
                            ROWdepth=ROWdepth,
                            COLnum=COLnum,
                            ROWnum=ROWnum,
                            COLagg=COLagg,
                            ROWagg=ROWagg,
                        )
                    )
                    dump_all_records.append(dump_rows[-1])

            # ------------------- TEMPLATES -------------------
            # T1
            ind_row = pick_indices(row_headers, 1, 1)
            ind_col = pick_indices(col_headers, 1, 1)
            row_clause = clause_for_selection(row_headers, ind_row)
            col_clause = clause_for_selection(col_headers, ind_col)
            sql = f"SELECT Value FROM {table} WHERE {col_clause} AND {row_clause}"
            res = run_sql(conn, sql)
            tpl = 1
            nlqs = []
            if generate_nlq and f"template_{tpl}" in nlq_block:
                for patt in nlq_block[f"template_{tpl}"]:
                    nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], row_headers.iloc[ind_row]))
            _record(tpl, nlqs, sql, res, ind_row, ind_col)

            # T2
            ind_row = pick_indices(row_headers, 2, min(10, len(row_headers)))
            ind_col = pick_indices(col_headers, 1, 1)
            row_clause = clause_for_selection(row_headers, ind_row)
            col_clause = clause_for_selection(col_headers, ind_col)
            sql = f"SELECT Value FROM {table} WHERE {col_clause} AND {row_clause}"
            res = run_sql(conn, sql)
            tpl = 2
            nlqs = []
            if generate_nlq and f"template_{tpl}" in nlq_block:
                for patt in nlq_block[f"template_{tpl}"]:
                    nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], row_headers.iloc[ind_row]))
            _record(tpl, nlqs, sql, res, ind_row, ind_col)

            # T3
            ind_row = pick_indices(row_headers, 1, 1)
            ind_col = pick_indices(col_headers, 2, min(10, len(col_headers)))
            row_clause = clause_for_selection(row_headers, ind_row)
            col_clause = clause_for_selection(col_headers, ind_col)
            sql = f"SELECT Value FROM {table} WHERE {col_clause} AND {row_clause}"
            res = run_sql(conn, sql)
            tpl = 3
            nlqs = []
            if generate_nlq and f"template_{tpl}" in nlq_block:
                for patt in nlq_block[f"template_{tpl}"]:
                    nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], row_headers.iloc[ind_row]))
            _record(tpl, nlqs, sql, res, ind_row, ind_col)

            # T4
            ind_row = pick_indices(row_headers, 1, 1)
            ind_col = pick_indices(col_headers, 2, min(10, len(col_headers)))
            row_clause = clause_for_selection(row_headers, ind_row)
            col_clause = clause_for_selection(col_headers, ind_col)
            exprs_all = ["sum", "min", "max", "avg"]
            exprs = [e for e in exprs_all if e != (agg_fun or "").lower()]
            if len(exprs) >= 2:
                exprs = random.sample(exprs, 2)
            elif not exprs:
                exprs = ["sum"]
            sql = f"SELECT {expr_clause(exprs)} FROM {table} WHERE {col_clause} AND {row_clause}"
            res = run_sql(conn, sql)
            tpl = 4
            nlqs = []
            if generate_nlq and f"template_{tpl}" in nlq_block:
                for patt in nlq_block[f"template_{tpl}"]:
                    nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], row_headers.iloc[ind_row]))
            _record(tpl, nlqs, sql, res, ind_row, ind_col, exprs=exprs, tableAggFun=agg_fun)

            # T6
            ind_row = pick_indices(row_headers, 2, min(4, len(row_headers)))
            ind_col = pick_indices(col_headers, 2, min(5, len(col_headers)))
            row_clause = clause_for_selection(row_headers, ind_row)
            col_clause = clause_for_selection(col_headers, ind_col)
            sql = f"SELECT Value FROM {table} WHERE {col_clause} AND {row_clause}"
            res = run_sql(conn, sql)
            tpl = 6
            nlqs = []
            if generate_nlq and f"template_{tpl}" in nlq_block:
                for patt in nlq_block[f"template_{tpl}"]:
                    nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], row_headers.iloc[ind_row]))
            _record(tpl, nlqs, sql, res, ind_row, ind_col)

            # T7
            ind_row = pick_indices(row_headers, 2, min(5, len(row_headers)))
            ind_col = pick_indices(col_headers, 1, 1)
            row_clause = clause_for_selection(row_headers, ind_row)
            col_clause = clause_for_selection(col_headers, ind_col)
            exprs = random.sample(["sum", "min", "max", "avg"], 2)
            sql = f"SELECT {expr_clause(exprs)} FROM {table} WHERE {col_clause} AND {row_clause}"
            res = run_sql(conn, sql)
            tpl = 7
            nlqs = []
            if generate_nlq and f"template_{tpl}" in nlq_block:
                for patt in nlq_block[f"template_{tpl}"]:
                    nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], row_headers.iloc[ind_row]))
            _record(tpl, nlqs, sql, res, ind_row, ind_col, exprs=exprs, tableAggFun=agg_fun)

            # T8
            ind_row = pick_indices(row_headers, 2, min(5, len(row_headers)))
            ind_col = pick_indices(col_headers, 2, min(5, len(col_headers)))
            row_clause = clause_for_selection(row_headers, ind_row)
            col_clause = clause_for_selection(col_headers, ind_col)
            group_cols = ", ".join(col_headers.columns)
            exprs = random.sample(["sum", "min", "max", "avg"], 2)
            sql = f"SELECT {group_cols}, {expr_clause(exprs)} FROM {table} WHERE {col_clause} AND {row_clause} GROUP BY {group_cols}"
            res = run_sql(conn, sql)
            tpl = 8
            nlqs = []
            if generate_nlq and f"template_{tpl}" in nlq_block:
                for patt in nlq_block[f"template_{tpl}"]:
                    nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], row_headers.iloc[ind_row]))
            _record(tpl, nlqs, sql, res, ind_row, ind_col, exprs=exprs, tableAggFun=agg_fun)

            # T9
            if ROWdepth >= 1:
                lvl1_col = row_headers.columns[0]
                lvl1 = df[[lvl1_col]].drop_duplicates().reset_index(drop=True)
                ind_row = pick_indices(lvl1, max(1, min(2, len(lvl1))), min(2, len(lvl1)))
                ind_col = pick_indices(col_headers, 1, 1)
                row_clause = clause_for_selection(lvl1, ind_row)
                col_clause = clause_for_selection(col_headers, ind_col)
                sql = f"SELECT min(Value) AS min_Value FROM {table} WHERE {col_clause} AND {row_clause} GROUP BY {lvl1_col}"
                res = run_sql(conn, sql)
                tpl = 9
                nlqs = []
                if generate_nlq and f"template_{tpl}" in nlq_block:
                    for patt in nlq_block[f"template_{tpl}"]:
                        nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], lvl1.iloc[ind_row]))
                _record(tpl, nlqs, sql, res, ind_row, ind_col, exprs=["min"])

            # T10
            if ROWdepth >= 1:
                lvl1_col = row_headers.columns[0]
                lvl1 = df[[lvl1_col]].drop_duplicates().reset_index(drop=True)
                ind_row = pick_indices(lvl1, max(1, min(2, len(lvl1))), min(2, len(lvl1)))
                ind_col = pick_indices(col_headers, 1, 1)
                row_clause = clause_for_selection(lvl1, ind_row)
                col_clause = clause_for_selection(col_headers, ind_col)
                expr = random.choice(["sum", "min", "max", "avg"])
                sql = f"SELECT {lvl1_col}, {expr}(Value) AS {expr}_Value FROM {table} WHERE {col_clause} AND {row_clause} GROUP BY {lvl1_col}"
                res = run_sql(conn, sql)
                tpl = 10
                nlqs = []
                if generate_nlq and f"template_{tpl}" in nlq_block:
                    for patt in nlq_block[f"template_{tpl}"]:
                        nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], lvl1.iloc[ind_row]))
                _record(tpl, nlqs, sql, res, ind_row, ind_col, exprs=[expr])

            # T11
            if ROWdepth >= 1:
                lvl1_col = row_headers.columns[0]
                lvl1 = df[[lvl1_col]].drop_duplicates().reset_index(drop=True)
                ind_row = pick_indices(lvl1, max(1, min(2, len(lvl1))), min(2, len(lvl1)))
                ind_col = pick_indices(col_headers, 2, min(3, len(col_headers)))
                row_clause = clause_for_selection(lvl1, ind_row)
                col_clause = clause_for_selection(col_headers, ind_col)
                expr = random.choice(["sum", "min", "max", "avg"])
                group_cols = ", ".join([lvl1_col] + col_headers.columns.tolist())
                sql = f"SELECT {group_cols}, {expr}(Value) AS {expr}_Value FROM {table} WHERE {col_clause} AND {row_clause} GROUP BY {group_cols}"
                res = run_sql(conn, sql)
                tpl = 11
                nlqs = []
                if generate_nlq and f"template_{tpl}" in nlq_block:
                    for patt in nlq_block[f"template_{tpl}"]:
                        nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], lvl1.iloc[ind_row]))
                _record(tpl, nlqs, sql, res, ind_row, ind_col, exprs=[expr])

            # T12
            if ROWdepth >= 1:
                lvl1_col = row_headers.columns[0]
                lvl1 = df[[lvl1_col]].drop_duplicates().reset_index(drop=True)
                ind_row = pick_indices(lvl1, 2, min(2, len(lvl1)))
                ind_col = pick_indices(col_headers, 1, 1)
                row_clause = clause_for_selection(lvl1, ind_row)
                col_clause = clause_for_selection(col_headers, ind_col)
                tmp_sql = f"SELECT Value FROM {table} WHERE {col_clause} AND {row_clause}"
                tmp_res = run_sql(conn, tmp_sql)
                nvals = max(0, len(tmp_res))
                if nvals == 0:
                    k = 0
                else:
                    k = max(2, min(5, nvals - 1))
                order = random.choice(["ASC", "DESC"])
                sql = f"SELECT Value FROM {table} WHERE {col_clause} AND {row_clause} ORDER BY Value {order} LIMIT {k}"
                res = run_sql(conn, sql)
                tpl = 12
                nlqs = []
                if generate_nlq and f"template_{tpl}" in nlq_block:
                    for patt in nlq_block[f"template_{tpl}"]:
                        nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], lvl1.iloc[ind_row]))
                _record(tpl, nlqs, sql, res, ind_row, ind_col, topk=k)

            # T13
            if ROWdepth >= 1:
                lvl1_col = row_headers.columns[0]
                lvl1 = df[[lvl1_col]].drop_duplicates().reset_index(drop=True)
                ind_row = pick_indices(lvl1, 2, min(2, len(lvl1)))
                ind_col = pick_indices(col_headers, 1, 1)
                row_clause = clause_for_selection(lvl1, ind_row)
                col_clause = clause_for_selection(col_headers, ind_col)
                order = random.choice(["ASC", "DESC"])
                sql = f"SELECT Value FROM {table} WHERE {col_clause} AND {row_clause} ORDER BY Value {order}"
                res = run_sql(conn, sql)
                tpl = 13
                nlqs = []
                if generate_nlq and f"template_{tpl}" in nlq_block:
                    for patt in nlq_block[f"template_{tpl}"]:
                        nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], lvl1.iloc[ind_row]))
                _record(tpl, nlqs, sql, res, ind_row, ind_col)

            # T14
            ind_col = pick_indices(col_headers, 1, 1)
            col_clause = clause_for_selection(col_headers, ind_col)
            threshold = float(pd.to_numeric(df["Value"], errors="coerce").mean())
            op = random.choice([">", "<"])
            sel_cols = ", ".join(row_headers.columns) if not row_headers.empty else ""
            sql = f"SELECT {sel_cols} FROM {table} WHERE {col_clause} AND Value {op} {threshold:.2f}"
            res14 = run_sql(conn, sql)
            tpl = 14
            nlqs = []
            if generate_nlq and f"template_{tpl}" in nlq_block:
                for patt in nlq_block[f"template_{tpl}"]:
                    nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col], None))
            _record(tpl, nlqs, sql, res14, None, ind_col)

            # T15
            if len(col_headers) >= 1:
                ind_col_2 = pick_indices(col_headers, 1, 1)
                if set(ind_col_2) == set(ind_col):
                    inds = list(set(range(len(col_headers))) - set(ind_col))
                    if inds:
                        ind_col_2 = [random.choice(inds)]
                col_clause_2 = clause_for_selection(col_headers, ind_col_2)
                attr_list_sql = ", ".join(row_headers.columns) if not row_headers.empty else ""
                inner = sql  # from T14
                sql15 = (
                    f"SELECT {attr_list_sql}, Value FROM {table} "
                    f"WHERE {col_clause_2} AND ({attr_list_sql}) IN ({inner})"
                )
                res15 = run_sql(conn, sql15)
                tpl = 15
                nlqs = []
                if generate_nlq and f"template_{tpl}" in nlq_block:
                    patt = nlq_block.get("template_1", ["What is the value ?"])[0]
                    nlqs.append(nlq_fill(patt, col_headers.iloc[ind_col_2], None))
                _record(tpl, nlqs, sql15, res15, None, ind_col_2)

            conn.close()

            # ----- Write per-table Q&A JSON  ----------------------------------------
            qna_path = SEMANTIC_QANDA_FOLDER / f"{root}_QandA.json"
            with open(qna_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "filename": str(csv_file),
                        "questions": sql_nlq_entries,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[WRITE] Q&A JSON saved -> {qna_path}")

            # ----- Build annotator big JSON chunk  ----------------------------------
            add_temp5 = any(e["name"] == "template_5" for e in sql_nlq_entries)
            template_ids = [1, 2, 3, 4] + ([5] if add_temp5 else []) + [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            task_list = [""] * len(template_ids)

            def first_or_blank(tnum: int) -> str:
                return nlq_firsts.get(tnum, "")

            def ans_or_blank(tnum: int) -> str:
                return ans_firsts.get(tnum, "")

            annot_obj = {
                **annot,
                "template_ids": template_ids,
                "task_list": task_list,
                "query_type_list": json_qna_items,
                "question_list": [first_or_blank(t) for t in template_ids],
                "answer_list": [ans_or_blank(t) for t in template_ids],
            }
            annotator_chunks.append(json.dumps({root: annot_obj}, ensure_ascii=False))

        # --- write per-prefix dump CSV
        if dump_rows:
            dump_df = pd.DataFrame(dump_rows)
            dump_csv_path = SEMANTIC_QANDA_FOLDER / f"DUMP_NLQ_{data_prefix}.csv"
            dump_df.to_csv(dump_csv_path, index=False, quoting=1)
            print(f"[WRITE] Dump CSV saved -> {dump_csv_path}")

    # --- write global dump CSV
    if dump_all_records:
        dump_all_path = SEMANTIC_QANDA_FOLDER / "DUMP_NLQ_ALL.csv"
        pd.DataFrame(dump_all_records).to_csv(dump_all_path, index=False, quoting=1)
        print(f"[WRITE] Global Dump CSV saved -> {dump_all_path}")

    # --- write Stage 3 annotator JSON (big merged object) in cache/S3_OUT
    big: Dict[str, dict] = {}
    for chunk in annotator_chunks:
        obj = json.loads(chunk)
        big.update(obj)
    annot_out = S3_OUT / "table_questions_answers_annotations.json"
    with open(annot_out, "w", encoding="utf-8") as f:
        json.dump(big, f, ensure_ascii=False, indent=2)
    print(f"[WRITE] Annotator JSON saved -> {annot_out}")

    print(f"[OK] Per-table Q&A JSONs folder: {SEMANTIC_QANDA_FOLDER}")
    print(f"[OK] Dumps folder: {SEMANTIC_QANDA_FOLDER}")
    print("[DONE] Stage 3 complete")


if __name__ == "__main__":
    main(generate_nlq=True)
