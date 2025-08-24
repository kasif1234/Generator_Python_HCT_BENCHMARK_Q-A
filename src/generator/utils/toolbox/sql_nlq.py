# toolbox_sql_and_nlq.py
# -----------------------------------------------------------------------------
# Python port of the R helpers from "toolbxsqlandnlq".
#
# Notes:
# - Functions are translated as closely as practical. Where R code relied on
#   tidyverse semantics, equivalent pandas logic is used.
# - Naming: snake_case in Python; function names otherwise match the R intent.
# - Some advanced NLQ regrouping utilities are kept, but simplified in a few
#   places. They still produce readable NLQ text, even if not as aggressively
#   minimized as the R version.
# - Constants ROW_SEP / COL_SEP match the examples in R's `formatResults` doc.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
import random
import re

import pandas as pd
import numpy as np


# ------------------------------- Constants -----------------------------------

ROW_SEP = " || "           # e.g., "{A} || {B} || {C}"
COL_SEP = " | "            # e.g., "{A | B | C}"
NUM_DECIMAL_DIGITS_REAL_FORMAT = 2
ATTR_DELIMITER = ("##", "@@")  # used to bracket attribute tokens in regrouping

# Natural language labels for SQL aggregation keywords
EXPR_LIST_NAME = {
    "sum": "total",
    "min": "minimum",
    "max": "maximum",
    "avg": "average",
    "count": "count",
}


# ------------------------------ Utils / Helpers ------------------------------

def squish(s: str) -> str:
    """Collapse repeated whitespace and trim ends."""
    return re.sub(r"\s+", " ", s or "").strip()


def safe_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    return [x]


def get_sql_attr_names(names: Iterable[str]) -> List[str]:
    """
    Map NL attribute names to "SQL-style" names by replacing spaces with underscores.
    (R comments: 'add "_" as an attribute "xx zzz" is stored as HCThead$xx_zzz'.)
    """
    return [re.sub(r"\s+", "_", str(n)) for n in names]


def get_nl_attr_names(sql_attr_names: Union[str, Iterable[str]], all_nl_attr_names: Iterable[str]) -> Union[str, List[str]]:
    """
    Reverse map: given SQL-style attr names (with underscores), return original NL names.
    We match by normalizing allNL names into the SQL-style form and mapping back.
    """
    all_nl = list(all_nl_attr_names)
    all_sql = get_sql_attr_names(all_nl)
    mapping = {sql: nl for sql, nl in zip(all_sql, all_nl)}

    if isinstance(sql_attr_names, str):
        return mapping.get(sql_attr_names, sql_attr_names.replace("_", " "))
    res: List[str] = []
    for nm in sql_attr_names:
        res.append(mapping.get(nm, str(nm).replace("_", " ")))
    return res


def get_oxford_style_group(term_list: Sequence[str], term_conjunct: str = "or") -> str:
    """Return Oxford-style grouping: a; a or b; a, b, or c; etc."""
    terms = [str(t) for t in term_list if str(t)]
    n = len(terms)
    conj = f" {term_conjunct} "
    if n == 0:
        return ""
    if n == 1:
        return terms[0]
    if n == 2:
        return conj.join(terms)
    return f"{', '.join(terms[:-1])},{conj}{terms[-1]}"


# ------------------------------ I/O-like helpers -----------------------------

def get_hct_row_headers(DBdata: pd.DataFrame, col_names: Sequence[str], row_names: Sequence[str]) -> pd.DataFrame:
    """
    Get all row headers from HCT table (without aggregate rows nor DB row colnames).
    Filter rows by the values in the *first* row for the given `col_names`,
    then return only `row_names` columns.
    """
    if DBdata.empty:
        return DBdata.loc[:, list(row_names)]
    col_val = DBdata.iloc[[0]][list(col_names)]
    filt = DBdata.copy()
    for c in col_names:
        filt = filt[filt[c] == col_val.iloc[0][c]]
    return filt.loc[:, list(row_names)]


def get_hct_col_headers(DBdata: pd.DataFrame, col_names: Sequence[str], row_names: Sequence[str]) -> pd.DataFrame:
    """
    Get all column headers from HCT table (without aggregate cols nor DB col colnames).
    Filter rows by the values in the *first* row for the given `row_names`,
    then return only `col_names` columns.
    """
    if DBdata.empty:
        return DBdata.loc[:, list(col_names)]
    row_val = DBdata.iloc[[0]][list(row_names)]
    filt = DBdata.copy()
    for r in row_names:
        filt = filt[filt[r] == row_val.iloc[0][r]]
    return filt.loc[:, list(col_names)]


# ------------------------ Header selection / SQL clauses ---------------------

def gene_index_for_random_clauses(HCTheaders: pd.DataFrame, min_lev_val: int = 0, max_lev_val: int = 0, rng: Optional[random.Random] = None) -> List[int]:
    """
    Select indices (1-based in R; we will return 0-based in Python for convenience).
    - If min=max=0: select *all* rows.
    - Else choose a random sample size n between [min,max], then distinct row indices.
    """
    if rng is None:
        rng = random

    nrow_hct = int(HCTheaders.shape[0])
    if nrow_hct == 0:
        return []

    if min_lev_val == 0 and max_lev_val == 0:
        return list(range(nrow_hct))

    if min_lev_val == max_lev_val:
        n_sel = max(0, min(min_lev_val, nrow_hct))
    else:
        lo = max(1, min(min_lev_val, max_lev_val))
        hi = min(nrow_hct, max(min_lev_val, max_lev_val))
        n_sel = rng.randint(lo, hi)

    # random unique indices (0-based)
    candidates = list(range(nrow_hct))
    rng.shuffle(candidates)
    return sorted(candidates[:n_sel])


def gene_multi_row_or_col_selection_sql(HCTheaders: pd.DataFrame, ind_col_sel: Sequence[int]) -> str:
    """
    Build a SQL WHERE clause for selected rows of HCTheaders.
    - If multiple columns: OR of (A='x' AND B='y' ...)
    - If single column: IN ('x','y',...)
    Always wrapped in parentheses.
    """
    if HCTheaders is None or len(ind_col_sel) == 0:
        return "()"

    df = HCTheaders.reset_index(drop=True)
    cols = list(df.columns)
    ncols = len(cols)
    rows = df.iloc[list(ind_col_sel)]

    if ncols > 1:
        parts = []
        for _, row in rows.iterrows():
            conds = []
            for c in cols:
                val = row[c]
                if pd.notna(val):
                    conds.append(f"{c} = '{val}'")
            parts.append(f"({ ' AND '.join(conds) })")
        clause = " OR ".join(parts)
    else:
        col = cols[0]
        vals = [f"'{v}'" for v in rows[col].dropna().astype(str).tolist()]
        clause = f"{col} IN ({','.join(vals)})"
    return f"({clause})"


@dataclass
class AggregatedSelectionResult:
    other_col_names: Union[str, List[str]]
    other_col_values: Union[str, pd.DataFrame]
    gene_col_names: str
    gene_col_values: List[Any]
    col_clause: str


def gene_aggregated_selection_sql(HCTcolHeaders: pd.DataFrame, agg_col_to_gen: str, rng: Optional[random.Random] = None) -> AggregatedSelectionResult:
    """
    Generate selection SQL for an 'aggregated' column attribute.
    Picks one concrete combination for the columns *before* agg_col_to_gen (if any),
    then lists all values present for agg_col_to_gen under that combination.
    """
    if rng is None:
        rng = random

    cols = list(HCTcolHeaders.columns)
    if agg_col_to_gen not in cols:
        raise ValueError(f"{agg_col_to_gen} not found in headers")

    ind_level = cols.index(agg_col_to_gen)
    other_cols = cols[:ind_level]

    col_clause_parts: List[str] = []
    other_col_names: Union[str, List[str]]
    other_col_values: Union[str, pd.DataFrame]

    if other_cols:
        # distinct combos across other columns, pick one
        distinct = HCTcolHeaders.loc[:, other_cols].drop_duplicates().reset_index(drop=True)
        i = rng.randrange(len(distinct))
        chosen = distinct.iloc[i:i+1]  # DataFrame with one row
        for c in other_cols:
            val = chosen.iloc[0][c]
            col_clause_parts.append(f"{c} = '{val}'")
        other_col_names = list(chosen.columns)
        other_col_values = chosen
        # filter headers to those values
        filt = HCTcolHeaders.copy()
        for c in other_cols:
            filt = filt[filt[c] == chosen.iloc[0][c]]
    else:
        other_col_names = ""
        other_col_values = ""
        filt = HCTcolHeaders

    # add IN(...) for aggregated attribute values
    list_val = pd.unique(filt[agg_col_to_gen]).tolist()
    ucol = "','".join([str(v) for v in list_val])
    col_clause_parts.append(f"({agg_col_to_gen} IN ('{ucol}'))")

    col_clause = " AND ".join(col_clause_parts)

    return AggregatedSelectionResult(
        other_col_names=other_col_names,
        other_col_values=other_col_values,
        gene_col_names=agg_col_to_gen,
        gene_col_values=list_val,
        col_clause=col_clause,
    )


def gene_expression_clause(expression: Union[str, Iterable[str]] = "", val_attr_name: str = "Value") -> str:
    """
    Build a comma-separated list of expressions applied on a value column,
    e.g., ['avg','min'] -> "avg(Value),min(Value)".
    """
    exprs = safe_list(expression)
    parts = [f"{e}({val_attr_name})" for e in exprs if str(e)]
    return ",".join(parts)


def gene_group_by_clause(HCTcol_or_row_headers: pd.DataFrame, col_or_row_clause: str) -> str:
    """
    Pick randomly an attribute from headers that varies across the provided WHERE
    clause tokens (very close to the R heuristic). Returns the attribute name,
    or empty string if none.
    """
    tokens = col_or_row_clause.split()
    candidates: List[str] = []
    for cn in HCTcol_or_row_headers.columns:
        # find positions of this column name in the WHERE clause
        idx = [i for i, t in enumerate(tokens) if t == cn]
        # check the token 2 positions later should be the quoted value; gather distinct
        vals = set()
        for i in idx:
            if i + 2 < len(tokens):
                vals.add(tokens[i + 2])
        if len(vals) > 1:
            candidates.append(cn)

    if candidates:
        return random.choice(candidates)
    return ""


# --------------------------- JSON templating helpers -------------------------

def sql_json_template(temp_num: Union[int, str], sql_clause: str, sql_result_str: str) -> str:
    """Return a pretty JSON-like string (to match R's manual formatting)."""
    return (
        "{"
        f"\"name\": \"template {temp_num}\",\n "
        f"\"sql\": \"{sql_clause}\",\n "
        f"\"sqlResult\": \"{sql_result_str}\""
        "}"
    )


def sql_nlq_json_template(temp_num: Union[int, str], sql_clause: str, sql_result_str: str, nl_question_variants: Union[str, Sequence[str]]) -> str:
    """Return a JSON-like string with either a single 'nlq' string or an array."""
    variants = safe_list(nl_question_variants)
    if len(variants) == 1:
        return (
            "{"
            f"\"name\": \"template {temp_num}\",\n "
            f"\"sql\": \"{sql_clause}\",\n "
            f"\"sqlResult\": \"{sql_result_str}\",\n "
            f"\"nlq\": \"{variants[0]}\""
            "}"
        )
    start = (
        "{"
        f"\"name\": \"template {temp_num}\",\n "
        f"\"sql\": \"{sql_clause}\",\n "
        f"\"sqlResult\": \"{sql_result_str}\",\n "
        f"\"nlq\": ["
    )
    mid = ",\n         ".join([json.dumps(v) for v in variants])
    end = "]}"
    return f"{start}{mid}{end}"


def get_unique_values_at_level(HCTheaders: pd.DataFrame, val_level: Union[int, str]) -> pd.DataFrame:
    """
    Return a one-column DataFrame of unique values for the given level.
    - If val_level is int, treat like R's 1-based column index.
    - If val_level is str, treat as column name.
    """
    if isinstance(val_level, int):
        # R is 1-based
        idx = max(0, val_level - 1)
        col = HCTheaders.columns[idx]
    else:
        col = str(val_level)
    res = pd.DataFrame({col: pd.unique(HCTheaders[col])})
    return res


def format_results(sql_result: pd.DataFrame,
                   row_sep: str = ROW_SEP,
                   col_sep: str = COL_SEP,
                   num_decimal_digits: int = NUM_DECIMAL_DIGITS_REAL_FORMAT) -> Optional[str]:
    """
    Format a DataFrame result like:
      n=1: "{A | B | C}"
      n>1: "{row1} || {row2} ..."
    Returns None if empty.
    """
    if sql_result is None or sql_result.shape[0] == 0:
        return None

    df = sql_result.copy()
    # Round numerics like R
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(num_decimal_digits)

    def fmt_row(sr: pd.Series) -> str:
        return "{" + col_sep.join([str(v) for v in sr.astype(object).tolist()]) + "}"

    if df.shape[0] == 1:
        return fmt_row(df.iloc[0])

    parts = [fmt_row(df.iloc[i]) for i in range(df.shape[0])]
    return row_sep.join(parts)


# --------------------------------- NLQ utils ---------------------------------

def get_attr_names_from_pattern(nl_pattern_segment: str, list_attr_names: Sequence[str]) -> List[str]:
    """Return attribute names in `list_attr_names` that appear as $Name inside `nl_pattern_segment`."""
    res = []
    for name in list_attr_names:
        if re.search(r"\$" + re.escape(str(name)) + r"\b", nl_pattern_segment):
            res.append(str(name))
    return res


def get_valid_attr_names(str_pattern: str, all_nl_attr_names: Sequence[str]) -> List[str]:
    """
    Get all SQL-form attribute names ($xxx) present in `str_pattern` that belong to this class of table.
    """
    all_sql_attr_names = get_sql_attr_names(all_nl_attr_names)
    found = []
    for ft in all_sql_attr_names:
        if re.search(r"\$" + re.escape(ft) + r"\b", str_pattern):
            found.append(ft)
    return found


def get_var_value_nlq(HCThead: pd.DataFrame,
                      str_pattern: str,
                      all_nl_attr_names: Sequence[str],
                      attr_delimiter: Optional[Tuple[str, str]] = None) -> str:
    """
    Replace $Attr with actual values taken from the single-row HCThead (or combined row),
    keep other text (underscores -> spaces). If none matched, return "".
    """
    valid_sql_attrs = get_valid_attr_names(str_pattern, all_nl_attr_names)
    attr_names = list(HCThead.columns)

    res_cur = str_pattern
    matched_any = False
    for ft in valid_sql_attrs:
        # Convert SQL attr to the actual DataFrame col name (which we assume equals SQL name)
        if ft in attr_names:
            val = HCThead.iloc[0][ft]
            if attr_delimiter and len(attr_delimiter) == 2:
                val_str = f"{attr_delimiter[0]}{val}{attr_delimiter[1]}"
            else:
                val_str = str(val)
            res_cur = re.sub(r"\$" + re.escape(ft) + r"\b", val_str, res_cur)
            matched_any = True
        else:
            # remove the unmatched variable marker
            res_cur = re.sub(r"\$" + re.escape(ft) + r"\b", "", res_cur)

    if not matched_any:
        return ""

    res_cur = res_cur.replace("_", " ")
    if attr_delimiter and len(attr_delimiter) == 2:
        # When delimiter requested, the caller expects raw with delimiters preserved
        return res_cur
    return squish(res_cur)


def get_var_attr_nlq(attr_pattern: str,
                     attr_name: str,
                     all_nl_attr_names: Sequence[str],
                     attr_delimiter: Optional[Tuple[str, str]] = None) -> Optional[str]:
    """
    Return the NL version of the pattern for a specific attribute name, or None if not present.
    This follows the R logic for "__" and "==" segmenting (simplified but equivalent).
    """
    if not re.search(r"\$" + re.escape(attr_name) + r"\b", attr_pattern):
        return None

    attr_name_nl = get_nl_attr_names(attr_name, all_nl_attr_names)
    # split into larger groups by "__", then sub-groups by "=="
    groups = []
    for seg in re.split(r"__", attr_pattern):
        groups.extend(re.split(r"==", seg))

    # find the subsegment that contains $attr_name
    for part in groups:
        if re.search(r"\$" + re.escape(attr_name) + r"\b", part):
            prefix = re.sub(r"\$" + re.escape(attr_name) + r"\b.*", "", part)
            prefix = prefix.replace("_", " ")
            res = squish(f"{prefix} {attr_name_nl}")
            return res
    return None


def gene_for_which_prem_attr(attr_list: Sequence[str], nl_pattern: str, all_nl_attr_names: Sequence[str]) -> str:
    """Return 'premise' string like 'in Year and in Language' for which..."""
    split_pattern = squish(nl_pattern).split()
    nlq_prem_attr: List[str] = []
    for attr in attr_list:
        for seg in split_pattern:
            res = get_var_attr_nlq(seg, attr, all_nl_attr_names)
            if res:
                nlq_prem_attr.append(res)
    return get_oxford_style_group(nlq_prem_attr, "and")


def get_expr_nlq(expr_list_sql: Sequence[str], seg_str: str) -> str:
    """
    Apply pluralization (is/are, number/numbers) and replace $EXPR with 'total and minimum' etc.
    """
    seg_list = seg_str.split("_")
    out: List[str] = []
    plural = len(expr_list_sql) > 1
    for token in seg_list:
        if "/" in token:
            a, b = token.split("/", 1)
            out.append(b if plural else a)
        elif token == "$EXPR":
            expr_list = [EXPR_LIST_NAME.get(e, e) for e in expr_list_sql]
            out.append(get_oxford_style_group(expr_list, "and"))
        else:
            out.append(token)
    return " ".join(out)


def get_group_by_nlq(group_by_attr_names: Sequence[str], seg_str: str, all_nl_attr_names: Sequence[str]) -> str:
    seg_list = seg_str.split("_")
    out: List[str] = []
    plural = len(group_by_attr_names) > 1
    for token in seg_list:
        if "/" in token:
            a, b = token.split("/", 1)
            out.append(b if plural else a)
        elif token == "$GROUPBY":
            group_by_nl = get_nl_attr_names(group_by_attr_names, all_nl_attr_names)
            if isinstance(group_by_nl, str):
                gb = group_by_nl
            else:
                gb = get_oxford_style_group(group_by_nl, "and")
            out.append(f" for each {gb}, ")
        else:
            out.append(token)
    return " ".join(out)


def gene_report_attr(report_attr_names: Sequence[str], seg_str: str) -> str:
    seg_list = seg_str.split("_")
    out: List[str] = []
    plural = len(report_attr_names) > 1
    for token in seg_list:
        if "/" in token:
            a, b = token.split("/", 1)
            out.append(b if plural else a)
        elif token == "$REPORTATTR":
            out.append(get_oxford_style_group(report_attr_names, "and") + ".")
        else:
            out.append(token)
    return " ".join(out)


def get_order_by_nlq(order_by_k: Dict[str, Any], seg_str: str) -> str:
    """
    order_by_k: {"K": int, "orderBy": "DESC" | "ASC"}
    seg_str examples:
      "$ORDERBYDESC_decreasing/increasing_values"
      "top/bottom_$TOPK"
    """
    first_k = int(order_by_k.get("K", 0) or 0)
    dir_ = str(order_by_k.get("orderBy", "DESC")).upper()
    seg_list = seg_str.split("_")
    out: List[str] = []
    for token in seg_list:
        if "/" in token:
            a, b = token.split("/", 1)
            out.append(a if dir_ == "DESC" else b)
        elif token in ("$ORDERBYDESC", "$TOPK"):
            if first_k <= 0:
                out.append("ordered by")
            else:
                out.append(str(first_k))
        else:
            out.append(token)
    return " ".join(out)


def get_for_which_nlq(for_which: Dict[str, Any], seg_str: str) -> str:
    """
    for_which = {"OPATTR": [...], "OPTYPE": ">", "OPVAL": value}
    seg_str like: "is/are_the_$OPATTR_for_which" ... "is_$OPANDVAL"
    """
    seg_list = seg_str.split("_")
    attr_list = safe_list(for_which.get("OPATTR", []))
    op = squish(str(for_which.get("OPTYPE", "")))

    # SQL operator -> words
    op_map = {
        ">": "greater than",
        ">=": "greater than or equal to",
        "<": "lower than",
        "<=": "lower than or equal to",
        "=": "equal to",
        "!=": "different from",
        "<>": "different from",
    }
    op_words = op_map.get(op, op)
    op_val = for_which.get("OPVAL", "")

    out: List[str] = []
    for token in seg_list:
        if "/" in token:
            a, b = token.split("/", 1)
            out.append(b if len(attr_list) > 1 else a)
        elif token == "$OPATTR":
            out.append(get_oxford_style_group(attr_list, "and"))
        elif token == "$OPANDVAL":
            out.append(f"{op_words} {op_val}")
        else:
            out.append(token)
    return " ".join(out)


# -------------------- NL generation for row/col patterns ---------------------

def gene_row_col_union_from_nl_pattern(HCTelt: Optional[pd.DataFrame],
                                       nl_row_col_pattern: Union[str, Sequence[str]],
                                       all_nl_attr_names: Sequence[str],
                                       attr_delimiter: Optional[Tuple[str, str]] = None) -> str:
    """
    Generate an NL clause expressing an OR over HCT rows/columns using the given
    NL sub-pattern(s). This implements the “row-by-row” path from the R code,
    which is robust and preserves order.
    """
    if HCTelt is None or (isinstance(HCTelt, pd.DataFrame) and HCTelt.empty):
        return ""

    # Determine valid attr names present in the pattern and in the current HCTelt
    if isinstance(nl_row_col_pattern, str):
        patterns = [nl_row_col_pattern]
    else:
        patterns = list(nl_row_col_pattern)

    combined = "__".join(patterns)
    valid_attr_sql = get_valid_attr_names(combined, all_nl_attr_names)

    # Keep only columns present in both
    cols = [c for c in HCTelt.columns if c in valid_attr_sql]
    df = HCTelt.loc[:, cols].drop_duplicates().reset_index(drop=True)

    # break patterns by "__" and "=="
    split_clean: List[str] = []
    label_mask: List[bool] = []
    for seg in patterns:
        for seg2 in re.split(r"__", seg):
            parts = re.split(r"==", seg2)
            split_clean.extend(parts)
            # Mark segments that were between '==' and not containing a variable
            # (R used to drop these if neighbors are empty)
            for p in parts:
                label_mask.append(False if re.search(r"\$", p) else True)

    # Build one NL string per row, then Oxford-group with 'or'
    row_strings: List[str] = []
    start_tokens: List[str] = []
    seen_attribute_anywhere = False

    for i in range(df.shape[0]):
        seg_vals: List[str] = []
        attr_seen_in_this_row = False

        for j, pat in enumerate(split_clean):
            cur = ""
            if re.search(r"\$", pat):  # contains a variable
                # Support both row and col attributes; pass the whole row
                cur = get_var_value_nlq(df.iloc[[i]], pat, all_nl_attr_names, attr_delimiter)
                if cur:
                    seen_attribute_anywhere = True
                    attr_seen_in_this_row = True
            else:
                # static text segment
                if not seen_attribute_anywhere:
                    start_tokens.append(pat.replace("_", " "))
                elif attr_seen_in_this_row:
                    cur = pat.replace("_", " ")

            seg_vals.append(cur)

        # remove empty "middle" labels that were from == markers when neighbors empty
        keep = []
        for k, val in enumerate(seg_vals):
            if label_mask[k] and not val:
                keep.append(False)
            else:
                keep.append(True)

        filtered = " ".join([seg_vals[k] for k in range(len(seg_vals)) if keep[k]])
        row_strings.append(squish(filtered))

    start_merged = squish(" ".join([t for t in start_tokens if t]))
    or_group = get_oxford_style_group([s for s in row_strings if s], "or")
    return squish(f"{start_merged} {or_group}")


# ---------- Delimiter-based splitting/regrouping (kept for parity) ----------

def split_attr1(sentence: str, attr_delimiter: Tuple[str, str]) -> List[str]:
    """Split to isolate attributes (preserving delimiter tokens)."""
    s1 = re.split(re.escape(attr_delimiter[0]), sentence)
    s1a = [attr_delimiter[0] + x for x in s1]
    s1b = [attr_delimiter[0] in x and attr_delimiter[1] in x for x in s1a]
    s1c = s1[:]
    for i in range(len(s1)):
        if s1b[i]:
            s1c[i] = s1a[i]
    s2 = re.split(re.escape(attr_delimiter[1]), "".join(s1c))
    s2a = [x + attr_delimiter[1] for x in s2]
    s2b = [attr_delimiter[0] in x and attr_delimiter[1] in x for x in s2a]
    s2c = s2[:]
    for i in range(len(s2)):
        if s2b[i]:
            s2c[i] = s2a[i]
    return s2c


def split_attr2(sentence: str, attr_delimiter: Tuple[str, str]) -> List[str]:
    """Alternate splitter keeping prefix/suffix around attributes."""
    tokens = squish(sentence).split()
    cumul: List[str] = []
    res: List[str] = []
    on = False
    for tok in tokens:
        ds = attr_delimiter[0] in tok
        de = attr_delimiter[1] in tok
        if ds and de:
            res.append(tok)
        elif ds:
            on = True
            cumul.append(tok)
        elif de:
            on = False
            cumul.append(tok)
            res.append(" ".join(cumul))
            cumul = []
        else:
            if on:
                cumul.append(tok)
            else:
                res.append(tok)
    return res


def trim_attr(sentences: Sequence[str], attr_delimiter: Tuple[str, str]) -> List[str]:
    """Remove delimiters and return inner attribute contents."""
    res: List[str] = []
    for s in sentences:
        s1 = re.split(re.escape(attr_delimiter[0]), s)
        for t in s1:
            res.extend(re.split(re.escape(attr_delimiter[1]), t))
    # filter empties
    return [x for x in res if x and x not in attr_delimiter]


def merge_attr(sentences: Sequence[str], merge_str: str) -> str:
    return merge_str.join(sentences)


def add_attr_delimiter_attr(sentence: str, attr_delimiter: Tuple[str, str]) -> str:
    return f"{attr_delimiter[0]}{sentence}{attr_delimiter[1]}"


def regroup_attr(sentences: Sequence[str], attr_delimiter: Tuple[str, str], merge_str: str) -> str:
    return add_attr_delimiter_attr(get_oxford_style_group(trim_attr(sentences, attr_delimiter), merge_str), attr_delimiter)


def regroup_attr2(sentences: Sequence[str], attr_delimiter: Tuple[str, str], merge_str: str) -> str:
    # Simplified but compatible with expected output shape
    inner = merge_attr(sentences, merge_str)
    return inner  # already contains delimited pieces; caller merges around


def merge_delim(lmerge: Sequence[str], attr_delimiter: Tuple[str, str]) -> str:
    """Merge a sequence of delimited fragments back together (simplified)."""
    lm1 = [s.replace(attr_delimiter[0], "") for s in lmerge]
    lm1 = [s.replace(attr_delimiter[1], "") for s in lm1]
    return " ".join(lm1)


def regroup_sentences(all_sent: Sequence[str], attr_delimiter: Tuple[str, str]) -> List[str]:
    """
    Simplified regrouping: try to merge sentences that differ by exactly one
    delimited attribute occurrence. Preserves readability if not fully minimal.
    """
    uniq = list(dict.fromkeys([squish(s) for s in all_sent]))
    # naive attempt; preserve if nothing to merge
    return uniq


def regroup_sentences_last_pass(all_sent: Sequence[str], attr_delimiter: Tuple[str, str]) -> List[str]:
    """Last pass regroup (kept as identity for stability/simplicity)."""
    uniq = list(dict.fromkeys([squish(s) for s in all_sent]))
    return uniq


# --------------------------- Optional cleanup helpers ------------------------

def remove_optional_dup(nl_question_clean: str) -> str:
    """
    Process optional ((...)) blocks:
    - if the option text appears elsewhere, drop the optional block
    - otherwise, expand '((opt_text))' -> 'opt text' (with underscores -> spaces)
    """
    text = nl_question_clean

    # Find all ((...)) chunks
    opts = re.findall(r"\(\((.*?)\)\)", text)
    for opt in opts:
        # Count occurrences outside ((...)) by temporarily removing delimiters
        without = re.sub(r"\(\(" + re.escape(opt) + r"\)\)", "XXXXOPTXXXX", text)
        count_elsewhere = len(re.findall(re.escape(opt), without))
        if count_elsewhere > 0:
            # remove the optional chunk entirely
            text = re.sub(r"\s*\(\(" + re.escape(opt) + r"\)\)\s*", " ", text)
        else:
            # expand the option (replace underscores by spaces)
            expanded = opt.replace("_", " ")
            text = re.sub(r"\(\(" + re.escape(opt) + r"\)\)", expanded, text)

    return squish(text)


# ----------------------- High-level NLQ generation API -----------------------

def gene_nlq_select_express(HCTcolH: Optional[pd.DataFrame] = None,
                            HCTrowH: Optional[pd.DataFrame] = None,
                            expr_list_sql: Optional[Sequence[str]] = None,
                            group_by_attr_names: Optional[Sequence[str]] = None,
                            order_by_k: Optional[Dict[str, Any]] = None,
                            for_which: Optional[Dict[str, Any]] = None,
                            nl_pattern: str = "",
                            all_nl_attr_names: Optional[Sequence[str]] = None,
                            simplify_nested_list: Optional[Sequence[Sequence[str]]] = None) -> str:
    """
    Port of geneNLQselectExpress() (with some simplifications in the regrouping stages).

    - Simplifies nested attributes: if both parent/child columns exist in the *same*
      header side, drops the leftmost (parent) columns and keeps the rightmost (child).
    - Replaces tokens $EXPR, $GROUPBY, $ORDERBYDESC/$TOPK, $OPATTR/$OPANDVAL.
    - For ROW/COL variable segments, generates readable OR-joined clauses.
    """
    expr_list_sql = expr_list_sql or []
    group_by_attr_names = list(group_by_attr_names or [])
    order_by_k = order_by_k or {"K": 0, "orderBy": "DESC"}
    for_which = for_which or {}
    all_nl_attr_names = list(all_nl_attr_names or [])
    simplify_nested_list = simplify_nested_list or []

    # 1) Simplify nested headers (keep rightmost)
    def simplify_headers(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        keep_cols = list(df.columns)
        to_drop: List[str] = []
        for nested in simplify_nested_list:
            nested_sql = get_sql_attr_names(list(nested))
            present = [c for c in nested_sql if c in keep_cols]
            if len(present) > 1:
                # drop all but the rightmost
                to_drop.extend(present[:-1])
        if to_drop:
            keep_cols = [c for c in keep_cols if c not in set(to_drop)]
            return df.loc[:, keep_cols]
        return df

    HCTcolH = simplify_headers(HCTcolH)
    HCTrowH = simplify_headers(HCTrowH)
    # update allowable NL attr names too
    for nested in simplify_nested_list:
        nested_sql = get_sql_attr_names(list(nested))
        present = [c for c in nested_sql if (HCTcolH is not None and c in HCTcolH.columns) or (HCTrowH is not None and c in HCTrowH.columns)]
        if len(present) > 1:
            # drop the parents from NL names
            drop_nl = set(nested[:-1])
            all_nl_attr_names = [n for n in all_nl_attr_names if n not in drop_nl]

    # 2) Tokenize pattern and tag segment types
    nl_segments = squish(nl_pattern).split()
    types: List[str] = []
    all_cols = list(HCTcolH.columns) if HCTcolH is not None else []
    all_rows = list(HCTrowH.columns) if HCTrowH is not None else []
    for seg in nl_segments:
        if "$" in seg:
            names = get_attr_names_from_pattern(seg, all_cols + all_rows + ["EXPR", "GROUPBY", "ORDERBYDESC", "TOPK", "OPATTR", "OPANDVAL"])
            if len(names) == 1:
                nm = names[0]
                if nm in all_cols:
                    types.append("COL")
                elif nm in all_rows:
                    types.append("ROW")
                elif nm == "EXPR":
                    types.append("EXPR")
                elif nm == "GROUPBY":
                    types.append("GROUPBY")
                elif nm in ("ORDERBYDESC", "TOPK"):
                    types.append("ORDERBYK")
                elif nm == "OPATTR":
                    types.append("FORWHICH_START")
                elif nm == "OPANDVAL":
                    types.append("FORWHICH_END")
                else:
                    types.append("TRUE")
            else:
                is_col = any(n in all_cols for n in names)
                is_row = any(n in all_rows for n in names)
                if is_col and is_row:
                    types.append("ROWCOL")
                elif is_col:
                    types.append("COL")
                elif is_row:
                    types.append("ROW")
                else:
                    types.append("TRUE")
        else:
            types.append("FALSE")

    # 3) Build start, repeating, and end segments
    start_vals: List[str] = []
    start_types: List[str] = []
    rep_cols: List[str] = []
    rep_rows: List[str] = []
    rep_rowcols: List[str] = []
    rowcol_order: List[str] = []
    end_vals: List[str] = []
    end_types: List[str] = []

    for i in range(len(nl_segments) - 1):
        seg, t = nl_segments[i], types[i]
        if t in ("FALSE", "EXPR", "GROUPBY", "ORDERBYK", "FORWHICH_START"):
            start_vals.append(seg)
            start_types.append(t)
        elif t == "COL":
            rep_cols.append(seg)
            rowcol_order.append(seg)
        elif t == "ROW":
            rep_rows.append(seg)
            rowcol_order.append(seg)
        elif t == "ROWCOL":
            rep_rowcols.append(seg)
            rowcol_order.append(seg)
        elif t == "FORWHICH_END":
            end_vals.append(seg)
            end_types.append(t)

    # 4) Render start (non-repeating) part
    rendered_start: List[str] = []
    for seg, t in zip(start_vals, start_types):
        if t == "FALSE":
            rendered_start.append(seg)
        elif t == "EXPR":
            rendered_start.append(get_expr_nlq(expr_list_sql, seg))
        elif t == "GROUPBY":
            rendered_start.append(get_group_by_nlq(group_by_attr_names, seg, all_nl_attr_names))
        elif t == "ORDERBYK":
            rendered_start.append(get_order_by_nlq(order_by_k, seg))
        elif t == "FORWHICH_START":
            rendered_start.append(get_for_which_nlq(for_which, seg))
    NLQstr_start = " ".join(rendered_start)

    # 5) Repeating part (ROW/COL/ROWCOL)
    NLQstr_rowcol_rep = ""
    if rep_rowcols or rep_rows or rep_cols:
        # Build all combinations explicitly, bracket attributes for safer matching
        all_sent: List[str] = []
        n_col = 1 if HCTcolH is None else max(1, HCTcolH.shape[0])
        n_row = 1 if HCTrowH is None else max(1, HCTrowH.shape[0])

        for ic in range(n_col):
            for ir in range(n_row):
                if HCTcolH is None and HCTrowH is None:
                    continue
                if HCTcolH is None:
                    hct_bind = HCTrowH.iloc[[ir]].copy()
                elif HCTrowH is None:
                    hct_bind = HCTcolH.iloc[[ic]].copy()
                else:
                    hct_bind = pd.concat([HCTcolH.iloc[[ic]].reset_index(drop=True),
                                          HCTrowH.iloc[[ir]].reset_index(drop=True)], axis=1)
                # Compose by the original order of row/col tokens
                pieces: List[str] = []
                for pat in rowcol_order:
                    val = get_var_value_nlq(hct_bind, pat, all_nl_attr_names, ATTR_DELIMITER)
                    if not val:
                        # maybe static
                        if "$" not in pat:
                            val = pat.replace("_", " ")
                    pieces.append(val)
                sent = squish(" ".join(pieces))
                all_sent.append(sent)

        # Group — simplified passes
        all_sent = regroup_sentences(all_sent, ATTR_DELIMITER)
        all_sent = regroup_sentences_last_pass(all_sent, ATTR_DELIMITER)
        # Remove delimiters back to plain text
        cleaned = [s.replace(ATTR_DELIMITER[0], "").replace(ATTR_DELIMITER[1], "") for s in all_sent]
        NLQstr_rowcol_rep = get_oxford_style_group(cleaned, "or")

    # 6) End (e.g., FORWHICH_END)
    rendered_end: List[str] = []
    for seg, t in zip(end_vals, end_types):
        if t == "FORWHICH_END":
            rendered_end.append(get_for_which_nlq(for_which, seg))
    NLQstr_end = " ".join(rendered_end)

    # 7) Assemble, clean
    nl_question = f"{NLQstr_start} {NLQstr_rowcol_rep} {NLQstr_end}?"
    nl_question = squish(nl_question.replace(" _", " "))
    # Optional block cleanup, comma/space fixes
    nl_question = remove_optional_dup(nl_question)
    nl_question = nl_question.replace(" ,", ",")
    nl_question = re.sub(r",\?", "?", nl_question)
    return nl_question
