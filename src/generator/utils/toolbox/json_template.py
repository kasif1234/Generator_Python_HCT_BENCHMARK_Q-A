# toolbox_json_template.py
# -----------------------------------------------------------------------------
# Python port of:
#   - geneJSONformatQandA()
#   - aggregateNLQ()
# from the R script "toolboxjsontemplate"
#
# Notes:
# - Field names and values are kept to match the R output (including spaces and
#   slashes in keys, and numbers-as-strings where the R code used as.character()).
# - Booleans are real JSON booleans (true/false) rather than strings.
# - ROW_SEP defaults to ';' and COL_SEP defaults to ',' to match examples.
# - HCTrowHeaders / HCTcolHeaders can be pandas DataFrames, NumPy arrays,
#   list-of-lists, or similar; only number of columns is used.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence
import json


__all__ = [
    "gene_json_format_q_and_a",
    "aggregate_nlq",
]


def _ncols(obj: Any) -> int:
    """
    Robustly infer number of columns for common Python containers used as 'headers'.
    Returns 0 if unknown/empty.
    """
    if obj is None:
        return 0

    # Objects with .shape (e.g., pandas DataFrame / NumPy array)
    shape = getattr(obj, "shape", None)
    if shape and len(shape) >= 2:
        try:
            return int(shape[1])
        except Exception:
            pass

    # pandas DataFrame has .columns
    columns = getattr(obj, "columns", None)
    if columns is not None:
        try:
            return len(columns)
        except Exception:
            pass

    # list-like
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        if len(obj) == 0:
            return 0
        first = obj[0]
        if isinstance(first, dict):
            return len(first)
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray)):
            return len(first)
        # list of scalars -> treat as a single column
        return 1

    # dict -> number of keys
    if isinstance(obj, dict):
        return len(obj)

    return 0


def _parse_sql_result(sql_result: str, row_sep: str = ";", col_sep: str = ","):
    """
    Roughly match the R logic:
      allRowRes = strsplit(sqlResult, ROW_SEP)[[1]]
      nrowRes   = length(allRowRes)
      ncolRes   = length(strsplit(allRowRes[1], COL_SEP)[[1]])
    """
    if sql_result is None:
        return 0, 0

    rows = [r.strip() for r in str(sql_result).split(row_sep)]
    rows = [r for r in rows if r != ""] or [""]  # ensure at least one element
    nrow = len(rows)
    first_row_cols = [c.strip() for c in rows[0].split(col_sep)]
    ncol = len(first_row_cols) if rows[0] != "" else 1
    return nrow, ncol


def _agg_flags(expr_list_sql: Optional[Iterable[str]], table_agg_fun: str) -> Dict[str, bool]:
    exprs = set([e.strip().lower() for e in (expr_list_sql or [])])
    table_agg_fun = (table_agg_fun or "").strip().lower()

    agg_min = "min" in exprs
    agg_max = "max" in exprs
    agg_count = "count" in exprs
    agg_sum = "sum" in exprs
    agg_avg = "avg" in exprs

    agg_table_present = table_agg_fun in exprs
    agg_table_not_present = not agg_table_present

    return dict(
        agg_min=agg_min,
        agg_max=agg_max,
        agg_count=agg_count,
        agg_sum=agg_sum,
        agg_avg=agg_avg,
        agg_table_present=agg_table_present,
        agg_table_not_present=agg_table_not_present,
        num_aggs=len(exprs),
    )


def gene_json_format_q_and_a(
    num_template_qa: int,
    HCTrowHeaders: Any,
    HCTcolHeaders: Any,
    indRowSel: Sequence[Any],
    indColSel: Sequence[Any],
    sqlResult: Any,
    exprListSQL: Optional[Iterable[str]] = None,
    tableAggFun: str = "",
    firstK: Any = "",
    row_sep: str = ";",
    col_sep: str = ",",
) -> Dict[str, Any]:
    """
    Python port of geneJSONformatQandA().

    Returns a single dict (a “row” in the original R data.frame),
    ready to be added to a list and JSON-encoded by aggregate_nlq().
    """
    # Header depths / counts
    ncol_row_headers = _ncols(HCTrowHeaders)
    ncol_col_headers = _ncols(HCTcolHeaders)

    # SQL result sizes
    nrow_res, ncol_res = _parse_sql_result(str(sqlResult), row_sep=row_sep, col_sep=col_sep)

    # Helpers used in multiple templates
    def base_fields() -> Dict[str, Any]:
        return {
            # Row Filter
            "Row Filter": True,
            "Row Filter Condition Type Lookup": True,
            "Row Filter Condition Type Expression": False,
            "Row Filter Involved Columns Single": True,
            "Row Filter Involved Columns Multiple": False,
            "Row Filter Max Depth Of Involved Columns": "1",
            "Row Filter Retained Rows Single": True,   # will override where needed
            "Row Filter Retained Rows Multiple": False,  # will override where needed
            "Row Filter Num Of Conditions": str(ncol_row_headers),
            # Returned Columns
            "Returned Columns": True,
            "Returned Columns Project On Plain": True,
            "Returned Columns Project On Expression": False,
            "Returned Columns Max Depth Of Involved Columns": str(ncol_col_headers),
            "Returned Columns Expression In Table Present": False,
            "Returned Columns Expression In Table Not Present": False,
            "Returned Columns Num Of Output Columns": "1",
            # Yes/No (never used)
            "Yes/No": False,
            "Yes/No Scope Single": False,
            "Yes/No Scope Multiple": False,
            # Aggregation (defaults; will override where needed)
            "Aggregation": False,
            "Aggregation Aggregation On Plain": False,
            "Aggregation Aggregation On Expression": False,
            "Aggregation Aggregation Type Min": False,
            "Aggregation Aggregation Type Max": False,
            "Aggregation Aggregation Type Count": False,
            "Aggregation Aggregation Type Sum": False,
            "Aggregation Aggregation Type Avg": False,
            "Aggregation Grouping Local": False,
            "Aggregation Grouping Global": False,
            "Aggregation Aggregation In Table Present": False,
            "Aggregation Aggregation In Table Not Present": False,
            "Aggregation Num Of Aggregations": "",
            # Rank
            "Rank": False,
            "Rank Rank On Plain": False,
            "Rank Rank On Expression": False,
            "Rank Report Top": "",
        }

    # Convenience flags
    multi_row = len(indRowSel) != 1
    single_row = not multi_row

    if num_template_qa == 1:
        f = base_fields()
        f["Row Filter Retained Rows Single"] = True
        f["Row Filter Retained Rows Multiple"] = False
        f["Returned Columns Num Of Output Columns"] = "1"
        return f

    elif num_template_qa == 2:
        f = base_fields()
        f["Row Filter Retained Rows Single"] = single_row
        f["Row Filter Retained Rows Multiple"] = not single_row
        f["Returned Columns Num Of Output Columns"] = "1"
        return f

    elif num_template_qa == 3:
        f = base_fields()
        f["Row Filter Retained Rows Single"] = True
        f["Row Filter Retained Rows Multiple"] = False
        f["Returned Columns Num Of Output Columns"] = str(len(indColSel))
        return f

    elif num_template_qa == 4:
        f = base_fields()
        agg = _agg_flags(exprListSQL, tableAggFun)
        f.update({
            "Row Filter Retained Rows Single": True,
            "Row Filter Retained Rows Multiple": False,
            "Returned Columns Num Of Output Columns": str(len(indColSel)),
            "Aggregation": True,
            "Aggregation Aggregation On Plain": True,
            "Aggregation Aggregation On Expression": False,
            "Aggregation Aggregation Type Min": agg["agg_min"],
            "Aggregation Aggregation Type Max": agg["agg_max"],
            "Aggregation Aggregation Type Count": agg["agg_count"],
            "Aggregation Aggregation Type Sum": agg["agg_sum"],
            "Aggregation Aggregation Type Avg": agg["agg_avg"],
            "Aggregation Grouping Local": False,
            "Aggregation Grouping Global": True,
            "Aggregation Aggregation In Table Present": agg["agg_table_present"],
            "Aggregation Aggregation In Table Not Present": agg["agg_table_not_present"],
            "Aggregation Num Of Aggregations": str(agg["num_aggs"]),
        })
        return f

    elif num_template_qa == 5:
        f = base_fields()
        agg = _agg_flags(exprListSQL, tableAggFun)
        f.update({
            "Row Filter Retained Rows Single": True,
            "Row Filter Retained Rows Multiple": False,
            "Returned Columns Num Of Output Columns": str(len(indColSel)),
            "Aggregation": True,
            "Aggregation Aggregation On Plain": True,
            "Aggregation Aggregation On Expression": False,
            "Aggregation Aggregation Type Min": agg["agg_min"],
            "Aggregation Aggregation Type Max": agg["agg_max"],
            "Aggregation Aggregation Type Count": agg["agg_count"],
            "Aggregation Aggregation Type Sum": agg["agg_sum"],
            "Aggregation Aggregation Type Avg": agg["agg_avg"],
            "Aggregation Grouping Local": False,
            "Aggregation Grouping Global": True,
            "Aggregation Aggregation In Table Present": agg["agg_table_present"],
            "Aggregation Aggregation In Table Not Present": agg["agg_table_not_present"],
            "Aggregation Num Of Aggregations": "1",
        })
        return f

    elif num_template_qa == 6:
        f = base_fields()
        f["Row Filter Retained Rows Single"] = single_row
        f["Row Filter Retained Rows Multiple"] = not single_row
        f["Returned Columns Num Of Output Columns"] = str(len(indColSel))
        return f

    elif num_template_qa == 7:
        f = base_fields()
        agg = _agg_flags(exprListSQL, tableAggFun)
        f.update({
            "Row Filter Retained Rows Single": single_row,
            "Row Filter Retained Rows Multiple": not single_row,
            "Aggregation": True,
            "Aggregation Aggregation On Plain": True,
            "Aggregation Aggregation On Expression": False,
            "Aggregation Aggregation Type Min": agg["agg_min"],
            "Aggregation Aggregation Type Max": agg["agg_max"],
            "Aggregation Aggregation Type Count": agg["agg_count"],
            "Aggregation Aggregation Type Sum": agg["agg_sum"],
            "Aggregation Aggregation Type Avg": agg["agg_avg"],
            "Aggregation Grouping Local": False,
            "Aggregation Grouping Global": True,
            "Aggregation Aggregation In Table Present": agg["agg_table_present"],
            "Aggregation Aggregation In Table Not Present": agg["agg_table_not_present"],
            "Aggregation Num Of Aggregations": str(agg["num_aggs"]),
        })
        f["Returned Columns Num Of Output Columns"] = "1"
        return f

    elif num_template_qa == 8:
        f = base_fields()
        agg = _agg_flags(exprListSQL, tableAggFun)
        f.update({
            "Row Filter Retained Rows Single": single_row,
            "Row Filter Retained Rows Multiple": not single_row,
            "Aggregation": True,
            "Aggregation Aggregation On Plain": True,
            "Aggregation Aggregation On Expression": False,
            "Aggregation Aggregation Type Min": agg["agg_min"],
            "Aggregation Aggregation Type Max": agg["agg_max"],
            "Aggregation Aggregation Type Count": agg["agg_count"],
            "Aggregation Aggregation Type Sum": agg["agg_sum"],
            "Aggregation Aggregation Type Avg": agg["agg_avg"],
            "Aggregation Grouping Local": False,
            "Aggregation Grouping Global": True,
            "Aggregation Aggregation In Table Present": agg["agg_table_present"],
            "Aggregation Aggregation In Table Not Present": agg["agg_table_not_present"],
            "Aggregation Num Of Aggregations": str(agg["num_aggs"]),
        })
        f["Returned Columns Num Of Output Columns"] = str(len(indColSel))
        return f

    elif num_template_qa == 9:
        f = base_fields()
        agg = _agg_flags(exprListSQL, tableAggFun)
        f.update({
            "Row Filter Retained Rows Single": single_row,
            "Row Filter Retained Rows Multiple": not single_row,
            "Row Filter Num Of Conditions": "1",  # depth 1 grouping condition
            "Aggregation": True,
            "Aggregation Aggregation On Plain": True,
            "Aggregation Aggregation On Expression": False,
            "Aggregation Aggregation Type Min": agg["agg_min"],
            "Aggregation Aggregation Type Max": agg["agg_max"],
            "Aggregation Aggregation Type Count": agg["agg_count"],
            "Aggregation Aggregation Type Sum": agg["agg_sum"],
            "Aggregation Aggregation Type Avg": agg["agg_avg"],
            "Aggregation Grouping Local": True,
            "Aggregation Grouping Global": False,
            "Aggregation Aggregation In Table Present": agg["agg_table_present"],
            "Aggregation Aggregation In Table Not Present": agg["agg_table_not_present"],
            "Aggregation Num Of Aggregations": str(agg["num_aggs"]),
        })
        f["Returned Columns Num Of Output Columns"] = "1"
        return f

    elif num_template_qa == 10:
        f = base_fields()
        agg = _agg_flags(exprListSQL, tableAggFun)
        f.update({
            "Row Filter Retained Rows Single": single_row,
            "Row Filter Retained Rows Multiple": not single_row,
            "Row Filter Num Of Conditions": "1",  # depth 1 grouping condition
            "Aggregation": True,
            "Aggregation Aggregation On Plain": True,
            "Aggregation Aggregation On Expression": False,
            "Aggregation Aggregation Type Min": agg["agg_min"],
            "Aggregation Aggregation Type Max": agg["agg_max"],
            "Aggregation Aggregation Type Count": agg["agg_count"],
            "Aggregation Aggregation Type Sum": agg["agg_sum"],
            "Aggregation Aggregation Type Avg": agg["agg_avg"],
            "Aggregation Grouping Local": True,
            "Aggregation Grouping Global": False,
            "Aggregation Aggregation In Table Present": agg["agg_table_present"],
            "Aggregation Aggregation In Table Not Present": agg["agg_table_not_present"],
            "Aggregation Num Of Aggregations": str(agg["num_aggs"]),
        })
        f["Returned Columns Num Of Output Columns"] = str(len(indColSel))
        return f

    elif num_template_qa == 11:
        f = base_fields()
        agg = _agg_flags(exprListSQL, tableAggFun)
        f.update({
            "Row Filter Retained Rows Single": single_row,
            "Row Filter Retained Rows Multiple": not single_row,
            "Row Filter Num Of Conditions": "1",  # depth 1 grouping condition
            "Aggregation": True,
            "Aggregation Aggregation On Plain": True,
            "Aggregation Aggregation On Expression": False,
            "Aggregation Aggregation Type Min": agg["agg_min"],
            "Aggregation Aggregation Type Max": agg["agg_max"],
            "Aggregation Aggregation Type Count": agg["agg_count"],
            "Aggregation Aggregation Type Sum": agg["agg_sum"],
            "Aggregation Aggregation Type Avg": agg["agg_avg"],
            "Aggregation Grouping Local": True,
            "Aggregation Grouping Global": False,
            "Aggregation Aggregation In Table Present": agg["agg_table_present"],
            "Aggregation Aggregation In Table Not Present": agg["agg_table_not_present"],
            "Aggregation Num Of Aggregations": str(agg["num_aggs"]),
        })
        f["Returned Columns Num Of Output Columns"] = str(len(indColSel))
        return f

    elif num_template_qa == 12:
        f = base_fields()
        f.update({
            "Row Filter Retained Rows Single": single_row,
            "Row Filter Retained Rows Multiple": not single_row,
            "Aggregation": False,
            "Rank": True,
            "Rank Rank On Plain": True,
            "Rank Rank On Expression": False,
            "Rank Report Top": str(firstK),
        })
        f["Returned Columns Num Of Output Columns"] = "1"
        return f

    elif num_template_qa == 13:
        f = base_fields()
        f.update({
            "Row Filter Retained Rows Single": single_row,
            "Row Filter Retained Rows Multiple": not single_row,
            "Aggregation": False,
            "Rank": True,
            "Rank Rank On Plain": True,
            "Rank Rank On Expression": False,
            # "Rank Report Top" stays empty (ORDER BY only)
        })
        f["Returned Columns Num Of Output Columns"] = "1"
        return f

    elif num_template_qa == 14:
        f = base_fields()
        # Expression-based row filter, per R:
        f["Row Filter Condition Type Lookup"] = False
        f["Row Filter Condition Type Expression"] = True
        f["Row Filter Max Depth Of Involved Columns"] = str(ncol_row_headers)
        # nrowRes determines single vs multiple
        RRS = (nrow_res == 1)
        f["Row Filter Retained Rows Single"] = RRS
        f["Row Filter Retained Rows Multiple"] = not RRS
        num_cond = ncol_col_headers
        f["Row Filter Num Of Conditions"] = str(num_cond)
        f["Aggregation"] = False
        f["Aggregation Aggregation On Plain"] = False
        f["Returned Columns Num Of Output Columns"] = str(len(indColSel))
        return f

    elif num_template_qa == 15:
        f = base_fields()
        # Conditional (reuse template 14 style)
        f["Row Filter Condition Type Lookup"] = False
        f["Row Filter Condition Type Expression"] = True
        f["Row Filter Max Depth Of Involved Columns"] = str(ncol_row_headers)
        f["Row Filter Retained Rows Single"] = single_row
        f["Row Filter Retained Rows Multiple"] = not single_row
        num_cond = ncol_col_headers
        f["Row Filter Num Of Conditions"] = str(num_cond)
        f["Aggregation"] = False
        f["Aggregation Aggregation On Plain"] = False
        f["Returned Columns Num Of Output Columns"] = str(len(indColSel))
        return f

    else:
        raise ValueError(f"Unsupported num_template_qa: {num_template_qa}")


def aggregate_nlq(query_type_df_list: List[Dict[str, Any]]) -> str:
    """
    Python port of aggregateNLQ(): combine a list of per-template dicts and
    return a JSON string. In R they did some post-processing of the JSON
    (e.g., replacing 'TRUE'/'FALSE' strings). Here booleans are already proper.
    """
    # Just return a compact JSON array of objects, as in the R code.
    return json.dumps(query_type_df_list, ensure_ascii=False)
