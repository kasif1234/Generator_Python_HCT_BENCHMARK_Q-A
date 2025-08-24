# src/generator/utils/toolbox/tables.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

from generator.config import (
    NUM_DECIMAL_DIGITS_REAL_FORMAT as DEFAULT_DIGITS,
    get_sql_attr_names as _get_sql_attr_names,
)

# Back-compat alias expected by Stage 2
getSQLattrNames = _get_sql_attr_names  # noqa: N816

# ---------------------------
# STEP 1: names/values by code
# ---------------------------
def get_names_values(semantic_data: List[Dict[str, Any]], code: str) -> Dict[str, Any]:
    for entry in semantic_data:
        if entry.get("code") == code:
            return {"names": entry.get("names", []), "values": entry.get("values", [])}
    return {"names": [], "values": []}

# ---------------------------
# STEP 2: numeric/value sampler
# ---------------------------
def _make_distinct(vals: List[float], num_digits: int = DEFAULT_DIGITS) -> List[float]:
    eps = 10 ** -num_digits
    vals = list(vals)
    while len(set(vals)) != len(vals):
        vals.sort()
        for i in range(len(vals) - 1):
            if vals[i] == vals[i + 1]:
                vals[i + 1] += eps
    np.random.shuffle(vals)
    return vals

def get_sample_values(semantic_values: Dict[str, List[float]],
                      value_code: Any,
                      num_sample: int,
                      num_digits: int = DEFAULT_DIGITS) -> List[float]:
    """
    R's getSampleValues:
      - value_code can be a key into semantic_values (e.g., "int", "realUnit"),
        or a direct [lo, hi] pair.
      - Returns distinct numbers (ints or reals) of length num_sample.
    """
    if isinstance(value_code, str) and value_code in semantic_values:
        lo, hi = semantic_values[value_code]
    elif isinstance(value_code, (list, tuple)) and len(value_code) == 2:
        lo, hi = value_code
    else:
        raise ValueError(f"Invalid value_code: {value_code}")

    if float(lo).is_integer() and float(hi).is_integer():
        lo_i, hi_i = int(lo), int(hi)
        needed_max = max(hi_i, lo_i + num_sample - 1)
        pool = list(range(lo_i, needed_max + 1))
        vals = np.random.choice(pool, size=num_sample, replace=False).tolist()
        return _make_distinct(vals, 0)
    else:
        vals = np.round(np.random.uniform(lo, hi, size=num_sample), num_digits).tolist()
        return _make_distinct(vals, num_digits)

# ---------------------------
# STEP 3: keep/remove filters
# ---------------------------
def get_values_from_names(names_and_values: Dict[str, Any],
                          val_keep: List[str] | None,
                          val_remove: List[str] | None) -> List[str]:
    names = names_and_values.get("names", [])
    vals = names_and_values.get("values", [])
    flat: List[str] = []

    for i, v in enumerate(vals):
        nm = names[i] if i < len(names) else f"name{i}"
        if isinstance(v, list):
            for sub in v:
                flat.append(f"{nm}.{sub}")
        else:
            flat.append(f"{nm}.{v}")

    if val_keep:
        return [x for x in flat if any(k in x for k in val_keep)]
    if val_remove:
        return [x for x in flat if all(r not in x for r in val_remove)]
    return flat

# ---------------------------
# STEP 4: block sampler (lists)
# ---------------------------
def sample_values(val_list: List[str], val_sample: List[int] | Tuple[int, int] | None):
    """
    Pick a consecutive block from val_list.
    val_sample like [n, m] → choose size in [n, m]; anything else → return all.
    Returns {'listAttrNames': [...], 'isAggPossible': 'true'|'false'}.
    """
    if not (isinstance(val_sample, (list, tuple)) and len(val_sample) == 2):
        return {"listAttrNames": list(val_list), "isAggPossible": "true"}

    total = len(val_list)
    n, m = val_sample
    n = max(1, min(int(n), total))
    m = max(n, min(int(m), total))
    size = n if n == m else int(np.random.randint(n, m + 1))
    start = int(np.random.randint(0, total - size + 1))
    sel = val_list[start:start + size]
    return {"listAttrNames": sel, "isAggPossible": "false" if len(sel) == 1 else "true"}

# -----------------------------------------
# STEP 5: hierarchical sampler (A.B[.C...])
# -----------------------------------------
def sample_values_from_hierarchy(list_composite: List[str], val_sample: List[int] | Tuple[int, int]):
    split_rows = [row.split(".") for row in list_composite]
    if not split_rows:
        return {"listAttrNames": [], "isAggPossible": "true"}
    depth = len(split_rows[0])

    def uniq(seq):
        seen = set(); out = []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    # level 0
    tops = uniq(r[0] for r in split_rows)
    r0 = sample_values(tops, val_sample)
    agg_flags = [r0["isAggPossible"]]
    matrix = [r for r in split_rows if r[0] in r0["listAttrNames"]]

    # deeper levels
    for lvl in range(1, depth):
        next_mat: List[List[str]] = []
        lvl_flag = "true"
        # parents are unique prefixes up to lvl-1
        parents = uniq(".".join(r[:lvl]) for r in matrix)
        for p in parents:
            group = [r for r in matrix if ".".join(r[:lvl]) == p]
            vals = uniq(r[lvl] for r in group)
            r = sample_values(vals, val_sample)
            if r["isAggPossible"] == "false":
                lvl_flag = "false"
            allowed = set(r["listAttrNames"])
            next_mat.extend([row for row in group if row[lvl] in allowed])
        agg_flags.append(lvl_flag)
        matrix = next_mat

    collapsed = [".".join(row) for row in matrix]
    return {"listAttrNames": collapsed, "isAggPossible": ".".join(agg_flags)}

# -------------------------------------------
# STEP 6: first attr index matching any code(s)
# -------------------------------------------
def get_attr_from_codes(attr_list: List[Dict[str, Any]], codes: str | List[str] | Tuple[str, ...]) -> int | None:
    if isinstance(codes, str):
        codes = [codes]
    for i, attr in enumerate(attr_list):
        if attr.get("code") in codes:
            return i
    return None

# -------------------------------------------
# STEP 7: codes by attribute display name
# -------------------------------------------
def get_codes_from_name(semantic_obj: Dict[str, Any], attr_name: str) -> List[str]:
    codes: List[str] = []
    for entry in semantic_obj.get("data", []):
        if attr_name in entry.get("names", []):
            codes.append(entry.get("code"))
    return codes

# R-style back-compat aliases used by Stage 2
getNamesValues = get_names_values
getValuesFromNames = get_values_from_names
sampleValues = sample_values
sampleValuesFromHierarchy = sample_values_from_hierarchy
getAttrFromCodes = get_attr_from_codes
getCodesFromName = get_codes_from_name
getSampleValues = get_sample_values

__all__ = [
    "DEFAULT_DIGITS",
    "get_names_values",
    "get_values_from_names",
    "get_sample_values",
    "sample_values",
    "sample_values_from_hierarchy",
    "get_attr_from_codes",
    "get_codes_from_name",
    # back-compat exports
    "getNamesValues",
    "getValuesFromNames",
    "sampleValues",
    "sampleValuesFromHierarchy",
    "getAttrFromCodes",
    "getCodesFromName",
    "getSampleValues",
    "getSQLattrNames",
]
