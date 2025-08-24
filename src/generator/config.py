# src/generator/config.py
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
PATHS = {"input": DATA/"input", "cache": DATA/"cache", "output": DATA/"output"}

# PARAM files (in configs/)
PARAM_SEMANTICS_JSON        = CONFIGS / "PARAM_semantics.json"
PARAM_TABLE_TEMPLATES_JSON  = CONFIGS / "PARAM_tableTemplates.json"
PARAM_NLQ_TEMPLATES_JSON    = CONFIGS / "PARAM_NLquestionTemplates.json"
PARAM_TABLE_TO_GEN_JSON     = CONFIGS / "PARAM_tablesToGenerate.json"


# Stage output folders (cache + output) -> Intermediate Outputs for each stage
S1_OUT = PATHS["cache"] / "s1_tables"
S2_OUT = PATHS["cache"] / "s2_hct"
S3_OUT = PATHS["cache"] / "s3_sql_nlq"
S4_OUT = PATHS["cache"] / "s4_qna"
S5_OUT = PATHS["cache"] / "s5_clean"
S6_OUT = PATHS["cache"] / "s6_counts"

BENCHMARK_FOLDER = PATHS["output"] / "benchmark"
REPORTS_FOLDER   = PATHS["output"] / "reports"


# Semantic + Non-semantic folders (like in R config)
SEMANTIC_TABLES_FOLDER      = PATHS["output"] / "tables"
SEMANTIC_QANDA_FOLDER       = PATHS["output"] / "qanda"
NON_SEMANTIC_TABLES_FOLDER  = PATHS["output"] / "tables_nonsemantic"
NON_SEMANTIC_QANDA_FOLDER   = PATHS["output"] / "qanda_nonsemantic"

# ensure output folders exist
for folder in [
    SEMANTIC_TABLES_FOLDER,
    SEMANTIC_QANDA_FOLDER,
    NON_SEMANTIC_TABLES_FOLDER,
    NON_SEMANTIC_QANDA_FOLDER,
]:
    folder.mkdir(parents=True, exist_ok=True)


# Output format toggles
HTML_OK = True   # generate HTML
PDF_OK  = False  # costly, disabled by default
PNG_OK  = False  # needs PDF_OK first

# Formatting / naming
NAME_SEP = "_"
COL_SEP, ROW_SEP = " | ", " || "
ATTR_DELIMITER = ["##", "@@"]
NUM_DECIMAL_DIGITS_REAL_FORMAT = 2
STR_REAL_VAL_FORMAT = f"%.{NUM_DECIMAL_DIGITS_REAL_FORMAT}f"
STR_INT_VAL_FORMAT = "%d"

def get_sql_attr_names(attr_names, name_sep=NAME_SEP):
    sub = lambda s: re.sub(r"[.\s\-/]", name_sep, s)
    if isinstance(attr_names, str):
        return sub(attr_names)
    if isinstance(attr_names, (list, tuple)):
        return [sub(x) for x in attr_names]
    raise TypeError("attr_names must be str or list/tuple")

def print_cond(cond, msg):
    if cond:
        print(msg)


getSQLattrNames = get_sql_attr_names
