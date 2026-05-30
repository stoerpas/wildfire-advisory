"""
preprocess.py
Load and clean the Algerian Forest Fires dataset.
Produces a single clean DataFrame saved to data/processed/.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    ALGERIAN_RAW, ALGERIAN_CLEAN, FEATURE_COLS, TARGET_COL, DATA_PROCESSED
)


# ── Helpers ───────────────────────────────────────────────

def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and string values."""
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df


def _parse_region_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw CSV contains two region header rows mixed into the data
    (rows with 'Classes' == 'Classes' or containing region labels).
    Remove them and add a 'Region' column derived from their position.
    """
    # Rows that are actually header / region label rows
    mask_header = df.iloc[:, 0].astype(str).str.contains(
        r"day|Bejaia|Sidi", case=False, na=False
    )
    df = df[~mask_header].copy()
    return df


def load_algerian(path: Path = ALGERIAN_RAW) -> pd.DataFrame:
    """
    Read the raw Algerian Forest Fires CSV.

    The file has a peculiar structure:
      Line 0  : 'Bejaia Region Dataset'   (region label, no commas)
      Line 1  : column header row
      Lines 2-123 : Bejaia data (122 rows)
      Line 124 : blank
      Line 125 : 'Sidi-Bel Abbes Region Dataset'
      Line 126 : column header row (duplicate)
      Lines 127-248 : Sidi-Bel-Abbes data (122 rows)

    We read each section separately, tag with Region, and concatenate.
    """
    col_names = [
        "day", "month", "year", "Temperature", "RH", "Ws", "Rain",
        "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "Classes"
    ]

    # Read all lines, skip region-label and header rows manually
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [l.rstrip("\r\n") for l in f.readlines()]

    data_rows = []
    current_region = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "Bejaia" in stripped and "day" not in stripped:
            current_region = "Bejaia"
            continue
        if "Sidi" in stripped and "day" not in stripped:
            current_region = "Sidi-Bel-Abbes"
            continue
        if stripped.startswith("day"):  # header row
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) == len(col_names):
            data_rows.append(parts + [current_region])

    all_cols = col_names + ["Region"]
    df = pd.DataFrame(data_rows, columns=all_cols)

    # Standardise the Classes label
    df[TARGET_COL] = (
        df[TARGET_COL]
        .str.strip()
        .str.lower()
        .map({"fire": 1, "not fire": 0})
    )

    # Cast numeric columns
    numeric_cols = FEATURE_COLS + ["day", "month", "year"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any remaining rows with NaN in features or target
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df.reset_index(drop=True)


def save_clean(df: pd.DataFrame, path: Path = ALGERIAN_CLEAN) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned dataset → {path}  ({len(df)} rows)")


def load_clean(path: Path = ALGERIAN_CLEAN) -> pd.DataFrame:
    return pd.read_csv(path)


# ── CLI entry point ───────────────────────────────────────
if __name__ == "__main__":
    df = load_algerian()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Class balance:\n{df[TARGET_COL].value_counts()}")
    save_clean(df)