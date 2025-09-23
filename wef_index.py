"""Utilities for computing the Water-Energy-Food (WEF) index."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS: Iterable[str] = ("water", "energy", "food")


def _validate_columns(columns: Iterable[str]) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(sorted(missing))
        )


def _normalize_series(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum):
        return series
    if minimum == maximum:
        return pd.Series(0.0, index=series.index)
    return (series - minimum) / (maximum - minimum)


def calculate_wef_index(csv_path: str | Path) -> pd.DataFrame:
    """Calculate the Water-Energy-Food (WEF) index for a CSV file.

    Parameters
    ----------
    csv_path:
        Path to the input CSV file containing ``water``, ``energy`` and ``food``
        columns.

    Returns
    -------
    pandas.DataFrame
        The normalized DataFrame with an added ``WEF_index`` column. The
        DataFrame is also saved to ``wef_index_output.csv`` in the current
        working directory.
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    _validate_columns(df.columns)

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().all():
            raise ValueError(f"Column '{column}' contains only missing values.")
        mean_value = df[column].mean()
        df[column] = df[column].fillna(mean_value)
        df[column] = _normalize_series(df[column])

    df["WEF_index"] = df[list(REQUIRED_COLUMNS)].mean(axis=1)
    output_path = Path("wef_index_output.csv")
    df.to_csv(output_path, index=False)
    return df


__all__ = ["calculate_wef_index"]
