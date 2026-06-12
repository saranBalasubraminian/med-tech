import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BP_CANDIDATES,
    DEFAULT_CLEAN_DATA,
    DEFAULT_RAW_DATA,
    GLUCOSE_CANDIDATES,
    TARGET_CANDIDATES,
)


def first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    columns_by_lower = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        match = columns_by_lower.get(candidate.lower())
        if match:
            return match
    return None


def clean_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df.columns = [column.strip() for column in df.columns]

    target_column = first_existing_column(df, TARGET_CANDIDATES)
    if target_column is None:
        raise ValueError(
            "No target column found. Expected one of: "
            + ", ".join(TARGET_CANDIDATES)
        )
    df = df.dropna(subset=[target_column]).copy()

    anomaly_columns = [
        column
        for column in (
            first_existing_column(df, GLUCOSE_CANDIDATES),
            first_existing_column(df, BP_CANDIDATES),
        )
        if column is not None
    ]
    for column in anomaly_columns:
        df[column] = df[column].replace(0, np.nan)

    for column in df.columns:
        if column == target_column:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            mode = df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[column] = df[column].fillna(fill_value)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Cleaned dataset saved to:", output_path)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values after cleaning:")
    print(df.isna().sum())
    if anomaly_columns:
        print("\nZero-value anomaly columns fixed:", ", ".join(anomaly_columns))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean disease prediction CSV data.")
    parser.add_argument("--input", default=str(DEFAULT_RAW_DATA), help="Raw CSV path.")
    parser.add_argument(
        "--output", default=str(DEFAULT_CLEAN_DATA), help="Cleaned CSV output path."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_dataset(args.input, args.output)
