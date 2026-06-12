import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_CLEAN_DATA, DEFAULT_SPLIT_DATA, TARGET_CANDIDATES
from data_cleaning import first_existing_column


def build_train_test_split(input_path: str, output_path: str = str(DEFAULT_SPLIT_DATA)) -> dict:
    df = pd.read_csv(input_path)
    target_column = first_existing_column(df, TARGET_CANDIDATES)
    if target_column is None:
        raise ValueError("No target column found for feature engineering.")

    X = df.drop(columns=[target_column])
    X = pd.get_dummies(X, drop_first=True)
    y = df[target_column]

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    payload = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": X.columns.tolist(),
        "target_column": target_column,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)

    print("Train/test artifact saved to:", output_path)
    print("X_train shape:", X_train_scaled.shape)
    print("X_test shape:", X_test_scaled.shape)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare scaled train/test datasets.")
    parser.add_argument("--input", default=str(DEFAULT_CLEAN_DATA), help="Cleaned CSV path.")
    parser.add_argument(
        "--output", default=str(DEFAULT_SPLIT_DATA), help="Joblib output path."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_train_test_split(args.input, args.output)
