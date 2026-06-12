import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from config import DEFAULT_METRICS, DEFAULT_SPLIT_DATA, MODELS


def train_models(input_path: str = str(DEFAULT_SPLIT_DATA), metrics_path: str = str(DEFAULT_METRICS)) -> dict:
    data = joblib.load(input_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    metrics = {}
    MODELS.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions)

        metrics[name] = {
            "accuracy": round(float(accuracy), 4),
            "confusion_matrix": matrix.tolist(),
        }
        joblib.dump(model, MODELS / f"{name}.joblib")

        print(f"\n{name}")
        print("Accuracy:", round(float(accuracy), 4))
        print("Confusion matrix:")
        print(matrix)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nMetrics saved to:", metrics_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train first disease prediction models.")
    parser.add_argument(
        "--input", default=str(DEFAULT_SPLIT_DATA), help="Train/test joblib artifact."
    )
    parser.add_argument(
        "--metrics", default=str(DEFAULT_METRICS), help="Metrics JSON output path."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_models(args.input, args.metrics)
