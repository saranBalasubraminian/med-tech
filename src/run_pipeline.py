import argparse

from config import DEFAULT_CLEAN_DATA, DEFAULT_RAW_DATA, DEFAULT_SPLIT_DATA
from data_cleaning import clean_dataset
from eda_visualizations import save_eda_charts
from feature_engineering import build_train_test_split
from train_models import train_models


def run_pipeline(input_path: str) -> None:
    clean_dataset(input_path, str(DEFAULT_CLEAN_DATA))
    save_eda_charts(str(DEFAULT_CLEAN_DATA))
    build_train_test_split(str(DEFAULT_CLEAN_DATA), str(DEFAULT_SPLIT_DATA))
    train_models(str(DEFAULT_SPLIT_DATA))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full disease prediction pipeline.")
    parser.add_argument("--input", default=str(DEFAULT_RAW_DATA), help="Raw CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input)
