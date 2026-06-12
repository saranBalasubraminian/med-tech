import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DEFAULT_CLEAN_DATA, FIGURES, TARGET_CANDIDATES
from data_cleaning import first_existing_column


def save_eda_charts(input_path: str, figures_dir: str = str(FIGURES)) -> None:
    df = pd.read_csv(input_path)
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_column = first_existing_column(df, TARGET_CANDIDATES)
    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.4)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

    numeric_columns = [column for column in numeric_df.columns if column != target_column]
    if numeric_columns:
        cols = 3
        rows = math.ceil(len(numeric_columns) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(14, max(4, rows * 3)))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for index, column in enumerate(numeric_columns):
            sns.histplot(df[column], kde=True, ax=axes[index], color="#276FBF")
            axes[index].set_title(column)
        for index in range(len(numeric_columns), len(axes)):
            axes[index].axis("off")
        fig.suptitle("Feature Distributions", y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / "distribution_plots.png", dpi=160, bbox_inches="tight")
        plt.close()

    if target_column:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=target_column, palette="Set2", hue=target_column, legend=False)
        plt.title("Class Balance")
        plt.tight_layout()
        plt.savefig(output_dir / "class_balance.png", dpi=160)
        plt.close()

    print("EDA charts saved to:", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create EDA charts for README.")
    parser.add_argument("--input", default=str(DEFAULT_CLEAN_DATA), help="Cleaned CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_eda_charts(args.input)
