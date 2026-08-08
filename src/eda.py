import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED_DATA_PATH, FIGURES_DIR, TARGET


def save_plot(filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def run_eda():
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError("Processed dataset not found. Run: python src/data_preparation.py")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().mean().sort_values(ascending=False).head(15))

    print("\nTarget distribution:")
    print(df[TARGET].value_counts(normalize=True))

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Target Distribution: Loan Default")
    save_plot("target_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="loan_amnt", hue=TARGET, bins=40, kde=False)
    plt.title("Loan Amount Distribution by Default Status")
    save_plot("loan_amount_by_default.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET, y="int_rate")
    plt.title("Interest Rate by Default Status")
    save_plot("interest_rate_by_default.png")

    plt.figure(figsize=(10, 5))
    order = df["grade"].dropna().sort_values().unique()
    sns.countplot(data=df, x="grade", hue=TARGET, order=order)
    plt.title("Loan Grade vs Default Status")
    save_plot("grade_vs_default.png")

    numeric_df = df.select_dtypes(include=["number"])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    save_plot("correlation_heatmap.png")


if __name__ == "__main__":
    run_eda()
