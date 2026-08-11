import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    FEATURES,
    FICO_SOURCE_COLUMNS,
    TARGET,
    GOOD_STATUS,
    BAD_STATUS,
    SAMPLE_N,
    RANDOM_STATE,
)


def clean_percentage(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.replace("%", "").strip()
    return pd.to_numeric(value, errors="coerce")


def clean_emp_length(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value == "10+ years":
        return 10
    if value == "< 1 year":
        return 0
    if value == "1 year":
        return 1
    return pd.to_numeric(value.replace(" years", "").replace(" year", ""), errors="coerce")


def add_credit_history_length(df):
    df = df.copy()
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
    # Convert credit history start date into years before issue date if available.
    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
        df["credit_history_years"] = ((df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25).clip(lower=0)
    else:
        max_date = df["earliest_cr_line"].max()
        df["credit_history_years"] = ((max_date - df["earliest_cr_line"]).dt.days / 365.25).clip(lower=0)
    return df


def add_fico_and_debt_ratio_features(df):
    """
    Adds:
      - fico_score: mean of fico_range_low / fico_range_high (both are
        origination-time FICO band, safe to use - NOT last_fico_range_*,
        which reflects post-outcome updates and would leak the label).
      - loan_to_income: loan_amnt / annual_inc
      - installment_to_income: installment / annual_inc

    annual_inc == 0 (a small number of self-reported rows) is treated
    as missing before the division, so the ratios come out as NaN
    instead of inf - NaN then flows through the normal imputer later
    in the sklearn pipeline like any other missing value.
    """
    df = df.copy()

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    else:
        df["fico_score"] = np.nan

    safe_income = df["annual_inc"].replace(0, np.nan)
    df["loan_to_income"] = df["loan_amnt"] / safe_income
    df["installment_to_income"] = df["installment"] / safe_income

    return df


def load_raw_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw file not found: {RAW_DATA_PATH}\n"
            "Download accepted_2007_to_2018Q4.csv from Kaggle and place it in data/raw/."
        )

    usecols = list(set(FEATURES + FICO_SOURCE_COLUMNS + ["loan_status", "issue_d"]))
    df = pd.read_csv(RAW_DATA_PATH, usecols=lambda col: col in usecols, low_memory=False)

    if SAMPLE_N is not None and len(df) > SAMPLE_N:
        df = df.sample(n=SAMPLE_N, random_state=RANDOM_STATE)

    return df


def prepare_data():
    df = load_raw_data()

    df = df[df["loan_status"].isin([GOOD_STATUS, BAD_STATUS])].copy()
    df[TARGET] = np.where(df["loan_status"] == BAD_STATUS, 1, 0)

    df["int_rate"] = df["int_rate"].apply(clean_percentage)
    df["revol_util"] = df["revol_util"].apply(clean_percentage)
    df["emp_length"] = df["emp_length"].apply(clean_emp_length)

    df = add_credit_history_length(df)
    df = add_fico_and_debt_ratio_features(df)

    # Replace original date column with engineered numeric column.
    if "earliest_cr_line" in df.columns:
        df = df.drop(columns=["earliest_cr_line"])

    selected_features = [col for col in FEATURES if col != "earliest_cr_line"] + ["credit_history_years"]
    final_cols = selected_features + [TARGET]
    df = df[final_cols]

    # Drop rows where target is missing and remove extreme impossible values.
    df = df.dropna(subset=[TARGET])
    df["annual_inc"] = pd.to_numeric(df["annual_inc"], errors="coerce")
    df["dti"] = pd.to_numeric(df["dti"], errors="coerce")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Cleaned data saved to: {PROCESSED_DATA_PATH}")
    print(f"Shape: {df.shape}")
    print("Target distribution:")
    print(df[TARGET].value_counts(normalize=True).rename("proportion"))


if __name__ == "__main__":
    prepare_data()
