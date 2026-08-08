from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lending_club_cleaned.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "lending_club_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Use a sample by default because the Kaggle file is very large.
# Set SAMPLE_N = None if your machine has enough RAM.
SAMPLE_N = 120000

FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
]

TARGET = "loan_default"

GOOD_STATUS = "Fully Paid"
BAD_STATUS = "Charged Off"
