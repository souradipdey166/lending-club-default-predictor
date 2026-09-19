import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from config import (
    PROCESSED_DATA_PATH,
    MODELS_DIR,
    MODEL_PATH_TEMPLATE,
    MODEL_DISPLAY_NAMES,
    SELECTED_MODEL_NAME,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
)
from preprocessing import OutlierCapper


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def build_models() -> dict:
    """
    Hyperparameters below are the winning values found via
    RandomizedSearchCV in notebooks/00_experiment.ipynb (scoring="f1",
    StratifiedKFold cv). The search itself is NOT repeated here - it
    was expensive (full 1.3M-row dataset) and only needs to run once;
    this just bakes in the result so train.py stays fast and
    reproducible.

    gradient_boosting was intentionally left OUT of tuning (it has no
    n_jobs / no class_weight support, and was by far the slowest model
    to search over) - it's kept here only as an UNTUNED baseline for
    comparison, not a candidate for SELECTED_MODEL_NAME.
    """
    return {
        "logistic_regression": LogisticRegression(
            # Best params (CV f1: 0.4265):
            # {'LR__solver': 'liblinear', 'LR__penalty': 'l2', 'LR__C': 0.5}
            solver="liblinear",
            penalty="l2",
            C=0.5,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            # Best params (CV f1: 0.4180):
            # {'RF__n_estimators': 300, 'RF__min_samples_split': 5,
            #  'RF__min_samples_leaf': 5, 'RF__max_features': 'log2',
            #  'RF__max_depth': 6}
            n_estimators=300,
            min_samples_split=5,
            min_samples_leaf=5,
            max_features="log2",
            max_depth=6,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            # NOT tuned - untuned baseline only, see docstring above.
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            # Best params (test F1: 0.4377, ROC-AUC: 0.7235):
            # subsample=0.6, n_estimators=300, min_child_weight=2,
            # max_depth=5, learning_rate=0.1, colsample_bytree=1.0
            subsample=0.6,
            n_estimators=300,
            min_child_weight=2,
            max_depth=5,
            learning_rate=0.1,
            colsample_bytree=1.0,
            scale_pos_weight=4.0094,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_models():
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError("Processed dataset not found. Run: python src/data_preparation.py")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Same split (same TEST_SIZE / RANDOM_STATE) is reproduced in
    # evaluate.py, so evaluate.py sees the exact same train/test rows
    # without needing to persist them separately.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # NOTE: OutlierCapper.fit() only ever sees X_train (it's the first
    # Pipeline step, and pipeline.fit(X_train, ...) only fits on
    # X_train). Bounds learned here are reused unchanged on X_test and
    # on any new row in prediction.py - leak-free.
    preprocessor = build_preprocessor(X_train)
    models = build_models()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Only trains SELECTED_MODEL_NAME (config.py) - the model already
    # chosen from notebook experiments. The other candidates (LR, RF,
    # GBM) don't need to be retrained here every time; if you want a
    # fresh full comparison across all models again, change this back
    # to `for name in MODEL_NAMES:` (and re-import MODEL_NAMES).
    name = SELECTED_MODEL_NAME
    model = models[name]
    display_name = MODEL_DISPLAY_NAMES.get(name, name)
    print(f"Training: {display_name}")

    pipeline = Pipeline(
        steps=[
            ("outlier_capper", OutlierCapper()),
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    out_path = MODEL_PATH_TEMPLATE.format(model_name=name)
    joblib.dump(pipeline, out_path)
    print(f"  Saved -> {out_path}")

    print(f"\n{display_name} trained and saved.")
    print("Run 'python src/evaluate.py' to review train/test metrics,")
    print("confusion matrix, and ROC curve for this model.")


if __name__ == "__main__":
    train_models()
