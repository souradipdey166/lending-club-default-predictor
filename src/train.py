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
    MODEL_NAMES,
    MODEL_DISPLAY_NAMES,
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
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight={0: .2, 1: .9},
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_split=20,
            class_weight={0: .3, 1: .9},
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,
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

    for name in MODEL_NAMES:
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

    print("\nAll models trained and saved separately.")
    print("Run 'python src/evaluate.py' next, then review reports/model_metrics.json")
    print("and reports/figures/ to manually choose the best model.")
    print("Set SELECTED_MODEL_NAME in config.py to whichever one you pick.")


if __name__ == "__main__":
    train_models()
