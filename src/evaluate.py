"""
evaluate.py

Loads every model saved by train.py (config.MODEL_NAMES) and, for EACH
model, evaluates it on BOTH the train split and the test split so you
can compare them side by side (and spot overfitting) before manually
choosing which model to use.

For each model, this produces:
  - Classification report (train AND test)
  - Confusion matrix plot, labeled with class names, for train AND test
        reports/figures/confusion_matrix_<model_name>_train.png
        reports/figures/confusion_matrix_<model_name>_test.png
  - ROC curve plot, for train AND test
        reports/figures/roc_curve_<model_name>_train.png
        reports/figures/roc_curve_<model_name>_test.png
  - AUC score (train AND test)

All of it is written to reports/model_metrics.json so you can compare
every model's numbers in one place. Nothing here auto-picks a "best"
model - that's a judgment call for you to make from this output, then
set config.SELECTED_MODEL_NAME accordingly.

Run:
    python src/evaluate.py
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH_TEMPLATE,
    MODEL_NAMES,
    MODEL_DISPLAY_NAMES,
    CLASS_NAMES,
    CONFUSION_MATRIX_TEMPLATE,
    ROC_CURVE_TEMPLATE,
    METRICS_PATH,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_split():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Same TEST_SIZE / RANDOM_STATE / stratify as train.py, so this
    # reproduces the exact same train/test rows without needing to
    # persist them to disk separately.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, model_name: str, split: str) -> str:
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")

    ax.set_title(f"Confusion Matrix — {display_name} ({split})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out_path = CONFUSION_MATRIX_TEMPLATE.format(model_name=model_name, split=split)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved confusion matrix ({split}) -> {out_path}")
    return out_path


def plot_roc_curve(pipeline, X, y_true, model_name: str, split: str) -> str:
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipeline, X, y_true, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title(f"ROC Curve — {display_name} ({split})")
    ax.legend(loc="lower right")
    plt.tight_layout()

    out_path = ROC_CURVE_TEMPLATE.format(model_name=model_name, split=split)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved ROC curve ({split}) -> {out_path}")
    return out_path


def evaluate_split(pipeline, X, y_true, model_name, split):
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    cm_path = plot_confusion_matrix(y_true, y_pred, model_name, split)
    roc_path = plot_roc_curve(pipeline, X, y_true, model_name, split)
    metrics["confusion_matrix_plot"] = os.path.relpath(cm_path)
    metrics["roc_curve_plot"] = os.path.relpath(roc_path)

    print(
        f"  [{split}] Accuracy={metrics['accuracy']:.4f}  "
        f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  "
        f"F1={metrics['f1_score']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}"
    )
    return metrics


def evaluate_model(model_name, X_train, X_test, y_train, y_test):
    model_path = MODEL_PATH_TEMPLATE.format(model_name=model_name)
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    if not os.path.exists(model_path):
        print(f"Skipping {display_name}: no saved model at {model_path} (run train.py first)")
        return None

    print(f"\nEvaluating: {display_name}")
    pipeline = joblib.load(model_path)

    return {
        "model_display_name": display_name,
        "train": evaluate_split(pipeline, X_train, y_train, model_name, "train"),
        "test": evaluate_split(pipeline, X_test, y_test, model_name, "test"),
    }


def run():
    X_train, X_test, y_train, y_test = load_split()

    all_metrics = {}
    for model_name in MODEL_NAMES:
        result = evaluate_model(model_name, X_train, X_test, y_train, y_test)
        if result is not None:
            all_metrics[model_name] = result

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nAll metrics saved -> {METRICS_PATH}")
    print("\nTest-set summary (review this + the plots to pick a model yourself):")
    print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'ROC-AUC':>9}")
    for name, res in all_metrics.items():
        t = res["test"]
        print(
            f"{MODEL_DISPLAY_NAMES.get(name, name):<22}"
            f"{t['accuracy']:>10.4f}{t['precision']:>11.4f}"
            f"{t['recall']:>9.4f}{t['f1_score']:>8.4f}{t['roc_auc']:>9.4f}"
        )
    print(
        "\nNo model is auto-selected. Set SELECTED_MODEL_NAME in config.py "
        "to whichever model you judge best from this table + the confusion "
        "matrix / ROC curve plots in reports/figures/."
    )


if __name__ == "__main__":
    run()
