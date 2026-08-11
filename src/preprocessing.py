"""
preprocessing.py

Custom sklearn-compatible transformer for leak-free outlier capping
(winsorizing).

Why this exists:
    IQR-based outlier detection was tested per-class on the training
    data. Sparse / near-binary columns (term, delinq_2yrs, pub_rec)
    produced false "outliers" because their IQR is 0 or near 0 - those
    are excluded here. Genuinely continuous columns had similar,
    balanced outlier rates between classes (~0.4%-6%), so instead of
    dropping rows, extreme values are capped (winsorized) at the
    1st / 99th percentile.

Why a custom Transformer instead of capping in data_preparation.py:
    Capping thresholds (the 1st/99th percentile cutoffs) must be
    learned ONLY from the training split, then applied to the test
    split and to any new prediction row - otherwise information from
    the test set leaks into preprocessing. Wrapping this as a
    scikit-learn Transformer and placing it as the first step of the
    Pipeline guarantees:

      - pipeline.fit(X_train, y_train)  -> caps are learned on X_train only
      - pipeline.predict(X_test)        -> X_test is capped using the
                                            SAME thresholds learned from
                                            X_train (no leakage)
      - prediction.py / joblib.load(...) -> a single new row at
                                            inference time is capped
                                            with those same saved
                                            thresholds automatically

Columns capped (continuous, low/balanced outlier rate):
    loan_amnt, int_rate, installment, annual_inc, dti, revol_bal,
    open_acc, total_acc, credit_history_years, loan_to_income,
    installment_to_income

Columns intentionally NOT capped (sparse / near-binary, IQR is not
meaningful for them - flagging non-zero values as "outliers" would be
wrong):
    term, delinq_2yrs, pub_rec, revol_util

Columns intentionally NOT capped (naturally bounded, can't produce
real outliers):
    fico_score (bounded 300-850 by definition)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Columns where 1st/99th percentile capping is applied.
DEFAULT_CAP_COLUMNS = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "revol_bal",
    "open_acc",
    "total_acc",
    "credit_history_years",
    "loan_to_income",
    "installment_to_income",
]


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Winsorizes selected numeric columns at [lower_pct, upper_pct]
    percentiles learned from the data passed to fit().

    Rows are never dropped - values outside the learned bounds are
    clipped to the nearest bound.
    """

    def __init__(self, columns=None, lower_pct: float = 0.01, upper_pct: float = 0.99):
        self.columns = columns
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = self.columns if self.columns is not None else DEFAULT_CAP_COLUMNS
        self.columns_ = [c for c in cols if c in X.columns]

        self.bounds_ = {}
        for col in self.columns_:
            lower = X[col].quantile(self.lower_pct)
            upper = X[col].quantile(self.upper_pct)
            self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X

    def get_feature_names_out(self, input_features=None):
        # Passthrough transformer - column set is unchanged.
        return np.asarray(input_features)
