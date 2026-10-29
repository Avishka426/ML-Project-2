"""
modeling.py
-----------
Train/test splitting, the leakage-safe preprocessing pipeline, model
definitions, hyperparameter tuning and evaluation helpers.

Every step that *learns* something from data (median for imputation, mode
for imputation, mean/std for scaling, category list for one-hot encoding)
lives inside a scikit-learn Pipeline. When the Pipeline is fitted on the
training set, or on a cross-validation training fold, those values are
learned from that data only. The test set is never seen during fitting.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier  # noqa: F401  (used in the notebook baseline)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config


def split_data(X, y):
    """Stratified 80/20 split so both sets keep the same churn rate."""
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )


def build_preprocessor(numeric_features=None, categorical_features=None, scale_numeric=True,
                       onehot_drop="if_binary"):
    """
    ColumnTransformer that prepares raw feature columns for a model.

    numeric     : median imputation -> (optional) standard scaling
    categorical : most-frequent imputation -> one-hot encoding

    onehot_drop="if_binary" keeps a single column for Yes/No variables (e.g.
    Partner_Yes) but one column per level for multi-level variables.
    onehot_drop="first" also drops the first level of multi-level variables,
    so logistic regression coefficients compare each level with that
    reference level (e.g. "One year" vs "Month-to-month" contract).
    handle_unknown="ignore" stops the model crashing on an unseen category.
    """
    numeric_features = numeric_features or config.NUMERIC_FEATURES
    categorical_features = categorical_features or config.CATEGORICAL_FEATURES

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop=onehot_drop, handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), list(numeric_features)),
            ("cat", Pipeline(categorical_steps), list(categorical_features)),
        ],
        remainder="drop",  # anything not listed (e.g. customerID) is never used
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

# Metrics recorded during cross-validation. "Positive class" = churn (1).
CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

# Order of interpretability, used only to break near-ties when selecting a model
INTERPRETABILITY_RANK = {"Logistic Regression": 1, "Decision Tree": 2, "Random Forest": 3}


def lr_feature_lists(include_charges=False):
    """Numeric / categorical feature lists for logistic regression."""
    numeric = list(config.NUMERIC_FEATURES) if include_charges else [
        f for f in config.NUMERIC_FEATURES if f not in config.LR_EXCLUDED_FEATURES]
    return numeric, list(config.CATEGORICAL_FEATURES)


def make_pipeline(model, numeric_features=None, categorical_features=None,
                  scale_numeric=True, onehot_drop="if_binary"):
    """Chain preprocessing and a model, so both are always fitted together on training data."""
    return Pipeline([
        ("prep", build_preprocessor(numeric_features, categorical_features,
                                    scale_numeric=scale_numeric, onehot_drop=onehot_drop)),
        ("model", model),
    ])


def get_model_specs():
    """
    The candidate models and their hyperparameter grids.

    Logistic Regression : interpretable baseline. Reference-level ("first")
                          one-hot coding so coefficients compare against a
                          baseline category; MonthlyCharges and TotalCharges are
                          excluded because of multicollinearity (VIF analysis).
    Decision Tree       : human-readable if/then rules; depth is limited to
                          control overfitting.
    Random Forest       : many de-correlated trees averaged together; captures
                          non-linear patterns and interactions.
    class_weight="balanced" is used for EVERY model (decided in advance to
    handle the 26.5% / 73.5% class imbalance). Using the same setting for all
    models keeps their default 0.5 decision thresholds comparable.
    """
    rs = config.RANDOM_STATE
    lr_num, lr_cat = lr_feature_lists(include_charges=False)
    return {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=5000, class_weight="balanced"),
                          lr_num, lr_cat, scale_numeric=True, onehot_drop="first"),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            make_pipeline(DecisionTreeClassifier(random_state=rs, class_weight="balanced"),
                          scale_numeric=False),
            {"model__max_depth": [3, 4, 5, 6, 8],
             "model__min_samples_leaf": [1, 10, 25, 50]},
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1,
                                                 class_weight="balanced"),
                          scale_numeric=False),
            {"model__max_depth": [None, 8, 12],
             "model__min_samples_leaf": [1, 5, 10],
             "model__max_features": ["sqrt", 0.5]},
        ),
    }


def make_cv():
    """Stratified 5-fold cross-validation (same folds for every model)."""
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def tune_model(pipeline, param_grid, X_train, y_train):
    """
    Grid search with stratified 5-fold CV on the TRAINING set only.
    The best combination is chosen by mean ROC-AUC and then refitted on
    the whole training set.
    """
    search = GridSearchCV(pipeline, param_grid, scoring=CV_SCORING, refit="roc_auc",
                          cv=make_cv(), n_jobs=-1, return_train_score=True)
    search.fit(X_train, y_train)
    return search


def cv_summary(search, name):
    """Mean and SD of every CV metric for the best hyperparameter combination."""
    i = search.best_index_
    res = search.cv_results_
    row = {"model": name}
    for metric in CV_SCORING:
        row[f"cv_{metric}_mean"] = res[f"mean_test_{metric}"][i]
        row[f"cv_{metric}_sd"] = res[f"std_test_{metric}"][i]
    row["cv_train_roc_auc_mean"] = res["mean_train_roc_auc"][i]
    row["best_params"] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return row


def select_best_model(cv_table, tolerance=0.01):
    """
    Selection rule fixed in advance:
    highest mean CV ROC-AUC; if other models are within `tolerance` of the
    best, choose the most interpretable of them.
    """
    candidates = cv_table[cv_table["model"].isin(INTERPRETABILITY_RANK)]
    best_auc = candidates["cv_roc_auc_mean"].max()
    near = candidates[candidates["cv_roc_auc_mean"] >= best_auc - tolerance]
    return min(near["model"], key=INTERPRETABILITY_RANK.get)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def classification_metrics(y_true, proba, threshold=0.5):
    """All evaluation metrics for one set of predicted churn probabilities."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def bootstrap_ci(y_true, proba, metric=roc_auc_score, n_boot=1000, seed=None):
    """
    95% bootstrap confidence interval for a metric on the test set:
    resample test customers with replacement many times and recompute the metric.
    """
    rng = np.random.default_rng(config.RANDOM_STATE if seed is None else seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        scores.append(metric(y_true[idx], proba[idx]))
    return np.percentile(scores, [2.5, 97.5])


def paired_bootstrap_difference(y_true, proba_a, proba_b, metric=roc_auc_score, n_boot=1000):
    """95% CI of metric(A) - metric(B) using the SAME resampled customers for both models."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    y_true, proba_a, proba_b = map(np.asarray, (y_true, proba_a, proba_b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(metric(y_true[idx], proba_a[idx]) - metric(y_true[idx], proba_b[idx]))
    return np.mean(diffs), np.percentile(diffs, [2.5, 97.5])


def threshold_table(y_true, proba, thresholds=None):
    """Precision, recall, F1 and share of customers flagged at each probability threshold."""
    # 0.05 - 0.90: above ~0.9 almost no customers are flagged and precision becomes unstable
    thresholds = np.round(np.arange(0.05, 0.91, 0.01), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({"threshold": t,
                     "precision": precision_score(y_true, pred, zero_division=0),
                     "recall": recall_score(y_true, pred),
                     "f1": f1_score(y_true, pred, zero_division=0),
                     "share_flagged": pred.mean()})
    return pd.DataFrame(rows)
