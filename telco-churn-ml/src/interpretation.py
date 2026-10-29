"""
interpretation.py
-----------------
Tools for explaining what the models learned.

* Odds ratios from the logistic regression (with bootstrap confidence intervals)
* Permutation importance (model-agnostic: how much does test ROC-AUC drop when
  one column is randomly shuffled?)
* Cumulative gains / lift table (how well the risk ranking concentrates churners)
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from src import config


def odds_ratio_table(lr_pipeline):
    """
    Coefficients of a fitted logistic-regression Pipeline turned into odds ratios.

    odds ratio = exp(coefficient)
    * > 1 : associated with HIGHER odds of churn
    * < 1 : associated with LOWER odds of churn
    Numeric features are standardised, so their odds ratio is per 1 standard
    deviation. One-hot features compare a category with the reference
    (first, alphabetical) category of the same variable, holding the other
    features constant.
    """
    prep = lr_pipeline.named_steps["prep"]
    model = lr_pipeline.named_steps["model"]
    names = prep.get_feature_names_out()
    coefs = model.coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefs, "odds_ratio": np.exp(coefs)})
    return table


def bootstrap_odds_ratios(lr_pipeline, X_train, y_train, n_boot=200):
    """
    95% bootstrap confidence intervals for the odds ratios: refit the whole
    Pipeline on many resamples of the TRAINING customers.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    names = lr_pipeline.named_steps["prep"].get_feature_names_out()
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X_train), len(X_train))
        fitted = clone(lr_pipeline).fit(X_train.iloc[idx], y_train.iloc[idx])
        boot_names = fitted.named_steps["prep"].get_feature_names_out()
        coefs = pd.Series(fitted.named_steps["model"].coef_.ravel(), index=boot_names)
        draws.append(coefs.reindex(names))
    draws = pd.DataFrame(draws)
    return pd.DataFrame({
        "feature": names,
        "or_ci_low": np.exp(draws.quantile(0.025).to_numpy()),
        "or_ci_high": np.exp(draws.quantile(0.975).to_numpy()),
    })


def reference_levels(lr_pipeline):
    """The dropped (reference) category for each one-hot encoded variable."""
    encoder = lr_pipeline.named_steps["prep"].named_transformers_["cat"].named_steps["encode"]
    columns = lr_pipeline.named_steps["prep"].transformers_[1][2]
    return {col: cats[idx] for col, cats, idx in zip(columns, encoder.categories_, encoder.drop_idx_)}


def permutation_importance_table(pipeline, X, y, n_repeats=20):
    """
    Permutation importance on the raw input columns.

    Each column is shuffled n_repeats times; the average drop in ROC-AUC is
    its importance. Because the Pipeline is applied after shuffling, every
    categorical variable is shuffled as ONE unit (all its dummies together).
    Caveat: when two columns carry the same information (e.g. tenure and
    TotalCharges), shuffling one of them costs little, so both can look less
    important than they really are.
    """
    result = permutation_importance(pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats,
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns,
                          "importance_mean": result.importances_mean,
                          "importance_sd": result.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def gains_table(y_true, proba, n_groups=10):
    """
    Sort customers from highest to lowest predicted risk, cut them into
    10 equal groups (deciles) and summarise each group.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(np.arange(len(df)), n_groups, labels=range(1, n_groups + 1))
    base_rate = df["y"].mean()
    out = df.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"),
                                                  mean_predicted_probability=("p", "mean"))
    out["actual_churn_rate"] = out["churners"] / out["customers"]
    out["cumulative_share_of_customers"] = out["customers"].cumsum() / out["customers"].sum()
    out["cumulative_share_of_churners_captured"] = out["churners"].cumsum() / out["churners"].sum()
    out["lift"] = out["actual_churn_rate"] / base_rate
    return out


def tree_leaf_table(tree_pipeline, X_train, y_train):
    """
    One row per final group (leaf) of a fitted decision-tree Pipeline:
    the plain-language rule that leads to it, its size and its actual churn rate
    in the training data.
    """
    prep = tree_pipeline.named_steps["prep"]
    tree = tree_pipeline.named_steps["model"].tree_
    names = list(prep.get_feature_names_out())
    leaves = tree_pipeline.named_steps["model"].apply(prep.transform(X_train))
    y = np.asarray(y_train)

    rows = []

    def walk(node, conditions):
        if tree.children_left[node] == -1:
            mask = leaves == node
            rows.append({"rule": " AND ".join(conditions), "customers": int(mask.sum()),
                         "churn_rate": float(y[mask].mean())})
            return
        feat, thr = names[tree.feature[node]], tree.threshold[node]
        if feat in config.NUMERIC_FEATURES:
            left, right = f"{feat} <= {thr:.1f}", f"{feat} > {thr:.1f}"
        else:
            var, level = feat.split("_", 1)
            left, right = f"{var} is not {level}", f"{var} = {level}"
        walk(tree.children_left[node], conditions + [left])
        walk(tree.children_right[node], conditions + [right])

    walk(0, [])
    return pd.DataFrame(rows).sort_values("churn_rate", ascending=False).reset_index(drop=True)
