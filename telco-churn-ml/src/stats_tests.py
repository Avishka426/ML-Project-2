"""
stats_tests.py
--------------
Statistical tests used to examine relationships with churn.

* Chi-square test of independence  - categorical variable vs churn
  (+ Cramer's V as the effect size)
* Mann-Whitney U test              - numeric variable, churners vs non-churners
  (+ rank-biserial correlation as the effect size)
* Variance Inflation Factor (VIF)  - multicollinearity among predictors

A p-value answers "could this difference be due to chance?". With ~7,000
customers even tiny differences give very small p-values, so every test is
reported together with an effect size that answers "how big is it?".
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant


# ---------------------------------------------------------------------------
# Effect-size labels (Cohen, 1988). For a 2-column table (churn Yes/No) the
# thresholds for Cramer's V are 0.10 small, 0.30 medium, 0.50 large; the same
# cut-offs are commonly used for the rank-biserial correlation.
# ---------------------------------------------------------------------------
def effect_size_label(value):
    """Describe the magnitude of an effect size (absolute value)."""
    v = abs(value)
    if v < 0.10:
        return "negligible"
    if v < 0.30:
        return "small"
    if v < 0.50:
        return "medium"
    return "large"


def cramers_v(table):
    """Cramer's V for a contingency table: 0 = no association, 1 = perfect association."""
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return np.sqrt(chi2 / (n * k))


def chi_square_table(df, columns, target="Churn", n_tests=None, alpha=0.05):
    """
    Chi-square test of independence between each categorical column and the target.

    H0: the column and churn are independent (churn rate is the same in every level).
    H1: they are associated.
    Assumptions checked: independent observations (one row per customer) and
    expected counts of at least 5 in every cell.

    n_tests: number of tests in the family for the Bonferroni correction
             (defaults to the number of columns tested here).
    """
    n_tests = n_tests or len(columns)
    rows = []
    for col in columns:
        table = pd.crosstab(df[col], df[target])          # missing values are excluded
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
        v = cramers_v(table)
        rates = (table["Yes"] / table.sum(axis=1) * 100)
        rows.append({
            "variable": col,
            "n": int(table.to_numpy().sum()),
            "levels": table.shape[0],
            "chi2": chi2,
            "dof": dof,
            "p_value": p,
            "p_bonferroni": min(p * n_tests, 1.0),
            "significant_after_bonferroni": p * n_tests < alpha,
            "min_expected_count": expected.min(),
            "cramers_v": v,
            "effect_size": effect_size_label(v),
            "lowest_churn_level": f"{rates.idxmin()} ({rates.min():.1f}%)",
            "highest_churn_level": f"{rates.idxmax()} ({rates.max():.1f}%)",
        })
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def mann_whitney_table(df, columns, target="Churn", n_tests=None, alpha=0.05):
    """
    Mann-Whitney U test comparing churners with non-churners for numeric columns.

    H0: the two groups have the same distribution (a randomly chosen churner is
        equally likely to have a higher or lower value than a non-churner).
    Used instead of the t-test because tenure and TotalCharges are clearly
    non-normal (skewed / U-shaped); it compares ranks, not means.

    Effect sizes
    * prob_churner_higher  = U / (n1 * n2): probability that a random churner
      has a higher value than a random non-churner (0.5 = no difference).
    * rank_biserial        = 2 * prob - 1, from -1 to +1 (positive = churners higher).
    """
    n_tests = n_tests or len(columns)
    rows = []
    churned = df[target] == "Yes"
    for col in columns:
        x1 = df.loc[churned, col].dropna()
        x0 = df.loc[~churned, col].dropna()
        u, p = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        prob = u / (len(x1) * len(x0))
        r_rb = 2 * prob - 1
        rows.append({
            "variable": col,
            "median_churned": x1.median(),
            "median_stayed": x0.median(),
            "mean_churned": x1.mean(),
            "mean_stayed": x0.mean(),
            "skewness": df[col].skew(),
            "U": u,
            "p_value": p,
            "p_bonferroni": min(p * n_tests, 1.0),
            "significant_after_bonferroni": p * n_tests < alpha,
            "prob_churner_higher": prob,
            "rank_biserial": r_rb,
            "effect_size": effect_size_label(r_rb),
            "point_biserial_r": stats.pointbiserialr(churned[df[col].notna()], df[col].dropna())[0],
        })
    return pd.DataFrame(rows)


def vif_table(X):
    """
    Variance Inflation Factor for every column of a numeric design matrix.

    VIF_j = 1 / (1 - R^2_j), where R^2_j comes from regressing column j on all
    the other columns. An intercept is always added - leaving it out gives
    misleading VIFs. Rule of thumb: VIF > 5 is worth noting, VIF > 10 is serious.
    """
    X = X.astype(float)
    with_const = add_constant(X, has_constant="add")
    values = [variance_inflation_factor(with_const.to_numpy(), i + 1) for i in range(X.shape[1])]
    out = pd.DataFrame({"feature": X.columns, "VIF": values})
    out["concern"] = np.select([out["VIF"] > 10, out["VIF"] > 5], ["serious (>10)", "moderate (>5)"], "low")
    return out.sort_values("VIF", ascending=False).reset_index(drop=True)


def vif_without_intercept(X):
    """VIF computed WITHOUT an intercept - included only to show why this is a mistake."""
    X = X.astype(float).to_numpy()
    return [variance_inflation_factor(X, i) for i in range(X.shape[1])]
