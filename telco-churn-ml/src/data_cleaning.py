"""
data_cleaning.py
----------------
Functions to load, audit and clean the Telco Customer Churn dataset.

Design principles
* The raw file is never modified. We read it, then work on copies.
* The raw file is first read with every value as text, so that blanks
  (" "), invalid tokens ("INVALID_VAL") and sentinel codes (-99) are visible
  instead of being silently converted by pandas.
* Every cleaning rule here works one row at a time and learns nothing from
  other rows (no means, medians or modes). That makes these rules safe to
  apply before the train/test split: they cannot leak information from the
  test set. Statistical imputation (median / most frequent) happens later,
  inside the scikit-learn Pipeline, using training data only.
"""

import hashlib

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def file_sha256(path):
    """Return the SHA-256 fingerprint of a file (used to prove it is unchanged)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path=config.RAW_DATA_PATH):
    """Read the raw CSV with every column as text and no automatic NaN conversion."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_number(series):
    """Convert text to numbers; anything non-numeric (blank, 'INVALID_VAL') becomes NaN."""
    return pd.to_numeric(series.str.strip(), errors="coerce")


def _is_blank(series):
    """True where a text value is empty or only whitespace."""
    return series.str.strip() == ""


def _invalid_tenure_mask(tenure):
    """Tenure is invalid if it is negative or not a whole number of months."""
    return (tenure < 0) | (tenure != tenure.round())


def _invalid_charge_mask(monthly):
    """Monthly charges above the plausible ceiling are data-entry errors."""
    return monthly > config.MAX_PLAUSIBLE_MONTHLY_CHARGE


# ---------------------------------------------------------------------------
# Auditing (describe problems BEFORE changing anything)
# ---------------------------------------------------------------------------
def missing_value_table(raw):
    """
    Compare what pandas would report as missing with the *true* number of
    missing or invalid entries in each column.

    A default pd.read_csv() does not treat " " or "INVALID_VAL" as missing,
    so a naive df.isnull().sum() under-reports the problem.
    """
    naive = pd.read_csv(config.RAW_DATA_PATH)  # default pandas behaviour
    rows = []
    for col in raw.columns:
        values = raw[col]
        blank = int(_is_blank(values).sum())
        invalid_label = 0
        if col in config.VALID_CATEGORIES:
            invalid_label = int((~values.isin(config.VALID_CATEGORIES[col]) & ~_is_blank(values)).sum())
        non_numeric = 0
        if col in config.NUMERIC_FEATURES:
            non_numeric = int((_to_number(values).isna() & ~_is_blank(values)).sum())
        rows.append({
            "column": col,
            "pandas_isnull_default": int(naive[col].isnull().sum()),
            "blank_or_whitespace": blank,
            "invalid_label": invalid_label,
            "non_numeric_text": non_numeric,
        })
    table = pd.DataFrame(rows)
    table["total_missing_or_invalid_text"] = table[
        ["blank_or_whitespace", "invalid_label", "non_numeric_text"]].sum(axis=1)
    table["pct_of_rows"] = table["total_missing_or_invalid_text"] / len(raw) * 100
    return table


def issue_counts_by_type(raw):
    """
    Count problem values per column, split into three kinds:
    blank, invalid text/label, and impossible numeric value.
    Used for the 'missing values before cleaning' figure.
    """
    rows = {}
    for col in raw.columns:
        values = raw[col]
        blank = _is_blank(values)
        invalid_text = pd.Series(False, index=raw.index)
        impossible = pd.Series(False, index=raw.index)
        if col in config.VALID_CATEGORIES:
            invalid_text = ~values.isin(config.VALID_CATEGORIES[col]) & ~blank
        if col in config.NUMERIC_FEATURES:
            number = _to_number(values)
            invalid_text = number.isna() & ~blank
            # only real numbers can be "impossible" (blanks are already counted as blank)
            if col == "tenure":
                impossible = _invalid_tenure_mask(number) & number.notna()
            elif col == "MonthlyCharges":
                impossible = _invalid_charge_mask(number) & number.notna()
        rows[col] = {"Blank": int(blank.sum()),
                     "Invalid text / label": int(invalid_text.sum()),
                     "Impossible value": int(impossible.sum())}
    return pd.DataFrame(rows).T


def audit_dataset(raw):
    """
    Produce a table listing every data-quality issue found in the raw file,
    with counts, examples and the planned treatment.
    """
    df = raw.copy()
    issues = []

    def add(check, column, count, example, treatment):
        issues.append({"check": check, "column": column, "count": int(count),
                       "example": example, "planned_treatment": treatment})

    # Duplicates
    n_exact = df.duplicated().sum()
    n_dup_ids = df[config.ID_COL].duplicated().sum()
    add("Exact duplicate rows", "all", n_exact, "same customerID and identical values",
        "Remove extra copies")
    add("Repeated customerID", config.ID_COL, n_dup_ids,
        "every repeated ID is an exact duplicate row", "Resolved by removing exact duplicates")
    dedup = df.drop_duplicates()
    same_profile = dedup.drop(columns=config.ID_COL).duplicated().sum()
    add("Identical profile, different customerID", "all", same_profile,
        "e.g. new customers with only a basic phone plan",
        "Keep - genuinely different customers")

    # Invalid category labels
    for col, valid in config.VALID_CATEGORIES.items():
        bad = ~dedup[col].isin(valid)
        if bad.any():
            labels = dedup.loc[bad, col].replace("", "<blank>").value_counts().to_dict()
            add("Invalid / blank category", col, bad.sum(), str(labels), "Set to missing")

    # Numeric columns
    tenure = _to_number(dedup["tenure"])
    add("Blank value", "tenure", tenure.isna().sum(), "''", "Set to missing")
    neg = tenure < 0
    add("Negative sentinel value", "tenure", neg.sum(),
        str(sorted(tenure[neg].unique().tolist())), "Set to missing")
    frac = (tenure >= 0) & (tenure != tenure.round())
    add("Impossible non-integer value", "tenure", frac.sum(),
        f"{tenure[frac].min():.1f} to {tenure[frac].max():.1f} months", "Set to missing")
    add("Zero tenure (new customer)", "tenure", (tenure == 0).sum(), "0",
        "Keep - valid new customers")

    monthly = _to_number(dedup["MonthlyCharges"])
    add("Blank value", "MonthlyCharges", monthly.isna().sum(), "''", "Set to missing")
    big = _invalid_charge_mask(monthly)
    add("Impossible value (> plausible ceiling)", "MonthlyCharges", big.sum(),
        f"{monthly[big].min():,.0f} to {monthly[big].max():,.0f}", "Set to missing")

    total_txt = dedup["TotalCharges"]
    blank_tc = _is_blank(total_txt)
    add("Blank value", "TotalCharges", blank_tc.sum(),
        f"all have tenure = {sorted(tenure[blank_tc].unique().tolist())}",
        "Set to 0 (not billed yet)")
    bad_tc = ~blank_tc & _to_number(total_txt).isna()
    add("Non-numeric text", "TotalCharges", bad_tc.sum(),
        str(total_txt[bad_tc].unique().tolist()), "Set to missing")

    return pd.DataFrame(issues)


def iqr_outlier_table(df, columns=config.NUMERIC_FEATURES):
    """Count values outside the Tukey fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR) for each column."""
    rows = []
    for col in columns:
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({
            "column": col, "n": len(x), "min": x.min(), "Q1": q1, "median": x.median(),
            "Q3": q3, "max": x.max(), "lower_fence": low, "upper_fence": high,
            "n_below": int((x < low).sum()), "n_above": int((x > high).sum()),
            "skewness": x.skew(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_dataset(raw):
    """
    Apply the documented, row-wise cleaning rules.

    Returns
    -------
    clean : DataFrame
        One row per customer with numeric columns converted to numbers.
        Any value that could not be repaired is left as NaN for the
        Pipeline imputer to handle (fitted on training data only).
    change_log : DataFrame
        One row per changed cell: customerID, column, old value, new value, rule.
    """
    log = []

    def record(mask, col, old, new, rule):
        for idx in df.index[mask]:
            log.append({"customerID": df.at[idx, config.ID_COL], "column": col,
                        "old_value": old[idx], "new_value": new[idx], "rule": rule})

    # Step 1 - remove exact duplicate rows (keep the first copy)
    df = raw.drop_duplicates().reset_index(drop=True).copy()

    # Step 2 - invalid or blank category labels -> missing
    for col, valid in config.VALID_CATEGORIES.items():
        bad = ~df[col].isin(valid)
        old = df[col].copy()
        df.loc[bad, col] = np.nan
        record(bad, col, old, df[col], "invalid/blank label -> missing")

    # Step 3 - numeric conversion with validity rules
    tenure_txt, monthly_txt, total_txt = df["tenure"].copy(), df["MonthlyCharges"].copy(), df["TotalCharges"].copy()

    tenure = _to_number(tenure_txt)
    bad_tenure = tenure.isna() | _invalid_tenure_mask(tenure)
    tenure[bad_tenure] = np.nan

    monthly = _to_number(monthly_txt)
    bad_monthly = monthly.isna() | _invalid_charge_mask(monthly)
    monthly[bad_monthly] = np.nan

    total = _to_number(total_txt)
    # Blank TotalCharges belong to customers with tenure 0: they have not been billed yet.
    not_billed = _is_blank(total_txt) & (tenure == 0)
    total[not_billed] = 0.0
    bad_total = total.isna()

    # Step 4 - repair numeric values using the billing identity
    #          TotalCharges ~= tenure x MonthlyCharges
    # Each repair uses only the other two values in the SAME row.
    fix_total = bad_total & tenure.notna() & monthly.notna()
    total[fix_total] = (tenure * monthly)[fix_total].round(2)

    fix_tenure = bad_tenure & monthly.notna() & total.notna() & (monthly > 0)
    est_tenure = (total / monthly).round().clip(lower=0, upper=config.MAX_TENURE_MONTHS)
    tenure[fix_tenure] = est_tenure[fix_tenure]

    fix_monthly = bad_monthly & tenure.notna() & total.notna() & (tenure > 0)
    monthly[fix_monthly] = (total / tenure)[fix_monthly].round(2)

    df["tenure"], df["MonthlyCharges"], df["TotalCharges"] = tenure, monthly, total

    # Log numeric changes
    record(bad_tenure & ~fix_tenure, "tenure", tenure_txt, df["tenure"], "invalid -> missing (not repairable)")
    record(fix_tenure, "tenure", tenure_txt, df["tenure"], "invalid -> round(TotalCharges / MonthlyCharges)")
    record(bad_monthly & ~fix_monthly, "MonthlyCharges", monthly_txt, df["MonthlyCharges"],
           "invalid -> missing (not repairable)")
    record(fix_monthly, "MonthlyCharges", monthly_txt, df["MonthlyCharges"],
           "invalid -> TotalCharges / tenure")
    record(not_billed, "TotalCharges", total_txt, df["TotalCharges"], "blank with tenure 0 -> 0")
    record(fix_total, "TotalCharges", total_txt, df["TotalCharges"], "invalid -> tenure x MonthlyCharges")
    record(bad_total & ~fix_total, "TotalCharges", total_txt, df["TotalCharges"],
           "invalid -> missing (not repairable)")

    # Step 5 - consistent Yes/No coding for SeniorCitizen (was 0/1)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({"0": "No", "1": "Yes"})

    change_log = pd.DataFrame(log, columns=["customerID", "column", "old_value", "new_value", "rule"])
    change_log["old_value"] = change_log["old_value"].replace("", "<blank>").fillna("<blank>")
    return df, change_log


def reconstruction_accuracy(raw):
    """
    Check how accurate the billing identity is on rows where all three
    numeric values are valid. This justifies using it for repairs.
    """
    df = raw.drop_duplicates()
    t, m, c = (_to_number(df[col]) for col in ["tenure", "MonthlyCharges", "TotalCharges"])
    ok = (t >= 1) & ~_invalid_tenure_mask(t) & (m <= config.MAX_PLAUSIBLE_MONTHLY_CHARGE) & c.notna()
    t, m, c = t[ok], m[ok], c[ok]

    estimates = {
        "TotalCharges": ("tenure x MonthlyCharges", t * m, c),
        "tenure": ("round(TotalCharges / MonthlyCharges)",
                   (c / m).round().clip(0, config.MAX_TENURE_MONTHS), t),
        "MonthlyCharges": ("TotalCharges / tenure", c / t, m),
    }
    rows = []
    for name, (method, estimate, actual) in estimates.items():
        rows.append({
            "value_repaired": name,
            "method": method,
            "repair_mean_abs_error": (estimate - actual).abs().mean(),
            "repair_median_abs_pct_error": ((estimate - actual).abs() / actual).median() * 100,
            "median_imputation_mean_abs_error": (actual.median() - actual).abs().mean(),
            "median_imputation_median_abs_pct_error": ((actual.median() - actual).abs() / actual).median() * 100,
        })
    table = pd.DataFrame(rows)
    table.insert(1, "rows_checked", int(ok.sum()))
    return table


def validate_clean(clean):
    """Assertions that must hold after cleaning. Raises AssertionError if any fail."""
    assert len(clean) == clean[config.ID_COL].nunique(), "customerID is not unique"
    assert not clean.duplicated().any(), "duplicate rows remain"
    for col, valid in config.VALID_CATEGORIES.items():
        valid_after = ["No", "Yes"] if col == "SeniorCitizen" else valid
        observed = set(clean[col].dropna().unique())
        assert observed <= set(valid_after), f"{col} has invalid labels: {observed - set(valid_after)}"
    for col in config.NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(clean[col]), f"{col} is not numeric"
    t = clean["tenure"].dropna()
    assert ((t >= 0) & (t <= config.MAX_TENURE_MONTHS) & (t == t.round())).all(), "invalid tenure remains"
    assert (clean["MonthlyCharges"].dropna() <= config.MAX_PLAUSIBLE_MONTHLY_CHARGE).all(), "invalid charges remain"
    assert (clean["TotalCharges"].dropna() >= 0).all(), "negative TotalCharges"
    assert clean[config.TARGET].notna().all(), "target has missing values"
    return True


# ---------------------------------------------------------------------------
# Preparing model inputs
# ---------------------------------------------------------------------------
def prepare_model_data(clean):
    """
    Split the cleaned table into features X and binary target y (1 = churned).

    "No internet service" and "No phone service" are recoded to "No" here.
    That information is already captured by InternetService and PhoneService,
    so keeping the extra label would create several perfectly duplicated
    dummy columns. The original labels are kept in `clean` for EDA.
    """
    X = clean[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES].copy()
    for col in config.ADDON_SERVICES:
        X[col] = X[col].replace("No internet service", "No")
    X["MultipleLines"] = X["MultipleLines"].replace("No phone service", "No")
    y = (clean[config.TARGET] == "Yes").astype(int)
    return X, y


def build_data_dictionary(raw, clean):
    """Data dictionary: role, type, allowed values and business relevance of each column."""
    rows = []
    for col in raw.columns:
        group, meaning, relevance = config.FEATURE_DESCRIPTIONS[col]
        if col in config.NUMERIC_FEATURES:
            dtype = "Numeric (continuous)" if col != "tenure" else "Numeric (whole months)"
            s = clean[col]
            values = f"{s.min():,.2f} to {s.max():,.2f}"
        elif col == config.ID_COL:
            dtype, values = "Text (ID)", f"{clean[col].nunique():,} unique codes"
        else:
            dtype = "Binary categorical" if clean[col].nunique() == 2 else "Nominal categorical"
            values = " / ".join(sorted(clean[col].dropna().unique()))
        rows.append({"variable": col, "group": group, "type_after_cleaning": dtype,
                     "values_after_cleaning": values, "description": meaning,
                     "relevance_to_churn": relevance,
                     "missing_after_cleaning": int(clean[col].isna().sum())})
    return pd.DataFrame(rows)
