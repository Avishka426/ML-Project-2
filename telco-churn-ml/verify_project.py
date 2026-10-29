"""
verify_project.py
-----------------
Final quality-assurance checks. Run after the notebook, report and
presentation have been generated:

    python verify_project.py

Each check prints PASS or FAIL. The script exits with an error if any fail.
"""

import json
import re
import sys

import joblib
import numpy as np
import pandas as pd
from docx import Document
from pptx import Presentation

from src import config, data_cleaning as dc, modeling as md


def p_is_chapter(heading):
    """True for top-level headings such as '2. Data Cleaning' or 'References'."""
    return heading == "References" or (heading[:1].isdigit() and heading[1:3] == ". ")

ROOT = config.PROJECT_ROOT
REPORT = ROOT / "reports" / "Telco_Churn_Report.docx"
DECK = ROOT / "presentation" / "Telco_Churn_Presentation.pptx"
results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" – {detail}" if detail else ""))


# 1. Raw data untouched -----------------------------------------------------
check("Raw CSV unchanged (SHA-256)", dc.file_sha256(config.RAW_DATA_PATH) == config.RAW_DATA_SHA256)

# 2. Cleaning reproduces the saved processed file ----------------------------
raw = dc.load_raw()
clean, _ = dc.clean_dataset(raw)
saved = pd.read_csv(config.PROCESSED_DATA_PATH)
check("Cleaning rules pass validation", dc.validate_clean(clean))
check("Processed file matches a fresh cleaning run", clean.shape == saved.shape and
      np.allclose(clean[config.NUMERIC_FEATURES].to_numpy(dtype=float),
                  saved[config.NUMERIC_FEATURES].to_numpy(dtype=float), equal_nan=True),
      f"shape {saved.shape}")

# 3. Leakage checks ----------------------------------------------------------
X, y = dc.prepare_model_data(clean)
X_train, X_test, y_train, y_test = md.split_data(X, y)
check("Train and test sets do not overlap", set(X_train.index).isdisjoint(X_test.index))
check("customerID is not a model feature", config.ID_COL not in X.columns)
check("Stratified split keeps the churn rate", abs(y_train.mean() - y_test.mean()) < 0.001,
      f"train {y_train.mean():.4f} vs test {y_test.mean():.4f}")

# 4. Saved model reproduces the reported test metrics ------------------------
model = joblib.load(config.MODEL_DIR / "final_churn_model.joblib")
metrics = json.loads((config.MODEL_DIR / "final_model_metrics.json").read_text())
proba = model.predict_proba(X_test)[:, 1]
recomputed = md.classification_metrics(y_test, proba)
reported = metrics["test_metrics_threshold_0.5"]
check("Saved model reproduces test ROC-AUC", abs(recomputed["roc_auc"] - reported["roc_auc"]) < 1e-9,
      f"{recomputed['roc_auc']:.4f}")
check("Saved model reproduces the confusion matrix",
      all(recomputed[k] == reported[k] for k in ["TP", "FP", "FN", "TN"]))
prep_steps = [step[0] for step in model.named_steps["prep"].transformers_]
check("Preprocessing is inside the saved Pipeline", "num" in prep_steps and "cat" in prep_steps)
used = [c for _, _, cols in model.named_steps["prep"].transformers_[:2] for c in cols]
check("Logistic regression excludes the collinear charge columns",
      not set(config.LR_EXCLUDED_FEATURES) & set(used))

# 5. Tables agree with each other --------------------------------------------
test_table = pd.read_csv(config.TABLE_DIR / "t29_test_metrics.csv", index_col=0)
check("t29 test metrics match the saved model",
      abs(test_table.loc[metrics["selected_model"], "roc_auc"] - recomputed["roc_auc"]) < 1e-9)

# 6. Report -------------------------------------------------------------------
doc = Document(REPORT)
text = "\n".join(p.text for p in doc.paragraphs) + "\n".join(
    c.text for t in doc.tables for r in t.rows for c in r.cells)
for label, value in [("customers", f"{len(clean):,}"), ("test ROC-AUC", f"{recomputed['roc_auc']:.3f}"),
                     ("recall", f"{recomputed['recall']:.3f}"),
                     ("churn rate", f"{y.mean() * 100:.1f}%")]:
    check(f"Report states the correct {label} ({value})", value in text)
figures_in_report = len(doc.inline_shapes) - 1          # minus the university crest on the cover
check("Report contains 14 figures (plus the cover crest)", figures_in_report == 14, str(figures_in_report))
headings = [p.text for p in doc.paragraphs if p.style.name in ("Heading 1", "Heading 2", "Heading 3")]
check("Chapters follow the senior report layout",
      [h for h in headings if p_is_chapter(h)] == ["1. Introduction", "2. Data Cleaning", "3. Exploratory Data Analysis (EDA)",
                                                  "4. Model Fitting", "5. Conclusion", "References"],
      str([h for h in headings if p_is_chapter(h)]))
check("Data cleaning and model sections present (2.1–2.4, 4.1–4.4.3)",
      all(h in headings for h in ["2.1 Handling missing and invalid values", "2.2 Handling outliers",
                                  "2.3 Feature engineering and encoding", "2.4 Multicollinearity",
                                  "4.1 Logistic regression model", "4.1.1 Logistic regression model equation",
                                  "4.2 Decision tree model", "4.3 Random forest model", "4.4.1 Performance results",
                                  "4.4.2 Interpretation of results", "4.4.3 Actual vs predicted values"]))
check("Lists of figures and tables present", "Table of Figures" in text and "Table of Tables" in text)
eda_headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading") and p.text.startswith("3.") and not p.text.startswith("3. ")]
check("EDA section follows the senior structure (3.1–3.5.2)",
      eda_headings == ["3.1 Summary statistics", "3.2 Distribution of numeric variables", "3.3 Target variable analysis",
                       "3.4 Categorical variable analysis", "3.5 Correlation analysis", "3.5.1 Correlation heatmap",
                       "3.5.2 Relationship plots", "3.6 Statistical tests of association"], str(eda_headings))
check("Report contains the fitted model equation", "log-odds(churn) =" in text)
from docx.oxml.ns import qn
formats = [s._sectPr.find(qn("w:pgNumType")).get(qn("w:fmt")) for s in doc.sections]
check("Front matter numbered i, ii…; body numbered 1, 2…", formats == ["lowerRoman", "decimal"], str(formats))
unfilled = sorted(set(re.findall(r"\[(?:Group name|Member name|Index no\.|Submission date)\]", text)))
print(f"[INFO] Cover placeholders still to fill in by the group: {unfilled}")

# 7. Presentation ------------------------------------------------------------
deck = Presentation(DECK)
check("Presentation has 14 slides", len(deck.slides) == 14, str(len(deck.slides)))
check("Every slide has speaker notes",
      all(s.has_notes_slide and len(s.notes_slide.notes_text_frame.text) > 50 for s in deck.slides))
deck_text = " ".join(sh.text_frame.text for s in deck.slides for sh in s.shapes if sh.has_text_frame)
check("Slides state the correct test ROC-AUC", f"{recomputed['roc_auc']:.2f}" in deck_text)

# 8. Tables added for the senior-report comparison ---------------------------------
for name in ["t42_distribution_summary.csv", "t43_logistic_equation.txt", "t44_logit_summary.csv",
             "t45_logit_fit_statistics.csv", "t46_churn_rate_by_value.csv", "t47_imputation_values.csv",
             "t48_spearman_all_variables.csv"]:
    check(f"{name} exists", (config.TABLE_DIR / name).exists())
logit = pd.read_csv(config.TABLE_DIR / "t44_logit_summary.csv", index_col=0).drop(index="const")
check("statsmodels and final-model odds ratios agree (max diff < 0.15)",
      (logit["odds_ratio"] - logit["odds_ratio_sklearn_model"]).abs().max() < 0.15)

# 9. Figures referenced exist ---------------------------------------------------
missing = [f for f in ["fig04_numeric_distributions_by_churn.png", "fig05_churn_rate_key_segments.png",
                       "fig14_roc_pr_curves.png", "fig16_odds_ratios.png", "fig19_gains_and_deciles.png",
                       "fig07_correlation_heatmap.png", "fig20_vif_before_after.png",
                       "fig21_churn_rate_scatter_trend.png", "fig03_churn_distribution.png", "fig10_customer_attributes_churn.png"]
           if not (config.FIGURE_DIR / f).exists()]
check("Key figures exist", not missing, ", ".join(missing))

print(f"\n{sum(results)} of {len(results)} checks passed.")
sys.exit(0 if all(results) else 1)
