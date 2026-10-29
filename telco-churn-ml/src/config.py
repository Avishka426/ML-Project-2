"""
config.py
---------
Central place for file paths, constants and column definitions.

Keeping these in one file means every notebook section and script uses
exactly the same settings (e.g. the same random seed), which is essential
for reproducible results.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, so the project can be moved freely)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco_Customer_Churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"

# SHA-256 fingerprint of the original CSV. If this ever changes, the raw file
# has been modified and results may no longer be reproducible.
RAW_DATA_SHA256 = "901d7d1a43b2206e4a547fc717116695b33648bf3ee33337a634096c8c915ba4"

# ---------------------------------------------------------------------------
# Reproducibility and train/test split settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Column roles
# ---------------------------------------------------------------------------
ID_COL = "customerID"
TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# Columns left out of the LOGISTIC REGRESSION only, because of severe
# multicollinearity (see the VIF analysis in the notebook):
# * MonthlyCharges is almost fully determined by the subscribed services
# * TotalCharges is approximately tenure x MonthlyCharges
# Tree-based models keep all features (they are not affected in the same way).
LR_EXCLUDED_FEATURES = ["MonthlyCharges", "TotalCharges"]

# Optional internet add-on services (only available with an internet plan)
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# ---------------------------------------------------------------------------
# Validity rules used during cleaning
# ---------------------------------------------------------------------------
# The set of legitimate labels for each categorical column. Anything else
# (e.g. "Unknown", "N/A", blank) is treated as missing.
VALID_CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check",
    ],
    "Churn": ["No", "Yes"],
}

# tenure is a count of whole months, so it must be a non-negative integer.
# The legitimate values in this dataset run from 0 to 72 months.
MAX_TENURE_MONTHS = 72

# Legitimate monthly charges lie between about 18 and 119 (USD). The corrupt
# values start above 5,700, so any cut-off between ~120 and ~5,700 flags
# exactly the same rows. 200 is a generous, easily explained ceiling.
MAX_PLAUSIBLE_MONTHLY_CHARGE = 200.0

# ---------------------------------------------------------------------------
# Plain-English meaning of each variable (used for the data dictionary)
# ---------------------------------------------------------------------------
FEATURE_DESCRIPTIONS = {
    "customerID": ("Identifier", "Unique customer code.",
                   "Not a predictor - removed from model features."),
    "gender": ("Demographic", "Customer gender (Female / Male).",
               "Checks whether churn differs by gender; also a fairness check."),
    "SeniorCitizen": ("Demographic", "Whether the customer is 65 or older (0 = No, 1 = Yes).",
                      "Older customers may have different service needs."),
    "Partner": ("Demographic", "Whether the customer has a partner.",
                "Household ties may relate to account stability."),
    "Dependents": ("Demographic", "Whether the customer has dependents.",
                   "Family accounts may be harder to move."),
    "tenure": ("Account", "Number of months the customer has stayed with the company.",
               "Measures customer relationship length - a key churn indicator."),
    "PhoneService": ("Service", "Whether the customer has a phone service.",
                     "Part of the subscribed service bundle."),
    "MultipleLines": ("Service", "Whether the customer has multiple phone lines.",
                      "Indicates depth of phone subscription."),
    "InternetService": ("Service", "Internet type: DSL, Fiber optic or No internet.",
                        "Service type may relate to price and satisfaction."),
    "OnlineSecurity": ("Add-on", "Online security add-on subscribed.",
                       "Add-ons may increase engagement and switching cost."),
    "OnlineBackup": ("Add-on", "Online backup add-on subscribed.",
                     "Add-ons may increase engagement and switching cost."),
    "DeviceProtection": ("Add-on", "Device protection add-on subscribed.",
                         "Add-ons may increase engagement and switching cost."),
    "TechSupport": ("Add-on", "Technical support add-on subscribed.",
                    "Support access may relate to satisfaction."),
    "StreamingTV": ("Add-on", "Streaming TV add-on subscribed.",
                    "Entertainment services and their cost."),
    "StreamingMovies": ("Add-on", "Streaming movies add-on subscribed.",
                        "Entertainment services and their cost."),
    "Contract": ("Billing", "Contract term: Month-to-month, One year or Two year.",
                 "Contract commitment directly limits when a customer can leave."),
    "PaperlessBilling": ("Billing", "Whether the customer receives paperless bills.",
                         "Billing preference."),
    "PaymentMethod": ("Billing", "Payment method (electronic check, mailed check, "
                      "automatic bank transfer, automatic credit card).",
                      "Automatic vs manual payment may relate to engagement."),
    "MonthlyCharges": ("Billing", "Current monthly bill amount (USD).",
                       "Price level may relate to churn."),
    "TotalCharges": ("Billing", "Total amount billed over the customer's lifetime (USD).",
                     "Roughly tenure x monthly charges - strongly overlaps with tenure."),
    "Churn": ("Target", "Whether the customer left within the last month (Yes / No).",
              "The outcome to be predicted."),
}
