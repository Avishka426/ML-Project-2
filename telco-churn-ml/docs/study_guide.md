# Study Guide – Understanding and Explaining the Project
This guide explains each phase in plain English so every group member can answer the lecturer's questions. A new section is added after each phase is completed.

---

## Phase 2 – Data inspection, cleaning and preprocessing

### 1. Inspecting the data before changing it
**What we did**
- Loaded the raw CSV with every value kept as text.
- Listed every problem before touching anything.

**Why**
- If you let pandas convert values automatically, some problems become invisible:
  - `TotalCharges` contained `" "` and `"INVALID_VAL"`. A normal `read_csv` keeps the whole column as text, and `isnull()` reports **0** missing values.
  - `gender` contained `"Unknown"` and `"OtherXYZ"`, which `isnull()` does not count.
- Evidence first, decisions second.

**What we found** (raw file: 7,073 rows)

| Problem | Count | Our verdict |
|---|---|---|
| Exact duplicate rows | 30 | Remove. 7,043 unique customers remain. |
| `gender` = Unknown / N/A / OtherXYZ | 15 | Not a real category, so treat as missing |
| `Contract` blank | 35 | Missing |
| `tenure` blank / −99 / 545–988 (non-integer) | 40 / 10 / 10 | −99 is a "missing" code; 988 months is over 80 years. Both are errors. |
| `MonthlyCharges` blank / 5,765–9,960 | 50 / 10 | Real bills are 18–119 USD. These are errors. |
| `TotalCharges` blank (all tenure = 0) / INVALID_VAL | 11 / 12 | A blank means "not billed yet", so 0. INVALID_VAL is an error. |

**How to say it in the presentation**
> "Before cleaning, we audited every column. A simple `isnull()` check would have found almost nothing in TotalCharges. By inspecting the actual values we found blanks, text tokens, sentinel codes like −99 and impossible values like a 988-month tenure."

### 2. Cleaning rules
**Key idea: row-wise rules are leakage-free.**
- Every cleaning rule uses **only the values in the same customer's row**. For example, a missing TotalCharges becomes tenure × MonthlyCharges.
- No rule uses an average, median or mode of other customers.
- So applying the rules before the train/test split cannot leak information from the test set.

**The repair trick**
- `TotalCharges ≈ tenure × MonthlyCharges`: total billed ≈ months × monthly bill.
- We checked this on 6,900 complete customers:
  - Rebuilding TotalCharges this way is off by about 2%.
  - Filling in the median would be off by about 75%.
  - For tenure, the rebuild is off by less than 1 month on average, versus about 22 months for the median.
- No customer had two of the three values broken, so every numeric error could be repaired.

**What remained:** only 15 `gender` and 35 `Contract` values are still missing. These are filled with the most common value **inside the model Pipeline, learned from training data only**.

**Missingness check**
- The 180 customers with repaired values churned at 31.7%, compared with 26.4% for everyone else.
- The chi-square test does not find this difference statistically significant (p ≈ 0.14).
- We report it honestly. The affected group is only 2.6% of customers and none were deleted, so any effect is small.

### 3. Outliers: why we did not delete or winsorise
**Concept: an outlier is not automatically an error.**
- **Impossible values** (−99 months, a 9,960 USD monthly bill) are errors, so we fixed them.
- **Unusual but possible values** (a loyal 72-month customer with an 8,000 USD total) are real customers.
  - Removing them would hide exactly the long-tenure customers who rarely churn.
- After the fixes, the IQR rule finds **0 outliers** in tenure, MonthlyCharges and TotalCharges.
- `TotalCharges` is right-skewed, but every value is plausible.

**Likely lecturer question:** *"Why didn't you remove outliers?"*
> "We detected them with the IQR rule and boxplots. The extreme values in the raw file were impossible (data errors), so we repaired them. After repair, no values fall outside the IQR fences. Deleting or capping legitimate values would distort real customer behaviour."

### 4. Train/test split, class imbalance and the Pipeline
**Stratified 80/20 split** (`random_state=42`)
- 5,634 training and 1,409 test customers, each with a 26.5% churn rate.
- *Stratified* = the churn percentage is kept the same in both sets.

**Class imbalance**
- Only about 1 in 4 customers churned.
- A model that always says "No churn" is 73.5% accurate but useless, which is why accuracy alone is misleading.
- We will use `class_weight="balanced"`: during training, churners count about 3× more.
- **Why not SMOTE?** It creates synthetic customers. Class weighting achieves the same goal more simply. If SMOTE were used, it must happen *inside* cross-validation, never before the split.

**Why customerID was removed:** it is just a label. A model could memorise it, but it tells us nothing about new customers.

**Why one-hot encoding, not label encoding**
- Label encoding PaymentMethod as 0/1/2/3 tells a linear model that "credit card" is "3 times" "bank transfer". That is meaningless.
- One-hot encoding gives each category its own Yes/No column.

**Why "No internet service" → "No" for modelling:** whether a customer has internet is already stored in `InternetService`. Keeping the extra label would create six identical columns.

**What data leakage is, and how the Pipeline prevents it**
- *Leakage* means information from the test set influences training, which makes test scores look better than they really are.
- Examples of leakage:
  - computing the median for imputation on all 7,043 customers
  - fitting the scaler on all data
  - oversampling before splitting
- **Our solution:** imputer → scaler → encoder are placed in a scikit-learn `Pipeline` together with the model.
  - When we call `fit(X_train, y_train)`, all of these learn from training data only.
  - During cross-validation, the Pipeline is re-fitted on each training fold.

**How to say it**
> "All steps that learn from data are inside a scikit-learn Pipeline, fitted only on the training set. The test set is used exactly once, for final evaluation."

### 5. Files produced in Phase 2

| File | Contents |
|---|---|
| `outputs/tables/t01_raw_missing_values.csv` | What pandas reports vs the true missing/invalid counts |
| `t02_data_quality_audit.csv` | Every issue, its count, an example and the planned treatment |
| `t03_issue_counts_by_type.csv` | Data for Figure 1 |
| `t04_reconstruction_accuracy.csv` | Evidence that the repair rule beats median imputation |
| `t05_cleaning_summary.csv` | Number of values changed per rule |
| `t06_cleaning_change_log.csv` | Every individual change (customer, column, old value, new value, rule) |
| `t07_missingness_vs_churn.csv` | Churn rate of affected vs other customers |
| `t08_iqr_outliers_after_cleaning.csv` | IQR fences and outlier counts |
| `t09_data_dictionary.csv` | Data dictionary |
| `t10_split_summary.csv` | Train/test sizes and churn rates |
| `outputs/figures/fig01_…png` | Missing and invalid values before cleaning |
| `outputs/figures/fig02_…png` | Boxplots before and after cleaning |
| `data/processed/telco_churn_clean.csv` | Cleaned dataset (7,043 × 21) |

---

## Phase 3 – Exploratory Data Analysis (notebook section 4)

### What we did and why
- **What:** we described all 7,043 customers and compared churn rates across groups.
  - **Churn rate** = churners ÷ customers in a group.
  - Every chart shows the group size (n), so small groups are not over-interpreted.
- **Why:** to understand the data, form hypotheses for the statistical tests, and later check whether the model's important features make business sense.
- **How:**
  - `pd.cut` creates tenure groups (0–12, 13–24, 25–48, 49–72 months), exactly as in the Pandas tutorial.
  - `groupby().agg()` computes churn rates.
  - All plotting code lives in `src/eda_plots.py`, so every chart has the same style.

### Key findings (all are associations, not proof of cause)

| Pattern | Numbers |
|---|---|
| Class balance | 26.5% churned (1,869 of 7,043) |
| Contract | Month-to-month **42.7%** · one year 11.2% · two year 2.8% |
| Tenure | First year **47.5%** · 49–72 months 9.5% · median tenure of churners 10 vs 38 months |
| Internet | Fiber **41.9%** · DSL 19.0% · none 7.4% |
| Payment | Electronic check **45.3%** · other methods 15–19% |
| Add-ons (internet customers) | 0 add-ons 52.2% → 6 add-ons 5.3%; online security / tech support show the largest differences |
| Demographics | Senior 41.7% vs 23.6% · no partner 33.0% vs 19.7% · gender 26.9% vs 26.2% (no difference) |
| Monthly charges | Churners' median 79.6 vs 64.4 USD, but the pattern across price bands is uneven because bands correspond to service types |

### Concepts to be able to explain

**1. Why not just say "expensive plans cause churn"?**
- The ≤30 USD band is 92% customers without internet, while the >90 USD band is 97% fiber.
- Monthly charge and service type are tangled together, so we cannot separate a "price effect" from a "fiber effect" with this data.

**2. Confounding (Figure 8 and the add-on caution)**
- Month-to-month customers are also newer. To check whether contract type matters beyond tenure, we compared contracts *within* each tenure group.
- Result: month-to-month churn is far higher at every tenure level, so both carry information.
- Add-ons may partly reflect tenure, because long-standing customers have had time to add services.

**3. A single correlation can hide subgroup patterns**
- Overall, add-on count barely correlates with churn (−0.06).
- Among internet customers, however, churn falls from 52% to 5% as add-ons increase.
- The customers without internet (0 add-ons, low churn) mask the pattern. This is a classic example of why EDA needs more than one summary number.

**4. Why Spearman correlation?** It works on ranks, so it does not assume normal distributions or straight-line relationships. That suits skewed variables like TotalCharges.

**5. Categorical variables in the correlation matrix (Figure 7)**
- A correlation needs numbers, so each categorical variable is first one-hot encoded into 0/1 columns, the same encoding the models use. That gives 28 columns in total.
- Yes/No variables become one column (1 = Yes). "No internet service" and "No phone service" count as "No"; otherwise six add-on columns would be exact copies of `Internet: None`.
- Multi-level variables (Contract, InternetService, PaymentMethod) get one column per level, so each level's own link with churn is visible.
- A correlation with a 0/1 column is valid. Between two 0/1 columns it is the *phi coefficient*; with a numeric column it is a *point-biserial* correlation.
- Strongest links with churn: month-to-month **+0.41**, tenure **−0.37**, fiber optic **+0.31**, electronic check **+0.30**, two-year contract **−0.30**.
- Levels of the same variable are always negatively correlated (month-to-month vs two-year −0.62), because a customer can be in only one level. That is not a finding.

### Likely lecturer questions
- *"Is fiber optic causing churn?"*
  > "We can only say fiber customers churn more. They also pay more and may differ in other ways. Testing a cause would need an experiment or more data, such as service-quality records."
- *"Can categorical variables go into a correlation matrix?"*
  > "Yes, after one-hot encoding into 0/1 columns, the same encoding as the models. A correlation between 0/1 columns is the phi coefficient. We did not use label codes like 0/1/2/3, because they invent an order. The formal tests are chi-square with Cramér's V."
- *"Why did you include gender?"*
  > "To check it. It shows no meaningful difference (26.9% vs 26.2%), which is also reassuring from a fairness point of view."

### Files produced in Phase 3

| File | Contents |
|---|---|
| `fig03_churn_distribution.png` | Churn counts and percentages |
| `fig04_numeric_distributions_by_churn.png` | Tenure, monthly and total charge histograms by churn |
| `fig05_churn_rate_key_segments.png` | Churn rate by contract, internet, payment and tenure group |
| `fig06_monthly_charges_vs_churn.png` | Boxplot and churn rate by monthly charge band |
| `fig07_correlation_heatmap.png` | Spearman correlation heatmap of all 28 encoded variables (`t48`; numeric part in `t13`) |
| `fig08_churn_contract_by_tenure.png` | Contract × tenure churn heatmap |
| `fig09_addon_services_churn.png` | Add-on services and churn |
| `fig10_customer_attributes_churn.png` | Demographic and billing attributes |
| `t11`–`t16` (CSV) | Numeric summary by churn, charge bands with service mix, correlations, contract × tenure, churn rate for every category, customer profile |

---

## Phase 4 – Statistical analysis (notebook section 5)

### What we did and why

| Question | Test | Effect size | Why this test |
|---|---|---|---|
| Is a category linked to churn? | Chi-square test of independence | Cramér's V | Both variables are categorical; compares observed vs expected counts |
| Do churners differ on a numeric variable? | Mann-Whitney U | Rank-biserial r | Tenure and charges are not normally distributed (U-shaped, skewed), so we compare ranks instead of means |
| Do numeric variables move together? | Spearman ρ | ρ itself | Rank-based, so no normality or straight-line assumption |
| Are predictors redundant? | VIF | VIF value | Multicollinearity makes logistic regression coefficients unstable |

### Key concepts to explain

**1. Statistical significance vs practical importance**
- With 7,043 customers almost everything has p < 0.001. The p-value only says "probably not chance".
- The **effect size** says "how big". Cohen's guide: 0.10 small, 0.30 medium, 0.50 large.
- **Example:** paperless billing is highly significant (p < 0.001), but its effect is small (V = 0.19). Contract type has a medium effect (V = 0.41).

**2. Bonferroni correction**
- We ran 19 tests. At α = 0.05, about 1 in 20 tests could look "significant" by chance.
- Bonferroni requires p < 0.05/19 ≈ 0.0026 instead.
- After the correction, gender, phone service and multiple lines are **not** significant.

**3. Chi-square assumptions**
- Independent observations: guaranteed by one row per customer after duplicates were removed.
- Expected cell counts of at least 5: the smallest is 181, so the assumption is comfortably met.

**4. The "No internet service" trap**
- Streaming add-ons look associated with churn across all customers (V = 0.23).
- Among internet customers only, the association is negligible (V = 0.04).
- The apparent effect came from the "No internet service" level, which simply repeats InternetService = No.
- Protective add-ons remain meaningful: online security V = 0.28, tech support V = 0.27.

**5. Multicollinearity and VIF: the most important finding of this phase**
- **MonthlyCharges is almost exactly the sum of the service prices.** The service columns explain 99.9% of it, giving a VIF of about 836.
- **TotalCharges ≈ tenure × MonthlyCharges.** Its VIF is about 10, and its correlation with tenure is 0.89.
- **Decision:** logistic regression drops these two columns. Every VIF is then below 3, so the coefficients can be interpreted. The trees keep all features.
- The modelling phase will verify that the reduced model loses no predictive power.

**6. Why scaling does NOT fix multicollinearity**
- VIF is based on correlations between predictors, and re-scaling a variable does not change its correlations.
- We showed this in a table: with an intercept, the raw and standardised VIFs are identical.
- The previous-year report claimed scaling reduced VIF. That can only happen if VIF is computed *without an intercept*, which is a coding mistake.

### Hypothesis results

| | Hypothesis | Result |
|---|---|---|
| H1 | Contract type associated with churn | Supported: V = 0.41, the largest categorical effect |
| H2 | Shorter tenure associated with churn | Supported: r = −0.48, the largest numeric effect |
| H3 | Fiber optic and electronic check churn more | Supported: V = 0.32 and V = 0.30 |
| H4 | Protective add-ons associated with lower churn; streaming not | Supported |
| H5 | Higher monthly charges associated with churn | Partly supported: small effect (r = 0.24), non-linear, tied to service type |
| H6 | No gender association | Supported: V = 0.01, adjusted p = 1.0 |
| H7 | TotalCharges is redundant | Supported: VIF ≈ 10, ρ = 0.89 with tenure |

### Likely lecturer questions
- *"Why Mann-Whitney instead of a t-test?"*
  > "Tenure is U-shaped and TotalCharges is skewed, so the normality assumption behind the t-test is doubtful. Mann-Whitney compares ranks and needs no normality assumption."
- *"Did you check multicollinearity?"*
  > "Yes, with VIF on the training set. MonthlyCharges had a VIF over 800 because the services determine the bill. We removed MonthlyCharges and TotalCharges from the logistic regression, after which all VIFs are below 3. We also showed that scaling does not change VIF."
- *"Do these tests prove that contracts cause churn?"*
  > "No. They show association only. Each test also looks at one variable at a time, which is why we follow up with multivariable models."

### Files produced in Phase 4

| File | Contents |
|---|---|
| `fig11_effect_sizes.png` | Cramér's V and rank-biserial effect sizes |
| `t17_chi_square_tests.csv` | Chi-square tests for all 16 categorical variables |
| `t18_chi_square_addons_internet_only.csv` | Add-on tests among internet customers |
| `t19_mann_whitney_tests.csv` | Mann-Whitney tests for the numeric variables |
| `t20_spearman_tests.csv` | Spearman correlations with p-values |
| `t21_vif_numeric.csv` – `t24_vif_reduced_design.csv` | VIF analysis, including the scaling demonstration |
| `t25_hypothesis_summary.csv` | H1–H7 results |

---

## Phase 5 – Model development and evaluation (notebook section 6)

### The protocol, in one sentence
> "We tuned each model with 5-fold cross-validation on the training set, selected one using a rule we fixed in advance, and then evaluated all models once on the untouched test set."

### The models, simply explained

| Model | How it works | Strength | Weakness |
|---|---|---|---|
| Dummy | Always predicts "No churn" | A floor to beat | Catches no churners |
| Logistic Regression | Adds up weighted features and converts the total into a probability with the S-shaped logistic function | Transparent: each feature gets an odds ratio | Only captures effects the features express directly; interactions must be added by hand |
| Decision Tree | Asks a series of yes/no questions ("Contract = month-to-month?") and ends in groups with a churn rate | Rules are easy to read | Unstable and overfits if it grows too deep |
| Random Forest | Averages 300 trees, each built on a random sample of customers and features | Captures non-linear patterns and interactions; robust | A "black box"; harder to explain |

**Why not linear regression (from the tutorial)?** Churn is Yes/No. Linear regression can predict values below 0 or above 1. Logistic regression always outputs a valid probability between 0 and 1.

### Key results

| | CV ROC-AUC (train) | Test ROC-AUC | Test recall | Test precision | Test F1 |
|---|---|---|---|---|---|
| Dummy | 0.500 | 0.500 | 0 | 0 | 0 |
| **Logistic Regression (selected)** | **0.845** | **0.839** | **0.778** | **0.510** | **0.616** |
| Decision Tree | 0.831 | 0.836 | 0.794 | 0.526 | 0.633 |
| Random Forest | 0.849 | 0.845 | 0.791 | 0.534 | 0.638 |

### Concepts you must be able to explain

**1. Why not accuracy?** The dummy model is 73.5% accurate and useless. Our models are only about 74–76% accurate, but they catch 78–79% of churners instead of none.

**2. Precision vs recall in business terms**
- **Recall 0.78:** we catch 78 of every 100 customers who will leave.
- **Precision 0.51:** about half of the customers we flag will really leave. The other half receive an offer they did not need.
- A **false negative** (missed churner) means lost revenue. A **false positive** means a wasted retention offer.

**3. ROC-AUC = 0.84** means that if you pick one churner and one loyal customer at random, the model gives the churner a higher risk score 84% of the time. It does not depend on any threshold.

**4. Cross-validation:** the training set is split into 5 parts. The model trains on 4 and is scored on the 5th, five times, and the scores are averaged. This gives a more reliable estimate and leaves the test set untouched.

**5. Grid search:** try every combination of settings (e.g. tree depth 3, 4, 5, 6, 8) and keep the one with the best average CV score. It is done **on training data only**.

**6. Why choose Logistic Regression when Random Forest scored slightly higher?**
- Our rule, fixed in advance: highest CV ROC-AUC, but if another model is within 0.01, prefer the more interpretable one.
- Random Forest was ahead by only 0.004 in CV.
- On the test set, the paired bootstrap difference was +0.006, with a 95% CI of −0.001 to +0.014. The interval **includes zero**, so the advantage is not statistically reliable.
- Logistic regression also shows almost no overfitting and gives odds ratios managers can understand.

**7. Class weighting**
- `class_weight="balanced"` makes each churner count about 3× more during training.
- It did **not** change ROC-AUC (0.845 both ways).
- It moved the default decision point: recall rose from 0.54 to 0.79, and precision fell from 0.65 to 0.52.
- We used it for all models so that their 0.5 thresholds are comparable.

**8. Overfitting**
- Random Forest scored 0.890 on its training folds but 0.849 on validation folds, a moderate gap. Logistic regression's gap was only 0.003.
- For every model, test AUC was within 0.006 of the CV AUC, so all of them generalise to new customers.

**9. Threshold**
- The model gives a probability. The threshold decides who gets flagged.
- At 0.50, we flag 571 of 1,409 test customers and catch 291 of 374 churners.
- A higher threshold means fewer offers but more missed churners.
- The best F1 was at 0.55, which is practically the same as 0.50, so we keep 0.50. The threshold is a **business decision** based on budget and on the cost of losing a customer.

**10. Bootstrap confidence interval:** resample the 1,409 test customers 1,000 times and recompute AUC each time. The middle 95% gives 0.815–0.859 for logistic regression. This shows how much the score could vary with a different sample of customers.

### Likely lecturer questions
- *"Why is your accuracy only 74%?"*
  > "Because we deliberately favour catching churners. A model that never predicts churn is 73.5% accurate but useless. We judge models by ROC-AUC, recall and F1."
- *"How do you know there is no data leakage?"*
  > "All preprocessing is inside a Pipeline, so it is re-fitted on each training fold. Grid search and threshold choice used training data only. The test set was used once at the end."
- *"Isn't Random Forest better?"*
  > "Slightly on the test set, but the difference (0.006 AUC) is within the uncertainty: the bootstrap CI includes zero. We chose logistic regression by a rule set in advance because it is interpretable and barely overfits."

### Files produced in Phase 5

| File | Contents |
|---|---|
| `fig12_model_comparison.png` | All test metrics, all models |
| `fig13_confusion_matrices.png` | Confusion matrices (actual vs predicted) |
| `fig14_roc_pr_curves.png` | ROC and precision-recall curves |
| `fig15_threshold_tradeoff.png` | Precision, recall and F1 vs threshold |
| `t26_class_weight_check.csv`, `t27_lr_feature_set_check.csv` | Design checks |
| `t28_cv_results.csv` | Cross-validation results |
| `t29_test_metrics.csv` | Test metrics |
| `t30`, `t31` | Bootstrap confidence intervals and paired differences |
| `t32_overfitting_check.csv` | Train vs validation vs test AUC |
| `t33`, `t34` | Threshold analysis |
| `outputs/models/final_churn_model.joblib` + `final_model_metrics.json` | Saved final model |

---

## Phase 6 – Interpretation and recommendations (notebook sections 7–8)

### Three ways of asking "what drives churn?"

| Tool | What it tells you | Key result |
|---|---|---|
| **Odds ratios** (logistic regression) | Direction and size of each feature's link with churn, for otherwise identical customers | Two-year contract OR 0.25; +25 months of tenure OR 0.47; fiber optic OR 2.46; electronic check OR 1.50; online security OR 0.68 |
| **Permutation importance** | How much the test ROC-AUC drops if a column is scrambled | Both models rely mainly on tenure, internet service and contract |
| **Shallow decision tree** | Readable if/then rules | Month-to-month + fiber + first 13 months → 71% churn; two-year contract with bill ≤ 93.7 USD → 1% |

**How to read an odds ratio**
- OR = 1 means no difference.
- OR = 0.25 means a quarter of the odds.
- OR = 2.46 means about 2.5 times the odds.
- If the 95% interval crosses 1 (grey in the figure), the direction is uncertain.

**Why some results changed between the single-variable tests and the model**
- **Senior citizen:** significant on its own, but not once contract, tenure and services are included. Seniors simply have riskier contract and service profiles.
- **Streaming:** shows *higher* churn odds in the model. Because MonthlyCharges was removed, paid add-ons stand in for the price a customer pays.

**Why permutation importance rather than the random forest's built-in importance?** The built-in (impurity) importance favours continuous variables such as charges (Strobl et al., 2007). Permutation importance measures the real drop in performance.

**Cumulative gains (the business result)**
- If the company contacts customers from highest to lowest predicted risk:
  - the top 20% contains 50% of all churners
  - the top 30% contains 64%
- In the riskiest 10%, 75% of customers really churn.

**Error analysis:** missed churners look like loyal customers (median tenure 30 months), so they probably leave for reasons that are not in the data.

**Probability note:** balanced class weighting makes predicted probabilities higher than reality, so use the scores to **rank** customers.

---

## Phase 7 – The report (`reports/Telco_Churn_Report.docx`)

- **Generated, not typed by hand.** It is produced by `reports/build_report.py`, which reads every number from `outputs/tables`. If the analysis changes, re-running the script updates the report, so the text can never contradict the notebook.
- **Structure:**
  - cover and contents
  - Introduction
  - Methodology
  - EDA
  - Statistical and ML analysis
  - Results and discussion
  - Conclusion and recommendations
  - References
- **Contents:** 9 figures and 5 tables.
- **Length:** estimated at about 14 pages, under the strict 15-page limit including cover and references.
- **Before submitting:** follow the "Before submitting" steps in the README (cover details, updating the contents page in Word, exporting the PDF, checking the page count).

## Phase 8 – The presentation and final checks

- **Slides.** `presentation/Telco_Churn_Presentation.pptx` has 14 slides with a script in the speaker notes. The bar and line charts on slides 5 and 11 are native PowerPoint charts, so they can be edited.
- **Speaker guide.** `presentation/speaker_guide.md` assigns slides to Presenters A–E and gives timings for full (~23 min) and 15-minute versions. It also lists likely questions with model answers and a table of how we improved on the senior report's methods.
- **Final checks.** `verify_project.py` runs 20 checks:
  - the raw file is unchanged
  - cleaning is reproducible
  - there is no train/test overlap
  - the saved model reproduces the reported metrics
  - the report and slides state the correct numbers
  - every slide has notes

---

## Comparison with the previous-year ("Data Busters") report
The report now follows their chapter layout: 1 Introduction · 2 Data Cleaning (2.1–2.4) · 3 EDA (3.1–3.5.2, plus 3.6 statistical tests) · 4 Model Fitting (4.1–4.4.3) · 5 Conclusion · References, with a Table of Contents, Table of Figures and Table of Tables.

Every element of last year's example report is also in our project, usually in a stronger form. Use this table if the lecturer asks *"did you do X like the example report?"*

| Their report | Where it is in our project | Note |
|---|---|---|
| Cover with university crest, member table | Report cover and slide 1 (crest from `assets/uom_crest.png`) | Fill in names and index numbers |
| Table of contents, roman page numbers for front matter | Report pages i–ii, body numbered from 1 | Lists of figures/tables omitted to stay within 15 pages |
| Dataset overview, context, objectives | Report §1, notebook header | — |
| Missing-value table with % and total | Notebook §1.4 (`t01`, with `pct_of_rows`), §1.9 total: 223 problem values = 0.15% of all data points (report Table 1) | We also catch hidden blanks and invalid labels that `isnull()` misses |
| Mean/mode imputation | Row-wise repair + median/mode inside the Pipeline | Ours avoids leakage |
| IQR outliers + boxplots | Report §2.2, Figure 2 (`fig02`); IQR table `t08` in notebook | — |
| Winsorisation | Deliberately **not** used; explained in §2.4 | Impossible values repaired; real extremes kept |
| Label encoding | One-hot encoding | Label codes would impose a false order |
| VIF before/after charts | Report §2.4, Figure 3 (`fig20`) | Remedy = removing redundant columns, not scaling |
| Summary statistics table | Report §3.1, Table 3 (describe-style, `t11`) + Table 4 (churner profile) | — |
| Histograms + distribution-type table | Report §3.2, Figure 4 + Table 5 (same 4 columns: variable / observation / distribution type / insight) | Shapes computed from the data |
| Target variable analysis | Report §3.3, Figure 5 (`fig03`) | — |
| Categorical analysis by group | Report §3.4, Figures 6–7 (`fig05`, `fig10`); add-ons `fig09` in notebook | Ours adds chi-square significance and effect sizes in §4.1 |
| Correlation heatmap | Report §3.5.1, Figure 8 (`fig07`, full 28×28 Spearman, categories one-hot encoded) | Senior label-encoded the categories (false order); we use one 0/1 column per level |
| Scatter plots vs target | Report §3.5.2 / Figure 9 (`fig21`): churn rate per group vs tenure, monthly and total charges, with LOWESS trend lines (`t46`) | A raw scatter against a yes/no target shows only two lines of dots, so we plot each group's churn rate instead |
| Regression summary (coef, std err, p, CI) | Report §4.1, Table 6 (from `t44_logit_summary.csv`, statsmodels Logit) | 14 of 21 terms significant; odds ratios match our model (max difference 0.07) |
| Regression equation | Report §4.1.1 + full equation in `t43_logistic_equation.txt` | log-odds(churn) = −0.40 − 1.39·[two-year] + 0.90·[fiber] … |
| Random forest train vs test + feature importance | `t32` overfitting check, permutation importance (`fig17`) | — |
| Model comparison chart | Report §4.4.1, Figure 13 (`fig12`) + Table 8 | — |
| Actual vs predicted | Report §4.4.3: Figure 14 (`fig13`) + Table 9 (LR vs RF comparison); gains/deciles `fig19` in notebook | The classification versions of "actual vs predicted" |
| Conclusion, references | Report §6, 16 references | — |

**What McFadden's pseudo-R² means (≈ 0.285):** it is the logistic-regression counterpart of R², but on a different scale. Values of 0.2–0.4 indicate a good fit. Do **not** compare it with the senior report's linear-regression R² of 0.82: different models, different measures.
