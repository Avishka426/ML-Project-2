# Speaker Guide – Customer Churn Prediction

The slides are in `Telco_Churn_Presentation.pptx`. Every slide has its full speaking script in the **speaker notes** (View → Notes in PowerPoint). This guide shows who presents what, the timing, and how to answer likely questions.

> **Before presenting:**
> - Replace `[Group name]` on slide 1 and in the report cover.
> - Rename "Presenter A–E" to your members' names. The presenter appears in the footer of every slide and at the top of each speaker note.
> - If you have fewer than five members, merge two presenter blocks.

## 1. Who presents what

| Presenter | Slides | Topic | Full version | 15-min version |
|---|---|---|---|---|
| **A** | 1–3 | Title, problem & objectives, dataset | 4 min | 2.5 min |
| **B** | 4–6 | Data cleaning, who churns, careful patterns | 5.5 min | 3.5 min (shorten slide 6) |
| **C** | 7–8 | Statistical tests, ML method | 4 min | 3 min |
| **D** | 9–11 | Model comparison, drivers, business value | 5.5 min | 3.5 min |
| **E** | 12–14 | Recommendations, limitations, conclusion | 4.5 min | 2.5 min |
| | | **Total** | **~23 min** | **~15 min** |

**If the slot is 10 minutes:**
- Present slides 1, 3, 4, 5, 9, 10, 11, 12 and 14.
- Mention the statistical tests and limitations in one sentence each.

## 2. The story in one minute (everyone should know this)

1. **Data:** 7,073 rows of telecom customer data contained duplicates, blanks, impossible values and invalid labels. We audited, repaired and documented them. We deleted nothing except 30 exact duplicates, leaving 7,043 customers. 26.5% churned.
2. **Patterns:**
   - Churn is highest for month-to-month contracts (42.7%), first-year customers (47.5%), fiber-optic customers (41.9%) and electronic-check payers (45.3%).
   - Gender makes no difference.
3. **Statistics:** chi-square and Mann-Whitney tests with effect sizes show that contract (V = 0.41) and tenure (r = −0.48) have the strongest associations.
4. **Models:**
   - Logistic regression, decision tree and random forest all reach a test ROC-AUC of about 0.84 and catch about 78% of churners.
   - We chose logistic regression by a pre-set rule. The random forest's tiny advantage is not statistically reliable, and logistic regression is easier to explain.
5. **Value:** contacting the riskiest 20% of customers reaches 50% of all churners.
6. **Caution:** these are associations, not proof of cause. Offers should be tested with randomised trials.

## 3. Likely lecturer questions and model answers

**Data and cleaning**
- **"Why didn't you just delete rows with missing values?"**
  > "Only 2.7% of customers were affected, but deleting them could bias the data – their churn rate was 31.7% versus 26.4%. Instead we repaired the numeric errors with the billing identity TotalCharges ≈ tenure × MonthlyCharges. That is about 2% off, compared with about 75% for median filling."
- **"How did you treat outliers?"**
  > "We detected them with the IQR rule and boxplots. The extreme values, like a 988-month tenure or a 9,960 USD monthly bill, were impossible, so we treated them as errors and repaired them. After repair there were zero IQR outliers, so no real customer was removed or capped."
- **"What is data leakage and how did you prevent it?"**
  > "Leakage is when information from the test set influences training, which makes results look better than they are. Our cleaning rules use only each customer's own row. Imputation, scaling and encoding are inside a scikit-learn Pipeline fitted on training data only. Tuning used cross-validation on the training set, and the test set was used once."

**Statistics**
- **"Why Mann-Whitney and not a t-test?"**
  > "Tenure is U-shaped and TotalCharges is skewed, so normality is doubtful. Mann-Whitney compares ranks and does not need normality."
- **"Everything is significant – so what?"**
  > "With 7,000 customers even tiny differences are significant, so we report effect sizes. Contract has a medium effect (V = 0.41). Demographics are significant but small. Gender has no effect at all."
- **"What did you do about multicollinearity?"**
  > "VIF showed that MonthlyCharges is almost exactly the sum of the service prices, with a VIF of about 836, and that TotalCharges overlaps with tenure. We removed both from the logistic regression, after which all VIFs are below 3. Cross-validation showed almost no loss. Scaling does not fix multicollinearity: we checked, and the VIFs were identical."

**Modelling**
- **"Your accuracy is only 74% – isn't that bad?"**
  > "A model that always predicts 'no churn' is 73.5% accurate and useless. We optimise for catching churners: our model catches 78% of them, compared with 0% for the baseline. ROC-AUC, which is threshold-free, is 0.84."
- **"Why not the random forest, since it scored higher?"**
  > "Its advantage was 0.004 AUC in cross-validation and 0.006 on the test set. A paired bootstrap interval for the difference (−0.001 to 0.014) includes zero. Our rule, set in advance, prefers the interpretable model when the gap is under 0.01."
- **"What does class_weight='balanced' do?"**
  > "It makes each churner count about 3 times more during training. It didn't change ROC-AUC, but it raised recall from 0.54 to 0.79 at the default threshold."
- **"How did you choose the threshold?"**
  > "Using cross-validated predictions on the training set only. The F1-optimal threshold was 0.55, which performed the same as 0.50, so we kept 0.50. In practice the company would set it according to its retention budget."
- **"What does an odds ratio of 0.25 mean?"**
  > "For two otherwise identical customers, the one on a two-year contract has about a quarter of the odds of churning compared with a month-to-month customer."

**Interpretation and business**
- **"Does fiber optic cause churn?"**
  > "We can't say. Fiber customers churn more even after controlling for contract and tenure, which points to price or service quality. Proving a cause would need an experiment or more data."
- **"Which churners does the model miss?"**
  > "Customers who look loyal: a median tenure of 30 months, and only 55% on month-to-month contracts. They probably leave for reasons we don't observe, such as competitor offers or service problems."
- **"How would a company use this?"**
  > "Score customers monthly, contact the highest-risk group first, size the list to the budget, test offers with A/B trials, monitor performance, and retrain regularly. And never use gender or age to give worse terms."

## 4. Where the senior report's methods were improved
Useful if the lecturer compares the work with last year's example.

| Senior report practice | What we did instead | Why |
|---|---|---|
| Imputed and winsorised before the train/test split | Row-wise repairs only; learned imputation inside a training-only Pipeline | Prevents data leakage |
| Winsorised every numeric column at the 5th/95th percentiles | Detected outliers; repaired impossible values; kept real extremes | Winsorising distorts genuine values |
| Label-encoded nominal variables (0, 1, 2…) | One-hot encoding | Label codes impose a false order |
| Claimed scaling reduced VIF | Showed that VIF is unchanged by scaling; removed redundant variables | VIF depends only on correlations |
| Causal wording ("stress leads to…") | Associational wording throughout | Observational data cannot show causation |
| Single split, no baseline, overfitting not discussed | Baseline, 5-fold CV, train/validation/test gaps, bootstrap confidence intervals | Honest, reproducible evaluation |
| Impurity-based feature importance read as effect size | Permutation importance plus odds ratios | Impurity importance is biased |
