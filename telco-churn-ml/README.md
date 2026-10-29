# Customer Churn Prediction and Analysis Using Machine Learning
**TM3221 – Data Science for BPM · Group Assignment · University of Moratuwa**

**Research question:** *Which customer characteristics and service-related factors are most strongly associated with customer churn, and how effectively can machine learning models predict churn?*

The dataset is the Telco Customer Churn data: 7,073 raw rows, which become 7,043 unique customers after cleaning, with 21 columns. The task is binary classification: predict `Churn` (Yes/No).

## Project structure
```
telco-churn-ml/
├─ data/
│  ├─ raw/Telco_Customer_Churn.csv      original file – never modified (SHA-256 checked)
│  └─ processed/telco_churn_clean.csv   output of the cleaning rules
├─ notebooks/telco_churn_analysis.ipynb main analysis – run this
│  └─ telco_churn_analysis_colab.ipynb  copy of the main notebook + 1 setup cell, self-contained for Google Colab
├─ src/                                 reusable, commented helper code
│  ├─ config.py                         paths, random seed, column lists, validity rules
│  ├─ data_cleaning.py                  loading, auditing, cleaning, data dictionary
│  ├─ eda_plots.py                      consistent figure style and EDA plots
│  ├─ stats_tests.py                    chi-square, Mann-Whitney, VIF with effect sizes
│  ├─ modeling.py                       split, leakage-safe Pipelines, tuning, evaluation
│  ├─ model_plots.py                    model evaluation and interpretation figures
│  └─ interpretation.py                 odds ratios, permutation importance, gains, tree rules
├─ outputs/
│  ├─ figures/                          300-dpi PNG figures (fig01 – fig21)
│  ├─ tables/                           CSV/text tables (t01 – t47) – every number in the report comes from here
│  └─ models/                           final_churn_model.joblib + final_model_metrics.json
├─ reports/
│  ├─ build_report.py                   generates the Word report from outputs/
│  └─ Telco_Churn_Report.docx           academic report
├─ presentation/
│  ├─ build_presentation.py             generates the slides from outputs/
│  ├─ Telco_Churn_Presentation.pptx     14 slides with speaker notes (Presenters A–E)
│  └─ speaker_guide.md                  who presents what, timing, likely questions
├─ assets/uom_crest.png                 university crest (extracted from the previous-year example report) for the cover and title slide
├─ tools/build_colab_notebook.py        builds the single-file Colab notebook
├─ verify_project.py                    final quality-assurance checks
├─ docs/study_guide.md                  plain-English explanation of every step
├─ requirements.txt
└─ README.md
```

## Setup (one time)
You need **Python 3.10** (3.10–3.12 recommended).

**macOS / Linux**
```bash
cd telco-churn-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd telco-churn-ml
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the analysis
Interactively:
```bash
jupyter notebook notebooks/telco_churn_analysis.ipynb
```
Then use **Kernel → Restart & Run All**.

From the command line, running every cell from a clean state:
```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace telco_churn_analysis.ipynb
```

### Running on Google Colab (one file only)
Upload **only** `notebooks/telco_churn_analysis_colab.ipynb` (about 3 MB) to Colab or Google Drive, open it, and choose **Runtime → Run all**.
- It is an exact copy of the main notebook, including all saved outputs, with **one extra setup cell** below the title. That cell is collapsed and recreates the helper code (`src/`) and the dataset inside the Colab session, so no folder upload and no Drive mount is needed.
- Results are written to `/content/telco-churn-ml/outputs/`. Download them from the 📁 Files panel; they disappear when the runtime stops.
- Colab's library versions differ slightly from `requirements.txt`, so a few decimals may differ from the report.
- If you change the main notebook or `src/`, regenerate the Colab file with `python tools/build_colab_notebook.py`.
- Never upload `.venv/`: it is 677 MB of locally installed libraries that Colab already provides.

Running the notebook regenerates everything in `outputs/` and `data/processed/` (about 2 minutes). All randomness uses `random_state=42`, so the results are identical on every run.

Then rebuild the documents and run the checks:
```bash
python reports/build_report.py               # -> reports/Telco_Churn_Report.docx
python presentation/build_presentation.py    # -> presentation/Telco_Churn_Presentation.pptx
python verify_project.py                     # 33 PASS/FAIL checks
```

### Before submitting
1. **Fill in the cover details.** Open the report and the slides and replace `[Group name]`, `[Member name]`, `[Index no.]` and `[Submission date]`. Rename "Presenter A–E" in the slide footers and notes.
2. **Update the contents table.** Open the report in **Microsoft Word** and click **Yes** when asked to update fields, or right-click the table of contents and choose **Update Field**. This adds page numbers.
3. **Export the PDF:** File → Save As → PDF.
4. **Check the page count.** The limit is 15 pages including the cover, contents and references. The build script estimates about 14 pages, but exact pagination depends on Word.
5. **Confirm the dataset source.** The report cites the public Kaggle/IBM version. If your copy came from somewhere else, such as the course page, update the reference.

## Key methodological choices
- **Raw data preserved.** The original CSV is only read, never written. Its SHA-256 fingerprint is checked on every run.
- **Row-wise cleaning before the split.** These rules use only the values in the same row, never statistics computed from other customers, so they cannot leak test-set information:
  - remove duplicates
  - turn invalid labels into missing values
  - repair impossible values using TotalCharges ≈ tenure × MonthlyCharges
- **Statistical steps fitted on training data only.** Imputation, scaling and one-hot encoding sit inside a scikit-learn `Pipeline` / `ColumnTransformer`, so they are fitted only on training data or on training folds during cross-validation.
- **Outliers.** Impossible values are treated as data errors. Legitimate extreme values are kept, with no automatic deletion or winsorisation.
- **Class imbalance.** About 26.5% of customers churned. This is handled with `class_weight="balanced"`, not SMOTE, and evaluated with ROC-AUC, precision, recall and F1, not accuracy alone.
