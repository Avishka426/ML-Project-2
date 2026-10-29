"""
eda_plots.py
------------
Plotting helpers that give every figure the same clean, report-ready style.

All figures are saved at 300 dpi to outputs/figures so they can be placed
directly into the Word report and the PowerPoint slides.
"""

import matplotlib.pyplot as plt
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Colour palette (colour-blind-safe reference palette, fixed order)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"     # slot 1 - "Stayed" / single-series bars
ORANGE = "#eb6834"   # slot 2 - "Churned"
AQUA = "#1baf7a"     # slot 3
YELLOW = "#eda100"   # slot 4
GREY = "#9a9893"     # neutral reference lines
TEXT = "#2b2b2a"     # primary text
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"

CHURN_COLORS = {"No": BLUE, "Yes": ORANGE}
CHURN_LABELS = {"No": "Stayed", "Yes": "Churned"}


def set_style():
    """Apply one consistent Matplotlib style for all figures."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": GREY,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "text.color": TEXT,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def save_fig(fig, filename):
    """Save a figure as a 300-dpi PNG in outputs/figures and return its path."""
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURE_DIR / filename
    fig.savefig(path)
    return path


# ---------------------------------------------------------------------------
# Phase 2 figures - data quality
# ---------------------------------------------------------------------------
def plot_data_issues(issue_counts):
    """
    Horizontal stacked bar: number of problem values per column in the raw
    file, split by type of problem.

    issue_counts : DataFrame indexed by column name with the columns
                   'Blank', 'Invalid text / label', 'Impossible value'.
    """
    kinds = [("Blank", BLUE), ("Invalid text / label", ORANGE), ("Impossible value", AQUA)]
    data = issue_counts.loc[issue_counts.sum(axis=1) > 0].copy()
    data = data.loc[data.sum(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    left = pd.Series(0.0, index=data.index)
    for kind, colour in kinds:
        ax.barh(data.index, data[kind], left=left, color=colour, label=kind,
                height=0.6, edgecolor="white", linewidth=1.5)
        left += data[kind]
    for name, total in left.items():
        ax.text(total + 1.5, name, f"{int(total)}", va="center", fontsize=9, color=TEXT)

    ax.set_xlabel("Number of affected values (raw file, 7,073 rows)")
    ax.set_title("Missing and invalid values before cleaning")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, left.max() * 1.15)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_before_after_boxplots(before, after, columns=config.NUMERIC_FEATURES):
    """
    Boxplots of each numeric column before (top row) and after (bottom row)
    cleaning. The raw values are shown on the same kind of axis so the
    impossible values (e.g. tenure of 988 months) are clearly visible.
    """
    units = {"tenure": "months", "MonthlyCharges": "USD", "TotalCharges": "USD"}
    fig, axes = plt.subplots(2, len(columns), figsize=(9.5, 4.2))
    box_style = dict(
        vert=False, widths=0.5, patch_artist=True,
        boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE, linewidth=1),
        medianprops=dict(color=BLUE, linewidth=2),
        whiskerprops=dict(color=BLUE, linewidth=1),
        capprops=dict(color=BLUE, linewidth=1),
        flierprops=dict(marker="o", markersize=5, markerfacecolor=ORANGE,
                        markeredgecolor="white", markeredgewidth=0.6, alpha=0.9),
    )
    for j, col in enumerate(columns):
        for i, (label, frame) in enumerate([("Before", before), ("After", after)]):
            ax = axes[i, j]
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            ax.boxplot(values, **box_style)
            ax.set_yticks([])
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{label}: {col}", fontsize=10)
            ax.set_xlabel(units[col])
    fig.suptitle("Numeric variables before and after cleaning (outlier check)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Phase 3 - exploratory data analysis
# ---------------------------------------------------------------------------
TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0–12", "13–24", "25–48", "49–72"]


def add_eda_columns(df):
    """
    Add helper columns used only for EDA (never as model inputs):
    churn as 0/1, tenure groups (pd.cut binning, as in the Pandas tutorial)
    and the number of internet add-on services a customer has.
    """
    out = df.copy()
    out["churn_flag"] = (out[config.TARGET] == "Yes").astype(int)
    out["tenure_group"] = pd.cut(out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    out["n_addons"] = (out[config.ADDON_SERVICES] == "Yes").sum(axis=1)
    return out


def churn_rate_table(df, column):
    """Customers, churners and churn rate (%) for each level of one column."""
    table = (df.groupby(column, observed=True)["churn_flag"]
             .agg(customers="size", churned="sum", churn_rate="mean"))
    table["churn_rate"] = table["churn_rate"] * 100
    table["share_of_customers"] = table["customers"] / table["customers"].sum() * 100
    return table


def all_churn_rates(df, columns):
    """Stack churn_rate_table() for many columns into one long table."""
    parts = []
    for col in columns:
        t = churn_rate_table(df, col).reset_index().rename(columns={col: "level"})
        t.insert(0, "variable", col)
        t["level"] = t["level"].astype(str)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def customer_profile_table(df):
    """Key characteristics of all customers, those who stayed and those who churned."""
    groups = {"All customers": df,
              "Stayed": df[df["churn_flag"] == 0],
              "Churned": df[df["churn_flag"] == 1]}
    def share(g, col, value):
        """Percentage of customers with `value` in `col` (missing values excluded)."""
        return (g[col].dropna() == value).mean() * 100

    rows = {}
    for name, g in groups.items():
        rows[name] = {
            "Customers": len(g),
            "Share of all customers (%)": len(g) / len(df) * 100,
            "Median tenure (months)": g["tenure"].median(),
            "Mean monthly charge (USD)": g["MonthlyCharges"].mean(),
            "Median total charges (USD)": g["TotalCharges"].median(),
            "Month-to-month contract (%)": share(g, "Contract", "Month-to-month"),
            "Fiber optic internet (%)": share(g, "InternetService", "Fiber optic"),
            "Pays by electronic check (%)": share(g, "PaymentMethod", "Electronic check"),
            "Paperless billing (%)": share(g, "PaperlessBilling", "Yes"),
            "Senior citizen (%)": share(g, "SeniorCitizen", "Yes"),
            "Has partner (%)": share(g, "Partner", "Yes"),
            "Has dependents (%)": share(g, "Dependents", "Yes"),
            "Has tech support (%)": share(g, "TechSupport", "Yes"),
            "Mean number of add-ons": g["n_addons"].mean(),
        }
    return pd.DataFrame(rows)


LABEL_BOX = dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9)


def describe_shape(values, skew_cutoff=0.5, bins=20):
    """
    Describe the shape of a distribution from the data itself:
    count the peaks of a lightly smoothed histogram (edges included), then
    use skewness. Returns e.g. 'U-shaped', 'bimodal', 'right-skewed'.
    """
    import numpy as np
    counts, _ = np.histogram(values, bins=bins)
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    padded = np.concatenate([[0], smooth, [0]])
    peaks = [i - 1 for i in range(1, len(padded) - 1)
             if padded[i] > padded[i - 1] and padded[i] >= padded[i + 1] and padded[i] > 0.5 * smooth.max()]
    skew = pd.Series(values).skew()
    if len(peaks) >= 2 and peaks[0] <= 1 and peaks[-1] >= bins - 2:
        return "U-shaped (peaks at both ends)"
    if len(peaks) >= 2:
        return "bimodal (two peaks)"
    if skew > skew_cutoff:
        return "right-skewed"
    if skew < -skew_cutoff:
        return "left-skewed"
    return "roughly symmetric"


def distribution_summary(df, columns, units):
    """
    One row per numeric variable: descriptive statistics, the shape of the
    distribution and how churners differ (medians), all computed from the data.
    """
    rows = []
    for col in columns:
        x = df[col].dropna()
        med_churn = df.loc[df["churn_flag"] == 1, col].median()
        med_stay = df.loc[df["churn_flag"] == 0, col].median()
        direction = "higher" if med_churn > med_stay else "lower"
        rows.append({
            "variable": col, "unit": units[col], "mean": x.mean(), "median": x.median(), "sd": x.std(),
            "min": x.min(), "max": x.max(), "skewness": x.skew(), "shape": describe_shape(x),
            "churn_insight": f"churners {direction} (median {med_churn:,.1f} vs {med_stay:,.1f} {units[col]})",
        })
    return pd.DataFrame(rows)


def _reference_line(ax, value, orientation="h", legend_loc="upper right"):
    """Dashed line marking the overall churn rate, used as a visual benchmark."""
    label = f"Overall churn rate ({value:.1f}%)"
    if orientation == "h":
        ax.axhline(value, color=GREY, linewidth=1, linestyle="--", zorder=1, label=label)
    else:
        ax.axvline(value, color=GREY, linewidth=1, linestyle="--", zorder=1, label=label)
        # horizontal bar charts: the axis label explains the line (a legend box would collide)
        ax.set_xlabel(f"Churn rate (%)  ·  dashed line = overall churn rate ({value:.1f}%)")
        return
    ax.legend(loc=legend_loc, fontsize=8, handlelength=2.2)


def plot_churn_distribution(df):
    """Bar chart of customers who stayed vs churned, labelled with counts and percentages."""
    counts = df[config.TARGET].value_counts().reindex(["No", "Yes"])
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    labels = [CHURN_LABELS[k] for k in counts.index]
    ax.bar(labels, counts.values, color=[CHURN_COLORS[k] for k in counts.index], width=0.55)
    for i, (n, p) in enumerate(zip(counts.values, pct.values)):
        ax.text(i, n + counts.max() * 0.02, f"{n:,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_ylabel("Number of customers")
    ax.set_title("Customer churn distribution")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df):
    """Stacked histograms of tenure, monthly and total charges, split by churn status."""
    specs = [("tenure", "Tenure (months)", range(0, 75, 3)),
             ("MonthlyCharges", "Monthly charges (USD)", range(15, 125, 5)),
             ("TotalCharges", "Total charges (USD)", range(0, 9000, 300))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (col, label, bins) in zip(axes, specs):
        data = [df.loc[df[config.TARGET] == k, col] for k in ["No", "Yes"]]
        ax.hist(data, bins=list(bins), stacked=True, color=[BLUE, ORANGE],
                label=["Stayed", "Churned"], edgecolor="white", linewidth=0.6)
        ax.set_xlabel(label)
        ax.set_title(label.split(" (")[0])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Number of customers")
    axes[0].legend(loc="upper center")
    fig.suptitle("Distributions of tenure, monthly charges and total charges by churn status",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return fig


def _churn_rate_bars(ax, table, overall, title, xlabel=None, order=None, rotate=0):
    """Draw one churn-rate bar panel with value + n labels and an overall-rate line."""
    if order is not None:
        table = table.reindex(order)
    labels = [str(i) for i in table.index]
    ax.bar(labels, table["churn_rate"], color=ORANGE, width=0.6, zorder=2)
    for i, (rate, n) in enumerate(zip(table["churn_rate"], table["customers"])):
        ax.text(i, rate + 1.2, f"{rate:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=8,
                bbox=LABEL_BOX, zorder=3)
    _reference_line(ax, overall)
    ax.set_ylim(0, max(table["churn_rate"].max() * 1.38, overall * 1.4))
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)
    if rotate:
        ax.tick_params(axis="x", labelrotation=rotate)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")


def plot_key_segments(df):
    """2x2 panel: churn rate by contract, internet service, payment method and tenure group."""
    overall = df["churn_flag"].mean() * 100
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    _churn_rate_bars(axes[0, 0], churn_rate_table(df, "Contract"), overall,
                     "(a) Contract type", order=["Month-to-month", "One year", "Two year"])
    _churn_rate_bars(axes[0, 1], churn_rate_table(df, "InternetService"), overall,
                     "(b) Internet service", order=["DSL", "Fiber optic", "No"])
    pay = churn_rate_table(df, "PaymentMethod")
    pay.index = [s.replace(" (automatic)", "\n(automatic)") for s in pay.index]
    _churn_rate_bars(axes[1, 0], pay.sort_values("churn_rate", ascending=False), overall,
                     "(c) Payment method")
    _churn_rate_bars(axes[1, 1], churn_rate_table(df, "tenure_group"), overall,
                     "(d) Tenure group", xlabel="Months with the company")
    fig.suptitle("Churn rate by contract, internet service, payment method and tenure",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_monthly_charges_vs_churn(df):
    """(a) Boxplot of monthly charges by churn status; (b) churn rate by monthly-charge band."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    for i, k in enumerate(["No", "Yes"]):
        values = df.loc[df[config.TARGET] == k, "MonthlyCharges"]
        ax.boxplot(values, positions=[i], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="white", edgecolor=CHURN_COLORS[k], linewidth=1.4),
                   medianprops=dict(color=CHURN_COLORS[k], linewidth=2.2),
                   whiskerprops=dict(color=CHURN_COLORS[k]), capprops=dict(color=CHURN_COLORS[k]),
                   flierprops=dict(marker="o", markersize=3))
        ax.text(i + 0.3, values.median(), f"median\n{values.median():.1f}", va="center", fontsize=8.5)
    ax.set_xticks([0, 1], ["Stayed", "Churned"])
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylabel("Monthly charges (USD)")
    ax.set_title("(a) Monthly charges by churn status")
    ax.grid(axis="x", visible=False)

    bands = pd.cut(df["MonthlyCharges"], bins=[0, 30, 50, 70, 90, 120],
                   labels=["≤30", "30–50", "50–70", "70–90", ">90"])
    table = churn_rate_table(df.assign(charge_band=bands), "charge_band")
    _churn_rate_bars(axes[1], table, df["churn_flag"].mean() * 100,
                     "(b) Churn rate by monthly-charge band", xlabel="Monthly charges (USD)")
    fig.suptitle("Monthly charges and churn", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, table


def _diverging_cmap():
    """Blue - neutral grey - red diverging colour map (for correlations from -1 to +1)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "blue_grey_red", ["#104281", BLUE, "#f0efec", "#e34948", "#8f2222"])


def encode_for_correlation(df):
    """
    Numeric 0/1 version of the data for the correlation matrix (EDA only).

    * Yes/No variables become one 0/1 column (1 = Yes); gender becomes Male = 1.
    * "No internet service" / "No phone service" count as "No" (as in the
      model data), otherwise six add-on columns would duplicate "Internet: None".
    * Multi-level variables (InternetService, Contract, PaymentMethod) get one
      0/1 column per level (one-hot encoding), so every level is visible.
    * A customer with a missing category gets a missing value (not 0) in those
      columns; pandas then uses pairwise-complete observations.
    """
    out = pd.DataFrame(index=df.index)
    out["Churn (1 = Yes)"] = df["churn_flag"]
    for col in config.NUMERIC_FEATURES:
        out[col] = df[col]
    out["Add-on count"] = df["n_addons"]
    out["Male"] = (df["gender"] == "Male").astype(float).where(df["gender"].notna())
    yes_no = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
              *config.ADDON_SERVICES, "PaperlessBilling"]
    multi = {"InternetService": "Internet", "Contract": "Contract", "PaymentMethod": "Payment"}
    for col in config.CATEGORICAL_FEATURES:
        if col in yes_no:
            out[col] = (df[col] == "Yes").astype(float).where(df[col].notna())
        elif col in multi:
            for level in sorted(df[col].dropna().unique()):
                short = {"No": "None"}.get(level, level.replace(" (automatic)", " (auto)"))
                name = f"{multi[col]}: {short}"
                out[name] = (df[col] == level).astype(float).where(df[col].notna())
    return out


def plot_correlation_heatmap(corr, title):
    """
    Annotated heatmap of the full correlation matrix. The matrix is symmetric
    (the upper half mirrors the lower half) and the diagonal is 1.00 because
    each variable correlates perfectly with itself. Sizes and fonts scale with
    the number of variables, so both small and large matrices stay readable.
    """
    n = len(corr)
    large = n > 10
    fig, ax = plt.subplots(figsize=(8.6, 7.6) if large else (6.2, 4.9))
    im = ax.imshow(corr, cmap=_diverging_cmap(), vmin=-1, vmax=1)
    for i in range(n):
        for j in range(corr.shape[1]):
            v = corr.iloc[i, j]
            # large matrices drop the leading zero (".37") so the numbers fit their cells
            v = 0.0 if abs(v) < 0.005 else v                # avoid printing "−.00"
            label = f"{v:.2f}".replace("0.", ".", 1) if large else f"{v:.2f}"
            ax.text(j, i, label.replace("-", "−"), ha="center", va="center",
                    fontsize=5.6 if large else 9.5, color="white" if abs(v) > 0.6 else TEXT)
    ax.set_xticks(range(corr.shape[1]), corr.columns, rotation=60 if large else 30, ha="right",
                  rotation_mode="anchor", fontsize=7 if large else None)
    ax.set_yticks(range(n), corr.index, fontsize=7 if large else None)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Spearman correlation")
    cbar.outline.set_visible(False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_contract_tenure_heatmap(df):
    """Churn rate for every contract x tenure-group combination (with group sizes)."""
    from matplotlib.colors import LinearSegmentedColormap
    rate = pd.crosstab(df["tenure_group"], df["Contract"], values=df["churn_flag"], aggfunc="mean") * 100
    size = pd.crosstab(df["tenure_group"], df["Contract"])
    cmap = LinearSegmentedColormap.from_list("orange_seq", ["#fdeee6", "#f5a27f", ORANGE, "#a8401a"])
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    im = ax.imshow(rate.values, cmap=cmap, vmin=0, vmax=rate.values.max(), aspect="auto")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            v = rate.iloc[i, j]
            ax.text(j, i, f"{v:.1f}%\n(n={size.iloc[i, j]:,})", ha="center", va="center", fontsize=9,
                    color="white" if v > rate.values.max() * 0.6 else TEXT)
    ax.set_xticks(range(rate.shape[1]), rate.columns)
    ax.set_yticks(range(rate.shape[0]), rate.index)
    ax.set_ylabel("Tenure group (months)")
    ax.set_xlabel("Contract type")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Churn rate (%)")
    cbar.outline.set_visible(False)
    ax.set_title("Churn rate by contract type within each tenure group")
    fig.tight_layout()
    return fig, rate, size


def _horizontal_rate_bars(ax, table, overall, title):
    """Horizontal churn-rate bars (one row per label) with value labels."""
    y = range(len(table))
    ax.barh(list(y), table["churn_rate"], color=ORANGE, height=0.62, zorder=2)
    for i, (rate, n) in enumerate(zip(table["churn_rate"], table["customers"])):
        ax.text(rate + 0.8, i, f"{rate:.1f}%  (n={n:,})", va="center", fontsize=8.5,
                bbox=LABEL_BOX, zorder=3)
    _reference_line(ax, overall, orientation="v")
    ax.set_yticks(list(y), table.index)
    ax.invert_yaxis()
    ax.set_xlim(0, table["churn_rate"].max() * 1.5)   # room for value labels
    ax.set_title(title)
    ax.grid(axis="y", visible=False)


def plot_services_panel(df):
    """
    (a) churn rate by number of add-on services (internet customers only);
    (b) churn rate with vs without each add-on (internet customers only).
    """
    internet = df[df["InternetService"] != "No"]
    overall = df["churn_flag"].mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.25, 1.2]})

    by_count = churn_rate_table(internet, "n_addons")
    _churn_rate_bars(axes[0], by_count, overall, "(a) By number of add-on services",
                     xlabel="Number of add-on services (internet customers)")

    rows = []
    for col in config.ADDON_SERVICES:
        t = churn_rate_table(internet, col)
        for level in ["Yes", "No"]:
            rows.append({"label": f"{col}: {'with' if level == 'Yes' else 'without'}",
                         "customers": t.loc[level, "customers"], "churn_rate": t.loc[level, "churn_rate"]})
    addon_table = pd.DataFrame(rows).set_index("label")
    _horizontal_rate_bars(axes[1], addon_table, overall, "(b) With vs without each add-on")
    fig.suptitle("Add-on services and churn (customers with internet service)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, by_count, addon_table


def plot_customer_attributes(df):
    """Churn rate for demographic and billing attributes (one row per level)."""
    overall = df["churn_flag"].mean() * 100
    rows = []
    for col, nice in [("SeniorCitizen", "Senior citizen"), ("Partner", "Partner"),
                      ("Dependents", "Dependents"), ("gender", "Gender"),
                      ("PaperlessBilling", "Paperless billing"), ("PhoneService", "Phone service")]:
        t = churn_rate_table(df, col)
        for level in t.index:
            rows.append({"label": f"{nice}: {level}", "customers": t.loc[level, "customers"],
                         "churn_rate": t.loc[level, "churn_rate"]})
    table = pd.DataFrame(rows).set_index("label")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _horizontal_rate_bars(ax, table, overall, "Churn rate by customer and billing attributes")
    fig.tight_layout()
    return fig, table


# ---------------------------------------------------------------------------
# Phase 4 - statistical analysis
# ---------------------------------------------------------------------------
def plot_effect_sizes(chi_table, mw_table):
    """
    (a) Cramer's V for every categorical variable (grey = not significant after
        Bonferroni correction); (b) rank-biserial correlation for numeric variables.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    data = chi_table.sort_values("cramers_v")
    colours = [BLUE if sig else GREY for sig in data["significant_after_bonferroni"]]
    ax.barh(data["variable"], data["cramers_v"], color=colours, height=0.65, zorder=2)
    for i, v in enumerate(data["cramers_v"]):
        ax.text(v + 0.006, i, f"{v:.2f}", va="center", fontsize=8.5)
    for x, label in [(0.10, "small"), (0.30, "medium")]:
        ax.axvline(x, color=TEXT_MUTED, linewidth=0.8, linestyle=":", zorder=1)
        ax.text(x + 0.004, len(data) - 0.4, label, fontsize=8, color=TEXT_MUTED, va="bottom")
    ax.set_xlim(0, data["cramers_v"].max() * 1.18)
    ax.set_ylim(-0.6, len(data) + 0.3)
    ax.set_xlabel("Cramér's V (0 = no association, 1 = perfect)")
    ax.set_title("(a) Categorical variables: chi-square effect size")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Significant (Bonferroni-adjusted p < 0.05)"),
                       Patch(color=GREY, label="Not significant")], loc="lower right", fontsize=8)

    ax = axes[1]
    data = mw_table.set_index("variable")["rank_biserial"].sort_values()
    ax.barh(data.index, data.values, color=BLUE, height=0.5, zorder=2)
    for i, v in enumerate(data.values):
        ax.text(v + (0.015 if v >= 0 else -0.015), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color=TEXT_MUTED, linewidth=0.8)
    ax.set_xlim(-0.65, 0.45)
    ax.set_xlabel("Rank-biserial correlation\n(− churners lower · + churners higher)")
    ax.set_title("(b) Numeric variables: Mann-Whitney effect size")
    ax.grid(axis="y", visible=False)

    fig.suptitle("Strength of association with churn (effect sizes)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scatter-plot equivalent for a binary target
# ---------------------------------------------------------------------------
SCATTER_SPECS = [("tenure", "Tenure (months)", 1), ("MonthlyCharges", "Monthly charges (USD)", 5),
                 ("TotalCharges", "Total charges (USD)", 300)]


def churn_rate_by_value(df, specs=SCATTER_SPECS):
    """
    Churn rate (%) for each value of a numeric variable (tenure: every month;
    charges: fixed-width bins). This is what each point in the scatter plot shows.
    """
    import numpy as np
    parts = []
    for col, _, width in specs:
        x = df[col]
        centre = (np.floor(x / width) * width + (width / 2 if width > 1 else 0))
        g = (df.assign(_bin=centre).groupby("_bin")["churn_flag"]
             .agg(customers="size", churn_rate="mean").reset_index().rename(columns={"_bin": "value"}))
        g["churn_rate"] *= 100
        g.insert(0, "variable", col)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def plot_churn_rate_scatter(rates, spearman, overall, specs=SCATTER_SPECS, min_customers=10):
    """
    Three scatter panels: churn rate vs tenure / monthly charges / total charges.
    Point size = number of customers; navy line = LOWESS trend (weighted by group size);
    dashed line = overall churn rate. Groups with fewer than `min_customers` are not drawn
    (their rates are too noisy) but remain in the saved table.
    """
    import numpy as np
    from statsmodels.nonparametric.smoothers_lowess import lowess
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, (col, label, _) in zip(axes, specs):
        d = rates[(rates["variable"] == col) & (rates["customers"] >= min_customers)]
        ax.scatter(d["value"], d["churn_rate"], s=np.clip(d["customers"] / 4, 8, 120), color=ORANGE,
                   alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3, label="Churn rate of the group")
        # weighted LOWESS: repeat each point in proportion to its size (capped to keep it fast)
        reps = np.clip((d["customers"] / d["customers"].min()).round().astype(int), 1, 50)
        trend = lowess(np.repeat(d["churn_rate"].to_numpy(), reps), np.repeat(d["value"].to_numpy(), reps), frac=0.4)
        ax.plot(trend[:, 0], trend[:, 1], color="#1F3A5F", linewidth=2.2, zorder=4, label="Trend (LOWESS)")
        ax.axhline(overall, color=GREY, linestyle="--", linewidth=1, zorder=1, label=f"Overall ({overall:.1f}%)")
        ax.set_xlabel(label)
        ax.set_title(f"{label.split(' (')[0]} vs churn\nSpearman ρ = {spearman[col]:+.2f}", fontsize=10.5)
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].set_ylim(0, 100)
    axes[2].legend(loc="upper right", fontsize=8)
    fig.suptitle("Relationship between the numeric variables and churn (each point = a group of customers)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
