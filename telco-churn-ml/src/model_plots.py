"""
model_plots.py
--------------
Figures for model evaluation and interpretation, in the same style as eda_plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config
from src.eda_plots import AQUA, BLUE, GREY, ORANGE, TEXT, TEXT_MUTED, YELLOW

# One fixed colour per model, used in every figure (colour follows the model)
MODEL_COLORS = {"Logistic Regression": BLUE, "Decision Tree": ORANGE,
                "Random Forest": AQUA, "Dummy (always 'No churn')": GREY}

METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1-score", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


def plot_model_comparison(test_table, selected):
    """
    Small multiples: one panel per metric, one bar per model (test set).
    The selected model is highlighted; the others are grey.
    """
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    models = list(test_table.index)
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
    for ax, metric in zip(axes.ravel(), metrics):
        values = test_table[metric]
        colours = [BLUE if m == selected else "#c9c7c1" for m in models]
        ax.barh(models, values, color=colours, height=0.6, zorder=2)
        for i, v in enumerate(values):
            ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=8.5, color=TEXT)
        ax.set_xlim(0, 1.12)
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", visible=False)
    axes[0, 0].invert_yaxis()   # y-axes are shared, so this lists models top-to-bottom everywhere
    fig.suptitle(f"Test-set performance of all models (highlighted: selected model – {selected})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_confusion_matrices(results, threshold_note="threshold 0.5"):
    """
    One confusion matrix per model. Each cell shows the count and the
    percentage of that ACTUAL class (row percentage).
    """
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#9ec5f4", BLUE, "#104281"])
    fig, axes = plt.subplots(1, len(results), figsize=(3.3 * len(results), 3.5))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        row_pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=1)
        labels = [["True negative", "False positive"], ["False negative", "True positive"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({row_pct[i, j]:.0%})\n{labels[i][j]}",
                        ha="center", va="center", fontsize=8,
                        color="white" if row_pct[i, j] > 0.55 else TEXT)
        ax.set_xticks([0, 1], ["Pred: stay", "Pred: churn"])
        ax.set_yticks([0, 1], ["Actual: stay", "Actual: churn"])
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(name, fontsize=10)
    fig.suptitle(f"Confusion matrices on the test set ({threshold_note}; % of actual class)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_roc_pr_curves(y_true, probas):
    """(a) ROC curves and (b) precision-recall curves for every model on the test set."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    base_rate = np.mean(y_true)
    for name, proba in probas.items():
        colour = MODEL_COLORS.get(name, GREY)
        style = "--" if name.startswith("Dummy") else "-"
        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[0].plot(fpr, tpr, color=colour, linewidth=2, linestyle=style,
                     label=f"{name} (AUC = {roc_auc_score(y_true, proba):.3f})")
        if not name.startswith("Dummy"):
            prec, rec, _ = precision_recall_curve(y_true, proba)
            axes[1].plot(rec, prec, color=colour, linewidth=2,
                         label=f"{name} (AP = {average_precision_score(y_true, proba):.3f})")
    axes[1].axhline(base_rate, color=GREY, linestyle="--", linewidth=1.2,
                    label=f"No-skill baseline (churn rate {base_rate:.1%})")
    axes[0].set_xlabel("False positive rate (stayers wrongly flagged)")
    axes[0].set_ylabel("True positive rate (recall)")
    axes[0].set_title("(a) ROC curves")
    axes[1].set_xlabel("Recall (share of churners caught)")
    axes[1].set_ylabel("Precision (share of flagged who churn)")
    axes[1].set_title("(b) Precision-recall curves")
    axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right", fontsize=8)
        ax.set_xlim(0, 1)
    fig.suptitle("Ranking quality of the models on the test set", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_threshold_tradeoff(table, chosen_threshold, model_name):
    """Precision, recall and F1 as the decision threshold changes (cross-validated, training data)."""
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for metric, colour in [("precision", BLUE), ("recall", ORANGE), ("f1", AQUA)]:
        ax.plot(table["threshold"], table[metric], color=colour, linewidth=2, label=METRIC_LABELS[metric])
    ax.plot(table["threshold"], table["share_flagged"], color=GREY, linewidth=1.5, linestyle=":",
            label="Share of customers flagged")
    ax.axvline(0.5, color=TEXT_MUTED, linewidth=0.8)
    ax.text(0.495, 0.02, "default 0.5", fontsize=8, color=TEXT_MUTED, ha="right")
    ax.axvline(chosen_threshold, color=TEXT, linewidth=1.2, linestyle="--")
    ax.text(chosen_threshold + 0.005, 0.95, f"max-F1 threshold\n= {chosen_threshold:.2f}", fontsize=8,
            color=TEXT, va="top")
    ax.set_xlabel("Decision threshold (predicted churn probability)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(table["threshold"].min(), table["threshold"].max())
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Threshold trade-off – {model_name} (5-fold CV predictions, training set)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interpretation figures
# ---------------------------------------------------------------------------
def readable_feature(name, references):
    """Turn 'Contract_Two year' into 'Contract: Two year (vs Month-to-month)'."""
    if name == "tenure":
        return "Tenure (+1 SD ≈ 25 months)"
    for var, ref in references.items():
        if name.startswith(var + "_"):
            level = name[len(var) + 1:]
            return f"{var}: {level} (vs {ref})"
    return name


def plot_odds_ratios(table):
    """
    Forest plot of odds ratios with 95% bootstrap CIs (log scale).
    Orange = higher churn odds, blue = lower, grey = CI includes 1 (no clear effect).
    """
    data = table.sort_values("odds_ratio").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    for i, row in data.iterrows():
        if row["or_ci_low"] <= 1 <= row["or_ci_high"]:
            colour = GREY
        else:
            colour = ORANGE if row["odds_ratio"] > 1 else BLUE
        ax.plot([row["or_ci_low"], row["or_ci_high"]], [i, i], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(row["odds_ratio"], i, "o", color=colour, markersize=7, markeredgecolor="white", markeredgewidth=1)
        ax.text(row["or_ci_high"] * 1.06, i, f"{row['odds_ratio']:.2f}", va="center", fontsize=8, color=TEXT)
    ax.axvline(1, color=TEXT_MUTED, linewidth=1)
    ax.set_xscale("log")
    ticks = [0.15, 0.25, 0.5, 1, 2, 3]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(0.12, 4.2)
    ax.set_yticks(range(len(data)), data["label"])
    ax.set_xlabel("Odds ratio for churn (log scale) · < 1 lower odds · > 1 higher odds")
    ax.set_title("Logistic regression odds ratios with 95% bootstrap confidence intervals")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_permutation_importance(tables, top_n=10):
    """Side-by-side permutation importance (drop in test ROC-AUC) for several models."""
    fig, axes = plt.subplots(1, len(tables), figsize=(5.4 * len(tables), 4.2))
    for ax, (name, table) in zip(np.atleast_1d(axes), tables.items()):
        data = table.head(top_n).iloc[::-1]
        ax.barh(data["feature"], data["importance_mean"], xerr=data["importance_sd"],
                color=MODEL_COLORS.get(name, BLUE), height=0.6, zorder=2,
                error_kw=dict(ecolor=TEXT_MUTED, elinewidth=1, capsize=2))
        for i, v in enumerate(data["importance_mean"]):
            ax.text(v + data["importance_sd"].max() + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, table["importance_mean"].max() * 1.3)
        ax.set_xlabel("Mean drop in test ROC-AUC when shuffled")
        ax.set_title(name)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Permutation importance – top features (test set, 20 repeats)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_simple_tree(tree_pipeline, X_train, y_train):
    """
    Draw a shallow decision tree with plain-language rules. Each box shows the
    question, number of training customers and their ACTUAL churn rate;
    darker orange = higher churn rate.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.tree import plot_tree

    prep = tree_pipeline.named_steps["prep"]
    tree = tree_pipeline.named_steps["model"]
    names = list(prep.get_feature_names_out())
    Xp = prep.transform(X_train)
    paths = tree.decision_path(Xp)
    n_node = np.asarray(paths.sum(axis=0)).ravel()
    churn_node = np.asarray(paths.T @ np.asarray(y_train)).ravel()
    rate = churn_node / n_node
    cmap = LinearSegmentedColormap.from_list("orange_seq", ["#fdeee6", "#f5a27f", ORANGE, "#a8401a"])

    fig, ax = plt.subplots(figsize=(13, 6.2))
    annotations = plot_tree(tree, ax=ax, filled=False, impurity=False, proportion=False,
                            feature_names=names, fontsize=8, rounded=True)
    # plot_tree also returns the root's "True/False" arrow labels; relabel them and keep
    # only the boxed node labels, which are drawn in node-id order
    for ann in annotations:
        if ann.get_bbox_patch() is None:
            ann.set_text(ann.get_text().replace("True", "yes").replace("False", "no"))
    annotations = [ann for ann in annotations if ann.get_bbox_patch() is not None]
    t = tree.tree_
    assert len(annotations) == t.node_count
    for node, ann in enumerate(annotations):
        if t.children_left[node] != -1:                       # internal node: a question
            feat = names[t.feature[node]]
            if feat in config.NUMERIC_FEATURES:
                question = f"{feat} ≤ {t.threshold[node]:.1f}?"
            else:
                # one-hot column: "<= 0.5" (left branch) means the customer is NOT in this category
                var, level = feat.split("_", 1)
                question = f"{var} is NOT {level}?"
            text = f"{question}\n"
        else:
            text = "Final group\n"
        text += f"n = {n_node[node]:,} | churn {rate[node]:.0%}"
        ann.set_text(text)
        ann.get_bbox_patch().set_facecolor(cmap(min(rate[node] / 0.75, 1.0)))
        ann.get_bbox_patch().set_edgecolor("white")
        ann.set_color("white" if rate[node] > 0.45 else TEXT)
    ax.set_title("Simplified decision tree (depth 3) – churn rules learned from the training data "
                 "(left branch = yes, right = no)",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, rate, n_node


def plot_gains(y_true, probas, gains):
    """(a) cumulative gains curves; (b) actual churn rate by predicted-risk decile (selected model)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    y_true = np.asarray(y_true)
    for name, proba in probas.items():
        order = np.argsort(-np.asarray(proba))
        captured = np.cumsum(y_true[order]) / y_true.sum()
        share = np.arange(1, len(y_true) + 1) / len(y_true)
        ax.plot(share, captured, color=MODEL_COLORS.get(name, BLUE), linewidth=2, label=name)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.2, label="Random targeting")
    for pct in (0.1, 0.2, 0.3):
        val = gains.loc[gains["cumulative_share_of_customers"].round(2) == pct,
                        "cumulative_share_of_churners_captured"]
        if len(val):
            ax.plot(pct, val.iloc[0], "o", color=TEXT, markersize=5)
            ax.text(pct + 0.02, val.iloc[0] - 0.05, f"top {pct:.0%} → {val.iloc[0]:.0%}", fontsize=8)
    ax.set_xlabel("Share of customers contacted (highest risk first)")
    ax.set_ylabel("Share of all churners captured")
    ax.set_title("(a) Cumulative gains")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    rates = gains["actual_churn_rate"] * 100
    ax.bar([str(d) for d in gains.index], rates, color=ORANGE, width=0.65, zorder=2)
    for i, (r, n) in enumerate(zip(rates, gains["churners"])):
        ax.text(i, r + 1.5, f"{r:.0f}%", ha="center", fontsize=8)
    base = y_true.mean() * 100
    ax.axhline(base, color=GREY, linestyle="--", linewidth=1, label=f"Overall churn rate ({base:.1f}%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Risk decile (1 = highest predicted risk)")
    ax.set_ylabel("Actual churn rate (%)")
    ax.set_ylim(0, rates.max() * 1.18)
    ax.set_title("(b) Actual churn rate by predicted-risk decile")
    ax.grid(axis="x", visible=False)
    fig.suptitle("How well the risk ranking concentrates churners (test set)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_vif_comparison(vif_before, vif_after, title_before, title_after):
    """
    Horizontal VIF bars before and after removing the collinear charge columns
    (log scale). Orange = serious (> 10), yellow = moderate (5-10), blue = low.
    Reference lines at VIF 5 and 10.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharex=True)
    for ax, table, title in [(axes[0], vif_before, title_before), (axes[1], vif_after, title_after)]:
        data = table.sort_values("VIF")
        colours = [ORANGE if v > 10 else YELLOW if v > 5 else BLUE for v in data["VIF"]]
        ax.barh(data["feature"], data["VIF"], color=colours, height=0.65, zorder=2)
        for i, v in enumerate(data["VIF"]):
            ax.text(v * 1.08, i, f"{v:,.1f}", va="center", fontsize=8)
        for x in (5, 10):
            ax.axvline(x, color=TEXT_MUTED, linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.8, max(vif_before["VIF"].max(), vif_after["VIF"].max()) * 3)
        ax.set_xlabel("VIF (log scale) · dotted lines at 5 and 10")
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=ORANGE, label="Serious (> 10)"), Patch(color=YELLOW, label="Moderate (5–10)"),
                            Patch(color=BLUE, label="Low (≤ 5)")], loc="lower right", fontsize=8)
    fig.suptitle("Multicollinearity in the logistic-regression design (training data)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
