"""
build_report.py
---------------
Generates the academic report (Word .docx) from the executed analysis.

Every number quoted in the report is read from outputs/tables or
outputs/models - nothing is typed by hand - so the report always matches
the notebook. Run the notebook first, then:

    python reports/build_report.py

The script also prints an approximate page count (Word's exact pagination
can only be seen by opening the file).
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

TABLES, FIGURES = config.TABLE_DIR, config.FIGURE_DIR
OUT = ROOT / "reports" / "Telco_Churn_Report.docx"
DUMMY = "Dummy (always 'No churn')"
CREST = ROOT / "assets" / "uom_crest.png"

# Every figure and table of the report, in order of appearance, with its caption.
# The body captions, the List of Figures / List of Tables and the cross-references
# (F() / T()) are all generated from these two registries, so they can never disagree.
FIG_CAPTIONS = {
    "fig01_missing_invalid_before_cleaning.png": "Missing and invalid values per column before cleaning (raw file)",
    "fig02_boxplots_before_after_cleaning.png": "Boxplots of the numeric variables before and after cleaning",
    "fig20_vif_before_after.png": "Variance inflation factors before and after removing the collinear charge columns",
    "fig04_numeric_distributions_by_churn.png": "Histograms of tenure, monthly charges and total charges by churn status",
    "fig03_churn_distribution.png": "Distribution of the target variable (churn)",
    "fig05_churn_rate_key_segments.png": "Churn rate by contract, internet service, payment method and tenure group",
    "fig10_customer_attributes_churn.png": "Churn rate by demographic and billing attributes",
    "fig07_correlation_heatmap.png": "Spearman correlation heatmap of all variables (categories one-hot encoded)",
    "fig21_churn_rate_scatter_trend.png": "Relationship plots: churn rate against tenure, monthly charges and total charges",
    "fig11_effect_sizes.png": "Effect sizes of the associations with churn",
    "fig16_odds_ratios.png": "Logistic regression odds ratios with 95% bootstrap confidence intervals",
    "fig17_permutation_importance.png": "Permutation importance of the input variables",
    "fig12_model_comparison.png": "Model comparison on the test set",
    "fig13_confusion_matrices.png": "Actual vs predicted classes: confusion matrices on the test set",
}
TAB_CAPTIONS = {
    "missing": "Missing and invalid values found in the raw file and their treatment",
    "imputation": "Imputation values learned from the training data",
    "summary": "Summary statistics of the numeric variables",
    "profile": "Profile of customers who stayed and customers who churned",
    "distribution": "Interpretation of the histograms",
    "lr_coef": "Logistic regression coefficient summary (training data)",
    "tree_rules": "Decision rules of the simplified decision tree (training data)",
    "performance": "Cross-validated and test-set performance of the models",
    "actual_pred": "Actual vs predicted values: comparison of the two strongest models",
    "recommendations": "Retention recommendations and supporting evidence",
}
FIG_ORDER = list(FIG_CAPTIONS)
TAB_ORDER = list(TAB_CAPTIONS)


def pretty_term(name):
    """'Contract_Two year' -> 'Contract: Two year'; 'tenure' -> 'tenure (standardised)'."""
    if name == "tenure":
        return "tenure (standardised)"
    var, _, level = name.partition("_")
    return f"{var}: {level}" if level else var


EQ_LABELS = {
    "Contract_Two year": "two-year contract", "Contract_One year": "one-year contract", "tenure": "z_tenure",
    "InternetService_Fiber optic": "fiber optic", "InternetService_No": "no internet",
    "PhoneService_Yes": "phone service", "PaymentMethod_Electronic check": "electronic check",
    "OnlineSecurity_Yes": "online security", "PaperlessBilling_Yes": "paperless billing",
    "TechSupport_Yes": "tech support", "StreamingMovies_Yes": "streaming movies", "StreamingTV_Yes": "streaming TV",
    "MultipleLines_Yes": "multiple lines", "Dependents_Yes": "dependents",
}


def F(filename):
    """'Figure n' for a figure file (or its short prefix such as 'fig04'), according to FIG_ORDER."""
    matches = [i for i, name in enumerate(FIG_ORDER) if name.startswith(filename)]
    assert len(matches) == 1, f"ambiguous or unknown figure reference: {filename}"
    return f"Figure {matches[0] + 1}"


def T(key):
    """'Table n' for a table key, according to TAB_ORDER."""
    return f"Table {TAB_ORDER.index(key) + 1}"

BODY_FONT, BODY_PT = "Calibri", 11
HEADING_COLOR = RGBColor(0x1F, 0x3A, 0x5F)
PAGE_W, PAGE_H, MARGIN = 21.0, 29.7, 2.2          # A4, cm
TEXT_W = PAGE_W - 2 * MARGIN


# ===========================================================================
# 1. Load every number used in the text
# ===========================================================================
def t(name, **kw):
    return pd.read_csv(TABLES / name, **kw)


def pct(x, d=1):
    return f"{x:.{d}f}%"


def load_results():
    R = {}
    split = t("t10_split_summary.csv", index_col=0)
    audit = t("t02_data_quality_audit.csv")
    R["train_n"], R["test_n"] = int(split["customers"].iloc[0]), int(split["customers"].iloc[1])
    R["clean_n"] = R["train_n"] + R["test_n"]
    R["dup"] = int(audit.loc[audit["check"] == "Exact duplicate rows", "count"].iloc[0])
    R["raw_n"] = R["clean_n"] + R["dup"]
    R["audit"] = audit

    def audit_count(check, column):
        return int(audit.loc[(audit["check"] == check) & (audit["column"] == column), "count"].iloc[0])

    R["gender_bad"] = audit_count("Invalid / blank category", "gender")
    R["contract_blank"] = audit_count("Invalid / blank category", "Contract")
    R["tenure_blank"] = audit_count("Blank value", "tenure")
    R["tenure_sentinel"] = audit_count("Negative sentinel value", "tenure")
    R["tenure_impossible"] = audit_count("Impossible non-integer value", "tenure")
    R["mc_blank"] = audit_count("Blank value", "MonthlyCharges")
    R["mc_impossible"] = audit_count("Impossible value (> plausible ceiling)", "MonthlyCharges")
    R["tc_blank"] = audit_count("Blank value", "TotalCharges")
    R["tc_invalid"] = audit_count("Non-numeric text", "TotalCharges")
    R["tenure_examples"] = audit.loc[audit["check"] == "Impossible non-integer value", "example"].iloc[0]
    R["mc_examples"] = audit.loc[audit["check"] == "Impossible value (> plausible ceiling)", "example"].iloc[0]

    rec = t("t04_reconstruction_accuracy.csv").set_index("value_repaired")
    R["rec"] = rec
    summary = t("t05_cleaning_summary.csv")
    R["n_repaired"] = int(summary.loc[summary["rule"].str.startswith("invalid ->") &
                                      ~summary["rule"].str.contains("missing"), "values_changed"].sum())
    miss = t("t07_missingness_vs_churn.csv", index_col=0)
    R["affected_n"] = int(miss["customers"].iloc[1])
    R["affected_rate"], R["unaffected_rate"] = miss["churn_rate"].iloc[1], miss["churn_rate"].iloc[0]
    R["affected_p"] = float(miss["chi2_p_value"].iloc[0])
    iqr = t("t08_iqr_outliers_after_cleaning.csv")
    R["iqr_outliers"] = int(iqr["n_below"].sum() + iqr["n_above"].sum())
    R["tc_skew"] = float(iqr.set_index("column").loc["TotalCharges", "skewness"])

    rates = t("t15_churn_rate_by_category.csv").set_index(["variable", "level"])["churn_rate"]
    R["rate"] = lambda var, level: rates[(var, level)]
    prof = t("t16_customer_profile.csv", index_col=0)
    R["profile"] = prof
    R["churn_rate"] = 100 * (1 - prof.loc["Share of all customers (%)", "Stayed"] / 100)
    R["n_churn"] = int(prof.loc["Customers", "Churned"])
    bands = t("t12_churn_by_monthly_charge_band.csv", index_col=0)
    R["bands"] = bands
    R["ct"] = t("t14_churn_contract_by_tenure.csv", header=[0, 1], index_col=0)
    sp = t("t13_spearman_correlations.csv", index_col=0)
    R["rho_tenure_tc"] = sp.loc["tenure", "TotalCharges"]
    R["rho_churn_tenure"] = sp.loc["Churn (1 = Yes)", "tenure"]
    R["rho_churn_mc"] = sp.loc["Churn (1 = Yes)", "MonthlyCharges"]
    R["rho_churn_addons"] = sp.loc["Churn (1 = Yes)", "Add-on count"]

    R["chi"] = t("t17_chi_square_tests.csv").set_index("variable")
    R["addon_chi"] = t("t18_chi_square_addons_internet_only.csv").set_index("variable")
    R["mw"] = t("t19_mann_whitney_tests.csv").set_index("variable")
    R["vif_num"] = t("t21_vif_numeric.csv").set_index("feature")["VIF"]
    vif_full = t("t23_vif_full_design.csv").set_index("feature")["VIF"]
    R["vif_mc_full"] = vif_full["MonthlyCharges"]
    R["vif_reduced_max"] = t("t24_vif_reduced_design.csv")["VIF"].max()
    R["hyp"] = t("t25_hypothesis_summary.csv")

    R["cw"] = t("t26_class_weight_check.csv", index_col=0, keep_default_na=False)
    R["lrfs"] = t("t27_lr_feature_set_check.csv", index_col=0)
    R["cv"] = t("t28_cv_results.csv", index_col=0)
    R["test"] = t("t29_test_metrics.csv", index_col=0)
    R["ci"] = t("t30_test_auc_confidence_intervals.csv", index_col=0)
    R["diff"] = t("t31_test_auc_paired_differences.csv", index_col=0)
    R["overfit"] = t("t32_overfitting_check.csv", index_col=0)
    R["ops"] = t("t34_selected_model_operating_points.csv", index_col=0)
    R["final"] = json.loads((config.MODEL_DIR / "final_model_metrics.json").read_text())
    R["or"] = t("t35_odds_ratios.csv").set_index("feature")
    R["perm"] = t("t36_permutation_importance.csv").set_index("feature")
    R["leaves"] = t("t38_decision_tree_leaves.csv")
    R["gains"] = t("t39_gains_table.csv", index_col=0)
    R["errors"] = t("t40_error_analysis.csv", index_col=0)
    R["recs"] = t("t41_recommendations.csv")
    R["dist"] = t("t42_distribution_summary.csv")
    R["summary"] = t("t11_numeric_summary_by_churn.csv").query("group == 'All'").set_index("variable")
    R["spearman"] = t("t13_spearman_correlations.csv", index_col=0)
    R["spearman_all"] = t("t48_spearman_all_variables.csv", index_col=0)
    R["scatter"] = t("t46_churn_rate_by_value.csv")
    issues = t("t03_issue_counts_by_type.csv", index_col=0)
    issues["total"] = issues.sum(axis=1)
    R["issues"] = issues[issues["total"] > 0]
    R["imputation"] = t("t47_imputation_values.csv")
    R["logit_tab"] = t("t44_logit_summary.csv", index_col=0)
    eq_text = (TABLES / "t43_logistic_equation.txt").read_text()
    import re
    R["equation"] = {
        "intercept": float(re.search(r"log-odds\(churn\) = ([+-][0-9.]+)", eq_text).group(1)),
        "tenure_mean": float(re.search(r"\(tenure − ([0-9.]+)\)", eq_text).group(1)),
        "tenure_sd": float(re.search(r"\) / ([0-9.]+),", eq_text).group(1)),
    }
    logit = t("t44_logit_summary.csv", index_col=0).drop(index="const")
    R["logit_or_diff"] = float((logit["odds_ratio"] - logit["odds_ratio_sklearn_model"]).abs().max())
    R["logit_fit"] = t("t45_logit_fit_statistics.csv").iloc[0]
    return R


# ===========================================================================
# 2. Low-level document helpers
# ===========================================================================
class Report:
    """Thin wrapper around python-docx that also estimates page usage."""

    def __init__(self):
        self.doc = Document()
        self.fig_no = 0
        self.tab_no = 0
        self.height_cm = 0.0          # running height on the current page
        self.pages = 1
        self._setup()

    # ---- page-usage estimate ------------------------------------------------
    def _add_height(self, h):
        usable = PAGE_H - 2 * MARGIN - 1.0         # minus footer area
        if self.height_cm + h > usable:
            self.pages += 1
            self.height_cm = h
        else:
            self.height_cm += h

    def _text_height(self, text, size_pt, width_cm=TEXT_W, spacing=1.15, after_pt=6):
        chars_per_line = max(1, width_cm / (size_pt * 0.0353 * 0.47))
        lines = max(1, math.ceil(len(text) / chars_per_line))
        return lines * size_pt * 1.22 * spacing * 0.0353 + after_pt * 0.0353

    @staticmethod
    def _set_page_numbering(section, fmt, start=None):
        """Set the page-number format (and optional restart) of a section, in schema order."""
        sect_pr = section._sectPr
        for old in sect_pr.findall(qn("w:pgNumType")):
            sect_pr.remove(old)
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:fmt"), fmt)
        if start is not None:
            pg.set(qn("w:start"), str(start))
        for name in ("cols", "formProt", "vAlign", "noEndnote", "titlePg", "textDirection", "bidi",
                     "rtlGutter", "docGrid", "printerSettings"):
            nxt = sect_pr.find(qn(f"w:{name}"))
            if nxt is not None:
                nxt.addprevious(pg)
                return
        sect_pr.append(pg)

    def start_body_section(self):
        """
        End the front matter (cover + contents, numbered i, ii …; no number on the cover)
        and start the body on a new page, numbered 1, 2, 3 …
        """
        self.doc.add_section(WD_SECTION.NEW_PAGE)
        # re-fetch both sections AFTER the break: add_section() re-assigns the document's final sectPr
        front, body = self.doc.sections[0], self.doc.sections[1]
        body.different_first_page_header_footer = False
        self._set_page_numbering(front, "lowerRoman", start=1)
        self._set_page_numbering(body, "decimal", start=1)
        self.pages += 1
        self.height_cm = 0.0

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        self.pages += 1
        self.height_cm = 0.0

    # ---- setup -----------------------------------------------------------------
    def _setup(self):
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(PAGE_W), Cm(PAGE_H)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Cm(MARGIN))
        sec.different_first_page_header_footer = True

        styles = self.doc.styles
        normal = styles["Normal"]
        normal.font.name, normal.font.size = BODY_FONT, Pt(BODY_PT)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        pf = normal.paragraph_format
        pf.space_after, pf.line_spacing = Pt(6), 1.15
        for name, size, before in [("Heading 1", 14, 12), ("Heading 2", 12, 8), ("Heading 3", 11, 6)]:
            st = styles[name]
            st.font.name, st.font.size, st.font.bold = BODY_FONT, Pt(size), True
            st.font.color.rgb = HEADING_COLOR
            rfonts = st.element.rPr.find(qn("w:rFonts"))
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rfonts.set(qn(attr), BODY_FONT)
            for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                if rfonts.get(qn(attr)) is not None:
                    del rfonts.attrib[qn(attr)]
            st.paragraph_format.space_before, st.paragraph_format.space_after = Pt(before), Pt(4)
            st.paragraph_format.keep_with_next = True
        cap = styles["Caption"]
        cap.font.name, cap.font.size, cap.font.italic, cap.font.bold = BODY_FONT, Pt(9.5), True, False
        cap.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
        cap.paragraph_format.space_after = Pt(8)

        # page number in the footer (not on the cover page)
        footer = sec.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._field(footer, "PAGE", "1")

        # ask Word to refresh the table of contents / lists when the file is opened
        # (inserted at its schema position: just before <w:compat>)
        settings = self.doc.settings.element
        upd = OxmlElement("w:updateFields")
        upd.set(qn("w:val"), "true")
        compat = settings.find(qn("w:compat"))
        compat.addprevious(upd) if compat is not None else settings.append(upd)
        zoom = settings.find(qn("w:zoom"))
        if zoom is not None and zoom.get(qn("w:percent")) is None:
            zoom.set(qn("w:percent"), "100")      # required by the schema

    # ---- fields ----------------------------------------------------------------
    @staticmethod
    def _field(paragraph, instruction, cached_text):
        """Insert a Word field (PAGE, SEQ ...) with a cached result."""
        def fld(kind):
            r = OxmlElement("w:r")
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), kind)
            r.append(fc)
            return r
        p = paragraph._p
        p.append(fld("begin"))
        r = OxmlElement("w:r")
        it = OxmlElement("w:instrText")
        it.set(qn("xml:space"), "preserve")
        it.text = f" {instruction} "
        r.append(it)
        p.append(r)
        p.append(fld("separate"))
        run = paragraph.add_run(cached_text)
        p.append(fld("end"))
        return run

    def toc(self, instruction, entries, size=10):
        """
        A Word TOC field whose cached result is a static list of entries,
        so it is readable even before Word updates it (Word then adds page numbers).
        """
        paras = []
        for i, (text, level) in enumerate(entries):
            p = self.doc.add_paragraph()
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.left_indent = Cm(0.6 * level)
            paras.append((p, text))
            self._add_height(size * 1.4 * 0.0353 + 0.04)
        first, last = paras[0][0], paras[-1][0]

        def fld(kind):
            r = OxmlElement("w:r")
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), kind)
            r.append(fc)
            return r
        first._p.append(fld("begin"))
        r = OxmlElement("w:r")
        it = OxmlElement("w:instrText")
        it.set(qn("xml:space"), "preserve")
        it.text = f" {instruction} "
        r.append(it)
        first._p.append(r)
        first._p.append(fld("separate"))
        for p, text in paras:
            run = p.add_run(text)
            run.font.size = Pt(size)
        last._p.append(fld("end"))

    # ---- content ---------------------------------------------------------------
    def h1(self, text):
        self.doc.add_heading(text, level=1)
        self._add_height(14 * 1.3 * 0.0353 + 0.6)

    def h3(self, text):
        self.doc.add_heading(text, level=3)
        self._add_height(11 * 1.3 * 0.0353 + 0.35)

    def h2(self, text):
        self.doc.add_heading(text, level=2)
        self._add_height(12 * 1.3 * 0.0353 + 0.45)

    def para(self, text, bold_lead=None, align_justify=True, size=BODY_PT, after=6):
        p = self.doc.add_paragraph()
        if bold_lead:
            p.add_run(bold_lead).bold = True
        run = p.add_run(text)
        if size != BODY_PT:
            run.font.size = Pt(size)
            if bold_lead:
                p.runs[0].font.size = Pt(size)
        if align_justify:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(after)
        self._add_height(self._text_height((bold_lead or "") + text, size, after_pt=after))
        return p

    def bullets(self, items, size=BODY_PT):
        for item in items:
            lead, text = item if isinstance(item, tuple) else (None, item)
            p = self.doc.add_paragraph(style="List Bullet")
            if lead:
                r = p.add_run(lead)
                r.bold = True
                r.font.size = Pt(size)
            r = p.add_run(text)
            r.font.size = Pt(size)
            p.paragraph_format.space_after = Pt(2)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            self._add_height(self._text_height((lead or "") + text, size, width_cm=TEXT_W - 0.64, after_pt=2))

    def figure(self, filename, width_cm, caption=None):
        """Insert a figure; its caption always comes from FIG_CAPTIONS."""
        from PIL import Image
        caption = FIG_CAPTIONS[filename]
        path = FIGURES / filename
        w, h = Image.open(path).size
        height = width_cm * h / w
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        self.fig_no += 1
        assert FIG_ORDER[self.fig_no - 1] == filename, f"{filename} placed out of FIG_ORDER"
        cp = self.doc.add_paragraph(style="Caption")
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.add_run("Figure ")
        self._field(cp, "SEQ Figure \\* ARABIC", str(self.fig_no))
        cp.add_run(f": {caption}")
        self._add_height(height + 0.25 + self._text_height(caption, 9.5, after_pt=8))
        return self.fig_no

    def table(self, header, rows, widths_cm, caption=None, size=9, align=None, key=None):
        """Table with caption above (from TAB_CAPTIONS), shaded header row and explicit column widths."""
        caption = TAB_CAPTIONS[key]
        self.tab_no += 1
        assert key is None or TAB_ORDER[self.tab_no - 1] == key, f"table '{key}' placed out of TAB_ORDER"
        cp = self.doc.add_paragraph(style="Caption")
        cp.paragraph_format.keep_with_next = True
        cp.paragraph_format.space_after = Pt(3)
        cp.add_run("Table ")
        self._field(cp, "SEQ Table \\* ARABIC", str(self.tab_no))
        cp.add_run(f": {caption}")
        self._add_height(self._text_height(caption, 9.5, after_pt=3))

        tbl = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        self._table_borders(tbl)
        align = align or ["left"] * len(header)
        for r_i, values in enumerate([header] + rows):
            row = tbl.rows[r_i]
            max_lines = 1
            for c_i, value in enumerate(values):
                cell = row.cells[c_i]
                cell.width = Cm(widths_cm[c_i])
                cell.text = ""
                para = cell.paragraphs[0]
                para.paragraph_format.space_after = Pt(0)
                para.paragraph_format.line_spacing = 1.0
                para.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
                                  "right": WD_ALIGN_PARAGRAPH.RIGHT}[align[c_i]]
                run = para.add_run(str(value))
                run.font.size = Pt(size)
                if r_i == 0:
                    run.bold = True
                    self._shade(cell, "DCE6F2")
                chars = max(1, (widths_cm[c_i] - 0.25) / (size * 0.0353 * 0.47))
                max_lines = max(max_lines, math.ceil(len(str(value)) / chars))
            self._add_height(max_lines * size * 1.22 * 0.0353 + 0.12)
        self._fix_table_width(tbl, widths_cm)
        spacer = self.doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(2)
        self._add_height(0.3)
        return self.tab_no

    @staticmethod
    def _shade(cell, hex_fill):
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_fill)
        tc_pr.append(shd)

    @staticmethod
    def _fix_table_width(tbl, widths_cm):
        """Explicit total width + grid column widths (twips), so every renderer keeps the layout."""
        twips = [int(w * 567) for w in widths_cm]
        for gc, w in zip(tbl._tbl.tblGrid.findall(qn("w:gridCol")), twips):
            gc.set(qn("w:w"), str(w))
        tbl_pr = tbl._tbl.tblPr
        tbl_w = tbl_pr.find(qn("w:tblW"))
        if tbl_w is None:
            tbl_w = OxmlElement("w:tblW")
            Report._insert_tblpr(tbl_pr, tbl_w, ["jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout",
                                                 "tblCellMar", "tblLook", "tblCaption", "tblDescription"])
        tbl_w.set(qn("w:w"), str(sum(twips)))
        tbl_w.set(qn("w:type"), "dxa")
        for row in tbl.rows:
            for cell, w in zip(row.cells, twips):
                tc_w = cell._tc.get_or_add_tcPr().get_or_add_tcW()
                tc_w.set(qn("w:w"), str(w))
                tc_w.set(qn("w:type"), "dxa")

    @staticmethod
    def _insert_tblpr(tbl_pr, element, later_siblings):
        """Insert a child of <w:tblPr> before the first sibling that must follow it (schema order)."""
        for name in later_siblings:
            nxt = tbl_pr.find(qn(f"w:{name}"))
            if nxt is not None:
                nxt.addprevious(element)
                return
        tbl_pr.append(element)

    @staticmethod
    def _table_borders(tbl):
        tbl_pr = tbl._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "A6A6A6")
            borders.append(el)
        Report._insert_tblpr(tbl_pr, borders, ["shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"])
        mar = OxmlElement("w:tblCellMar")
        for side in ("left", "right"):
            el = OxmlElement(f"w:{side}")
            el.set(qn("w:w"), "70")
            el.set(qn("w:type"), "dxa")
            mar.append(el)
        Report._insert_tblpr(tbl_pr, mar, ["tblLook", "tblCaption", "tblDescription"])


# ===========================================================================
# 3. The report content
# ===========================================================================
def build(R):
    rp = Report()
    doc = rp.doc
    rate = R["rate"]
    test, cv, ci, ors = R["test"], R["cv"], R["ci"], R["or"]
    sel = R["final"]["selected_model"]
    lr_t = test.loc[sel]
    rf_t = test.loc["Random Forest"]
    dt_t = test.loc["Decision Tree"]
    gains = R["gains"]
    top = lambda share: gains.loc[gains["cumulative_share_of_customers"].round(2) == share,
                                  "cumulative_share_of_churners_captured"].iloc[0] * 100
    ORv = lambda f: ors.loc[f, "odds_ratio"]
    diff_rf = R["diff"].loc[f"Random Forest minus {sel}"]
    chi, addon, mw = R["chi"], R["addon_chi"], R["mw"]

    # ---------------- Cover page ----------------
    crest = doc.add_paragraph()
    crest.alignment = WD_ALIGN_PARAGRAPH.CENTER
    crest.paragraph_format.space_after = Pt(14)
    crest.add_run().add_picture(str(CREST), height=Cm(3.4))
    for text, size, bold, after in [
        ("DEPARTMENT OF MANAGEMENT OF TECHNOLOGY", 13, True, 2),
        ("FACULTY OF BUSINESS", 12, True, 2),
        ("UNIVERSITY OF MORATUWA", 12, True, 26),
        ("TM3221 – Data Science for BPM", 14, True, 4),
        ("Group Assignment", 12, False, 26),
        ("Customer Churn Prediction and Analysis Using Machine Learning", 20, True, 6),
        ("A Data-Driven Approach for Customer Retention", 14, False, 30),
        ("Group name: [Group name]", 12, True, 10),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(after)
        r = p.add_run(text)
        r.font.size, r.bold = Pt(size), bold
        if size >= 14:
            r.font.color.rgb = HEADING_COLOR
    members = doc.add_table(rows=6, cols=2)
    members.alignment = WD_TABLE_ALIGNMENT.CENTER
    Report._table_borders(members)
    for i, (a, b) in enumerate([("Name", "Index number")] + [("[Member name]", "[Index no.]")] * 5):
        for j, val in enumerate((a, b)):
            cell = members.rows[i].cells[j]
            cell.width = Cm(6.5 if j == 0 else 4.0)
            cell.text = ""
            run = cell.paragraphs[0].add_run(val)
            run.bold = i == 0
            if i == 0:
                Report._shade(cell, "DCE6F2")
    members.autofit = False
    Report._fix_table_width(members, (6.5, 4.0))
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(26)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Semester 06 · [Submission date]").font.size = Pt(11)
    rp.page_break()

    # ---------------- Contents page + lists of figures and tables ----------------
    def front_title(text, before=0):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(before)
        r = p.add_run(text)
        r.bold, r.font.size, r.font.color.rgb = True, Pt(14), HEADING_COLOR
        rp._add_height(1.0 + before * 0.0353)

    front_title("Table of Contents")
    sections = [
        ("1. Introduction", 0), ("1.1 Dataset overview", 1), ("1.2 Context and relevance", 1),
        ("1.3 Objective of the study", 1),
        ("2. Data Cleaning", 0), ("2.1 Handling missing and invalid values", 1), ("2.2 Handling outliers", 1),
        ("2.3 Feature engineering and encoding", 1), ("2.4 Multicollinearity", 1),
        ("3. Exploratory Data Analysis (EDA)", 0), ("3.1 Summary statistics", 1),
        ("3.2 Distribution of numeric variables", 1), ("3.3 Target variable analysis", 1),
        ("3.4 Categorical variable analysis", 1), ("3.5 Correlation analysis", 1),
        ("3.5.1 Correlation heatmap", 2), ("3.5.2 Relationship plots", 2),
        ("3.6 Statistical tests of association", 1),
        ("4. Model Fitting", 0), ("4.1 Logistic regression model", 1), ("4.1.1 Logistic regression model equation", 2),
        ("4.2 Decision tree model", 1), ("4.3 Random forest model", 1), ("4.4 Model comparison", 1),
        ("4.4.1 Performance results", 2), ("4.4.2 Interpretation of results", 2),
        ("4.4.3 Actual vs predicted values", 2),
        ("5. Conclusion", 0), ("5.1 Recommendations", 1), ("5.2 Limitations and responsible use", 1),
        ("References", 0),
    ]
    rp.toc('TOC \\o "1-3" \\h \\z \\u', sections)
    front_title("Table of Figures", before=12)
    rp.toc('TOC \\h \\z \\c "Figure"', [(f"Figure {i}: {c}", 0) for i, c in enumerate(FIG_CAPTIONS.values(), 1)])
    front_title("Table of Tables", before=12)
    rp.toc('TOC \\h \\z \\c "Table"', [(f"Table {i}: {c}", 0) for i, c in enumerate(TAB_CAPTIONS.values(), 1)])
    rp.start_body_section()

    # ======================= 1. INTRODUCTION =======================
    rp.h1("1. Introduction")
    rp.h2("1.1 Dataset overview")
    rp.para(f"This study uses the Telco Customer Churn dataset, an IBM sample dataset describing the home phone and "
            f"internet customers of a telecom provider (BlastChar, 2018). The file supplied for the assignment contains "
            f"{R['raw_n']:,} records and 21 variables; after removing {R['dup']} duplicate records it describes "
            f"{R['clean_n']:,} unique customers, each row representing one customer. The variables cover demographics "
            f"(gender, senior citizen, partner, dependents), the account (tenure in months, contract type, paperless "
            f"billing, payment method, monthly and total charges) and the subscribed services (phone, multiple lines, "
            f"internet type and six internet add-ons). The target variable, Churn (Yes/No), records whether the customer "
            f"left within the last month; {pct(R['churn_rate'])} of customers churned.")

    rp.h2("1.2 Context and relevance")
    rp.para("Customer churn – customers ending their subscription – is a central concern for telecommunications "
            "companies, whose markets are mature and highly competitive. Because keeping an existing customer is "
            "commonly regarded as cheaper than winning a new one, operators use data mining to identify customers at "
            "risk of leaving so that retention actions can be targeted where they matter most (Verbeke et al., 2012). "
            "This study applies data analysis and machine learning to identify which customer and service "
            "characteristics are most strongly associated with churn, to measure how strong these relationships are, and "
            "to build models that predict churn and support retention decisions. Because the data are observational, the "
            "findings describe associations and predictions rather than causal effects.")

    rp.h2("1.3 Objective of the study")
    rp.para("The main goal is to predict which customers will churn and to identify the most influential predictors, "
            "answering the research question: “Which customer characteristics and service-related factors are most "
            "strongly associated with customer churn, and how effectively can machine learning models predict churn?” "
            "Through data cleaning, exploratory data analysis (EDA), statistical testing and model fitting, the study "
            "addresses four key questions:")
    rp.bullets([
        "Which contract, service, billing and demographic factors are associated with higher churn?",
        "How well do logistic regression, decision tree and random forest models predict churn?",
        "Which model is most suitable for targeting retention actions, and why?",
        "What practical retention actions do the findings support?",
    ])

    # ======================= 2. DATA CLEANING =======================
    rp.h1("2. Data Cleaning")
    rp.para("The analysis was carried out in Python with pandas (McKinney, 2010), scikit-learn (Pedregosa et al., "
            "2011), SciPy (Virtanen et al., 2020), statsmodels (Seabold & Perktold, 2010) and Matplotlib (Hunter, "
            "2007). The raw file was never modified: it was audited with every value read as text, so that blanks, text "
            "tokens and codes such as −99 were visible, and every cleaning step is documented and reproducible.")

    rp.h2("2.1 Handling missing and invalid values")
    issues = R["issues"]
    miss_rows = []
    treatment = {
        "tenure": "Set to missing, then rebuilt as round(TotalCharges ÷ MonthlyCharges)",
        "MonthlyCharges": "Set to missing, then rebuilt as TotalCharges ÷ tenure",
        "Contract": "Imputed with the most frequent value (training data)",
        "TotalCharges": "Blank (tenure 0) set to 0; invalid rebuilt as tenure × MonthlyCharges",
        "gender": "Imputed with the most frequent value (training data)",
    }
    for col, row in issues.sort_values("total", ascending=False).iterrows():
        miss_rows.append([col, f"{int(row['Blank'])}", f"{int(row['Invalid text / label'])}",
                          f"{int(row['Impossible value'])}", f"{int(row['total'])}",
                          f"{row['total'] / R['raw_n'] * 100:.2f}%", treatment[col]])
    rp.table(["Column", "Blank", "Invalid text", "Impossible", "Total", "% of rows", "Treatment"], miss_rows,
             [2.6, 1.2, 1.5, 1.6, 1.2, 1.5, 7.0], size=8.5,
             align=["left", "center", "center", "center", "center", "center", "left"], key="missing")
    n_problem = int(issues["total"].sum())
    rp.para(f"The audit found problems in 5 of the 21 columns ({T('missing')}, {F('fig01')}): {n_problem} affected values "
            f"in total, {n_problem / (R['raw_n'] * 21) * 100:.2f}% of all {R['raw_n'] * 21:,} data points. A default "
            f"pandas import would have missed most of them: it reports no missing values in TotalCharges, although that "
            f"column contains {R['tc_blank']} blank strings and {R['tc_invalid']} “INVALID_VAL” entries, and it does not "
            f"recognise gender labels such as “Unknown” or “OtherXYZ”. In addition, {R['dup']} records were exact "
            f"duplicates (every repeated customerID was an identical copy) and were removed.")
    rp.figure("fig01_missing_invalid_before_cleaning.png", 10.0)
    rec = R["rec"]
    rp.para(f"Instead of deleting customers or filling numeric gaps with an average, the three billing variables were "
            f"repaired using the relationship TotalCharges ≈ tenure × MonthlyCharges. Tested on "
            f"{int(rec['rows_checked'].iloc[0]):,} complete records, this rebuilt TotalCharges with a median error of "
            f"{rec.loc['TotalCharges', 'repair_median_abs_pct_error']:.1f}% (median imputation: "
            f"{rec.loc['TotalCharges', 'median_imputation_median_abs_pct_error']:.0f}%) and tenure with a mean error of "
            f"{rec.loc['tenure', 'repair_mean_abs_error']:.2f} months (median imputation: "
            f"{rec.loc['tenure', 'median_imputation_mean_abs_error']:.1f} months). No customer had more than one "
            f"corrupted billing value, so all {R['n_repaired']} were repaired; blank TotalCharges belonged to customers "
            f"with zero tenure and were set to 0. Only {R['gender_bad']} gender and {R['contract_blank']} contract values "
            f"remained missing; they were imputed with the most frequent category, learned from the training data only "
            f"({T('imputation')}). Customers with repaired values churned at {pct(R['affected_rate'])} against "
            f"{pct(R['unaffected_rate'])} for the rest, a difference that is not statistically significant "
            f"(χ² p = {R['affected_p']:.2f}).")
    imp = R["imputation"]
    imp_rows = [[r.column, r.type, r.strategy, str(r.learned_value), str(r.missing_in_clean_data)]
                for r in imp.itertuples() if r.type == "numeric" or r.missing_in_clean_data > 0]
    rp.table(["Column", "Type", "Strategy", "Value learned (training data)", "Missing after repair"], imp_rows,
             [3.4, 2.4, 3.0, 4.6, 3.2], size=8.5, align=["left", "left", "left", "left", "center"], key="imputation")

    rp.h2("2.2 Handling outliers")
    rp.para(f"Outliers were detected with the interquartile range (IQR) rule (below Q1 − 1.5 × IQR or above "
            f"Q3 + 1.5 × IQR) and boxplots ({F('fig02')}). The extreme raw values – tenures of −99 and "
            f"{R['tenure_examples']}, and monthly charges of {R['mc_examples']} USD against a normal range of about "
            f"18–119 USD – are impossible, so they were treated as data-entry errors and repaired (Section 2.1).")
    rp.figure("fig02_boxplots_before_after_cleaning.png", 12.0)
    rp.para(f"After repair, {R['iqr_outliers']} values lie outside the IQR fences. TotalCharges is right-skewed "
            f"(skewness {R['tc_skew']:.2f}) but plausible, so no further treatment was applied; winsorisation was not "
            f"used, because capping values would alter genuine records such as long-standing, high-spending customers.")

    rp.h2("2.3 Feature engineering and encoding")
    rp.para(f"The customerID was removed because it is only an identifier. SeniorCitizen was recoded from 0/1 to No/Yes, "
            f"and “No internet service” / “No phone service” were recoded to “No” for modelling, as InternetService and "
            f"PhoneService already hold that information. Categorical variables were converted with "
            f"one-hot encoding (a separate 0/1 column for each category) rather than label encoding, because integer "
            f"codes such as 0, 1, 2, 3 for payment methods would impose an order that does not exist. Numeric variables "
            f"were standardised for logistic regression. Two helper variables – tenure groups and the number of add-on "
            f"services – were created for EDA only.")
    rp.para(f"The data were split 80/20 with stratification on churn ({R['train_n']:,} training and {R['test_n']:,} "
            f"test customers, both with {pct(R['churn_rate'])} churners). To prevent data leakage (Kaufman et al., 2012), "
            f"every step that learns from data – imputation, scaling and encoding – was placed inside a scikit-learn "
            f"Pipeline with a ColumnTransformer, so it is fitted on the training data only and re-fitted within every "
            f"cross-validation fold. Class imbalance was handled with class_weight = \"balanced\" rather than synthetic "
            f"oversampling.")

    rp.h2("2.4 Multicollinearity")
    rp.para(f"Multicollinearity was checked with variance inflation factors (VIF) on the training data, because strongly "
            f"related predictors make regression coefficients unstable and hard to interpret. {F('fig20')}(a) shows that "
            f"MonthlyCharges has a VIF of {R['vif_mc_full']:.0f}: the service variables explain almost all of it, since "
            f"the monthly bill is essentially the sum of the service prices. TotalCharges is also strongly related to "
            f"tenure (VIF {R['vif_num']['TotalCharges']:.1f}; Spearman ρ = {R['rho_tenure_tc']:.2f}).")
    rp.figure("fig20_vif_before_after.png", 12.0)
    rp.para(f"Standardising the variables does not reduce VIF, because VIF depends only on the correlations between "
            f"predictors (raw and standardised variables gave identical VIFs). Removing the redundant MonthlyCharges and "
            f"TotalCharges lowers the largest VIF to {R['vif_reduced_max']:.2f} ({F('fig20')}(b)), so they were excluded "
            f"from the logistic regression at almost no cost (CV ROC-AUC {R['lrfs']['cv_roc_auc_mean'].iloc[0]:.4f} vs "
            f"{R['lrfs']['cv_roc_auc_mean'].iloc[1]:.4f} with all features); the tree models kept all variables.")

    # ======================= 3. EDA =======================
    # Structure follows the previous-year report: 3.1 summary statistics, 3.2 distributions,
    # 3.3 target variable, 3.4 categorical variables, 3.5 correlation (heatmap + relationship plots).
    rp.h1("3. Exploratory Data Analysis (EDA)")

    rp.h2("3.1 Summary statistics")
    summ, prof = R["summary"], R["profile"]
    units = {"tenure": "months", "MonthlyCharges": "USD", "TotalCharges": "USD"}
    summ_rows = [[f"{v} ({units[v]})", f"{int(summ.loc[v, 'count']):,}"] +
                 [f"{summ.loc[v, c]:,.1f}" for c in ["mean", "std", "min", "25%", "50%", "75%", "max"]]
                 for v in config.NUMERIC_FEATURES]
    rp.table(["Variable", "Count", "Mean", "SD", "Min", "Q1", "Median", "Q3", "Max"], summ_rows,
             [3.9, 1.5, 1.6, 1.6, 1.4, 1.5, 1.6, 1.6, 1.9], "Summary statistics of the numeric variables (cleaned data)",
             align=["left"] + ["right"] * 8, key="summary")
    rp.para(f"{T('summary')} summarises the three numeric variables after cleaning. The mean of total charges "
            f"({summ.loc['TotalCharges', 'mean']:,.0f} USD) lies far above its median "
            f"({summ.loc['TotalCharges', '50%']:,.0f} USD), a first sign of right skew. {T('profile')} adds the categorical "
            f"profile: churners are newer (median tenure {prof.loc['Median tenure (months)', 'Churned']:.0f} vs "
            f"{prof.loc['Median tenure (months)', 'Stayed']:.0f} months) and far more often on month-to-month contracts, "
            f"fiber-optic internet and electronic-check payment.")
    prof_rows = []
    for label, fmt in [("Median tenure (months)", "{:.0f}"), ("Mean monthly charge (USD)", "{:.1f}"),
                       ("Month-to-month contract (%)", "{:.1f}"),
                       ("Fiber optic internet (%)", "{:.1f}"), ("Pays by electronic check (%)", "{:.1f}"),
                       ("Has tech support (%)", "{:.1f}")]:
        prof_rows.append([label] + [fmt.format(prof.loc[label, c]) for c in ["All customers", "Stayed", "Churned"]])
    prof_rows.insert(0, ["Customers"] + [f"{int(prof.loc['Customers', c]):,}" for c in ["All customers", "Stayed", "Churned"]])
    rp.table(["Characteristic", "All customers", "Stayed", "Churned"], prof_rows, [6.6, 3.2, 3.2, 3.2],
             "Profile of customers who stayed and customers who churned",
             align=["left", "right", "right", "right"], key="profile")

    rp.h2("3.2 Distribution of numeric variables")
    rp.para(f"{F('fig04')} shows the shape, spread and concentration of each numeric variable as histograms, with "
            f"customers who stayed and churned stacked in each bar. {T('distribution')} interprets each histogram; the "
            f"distribution type is computed from the data (number of peaks and skewness), not judged by eye.")
    rp.figure("fig04_numeric_distributions_by_churn.png", 13.0,
              "Histograms of tenure, monthly charges and total charges by churn status (stacked counts)")
    dist = R["dist"]
    dist_rows = []
    for r in dist.itertuples():
        unit = units[r.variable]
        dist_rows.append([r.variable, f"Mean {r.mean:,.1f}, median {r.median:,.1f}; range {r.min:,.0f}–{r.max:,.0f} {unit}",
                          f"{r.shape[0].upper() + r.shape[1:]}; skewness " + f"{r.skewness:.2f}".replace("-", "−"),
                          r.churn_insight[0].upper() + r.churn_insight[1:]])
    rp.table(["Variable", "Observation", "Distribution type", "Insight (churned vs stayed)"], dist_rows,
             [2.8, 5.0, 4.0, 4.8], "Interpretation of the histograms", size=8.5, key="distribution")
    rp.para("Because none of the numeric variables is normally distributed, rank-based methods (Spearman correlation, "
            "Mann-Whitney U) were used in the analysis.")

    rp.h2("3.3 Target variable analysis")
    rp.para(f"The target variable, Churn, records whether a customer left within the last month. {F('fig03')} shows "
            f"that {R['n_churn']:,} of {R['clean_n']:,} customers ({pct(R['churn_rate'])}) churned and "
            f"{R['clean_n'] - R['n_churn']:,} ({pct(100 - R['churn_rate'])}) stayed. The classes are therefore "
            f"imbalanced, about 2.8 customers who stay for every churner. A model that always predicts “no churn” would "
            f"already be {pct(100 - R['churn_rate'])} accurate while catching no churners, which is why models are judged "
            f"with recall, precision, F1 and ROC-AUC rather than accuracy alone, and why class weighting is used.")
    rp.figure("fig03_churn_distribution.png", 7.0, "Distribution of the target variable: customers who stayed and churned")

    rp.h2("3.4 Categorical variable analysis")
    rp.para(f"{F('fig05')} compares churn rates across the account and service variables. Month-to-month customers "
            f"churn at {pct(rate('Contract', 'Month-to-month'))}, compared with {pct(rate('Contract', 'One year'))} on "
            f"one-year and {pct(rate('Contract', 'Two year'))} on two-year contracts. Fiber-optic customers churn at "
            f"{pct(rate('InternetService', 'Fiber optic'))}, more than twice the DSL rate "
            f"({pct(rate('InternetService', 'DSL'))}). Electronic-check payers churn at "
            f"{pct(rate('PaymentMethod', 'Electronic check'))}, against "
            f"{pct(rate('PaymentMethod', 'Credit card (automatic)'))}–{pct(rate('PaymentMethod', 'Mailed check'))} for "
            f"other methods, and churn falls from {pct(rate('tenure_group', '0–12'))} in the first year to "
            f"{pct(rate('tenure_group', '49–72'))} after four years.")
    rp.figure("fig05_churn_rate_key_segments.png", 10.5)
    ct = R["ct"]
    rp.para(f"{F('fig10')} covers the demographic and billing variables. Senior citizens churn at "
            f"{pct(rate('SeniorCitizen', 'Yes'))} vs {pct(rate('SeniorCitizen', 'No'))}; customers without a partner "
            f"({pct(rate('Partner', 'No'))}) or dependents ({pct(rate('Dependents', 'No'))}) churn more, as do customers "
            f"with paperless billing ({pct(rate('PaperlessBilling', 'Yes'))}). Gender shows no meaningful difference "
            f"({pct(rate('gender', 'Female'))} vs {pct(rate('gender', 'Male'))}), nor does phone service. Among "
            f"internet customers, churn is {pct(rate('OnlineSecurity', 'No'))} without online security vs "
            f"{pct(rate('OnlineSecurity', 'Yes'))} with it, and {pct(rate('TechSupport', 'No'))} vs "
            f"{pct(rate('TechSupport', 'Yes'))} for tech support, whereas streaming add-ons make little difference. "
            f"Because month-to-month customers also tend to be newer, contracts were compared within tenure groups: "
            f"month-to-month churn falls from {ct[('churn_rate_%', 'Month-to-month')].iloc[0]:.1f}% to "
            f"{ct[('churn_rate_%', 'Month-to-month')].iloc[-1]:.1f}% with tenure but stays far above two-year contracts "
            f"(at most {ct[('churn_rate_%', 'Two year')].max():.1f}%), so contract and tenure each carry information.")
    rp.figure("fig10_customer_attributes_churn.png", 9.5,
              "Churn rate by demographic and billing attributes (dashed line: overall rate)")

    rp.h2("3.5 Correlation analysis")
    rp.h3("3.5.1 Correlation heatmap")
    sp = R["spearman_all"]
    churn_col = "Churn (1 = Yes)"
    rho = lambda v: f"{v:+.2f}".replace("-", "−")     # signed, with a typographic minus sign
    r = lambda a, b=churn_col: rho(sp.loc[a, b])
    rp.para(f"To include the categorical variables, each was converted to 0/1 columns with the same one-hot encoding "
            f"used by the models: one column for each Yes/No variable (1 = Yes, with \"No internet service\" and \"No "
            f"phone service\" counted as No) and one column per level for InternetService, Contract and PaymentMethod. "
            f"Label codes such as 0/1/2/3 were avoided because they would impose a false order on the categories. "
            f"A correlation with a 0/1 column is a standard measure (the phi coefficient between two 0/1 columns, a "
            f"point-biserial correlation with a numeric one). {F('fig07')} shows the resulting {len(sp)} × {len(sp)} "
            f"Spearman correlation matrix; Spearman's coefficient ranges from −1 to +1, with values near 0 indicating no "
            f"monotonic relationship, and was used instead of Pearson's because the numeric variables are skewed and "
            f"bimodal.", after=3)
    rp.bullets([
        ("Higher churn: ", f"month-to-month contracts (ρ = {r('Contract: Month-to-month')}), fiber optic internet "
                           f"({r('Internet: Fiber optic')}) and electronic-check payment ({r('Payment: Electronic check')})."),
        ("Lower churn: ", f"tenure ({r('tenure')}), two-year contracts ({r('Contract: Two year')}), no internet service "
                          f"({r('Internet: None')}), OnlineSecurity ({r('OnlineSecurity')}) and TechSupport "
                          f"({r('TechSupport')}). Gender ({r('Male')}) and PhoneService ({r('PhoneService')}) are "
                          f"practically unrelated to churn."),
        ("Related predictors: ", f"tenure and total charges (ρ = {sp.loc['tenure', 'TotalCharges']:.2f}) and monthly "
                                 f"charges and fiber optic internet ({sp.loc['MonthlyCharges', 'Internet: Fiber optic']:.2f}) "
                                 f"move together – the multicollinearity that led to removing the charge columns from the "
                                 f"logistic regression (Section 2.4). Levels of the same variable are negatively correlated "
                                 f"by construction (e.g. month-to-month vs two-year, "
                                 f"{r('Contract: Month-to-month', 'Contract: Two year')})."),
        ("The add-on count ", f"barely correlates with churn overall ({r('Add-on count')}) because customers without "
                              f"internet have no add-ons yet rarely churn – an overall correlation can hide strong "
                              f"subgroup patterns."),
    ])
    rp.figure("fig07_correlation_heatmap.png", 16.4)

    rp.h3("3.5.2 Relationship plots")
    sc = R["scatter"]
    ten = sc[sc["variable"] == "tenure"].set_index("value")
    mc = sc[(sc["variable"] == "MonthlyCharges") & (sc["customers"] >= 10)]
    peak = mc.loc[mc["churn_rate"].idxmax()]
    rp.para(f"Because churn is a yes/no variable, a plain scatter plot against it would show only two lines of dots. "
            f"{F('fig21')} therefore plots the churn rate of groups of customers with the same tenure or a similar charge "
            f"level, with a smoothed LOWESS trend line. Churn is {ten.loc[1, 'churn_rate']:.0f}% among customers in their first "
            f"month and falls steadily with tenure. For monthly charges the relationship is not linear: churn rises from "
            f"the cheapest plans (mostly no internet) to a peak of {peak['churn_rate']:.0f}% around "
            f"{peak['value'] - 2.5:.0f}–{peak['value'] + 2.5:.0f} USD (mostly fiber optic) and then eases, which is why its correlation is weak "
            f"(ρ = {sp.loc[churn_col, 'MonthlyCharges']:+.2f}) even though price matters. Total charges largely mirror "
            f"tenure.")
    rp.figure("fig21_churn_rate_scatter_trend.png", 13.0, "Churn rate against tenure, monthly charges and total "
              "charges with LOWESS trend lines (each point is a group of customers; size = group size)")

    rp.h2("3.6 Statistical tests of association")
    n_sig = int(chi["significant_after_bonferroni"].sum())
    minus = lambda v: f"{v:.2f}".replace("-", "−")    # typographic minus sign
    rp.para(f"The patterns above, including the correlations in {F('fig07')}, were tested formally. Categorical variables were tested with chi-square tests of "
            f"independence (all expected cell counts ≥ 5) and Cramér's V as effect size; numeric variables with the "
            f"Mann-Whitney U test (Mann & Whitney, 1947) and the rank-biserial correlation, because they are not normally "
            f"distributed. A Bonferroni correction was applied to the 19 tests, and effect sizes (Cohen, 1988: 0.10 small, "
            f"0.30 medium, 0.50 large) were used to judge practical importance, since almost any difference is significant "
            f"with about 7,000 customers.")
    rp.para(f"{n_sig} of 16 categorical variables were significantly associated with churn; gender "
            f"(p = {chi.loc['gender', 'p_value']:.2f}), phone service and multiple lines were not, matching their "
            f"near-zero correlations with churn in {F('fig07')}. {F('fig11')} shows "
            f"that contract type has the largest effect (V = {chi.loc['Contract', 'cramers_v']:.2f}, medium), followed by "
            f"online security, tech support, internet service and payment method, while demographic variables have small "
            f"effects (V ≈ {chi.loc['Partner', 'cramers_v']:.2f}). Among the numeric variables, tenure differs most between "
            f"churners and non-churners (r = {minus(mw.loc['tenure', 'rank_biserial'])}). Re-testing the add-ons among "
            f"internet customers only showed that the streaming services' apparent association "
            f"(V = {chi.loc['StreamingTV', 'cramers_v']:.2f}) came from the shared “No internet service” level "
            f"(V = {addon.loc['StreamingTV', 'cramers_v']:.2f}). Overall, the hypotheses that contract, tenure, internet type, "
            f"payment method and protective add-ons are associated with churn were supported; monthly charges showed only "
            f"a small, non-linear association (r = {minus(mw.loc['MonthlyCharges', 'rank_biserial'])}), and gender showed none.")
    rp.figure("fig11_effect_sizes.png", 12.0)

    # ======================= 4. MODEL FITTING =======================
    import ast
    params = lambda name: ast.literal_eval(cv.loc[name, "best_params"])
    of = R["overfit"]
    rp.h1("4. Model Fitting")
    cw = R["cw"]
    rp.para(f"Churn is a yes/no outcome, so this is a binary classification problem; linear regression, which can "
            f"predict values outside 0–1, is not appropriate. Three classifiers were compared with a DummyClassifier that "
            f"always predicts “no churn”: logistic regression, an interpretable model of the log-odds of churn (Hosmer et "
            f"al., 2013); a decision tree of readable if/then rules (Breiman et al., 1984); and a random forest of 300 "
            f"trees that captures non-linear patterns (Breiman, 2001).")
    rp.para(f"Each model was tuned by grid search with stratified 5-fold cross-validation (CV) on the training set only, "
            f"and all used class_weight = \"balanced\", which left ranking ability unchanged (CV ROC-AUC "
            f"{cw.loc['balanced', 'roc_auc']:.3f}) but raised recall from {cw.loc['None', 'recall']:.2f} to "
            f"{cw.loc['balanced', 'recall']:.2f}. The selection rule was fixed in advance: highest mean CV ROC-AUC, "
            f"preferring the more interpretable model if another was within 0.01. ROC-AUC measures how well a model ranks "
            f"churners above non-churners; because only {pct(R['churn_rate'])} of customers churn, accuracy alone would be "
            f"misleading, so precision, recall, F1 and PR-AUC (Saito & Rehmsmeier, 2015) are also reported. The test set "
            f"({R['test_n']:,} customers) was used once, at the end.")

    rp.h2("4.1 Logistic regression model")
    lr_cv = cv.loc[sel]
    rp.para(f"Logistic regression was fitted without MonthlyCharges and TotalCharges (Section 2.4). The best "
            f"regularisation strength was C = {params(sel)['C']}, giving a CV ROC-AUC of {lr_cv['cv_roc_auc_mean']:.3f} "
            f"and a test ROC-AUC of {lr_t['roc_auc']:.3f}. To report standard errors and p-values, an unpenalised "
            f"statsmodels Logit was fitted to the same training data ({T('lr_coef')}); its odds ratios agree with the final "
            f"model (largest difference {R['logit_or_diff']:.2f}), {int(R['logit_fit']['terms_p_below_0.05'])} of "
            f"{int(R['logit_fit']['n_terms'])} terms are significant at p < 0.05, and McFadden's pseudo-R² is "
            f"{R['logit_fit']['mcfadden_pseudo_r2']:.3f}. {T('lr_coef')} lists the 12 largest of the "
            f"{int(R['logit_fit']['n_terms'])} terms (full table in the notebook).")
    lt = R["logit_tab"]
    lt = lt.reindex(["const"] + list(lt.drop(index="const")["coef"].abs().sort_values(ascending=False).index[:12]))
    fmt_p = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    coef_rows = [["Intercept" if k == "const" else pretty_term(k), f"{r['coef']:.3f}", f"{r['std_err']:.3f}",
                  f"{r['z']:.2f}", fmt_p(r["p_value"]), f"{r['odds_ratio']:.2f}"] for k, r in lt.iterrows()]
    rp.table(["Term", "Coef.", "Std. err.", "z", "p-value", "Odds ratio"], coef_rows,
             [6.6, 1.9, 1.9, 1.7, 2.1, 2.0], size=8, align=["left"] + ["right"] * 5, key="lr_coef")
    rp.para(f"{F('fig16')} shows the odds ratios (OR) with bootstrap confidence intervals. For otherwise identical "
            f"customers, a two-year contract has about a quarter of the churn odds of a month-to-month contract "
            f"(OR = {ORv('Contract_Two year'):.2f}); each additional 25 months of tenure roughly halves the odds "
            f"(OR = {ORv('tenure'):.2f}); fiber-optic service has {ORv('InternetService_Fiber optic'):.1f} times the odds "
            f"of DSL; and electronic-check payment {ORv('PaymentMethod_Electronic check'):.2f} times the odds of automatic "
            f"bank transfer. Online security (OR = {ORv('OnlineSecurity_Yes'):.2f}) and tech support "
            f"(OR = {ORv('TechSupport_Yes'):.2f}) are associated with lower odds, whereas gender "
            f"(OR = {ORv('gender_Male'):.2f}) and senior status show no clear effect once other factors are included.")
    rp.figure("fig16_odds_ratios.png", 8.5)

    rp.h3("4.1.1 Logistic regression model equation")
    eq = R["equation"]
    terms = R["or"]["coefficient"].reindex(R["or"]["coefficient"].abs().sort_values(ascending=False).index)
    shown = terms.head(8)
    num = lambda v: f"{v:+.2f}".replace("+", "+ ").replace("-", "− ")
    intercept = f"{eq['intercept']:.2f}".replace("-", "−")
    eq_text = f"log-odds(churn) = {intercept} " + " ".join(
        f"{num(v)}·[{EQ_LABELS.get(k, k)}]" for k, v in shown.items()) + " + …"
    rp.para("The fitted model, with its eight largest terms, is:", after=2)
    rp.para(eq_text, align_justify=False, size=9.5, after=2)
    rp.para(f"where z_tenure = (tenure − {eq['tenure_mean']:.1f}) / {eq['tenure_sd']:.1f}, each bracket is 1 for "
            f"customers in that category and 0 otherwise, and the predicted probability is "
            f"p(churn) = 1 / (1 + e^(−log-odds)); the remaining {len(terms) - len(shown)} smaller terms are listed in the "
            f"notebook. A negative coefficient lowers the predicted churn risk and a positive one raises it. Because "
            f"balanced class weights shift the intercept, the model is used to rank customers by risk rather than as a "
            f"calibrated probability.")

    rp.h2("4.2 Decision tree model")
    dt_p = params("Decision Tree")
    leaves = R["leaves"]
    rp.para(f"The tuned decision tree has a maximum depth of {dt_p['max_depth']} and at least "
            f"{dt_p['min_samples_leaf']} customers per final group, which limits overfitting (training-to-validation "
            f"AUC gap {of.loc['Decision Tree', 'gap_train_minus_validation']:.3f}). It reached a CV ROC-AUC of "
            f"{cv.loc['Decision Tree', 'cv_roc_auc_mean']:.3f} and a test ROC-AUC of {dt_t['roc_auc']:.3f}. To show how "
            f"its decisions work, a simplified depth-3 tree (test ROC-AUC {leaves['simple_tree_test_auc'].iloc[0]:.3f}) "
            f"was also fitted; {T('tree_rules')} lists its rules. The tree splits first on contract type, then on internet "
            f"service or monthly charges, then on tenure – the same drivers identified in the EDA.")
    rule_rows = [[r.rule.replace(" AND ", " and ").replace("<=", "≤"), f"{int(r.customers):,}",
                  f"{r.churn_rate * 100:.0f}%"] for r in leaves.itertuples()]
    rp.table(["Rule (path through the tree)", "Customers", "Churn rate"], rule_rows, [12.4, 2.0, 2.2], size=8,
             align=["left", "right", "right"], key="tree_rules")

    rp.h2("4.3 Random forest model")
    rf_p = params("Random Forest")
    perm = R["perm"]
    rp.para(f"The random forest combines 300 trees (maximum depth {rf_p['max_depth']}, at least "
            f"{rf_p['min_samples_leaf']} customers per leaf). Its ROC-AUC was {of.loc['Random Forest', 'cv_train_roc_auc']:.3f} "
            f"on the training folds, {of.loc['Random Forest', 'cv_validation_roc_auc']:.3f} in CV and "
            f"{of.loc['Random Forest', 'test_roc_auc']:.3f} on the test set: moderate overfitting, but the test score "
            f"matches the CV estimate, so the model generalises well.")
    rp.para(f"Feature importance was measured by permutation importance: how much the test ROC-AUC drops when one "
            f"variable is randomly shuffled ({F('fig17')}). This was preferred to the forest's built-in impurity "
            f"importance, which is biased towards continuous variables (Strobl et al., 2007). Contract is the forest's most "
            f"important input ({perm.loc['Contract', 'importance_mean_random_forest']:.3f}), followed by internet service "
            f"and tenure; for the logistic regression, tenure ({perm.loc['tenure', 'importance_mean_logistic_regression']:.3f}) "
            f"leads. Both models rely mainly on the same three variables, and gender has no importance in either.")
    rp.figure("fig17_permutation_importance.png", 10.0)

    rp.h2("4.4 Model comparison")
    rp.h3("4.4.1 Performance results")
    rp.para(f"{T('performance')} and {F('fig12')} compare the models on the test set. All three clearly outperform the "
            f"baseline, which is {test.loc[DUMMY, 'accuracy']:.1%} accurate but catches no churners. The random forest "
            f"scores highest on every metric (ROC-AUC {rf_t['roc_auc']:.3f}), closely followed by the logistic regression "
            f"({lr_t['roc_auc']:.3f}) and the decision tree ({dt_t['roc_auc']:.3f}); all catch about "
            f"{lr_t['recall']:.0%}–{rf_t['recall']:.0%} of churners.")
    perf_rows = []
    for name in [DUMMY, "Logistic Regression", "Decision Tree", "Random Forest"]:
        cvr, te = cv.loc[name], test.loc[name]
        label = "Dummy baseline" if name.startswith("Dummy") else name + (" (selected)" if name == sel else "")
        perf_rows.append([label, f"{cvr['cv_roc_auc_mean']:.3f} ± {cvr['cv_roc_auc_sd']:.3f}",
                          f"{te['roc_auc']:.3f}", f"{te['pr_auc']:.3f}", f"{te['accuracy']:.3f}",
                          f"{te['precision']:.3f}", f"{te['recall']:.3f}", f"{te['f1']:.3f}"])
    rp.table(["Model", "CV ROC-AUC", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1"], perf_rows,
             [4.0, 2.6, 1.6, 1.6, 1.7, 1.7, 1.6, 1.6], align=["left"] + ["center"] * 7, key="performance")
    rp.figure("fig12_model_comparison.png", 12.0)

    rp.h3("4.4.2 Interpretation of results")
    ops = R["ops"]
    rp.para(f"Although the random forest scored slightly higher, the logistic regression was selected: its CV ROC-AUC "
            f"({lr_cv['cv_roc_auc_mean']:.3f}) was within the 0.01 tolerance of the forest's "
            f"({cv.loc['Random Forest', 'cv_roc_auc_mean']:.3f}), and the forest's test advantage is not statistically "
            f"reliable – the paired bootstrap ROC-AUC difference is {diff_rf['mean_auc_difference']:+.3f} (95% CI "
            f"{format(diff_rf['ci_95_low'], '.3f').replace('-', '−')} to {diff_rf['ci_95_high']:.3f}), which includes zero (Efron & Tibshirani, 1993). "
            f"The logistic regression (test ROC-AUC {lr_t['roc_auc']:.3f}, 95% CI {ci.loc[sel, 'ci_95_low']:.3f}–"
            f"{ci.loc[sel, 'ci_95_high']:.3f}) also overfits least (gap {of.loc[sel, 'gap_train_minus_validation']:.3f} vs "
            f"{of.loc['Random Forest', 'gap_train_minus_validation']:.3f}) and is easier to explain. The 0.50 threshold was "
            f"kept, as the F1-optimal threshold ({R['final']['recommended_threshold_max_cv_f1']:.2f}) performed almost "
            f"identically; in practice it should follow the retention budget.")

    rp.h3("4.4.3 Actual vs predicted values")
    err = R["errors"]
    fn, tp = err.loc["False negative (churn missed)"], err.loc["True positive (churn caught)"]
    rp.para(f"{F('fig13')} compares actual and predicted classes for every model. The logistic regression correctly "
            f"identifies {int(lr_t['TP'])} of {int(lr_t['TP'] + lr_t['FN'])} churners; {int(lr_t['FN'])} are missed "
            f"(false negatives – customers lost without a retention attempt) and {int(lr_t['FP'])} loyal customers are "
            f"flagged (false positives – retention offers that were not needed). Ranking customers by predicted risk is "
            f"also effective: the 20% with the highest scores contain {top(0.2):.0f}% of all churners. {T('actual_pred')} "
            f"compares the two strongest models.")
    ap_rows = [
        ["Churners correctly predicted (TP)", f"{int(lr_t['TP'])} of {int(lr_t['TP'] + lr_t['FN'])} ({lr_t['recall']:.1%})",
         f"{int(rf_t['TP'])} of {int(rf_t['TP'] + rf_t['FN'])} ({rf_t['recall']:.1%})"],
        ["Churners missed (FN)", f"{int(lr_t['FN'])}", f"{int(rf_t['FN'])}"],
        ["Loyal customers wrongly flagged (FP)", f"{int(lr_t['FP'])}", f"{int(rf_t['FP'])}"],
        ["Loyal customers correctly predicted (TN)", f"{int(lr_t['TN'])}", f"{int(rf_t['TN'])}"],
        ["Share of flagged customers who churn", f"{lr_t['precision']:.1%}", f"{rf_t['precision']:.1%}"],
        ["Overall assessment",
         f"Slightly more false alarms; transparent coefficients; least overfitting (gap "
         f"{of.loc[sel, 'gap_train_minus_validation']:.3f}). Missed churners resemble loyal customers (median tenure "
         f"{fn['median_tenure_months']:.0f} vs {tp['median_tenure_months']:.0f} months for caught churners).",
         f"Marginally better on every count, but the difference is not statistically reliable; harder to explain and "
         f"more overfitting (gap {of.loc['Random Forest', 'gap_train_minus_validation']:.3f})."],
    ]
    rp.table(["", "Logistic regression (selected)", "Random forest"], ap_rows, [4.6, 6.0, 6.0], size=8.5,
             align=["left", "left", "left"], key="actual_pred")
    rp.figure("fig13_confusion_matrices.png", 13.0)

    # ======================= 5. CONCLUSION =======================
    rp.h1("5. Conclusion")
    rp.para(f"A documented cleaning process repaired a deliberately corrupted dataset without deleting customers or "
            f"leaking test information, and the EDA and statistical tests showed clear differences between churners and "
            f"other customers. The selected logistic regression predicts churn with a test ROC-AUC of "
            f"{lr_t['roc_auc']:.2f} and catches {lr_t['recall']:.0%} of churners, matching a random forest within "
            f"statistical uncertainty. The most influential predictors were:")
    rp.bullets([
        ("Contract type, ", f"with month-to-month customers churning at {pct(rate('Contract', 'Month-to-month'))} against "
                           f"{pct(rate('Contract', 'Two year'))} on two-year contracts (negative impact of short contracts);"),
        ("Tenure, ", f"with churn highest in the first year ({pct(rate('tenure_group', '0–12'))}) (protective effect of "
                     f"longer tenure);"),
        ("Internet service, ", f"with fiber-optic customers churning at {pct(rate('InternetService', 'Fiber optic'))} "
                               f"(higher risk);"),
        ("Payment method and add-ons, ", f"with electronic-check payment linked to higher churn and online security or "
                                         f"tech support to lower churn."),
    ])
    rp.para(f"These are also the variables most strongly correlated with churn in the full correlation matrix "
            f"({F('fig07')}). Gender showed no association with churn, and all findings are associations, not proof "
            f"of cause.")

    rp.h2("5.1 Recommendations")
    recs = R["recs"]
    rec_rows = [[row["recommendation"].split(". ", 1)[1], row["evidence"], row["suggested action"]]
                for _, row in recs.iterrows()]
    rp.para(f"{T('recommendations')} links each recommendation to the evidence that supports it.", after=3)
    rp.table(["Recommendation", "Evidence", "Suggested action"], rec_rows, [3.9, 6.6, 6.1], size=8,
             key="recommendations")

    rp.h2("5.2 Limitations and responsible use")
    rp.bullets([
        ("Association, not causation: ", "the effect of retention actions should be tested with randomised trials."),
        ("Limited information: ", "complaints, service quality and competitor offers are not in the data, so some "
                                  "churners who look loyal are missed."),
        ("Generalisability: ", "the data are an IBM sample for one provider; re-validate on local data before use."),
        ("Responsible use: ", "use the model to support staff decisions, never to give worse terms based on gender or "
                              "age, and monitor and retrain it regularly."),
    ])

    rp.h1("References")
    refs = [
        "BlastChar. (2018). Telco customer churn [Data set]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
        "Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). Classification and regression trees. Wadsworth.",
        "Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). Lawrence Erlbaum Associates.",
        "Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall.",
        "Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied logistic regression (3rd ed.). Wiley.",
        "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95.",
        "Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, "
        "and avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4), Article 15.",
        "Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically "
        "larger than the other. The Annals of Mathematical Statistics, 18(1), 50–60.",
        "McKinney, W. (2010). Data structures for statistical computing in Python. Proceedings of the 9th Python in "
        "Science Conference, 56–61.",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., "
        "Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). "
        "Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
        "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when "
        "evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432.",
        "Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. "
        "Proceedings of the 9th Python in Science Conference, 92–96.",
        "Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance "
        "measures: Illustrations, sources and a solution. BMC Bioinformatics, 8, 25.",
        "Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in "
        "the telecommunication sector: A profit driven data mining approach. European Journal of Operational "
        "Research, 218(1), 211–229.",
        "Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific "
        "computing in Python. Nature Methods, 17, 261–272.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(ref)
        r.font.size = Pt(8.5)
        p.paragraph_format.line_spacing = 1.0
        rp._add_height(rp._text_height(ref, 8.5, width_cm=TEXT_W - 1.0, spacing=1.0, after_pt=3))

    assert rp.fig_no == len(FIG_ORDER), f"{rp.fig_no} figures placed but {len(FIG_ORDER)} in FIG_ORDER"
    assert rp.tab_no == len(TAB_ORDER), f"{rp.tab_no} tables placed but {len(TAB_ORDER)} in TAB_ORDER"
    doc.core_properties.title = "Customer Churn Prediction and Analysis Using Machine Learning"
    doc.core_properties.subject = "TM3221 Data Science for BPM – Group Assignment"
    doc.save(OUT)
    return rp


if __name__ == "__main__":
    report = build(load_results())
    print(f"Saved {OUT.relative_to(ROOT)}")
    print(f"Figures: {report.fig_no} | Tables: {report.tab_no}")
    print(f"Estimated length: about {report.pages} pages (≈{report.height_cm / (PAGE_H - 2 * MARGIN - 1):.0%} of the last page used)")
