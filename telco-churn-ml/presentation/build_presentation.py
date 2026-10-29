"""
build_presentation.py
---------------------
Generates the presentation (PowerPoint .pptx) from the executed analysis.

Like the report, every number is read from outputs/tables (via the report's
loader), so slides always match the notebook. Speaker notes are written on
every slide and divided between Presenters A-E.

    python presentation/build_presentation.py
"""

import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports"))
from src import config  # noqa: E402
from build_report import load_results  # noqa: E402

OUT = ROOT / "presentation" / "Telco_Churn_Presentation.pptx"
FIG = config.FIGURE_DIR
CREST = ROOT / "assets" / "uom_crest.png"

# ---------------------------------------------------------------------------
# Design system: navy dominant, signal orange = churn, blue = retained
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
NAVY_DARK = RGBColor(0x14, 0x26, 0x3F)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
GREY = RGBColor(0x9A, 0x98, 0x93)
INK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x52, 0x51, 0x4E)
TINT = RGBColor(0xEE, 0xF3, 0xF9)        # light navy tint for cards
ORANGE_TINT = RGBColor(0xFD, 0xEE, 0xE6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD_FONT, BODY_FONT = "Cambria", "Calibri"
SLIDE_W, SLIDE_H = 13.333, 7.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def text_box(slide, x, y, w, h, paragraphs, size=16, color=INK, font=BODY_FONT, bold=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None, line_spacing=1.1):
    """
    Add a text box. `paragraphs` is a string or a list whose items are strings or
    lists of (text, {options}) runs. Options: bold, color, size, italic, font.
    """
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        shape.name = name
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(0.04))
    paragraphs = [paragraphs] if isinstance(paragraphs, str) else paragraphs
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        runs = [(para, {})] if isinstance(para, str) else para
        for text, opt in runs:
            r = p.add_run()
            r.text = text
            r.font.name = opt.get("font", font)
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", bold)
            r.font.italic = opt.get("italic", False)
            r.font.color.rgb = opt.get("color", color)
    return shape


def bullets(slide, x, y, w, h, items, size=16, color=INK, space_after=8):
    """Real bulleted list (bullet characters added through paragraph properties)."""
    from pptx.oxml.ns import qn
    from lxml import etree
    shape = text_box(slide, x, y, w, h, [""], size=size, color=color)
    tf = shape.text_frame
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        runs = [(item, {})] if isinstance(item, str) else item
        if i == 0:
            for r in list(p.runs):
                r._r.getparent().remove(r._r)
        for text, opt in runs:
            r = p.add_run()
            r.text = text
            r.font.name, r.font.size = BODY_FONT, Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", False)
            r.font.color.rgb = opt.get("color", color)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.28)))
        pPr.set("indent", str(-Inches(0.28)))
        bu_font = etree.SubElement(pPr, qn("a:buFont"))
        bu_font.set("typeface", "Arial")
        bu = etree.SubElement(pPr, qn("a:buChar"))
        bu.set("char", "•")
    return shape


def rect(slide, x, y, w, h, fill, rounded=True, name=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    if rounded:
        shape.adjustments[0] = 0.08
    if name:
        shape.name = name
    return shape


def circle_number(slide, x, y, d, label, fill=NAVY, color=WHITE, size=16):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD_FONT, Pt(size), True, color
    return c


def picture(slide, filename, x, y, max_w, max_h, name=None):
    """Insert a figure scaled to fit inside the box, centred, keeping its aspect ratio."""
    path = FIG / filename
    w_px, h_px = Image.open(path).size
    scale = min(max_w / w_px, max_h / h_px)
    w, h = w_px * scale, h_px * scale
    pic = slide.shapes.add_picture(str(path), Inches(x + (max_w - w) / 2), Inches(y + (max_h - h) / 2),
                                   Inches(w), Inches(h))
    if name:
        pic.name = name
    return pic


def title(slide, text, subtitle=None):
    text_box(slide, 0.6, 0.38, 12.1, 0.75, text, size=32, color=NAVY, font=HEAD_FONT, bold=True, name="Title")
    if subtitle:
        text_box(slide, 0.6, 1.08, 12.1, 0.45, subtitle, size=16, color=MUTED, name="Subtitle")


def footer(slide, number, presenter):
    text_box(slide, 0.6, 7.02, 8.0, 0.3, f"TM3221 · Customer Churn Prediction · Presenter {presenter}",
             size=10, color=GREY, name="Footer")
    text_box(slide, 11.9, 7.02, 0.85, 0.3, str(number), size=10, color=GREY, align=PP_ALIGN.RIGHT,
             name="Slide number")


def stat(slide, x, y, w, number, label, color=NAVY, size=40, label_size=14, fill=TINT, h=1.55):
    rect(slide, x, y, w, h, fill)
    number_h = size * 1.35 / 72                     # one line of the large number
    text_box(slide, x + 0.2, y + 0.08, w - 0.4, number_h, number, size=size, color=color, font=HEAD_FONT, bold=True)
    text_box(slide, x + 0.2, y + 0.1 + number_h, w - 0.4, h - 0.14 - number_h, label, size=label_size, color=MUTED)


def style_bar_chart(chart, color, number_format='0.0"%"', axis_size=12, label_size=12, title_text=None):
    chart.has_legend = False
    chart.font.name, chart.font.size = BODY_FONT, Pt(axis_size)
    plot = chart.plots[0]
    plot.gap_width = 70
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = number_format, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.color.rgb = Pt(label_size), INK
    for series in plot.series:
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = color
    va = chart.value_axis
    va.visible = False
    va.has_major_gridlines = False
    va.minimum_scale = 0
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(axis_size)
    ca.tick_labels.font.color.rgb = MUTED
    ca.format.line.color.rgb = GREY
    if title_text:
        chart.has_title = True
        chart.chart_title.text_frame.text = title_text
        p = chart.chart_title.text_frame.paragraphs[0]
        p.runs[0].font.size, p.runs[0].font.bold = Pt(14), True
        p.runs[0].font.color.rgb = NAVY
        p.runs[0].font.name = BODY_FONT
    else:
        chart.has_title = False


def bar_chart(slide, x, y, w, h, categories, values, color, title_text, maximum=None):
    data = CategoryChartData()
    data.categories = categories
    data.add_series("Churn rate (%)", [round(v, 1) for v in values])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
    style_bar_chart(gf.chart, color, title_text=title_text)
    if maximum:
        gf.chart.value_axis.maximum_scale = maximum
    return gf


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------
def build(R):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
    blank = prs.slide_layouts[6]
    rate = R["rate"]
    test, cv = R["test"], R["cv"]
    sel = R["final"]["selected_model"]
    lr_t = test.loc[sel]
    gains = R["gains"]
    top = lambda share: gains.loc[gains["cumulative_share_of_customers"].round(2) == share,
                                  "cumulative_share_of_churners_captured"].iloc[0] * 100
    ors = R["or"]
    ORv = lambda f: ors.loc[f, "odds_ratio"]
    diff_rf = R["diff"].loc[f"Random Forest minus {sel}"]
    ci = R["ci"]
    n = 0

    def new_slide(dark=False):
        nonlocal n
        n += 1
        s = prs.slides.add_slide(blank)
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = NAVY_DARK if dark else WHITE
        return s

    # 1 ── Title ──────────────────────────────────────────────────────────
    s = new_slide(dark=True)
    # University crest on a white card (the image has a white background)
    rect(s, 0.9, 0.35, 1.1, 1.25, WHITE, name="Crest card")
    crest = s.shapes.add_picture(str(CREST), Inches(1.0), Inches(0.42), height=Inches(1.11))
    crest.name = "University crest"
    text_box(s, 0.9, 1.85, 11.5, 0.5, "TM3221 – DATA SCIENCE FOR BPM · GROUP ASSIGNMENT", size=16,
             color=RGBColor(0xB9, 0xC7, 0xDA), bold=True)
    text_box(s, 0.9, 2.4, 11.5, 1.9, "Customer Churn Prediction and Analysis Using Machine Learning",
             size=44, color=WHITE, font=HEAD_FONT, bold=True, line_spacing=1.0)
    text_box(s, 0.9, 4.4, 11.5, 0.6, "A data-driven approach for customer retention", size=22,
             color=RGBColor(0xF5, 0xA2, 0x7F))
    text_box(s, 0.9, 5.6, 11.5, 0.9, ["[Group name]",
                                      "Department of Management of Technology · Faculty of Business · University of Moratuwa"],
             size=16, color=RGBColor(0xDD, 0xE4, 0xEE))
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER A (about 1 minute)\n"
        "Good morning. Our project asks a practical business question: which telecom customers are likely to "
        "leave, and why? We analysed 7,043 customers of a telecom provider, cleaned a deliberately messy dataset, "
        "tested which factors are associated with churn, and built machine learning models to predict it. "
        "We will finish with recommendations the company could act on.")

    # 2 ── Problem & objectives ───────────────────────────────────────────
    s = new_slide()
    title(s, "Why churn matters and what we set out to answer")
    rect(s, 0.6, 1.45, 5.6, 4.9, TINT)
    text_box(s, 0.9, 1.65, 5.0, 0.4, "RESEARCH QUESTION", size=13, color=ORANGE, bold=True)
    text_box(s, 0.9, 2.1, 5.0, 2.6, "Which customer characteristics and service-related factors are most strongly "
             "associated with churn – and how effectively can machine learning predict it?",
             size=20, color=NAVY, font=HEAD_FONT, bold=True, line_spacing=1.05)
    text_box(s, 0.9, 4.75, 5.0, 1.4, "Churn = a customer leaving the company. Keeping customers is usually cheaper "
             "than winning new ones, so knowing who is at risk lets retention budgets go where they matter.",
             size=14, color=MUTED)
    objectives = [("Clean", "Audit and repair a messy dataset without leaking test data"),
                  ("Explore", "Find patterns linking demographics, services and billing to churn"),
                  ("Model", "Compare logistic regression, decision tree and random forest"),
                  ("Act", "Interpret the drivers and turn them into retention actions")]
    for i, (head, body) in enumerate(objectives):
        y = 1.5 + i * 1.22
        circle_number(s, 6.75, y + 0.05, 0.62, str(i + 1))
        text_box(s, 7.6, y - 0.02, 5.1, 0.42, head, size=18, color=NAVY, bold=True)
        text_box(s, 7.6, y + 0.38, 5.1, 0.7, body, size=14, color=INK)
    footer(s, n, "A")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER A (about 1.5 minutes)\n"
        "Read the research question. Explain that churn is costly because a lost customer stops paying and must be "
        "replaced. Walk through the four objectives: clean, explore, model, act. Stress one principle that runs through "
        "the whole project: the data are observational, so we talk about associations and predictions, not causes.")

    # 3 ── Dataset overview ───────────────────────────────────────────────
    s = new_slide()
    title(s, "The dataset: 7,043 telecom customers, 1 in 4 churned",
          "IBM Telco Customer Churn sample data (Kaggle) – the assignment file contained injected errors")
    stat(s, 0.6, 1.75, 3.8, f"{R['raw_n']:,}", f"raw rows → {R['clean_n']:,} customers after removing {R['dup']} duplicates")
    stat(s, 4.75, 1.75, 3.8, "21", "columns: demographics, account, services and the target")
    stat(s, 8.9, 1.75, 3.8, pct0(R["churn_rate"]), f"churned ({R['n_churn']:,} customers): imbalanced classes",
         color=ORANGE, fill=ORANGE_TINT)
    groups = [("Demographics", "gender, senior citizen, partner, dependents"),
              ("Account & billing", "tenure, contract, payment method, paperless billing, monthly & total charges"),
              ("Services", "phone, multiple lines, internet type, six internet add-ons"),
              ("Target", "Churn = Yes / No (left within the last month)")]
    for i, (head, body) in enumerate(groups):
        x = 0.6 + (i % 2) * 6.15
        y = 3.65 + (i // 2) * 1.5
        rect(s, x, y, 5.95, 1.3, TINT)
        text_box(s, x + 0.25, y + 0.15, 5.5, 0.4, head, size=16, color=NAVY, bold=True)
        text_box(s, x + 0.25, y + 0.55, 5.5, 0.7, body, size=14, color=INK)
    footer(s, n, "A")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER A (about 1.5 minutes)\n"
        f"The file had {R['raw_n']:,} rows and 21 columns. After removing {R['dup']} exact duplicates we have "
        f"{R['clean_n']:,} customers. {pct0(R['churn_rate'])} churned. Point out why that matters: a model that always "
        f"says 'no churn' is already {100 - R['churn_rate']:.1f}% accurate but useless, so we cannot judge models by "
        "accuracy alone. Hand over to Presenter B for data cleaning.")

    # 4 ── Data cleaning ──────────────────────────────────────────────────
    s = new_slide()
    title(s, "Data cleaning: repair, don't delete",
          "Problems were documented first; every rule works row by row, so nothing leaks from the test set")
    items = [
        [("30 ", {"bold": True, "color": NAVY}), ("exact duplicate rows removed", {})],
        [("Hidden missing values: ", {"bold": True, "color": NAVY}),
         (f"{R['tc_blank']} blanks and {R['tc_invalid']} 'INVALID_VAL' in TotalCharges – invisible to isnull()", {})],
        [("Impossible values: ", {"bold": True, "color": NAVY}),
         ("tenure of −99 or 988 months; monthly bills up to 9,960 USD", {})],
        [("Repaired with ", {"bold": True, "color": NAVY}),
         (f"TotalCharges ≈ tenure × MonthlyCharges: {R['rec'].loc['TotalCharges', 'repair_median_abs_pct_error']:.0f}% "
          f"error vs {R['rec'].loc['TotalCharges', 'median_imputation_median_abs_pct_error']:.0f}% for median filling", {})],
        [("Outliers: ", {"bold": True, "color": NAVY}),
         (f"{R['iqr_outliers']} remain after repair – real extremes kept, no winsorising", {})],
    ]
    bullets(s, 0.6, 1.75, 5.4, 4.9, items, size=15)
    picture(s, "fig02_boxplots_before_after_cleaning.png", 6.1, 1.65, 6.75, 4.5, name="Boxplots before and after")
    text_box(s, 6.2, 6.2, 6.6, 0.6, "Before: errors stretch the axes. After: plausible ranges and no IQR outliers.",
             size=12, color=MUTED, align=PP_ALIGN.CENTER)
    footer(s, n, "B")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER B (about 2 minutes)\n"
        "We never edited the raw file. We first loaded every value as text, which exposed problems that a normal "
        "isnull() check misses. Explain the key trick: total charges is roughly months times the monthly bill, so if "
        "one of the three values is broken we can rebuild it from the other two – far more accurate than filling in "
        "an average. Outliers: we used the IQR rule and boxplots; the extreme values were impossible errors, not real "
        "customers. After repair there are no outliers, so we deleted nothing. Leakage: these rules only use the "
        "customer's own row; anything that learns from data (median, scaling, encoding) happens later inside a "
        "Pipeline on training data only.")

    # 5 ── EDA: key segments (native charts) ──────────────────────────────
    s = new_slide()
    title(s, "Who churns? Contract, internet, payment and tenure",
          f"Churn rate (%) by group · overall churn rate {pct0(R['churn_rate'])}")
    panels = [
        ("Contract", ["Month-to-month", "One year", "Two year"],
         [rate("Contract", c) for c in ["Month-to-month", "One year", "Two year"]]),
        ("Internet service", ["Fiber optic", "DSL", "No internet"],
         [rate("InternetService", c) for c in ["Fiber optic", "DSL", "No"]]),
        ("Payment method", ["E-check", "Mailed", "Bank (auto)", "Card (auto)"],
         [rate("PaymentMethod", c) for c in ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                             "Credit card (automatic)"]]),
        ("Tenure (months)", ["0–12", "13–24", "25–48", "49–72"],
         [rate("tenure_group", c) for c in ["0–12", "13–24", "25–48", "49–72"]]),
    ]
    for i, (head, cats, vals) in enumerate(panels):
        x = 0.5 + i * 3.1
        bar_chart(s, x, 1.65, 3.0, 4.1, cats, vals, ORANGE, head, maximum=60)
    text_box(s, 0.6, 5.95, 12.1, 0.9,
             [[("Highest risk: ", {"bold": True, "color": NAVY}),
               (f"month-to-month ({pct0(rate('Contract', 'Month-to-month'))}), fiber optic "
                f"({pct0(rate('InternetService', 'Fiber optic'))}), electronic check "
                f"({pct0(rate('PaymentMethod', 'Electronic check'))}) and the first year "
                f"({pct0(rate('tenure_group', '0–12'))}). Gender shows no difference "
                f"({pct0(rate('gender', 'Female'))} vs {pct0(rate('gender', 'Male'))}).", {})]], size=15)
    footer(s, n, "B")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER B (about 2 minutes)\n"
        f"Churn rate = churners divided by customers in the group. Month-to-month customers churn at "
        f"{pct0(rate('Contract', 'Month-to-month'))} against {pct0(rate('Contract', 'Two year'))} on two-year contracts. "
        f"Fiber customers churn at more than twice the DSL rate. Electronic-check payers churn far more than those on "
        "automatic payment. Churn is concentrated in the first year. These are associations: fiber customers also pay "
        "more, so we cannot say fiber itself causes churn.")

    # 6 ── Important patterns ─────────────────────────────────────────────
    s = new_slide()
    title(s, "Patterns that need careful reading",
          "Contract matters at every tenure level · protective add-ons matter, streaming does not")
    picture(s, "fig08_churn_contract_by_tenure.png", 0.5, 1.65, 6.3, 4.6, name="Contract by tenure heatmap")
    ct = R["ct"]
    cards = [
        ("Contract beyond tenure",
         f"Even after 4+ years, month-to-month churn is {ct[('churn_rate_%', 'Month-to-month')].iloc[-1]:.1f}% vs "
         f"{ct[('churn_rate_%', 'Two year')].iloc[-1]:.1f}% on two-year contracts."),
        ("Security & support add-ons",
         f"Internet customers: {pct0(rate('OnlineSecurity', 'Yes'))} churn with online security vs "
         f"{pct0(rate('OnlineSecurity', 'No'))} without; tech support {pct0(rate('TechSupport', 'Yes'))} vs "
         f"{pct0(rate('TechSupport', 'No'))}."),
        ("Price is tangled with service",
         f"Bills over 90 USD are {R['bands'].loc['>90', '% Fiber optic']:.0f}% fiber customers – the 'price effect' "
         "is mostly a service-type effect."),
    ]
    for i, (head, body) in enumerate(cards):
        y = 1.7 + i * 1.6
        rect(s, 7.1, y, 5.65, 1.42, TINT)
        text_box(s, 7.35, y + 0.12, 5.2, 0.4, head, size=16, color=NAVY, bold=True)
        text_box(s, 7.35, y + 0.52, 5.2, 0.85, body, size=14, color=INK)
    footer(s, n, "B")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER B (about 1.5 minutes)\n"
        "The heatmap answers a fair question: month-to-month customers are newer, so is contract just a tenure "
        "effect? No – within every tenure group month-to-month churn is far higher. Second, among internet customers, "
        "online security and tech support go with much lower churn, while streaming add-ons barely matter. Third, "
        "monthly charges: the bands mostly reflect which service a customer has. Hand over to Presenter C.")

    # 7 ── Statistical tests ──────────────────────────────────────────────
    s = new_slide()
    chi, mw = R["chi"], R["mw"]
    title(s, "Significant is not the same as important",
          "Chi-square (categorical) and Mann-Whitney U (numeric) tests · Bonferroni-corrected · effect sizes")
    picture(s, "fig11_effect_sizes.png", 0.45, 1.6, 8.2, 4.4, name="Effect sizes")
    stat(s, 8.95, 1.6, 3.8, f"{int(chi['significant_after_bonferroni'].sum())} of 16",
         "categorical variables linked to churn (not gender, phone or multiple lines)", size=34, h=1.65)
    stat(s, 8.95, 3.35, 3.8, f"V = {chi.loc['Contract', 'cramers_v']:.2f}",
         f"contract: largest effect (medium); tenure r = {mw.loc['tenure', 'rank_biserial']:.2f}", size=34, h=1.65)
    stat(s, 8.95, 5.1, 3.8, f"VIF ≈ {R['vif_mc_full']:.0f}",
         "MonthlyCharges is set by the services: dropped from the logistic model",
         size=34, color=ORANGE, fill=ORANGE_TINT, h=1.65)
    footer(s, n, "C")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER C (about 2 minutes)\n"
        "With 7,000 customers almost everything is 'significant', so we also measured effect size: Cramér's V for "
        "categories and the rank-biserial correlation for numbers. Contract has the biggest effect; demographics are "
        "significant but small; gender has none. We used Mann-Whitney instead of a t-test because tenure and charges "
        "are not normally distributed, and Bonferroni because we ran 19 tests. Multicollinearity: the monthly bill is "
        "almost exactly the sum of the service prices (VIF about 836), so we removed MonthlyCharges and TotalCharges "
        "from the logistic regression. Note: scaling does not reduce VIF – a mistake in last year's example report.")

    # 8 ── ML methodology ─────────────────────────────────────────────────
    s = new_slide()
    title(s, "Method: a fair, leakage-safe model comparison",
          "Binary classification – predict Churn = Yes / No")
    steps = [("Split", f"Stratified 80/20\n{R['train_n']:,} train · {R['test_n']:,} test"),
             ("Pipeline", "Impute → scale → one-hot,\nfitted on training data only"),
             ("Tune", "Grid search with\nstratified 5-fold CV"),
             ("Select", "Highest CV ROC-AUC;\nwithin 0.01 → simpler model"),
             ("Test once", "All models on the same\nunseen 1,409 customers")]
    for i, (head, body) in enumerate(steps):
        x = 0.6 + i * 2.5
        rect(s, x, 1.75, 2.2, 2.05, TINT)
        circle_number(s, x + 0.8, 1.55, 0.6, str(i + 1), fill=ORANGE)
        text_box(s, x + 0.12, 2.3, 1.96, 0.42, head, size=17, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
        text_box(s, x + 0.12, 2.75, 1.96, 1.0, body.split("\n"), size=13, color=INK, align=PP_ALIGN.CENTER)
    models = [("Dummy baseline", "always predicts 'no churn' – the floor to beat"),
              ("Logistic regression", "interpretable: odds ratio per feature"),
              ("Decision tree", "readable if/then rules"),
              ("Random forest", "300 trees – non-linear patterns")]
    for i, (head, body) in enumerate(models):
        x = 0.6 + (i % 2) * 6.15
        y = 4.2 + (i // 2) * 1.1
        text_box(s, x, y, 5.9, 1.0, [[(head + ": ", {"bold": True, "color": NAVY}), (body, {})]], size=15)
    text_box(s, 0.6, 6.35, 12.1, 0.55, "Imbalance: class_weight = \"balanced\" for every model (no SMOTE) · "
             "metrics: ROC-AUC, PR-AUC, recall, precision, F1 – not accuracy alone", size=13, color=MUTED)
    footer(s, n, "C")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER C (about 2 minutes)\n"
        "Walk through the five steps. The key idea is that the test set is locked away until the very end. All "
        "preprocessing sits inside a scikit-learn Pipeline, so during cross-validation it is re-fitted on each "
        "training fold – this prevents data leakage. We fixed the selection rule before seeing test results: best "
        "cross-validated ROC-AUC, but if another model is within 0.01 we prefer the more interpretable one. ROC-AUC "
        "measures how well the model ranks churners above loyal customers. Hand over to Presenter D.")

    # 9 ── Model comparison (native table) ────────────────────────────────
    s = new_slide()
    title(s, "All models beat the baseline; differences are small",
          "Test set of 1,409 customers · threshold 0.5")
    header = ["Model", "CV ROC-AUC", "Test ROC-AUC", "Recall", "Precision", "F1", "Accuracy"]
    names = ["Dummy (always 'No churn')", "Logistic Regression", "Decision Tree", "Random Forest"]
    tbl = s.shapes.add_table(len(names) + 1, len(header), Inches(0.6), Inches(1.75), Inches(8.45),
                             Inches(2.6)).table
    widths = [2.25, 1.3, 1.4, 0.85, 1.05, 0.6, 1.0]
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for j, htext in enumerate(header):
        cell = tbl.cell(0, j)
        cell.text = htext
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        para = cell.text_frame.paragraphs[0]
        para.runs[0].font.size, para.runs[0].font.bold = Pt(13), True
        para.runs[0].font.color.rgb, para.runs[0].font.name = WHITE, BODY_FONT
        para.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
    for i, name in enumerate(names, start=1):
        te, cvr = test.loc[name], cv.loc[name]
        label = "Dummy baseline" if name.startswith("Dummy") else name
        values = [label + (" ★" if name == sel else ""), f"{cvr['cv_roc_auc_mean']:.3f}", f"{te['roc_auc']:.3f}",
                  f"{te['recall']:.2f}", f"{te['precision']:.2f}", f"{te['f1']:.2f}", f"{te['accuracy']:.2f}"]
        for j, v in enumerate(values):
            cell = tbl.cell(i, j)
            cell.text = v
            cell.fill.solid()
            cell.fill.fore_color.rgb = ORANGE_TINT if name == sel else (TINT if i % 2 else WHITE)
            para = cell.text_frame.paragraphs[0]
            para.runs[0].font.size, para.runs[0].font.name = Pt(13), BODY_FONT
            para.runs[0].font.bold = name == sel
            para.runs[0].font.color.rgb = INK
            para.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
    picture(s, "fig13_confusion_matrices.png", 0.5, 4.65, 8.5, 2.2, name="Confusion matrices")
    stat(s, 9.25, 1.75, 3.5, f"{lr_t['roc_auc']:.2f}", f"test ROC-AUC, logistic regression "
         f"(95% CI {ci.loc[sel, 'ci_95_low']:.2f}–{ci.loc[sel, 'ci_95_high']:.2f})", size=40)
    stat(s, 9.25, 3.5, 3.5, f"{lr_t['recall']:.0%}", f"of churners caught ({int(lr_t['TP'])} of "
         f"{int(lr_t['TP'] + lr_t['FN'])}); the dummy catches 0%", size=40, color=ORANGE, fill=ORANGE_TINT)
    text_box(s, 9.25, 5.35, 3.5, 1.5, f"★ Selected: within 0.01 of the random forest in CV. On the test set the "
             f"forest is +{diff_rf['mean_auc_difference']:.3f} AUC, but the 95% CI "
             f"({diff_rf['ci_95_low']:.3f} to {diff_rf['ci_95_high']:.3f}) includes zero.", size=12, color=MUTED)
    footer(s, n, "D")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER D (about 2 minutes)\n"
        "The baseline is 73.5% accurate but catches no churners – that is why accuracy is misleading. All three "
        "models reach a test ROC-AUC of about 0.84 and catch roughly 78–79% of churners. The random forest was "
        "slightly better, but within our tolerance in cross-validation, and a bootstrap test shows the difference "
        "on the test set is not statistically reliable. So by our pre-set rule we chose logistic regression, which "
        f"is easier to explain and overfits least. Confusion matrix: {int(lr_t['FN'])} churners missed (lost revenue) "
        f"and {int(lr_t['FP'])} loyal customers flagged (unneeded offers) – a trade-off the threshold can adjust.")

    # 10 ── Interpretation ────────────────────────────────────────────────
    s = new_slide()
    title(s, "What drives the predictions?",
          "Logistic regression odds ratios (95% bootstrap intervals) – only clear effects shown; full chart in the report")
    picture(s, slide_odds_ratio_figure(R["or"]), 0.45, 1.55, 7.5, 5.3, name="Odds ratios")
    drivers = [(f"×{ORv('Contract_Two year'):.2f}", "odds for a two-year vs month-to-month contract", BLUE),
               (f"×{ORv('tenure'):.2f}", "odds for every extra ~25 months of tenure", BLUE),
               (f"×{ORv('InternetService_Fiber optic'):.1f}", "odds for fiber optic vs DSL", ORANGE),
               (f"×{ORv('PaymentMethod_Electronic check'):.1f}", "odds for electronic check vs bank transfer", ORANGE)]
    for i, (num, label, col) in enumerate(drivers):
        y = 1.65 + i * 1.12
        text_box(s, 8.3, y, 1.75, 0.8, num, size=30, color=col, font=HEAD_FONT, bold=True)
        text_box(s, 10.05, y + 0.08, 2.7, 0.9, label, size=14, color=INK)
    text_box(s, 8.3, 6.15, 4.45, 0.8, "Same top drivers in permutation importance, decision-tree rules and EDA: "
             "contract, tenure, internet type.", size=12, color=MUTED)
    footer(s, n, "D")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER D (about 2 minutes)\n"
        "An odds ratio below 1 means lower churn odds, above 1 higher. Holding everything else constant: two-year "
        "contracts have about a quarter of the odds of month-to-month; every extra two years of tenure roughly halves "
        "the odds; fiber optic has about 2.5 times the odds of DSL; electronic check about 1.5 times. Grey items, "
        "including gender and senior citizen, have no clear effect once other factors are included. Permutation "
        "importance and a simple decision tree agree: the riskiest group is month-to-month fiber customers in their "
        f"first 13 months, with {R['leaves'].iloc[0]['churn_rate']:.0%} churn.")

    # 11 ── Business value (native line chart) ────────────────────────────
    s = new_slide()
    title(s, "Business value: target the riskiest customers first",
          "Cumulative gains on the test set – customers contacted in order of predicted risk")
    data = CategoryChartData()
    data.categories = [f"{round(v * 100):.0f}%" for v in gains["cumulative_share_of_customers"]]
    data.add_series("Logistic regression", [round(v * 100, 1) for v in gains["cumulative_share_of_churners_captured"]])
    data.add_series("Random targeting", [round(v * 100, 1) for v in gains["cumulative_share_of_customers"]])
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.5), Inches(1.6), Inches(7.6), Inches(5.2), data)
    ch = gf.chart
    ch.font.name, ch.font.size = BODY_FONT, Pt(12)
    ch.has_legend = True
    ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.BOTTOM, False
    ch.legend.font.size = Pt(12)
    for series, col in zip(ch.plots[0].series, (BLUE, GREY)):
        series.format.line.color.rgb = col
        series.format.line.width = Pt(2.5)
        series.marker.format.fill.solid()
        series.marker.format.fill.fore_color.rgb = col
        series.marker.format.line.color.rgb = col
        series.smooth = False
    va = ch.value_axis
    va.maximum_scale, va.minimum_scale = 100, 0
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xE4, 0xE3, 0xDF)
    va.tick_labels.font.size, va.tick_labels.font.color.rgb = Pt(12), MUTED
    va.has_title = True
    va.axis_title.text_frame.text = "% of all churners reached"
    va.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    ca = ch.category_axis
    ca.tick_labels.font.size, ca.tick_labels.font.color.rgb = Pt(12), MUTED
    ca.has_title = True
    ca.axis_title.text_frame.text = "% of customers contacted (highest risk first)"
    ca.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    stat(s, 8.5, 1.75, 4.25, f"{top(0.2):.0f}%", "of all churners are in the top 20% of risk scores "
         "(random targeting: 20%)", size=44, color=ORANGE, fill=ORANGE_TINT, h=1.75)
    stat(s, 8.5, 3.7, 4.25, f"{gains['actual_churn_rate'].iloc[0]:.0%}", "actual churn rate in the highest-risk "
         f"10% – {gains['lift'].iloc[0]:.1f}× the average", size=44, h=1.75)
    text_box(s, 8.5, 5.65, 4.25, 1.2, "Use the scores to rank customers and size the contact list to the "
             "retention budget; the threshold is a business setting.", size=14, color=MUTED)
    footer(s, n, "D")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER D (about 1.5 minutes)\n"
        f"This is the most practical result. If the retention team calls the 20% of customers the model ranks as "
        f"riskiest, they reach {top(0.2):.0f}% of everyone who would churn – random calling would reach only 20%. "
        f"In the top 10%, three in four customers really churn. Hand over to Presenter E.")

    # 12 ── Recommendations ───────────────────────────────────────────────
    s = new_slide()
    title(s, "Recommendations for customer retention", "Each action follows from the evidence – and should be tested")
    recs = [
        ("Longer contracts", f"Offer incentives to move month-to-month customers to 1–2 year plans "
                             f"(OR {ORv('Contract_Two year'):.2f})."),
        ("First-year onboarding", f"Welcome calls and early check-ins – churn is "
                                  f"{pct0(rate('tenure_group', '0–12'))} in year one."),
        ("Fix the fiber experience", "Review fiber pricing, reliability and support; survey leavers "
                                     f"(OR {ORv('InternetService_Fiber optic'):.1f})."),
        ("Automatic payment", f"Nudge electronic-check payers ({pct0(rate('PaymentMethod', 'Electronic check'))} "
                              "churn) to automatic bank or card payment."),
        ("Security & support bundles", f"Trials of online security and tech support "
                                       f"({pct0(rate('OnlineSecurity', 'Yes'))} vs {pct0(rate('OnlineSecurity', 'No'))} churn)."),
        ("Prioritise and test", "Score monthly, call the top-risk group first, and A/B-test offers to measure "
                                "their real effect."),
    ]
    for i, (head, body) in enumerate(recs):
        x = 0.6 + (i % 3) * 4.1
        y = 1.75 + (i // 3) * 2.55
        rect(s, x, y, 3.85, 2.3, TINT if i < 5 else ORANGE_TINT)
        circle_number(s, x + 0.25, y + 0.25, 0.55, str(i + 1), fill=NAVY if i < 5 else ORANGE)
        text_box(s, x + 0.95, y + 0.27, 2.8, 0.55, head, size=16, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text_box(s, x + 0.25, y + 0.95, 3.4, 1.3, body, size=14, color=INK)
    footer(s, n, "E")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER E (about 2 minutes)\n"
        "Each recommendation comes from a specific finding. Contracts and tenure are the strongest drivers, so the "
        "first two actions target month-to-month and new customers. Fiber churn stays high even after controlling for "
        "other factors, which points to a service or price issue worth investigating. Automatic payment and security "
        "or support add-ons are associated with lower churn. Finally, because our data are observational, the "
        "company should test each offer with a randomised trial before rolling it out.")

    # 13 ── Limitations & responsible use ─────────────────────────────────
    s = new_slide()
    title(s, "Limitations and responsible use")
    cols = [
        ("Limitations", NAVY, [
            "Observational data: associations, not causes",
            "Missing drivers: complaints, service quality, competitor offers – about 1 in 5 churners is hard to predict",
            f"Repaired records churned at {pct0(R['affected_rate'])} vs {pct0(R['unaffected_rate'])} (not significant)",
            "IBM sample data for one provider – re-validate before real use",
            "Balanced weights inflate probabilities: use scores for ranking",
        ]),
        ("Responsible use", ORANGE, [
            "Decision support, not automatic decisions",
            "Never use gender or age to offer worse terms – gender has no predictive value here anyway",
            "Protect customer data and use it only for agreed purposes",
            "Monitor performance monthly and retrain regularly",
        ]),
    ]
    for i, (head, col, items) in enumerate(cols):
        x = 0.6 + i * 6.15
        rect(s, x, 1.5, 5.95, 5.25, TINT if i == 0 else ORANGE_TINT)
        text_box(s, x + 0.3, 1.7, 5.4, 0.5, head, size=20, color=col, font=HEAD_FONT, bold=True)
        bullets(s, x + 0.3, 2.35, 5.4, 4.3, items, size=15, space_after=10)
    footer(s, n, "E")
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER E (about 1.5 minutes)\n"
        "Be open about limits: we can predict and describe associations, but not prove causes. The model misses "
        "churners who look loyal – probably people leaving for reasons not in the data, like competitor offers. "
        "On responsible use: the model should help staff prioritise, not decide alone, and must not disadvantage "
        "customers because of gender or age.")

    # 14 ── Conclusion ────────────────────────────────────────────────────
    s = new_slide(dark=True)
    text_box(s, 0.9, 0.8, 11.5, 0.8, "Conclusion", size=40, color=WHITE, font=HEAD_FONT, bold=True)
    takeaways = [
        ("Drivers", "Short contracts, short tenure, fiber optic, electronic check and missing security/support "
                    "add-ons are most associated with churn; gender is not."),
        ("Prediction", f"A leakage-safe logistic regression reaches test ROC-AUC {lr_t['roc_auc']:.2f} and catches "
                       f"{lr_t['recall']:.0%} of churners – as good as a random forest, and explainable."),
        ("Action", f"Ranking customers by risk puts {top(0.2):.0f}% of churners in the top 20% – focus retention "
                   "budgets there and test offers."),
    ]
    for i, (head, body) in enumerate(takeaways):
        y = 1.95 + i * 1.45
        circle_number(s, 0.9, y + 0.05, 0.7, str(i + 1), fill=ORANGE, size=18)
        text_box(s, 1.9, y, 10.5, 0.45, head, size=20, color=RGBColor(0xF5, 0xA2, 0x7F), bold=True)
        text_box(s, 1.9, y + 0.45, 10.5, 0.9, body, size=17, color=WHITE)
    text_box(s, 0.9, 6.35, 11.5, 0.6, "Thank you – questions are welcome", size=22,
             color=RGBColor(0xDD, 0xE4, 0xEE), font=HEAD_FONT, bold=True)
    s.notes_slide.notes_text_frame.text = (
        "PRESENTER E (about 1 minute)\n"
        "Summarise the three takeaways: what drives churn, how well we can predict it, and how the company can use "
        "it. Invite questions. Likely questions and answers are in presentation/speaker_guide.md.")

    prs.core_properties.title = "Customer Churn Prediction and Analysis Using Machine Learning"
    prs.core_properties.subject = "TM3221 Data Science for BPM – Group Assignment"
    prs.save(OUT)
    return n


def pct0(x):
    return f"{x:.1f}%"


SHORT_LABELS = {
    "Contract_Two year": "Two-year contract", "Contract_One year": "One-year contract",
    "tenure": "Tenure (+25 months)", "InternetService_No": "No internet (vs DSL)",
    "InternetService_Fiber optic": "Fiber optic (vs DSL)", "PhoneService_Yes": "Phone service",
    "OnlineSecurity_Yes": "Online security", "TechSupport_Yes": "Tech support",
    "Dependents_Yes": "Has dependents", "MultipleLines_Yes": "Multiple lines",
    "StreamingTV_Yes": "Streaming TV", "StreamingMovies_Yes": "Streaming movies",
    "PaperlessBilling_Yes": "Paperless billing", "PaymentMethod_Electronic check": "Electronic check",
}


def slide_odds_ratio_figure(or_table):
    """
    Slide-sized version of the odds-ratio chart, drawn from the saved table
    (outputs/tables/t35_odds_ratios.csv): only effects whose 95% CI excludes 1,
    short labels and larger text so it is readable when projected.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src import eda_plots as ep
    ep.set_style()
    data = or_table[~or_table["ci_includes_1"]].copy()
    data["short"] = [SHORT_LABELS.get(f, f) for f in data.index]
    data = data.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for i, (_, row) in enumerate(data.iterrows()):
        colour = ep.ORANGE if row["odds_ratio"] > 1 else ep.BLUE
        ax.plot([row["or_ci_low"], row["or_ci_high"]], [i, i], color=colour, linewidth=2.5)
        ax.plot(row["odds_ratio"], i, "o", color=colour, markersize=9, markeredgecolor="white")
        ax.text(row["or_ci_high"] * 1.07, i, f"{row['odds_ratio']:.2f}", va="center", fontsize=12)
    ax.axvline(1, color=ep.TEXT_MUTED, linewidth=1)
    ax.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2, 3]
    ax.set_xticks(ticks, [str(t) for t in ticks], fontsize=12)
    ax.set_xlim(0.14, 4.3)
    ax.set_yticks(range(len(data)), data["short"], fontsize=13)
    ax.set_xlabel("Odds ratio (log scale): < 1 lower churn odds, > 1 higher", fontsize=12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = config.FIGURE_DIR / "slide_odds_ratios.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path.name


if __name__ == "__main__":
    slides = build(load_results())
    print(f"Saved {OUT.relative_to(ROOT)} with {slides} slides")
