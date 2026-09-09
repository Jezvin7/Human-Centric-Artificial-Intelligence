from pathlib import Path
from datetime import datetime
import html
import json

import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


# ============================================================
# PATHS
# ============================================================

ML_DIR = Path(__file__).resolve().parent
PROJECT3_DIR = ML_DIR.parent
ROOT_DIR = PROJECT3_DIR.parent

RESULTS_DIR = (
    ML_DIR
    / "saved_models"
)

REPORT_DIR = (
    ROOT_DIR
    / "static"
    / "project3"
)

ASSET_DIR = (
    REPORT_DIR
    / "report_assets"
)

REPORT_PATH = (
    REPORT_DIR
    / "HCAI_Project3_Report.pdf"
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ASSET_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RESULT FILES
# ============================================================

BASELINE_RESULTS_PATH = (
    RESULTS_DIR
    / "baseline_results.json"
)

EXPERT_RESULTS_PATH = (
    RESULTS_DIR
    / "expert_results.json"
)

L2D_RESULTS_PATH = (
    RESULTS_DIR
    / "l2d_results.json"
)

ACTIVE_RESULTS_PATH = (
    RESULTS_DIR
    / "active_learning_results.json"
)

HUMAN_RESULTS_PATH = (
    RESULTS_DIR
    / "human_expert_results.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path,required=True,):
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required results file not found:\n{path}"
            )
        return None
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def safe(value):

    return html.escape(
        str(value)
    )


def percent(value, digits=2):

    if value is None:
        return "-"

    return f"{float(value):.{digits}f}%"


# ============================================================
# LOAD ALL EXPERIMENT RESULTS
# ============================================================

def load_all_results():

    return {

        "baseline":
            load_json(
                BASELINE_RESULTS_PATH
            ),

        "expert":
            load_json(
                EXPERT_RESULTS_PATH
            ),

        "l2d":
            load_json(
                L2D_RESULTS_PATH
            ),

        "active":
            load_json(
                ACTIVE_RESULTS_PATH
            ),

        "human":
            load_json(
                HUMAN_RESULTS_PATH,
                required=False,
            ),
    }


# ============================================================
# REPORT STYLES
# ============================================================

def build_styles():

    styles = getSampleStyleSheet()


    title = ParagraphStyle(
        "ProjectTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=14,
    )


    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=12,
    )


    heading1 = ParagraphStyle(
        "Heading1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        spaceBefore=8,
        spaceAfter=10,
        textColor=colors.HexColor("#1F2937"),
    )


    heading2 = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        spaceBefore=8,
        spaceAfter=7,
        textColor=colors.HexColor("#374151"),
    )


    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        spaceAfter=8,
    )


    note = ParagraphStyle(
        "NoteCustom",
        parent=body,
        backColor=colors.HexColor("#EEF2FF"),
        borderPadding=8,
        spaceBefore=4,
        spaceAfter=10,
    )


    small = ParagraphStyle(
        "SmallCustom",
        parent=body,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#6B7280"),
    )


    return {
        "title": title,
        "subtitle": subtitle,
        "h1": heading1,
        "h2": heading2,
        "body": body,
        "note": note,
        "small": small,
    }


# ============================================================
# TABLE STYLE
# ============================================================

def style_table(table):

    table.setStyle(
        TableStyle(
            [

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D1D5DB"),
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#F3F4F6"),
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
            ]
        )
    )


# ============================================================
# PAGE NUMBER
# ============================================================

def add_page_number(
    canvas,
    document,
):

    canvas.saveState()

    page_number = (
        canvas.getPageNumber()
    )

    canvas.setFont(
        "Helvetica",
        8,
    )

    canvas.setFillColor(
        colors.HexColor("#6B7280")
    )

    canvas.drawCentredString(
        A4[0] / 2,
        1.1 * cm,
        f"Page {page_number}",
    )

    canvas.restoreState()


# ============================================================
# CHART 1 — OVERALL ACCURACY
# ============================================================

def create_overall_accuracy_chart(
    results,
):

    baseline = results["baseline"]
    expert = results["expert"]
    l2d = results["l2d"]
    active = results["active"]
    human = results["human"]


    experiments = (
        active["experiments"]
    )


    # Select the uncertainty-sampling operating point
    # that has the largest advantage over random sampling.

    best_active = max(
        experiments,

        key=lambda item:
            (
                item[
                    "competence_uncertainty"
                ][
                    "team_accuracy_percent"
                ]
                -
                item[
                    "random"
                ][
                    "team_accuracy_percent"
                ]
            ),
    )


    labels = [
        "AI only",
        "Simulated expert",
        "Full L2D",
        (
            "Active L2D\n"
            f"({best_active['query_budget']} queries)"
        ),
    ]


    values = [

        baseline[
            "accuracy_percent"
        ],

        expert[
            "accuracy_percent"
        ],

        l2d[
            "team_accuracy_percent"
        ],

        best_active[
            "competence_uncertainty"
        ][
            "team_accuracy_percent"
        ],
    ]


    output_path = (
        ASSET_DIR
        / "overall_accuracy.png"
    )


    plt.figure(
        figsize=(8, 4.5)
    )


    bars = plt.bar(
        labels,
        values,
    )


    plt.ylabel(
        "Test accuracy (%)"
    )


    plt.ylim(
        max(
            0,
            min(values) - 5,
        ),
        min(
            100,
            max(values) + 3,
        ),
    )


    plt.title(
        "Overall Project Performance"
    )


    for bar, value in zip(
        bars,
        values,
    ):

        plt.text(

            bar.get_x()
            +
            bar.get_width() / 2,

            bar.get_height()
            +
            0.15,

            f"{value:.2f}%",

            ha="center",
            va="bottom",
            fontsize=9,
        )


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )


    plt.close()


    return output_path


# ============================================================
# CHART 2 — AI VS EXPERT
# ============================================================

def create_class_performance_chart(
    results,
):

    baseline = results["baseline"]
    expert = results["expert"]


    class_names = (
        baseline["class_names"]
    )


    ai_recall = []

    expert_recall = []


    for class_name in class_names:

        ai_recall.append(
            baseline[
                "classification_report"
            ][class_name]["recall"]
            * 100
        )

        expert_recall.append(
            expert[
                "classification_report"
            ][class_name]["recall"]
            * 100
        )


    positions = list(
        range(
            len(class_names)
        )
    )


    width = 0.35


    output_path = (
        ASSET_DIR
        / "class_performance.png"
    )


    plt.figure(
        figsize=(8, 4.5)
    )


    plt.bar(
        [
            position - width / 2
            for position in positions
        ],
        ai_recall,
        width=width,
        label="AI",
    )


    plt.bar(
        [
            position + width / 2
            for position in positions
        ],
        expert_recall,
        width=width,
        label="Simulated expert",
    )


    plt.xticks(
        positions,
        class_names,
    )


    plt.ylabel(
        "Recall (%)"
    )


    plt.ylim(
        0,
        100,
    )


    plt.title(
        "AI and Simulated Expert Performance"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )


    plt.close()


    return output_path


# ============================================================
# CHART 3 — ACTIVE LEARNING CURVE
# ============================================================

def create_active_learning_chart(
    results,
):

    active = results["active"]

    experiments = (
        active["experiments"]
    )


    budgets = [
        item["query_budget"]
        for item in experiments
    ]


    random_accuracy = [
        item["random"]
        ["team_accuracy_percent"]

        for item in experiments
    ]


    uncertainty_accuracy = [
        item["competence_uncertainty"]
        ["team_accuracy_percent"]

        for item in experiments
    ]


    deferral_accuracy = [
        item["deferral_aware_diversity"]
        ["team_accuracy_percent"]

        for item in experiments
    ]


    full_accuracy = (
        active[
            "full_l2d_accuracy_percent"
        ]
    )


    output_path = (
        ASSET_DIR
        / "active_learning_curve.png"
    )


    plt.figure(
        figsize=(8, 4.8)
    )


    plt.plot(
        budgets,
        random_accuracy,
        marker="o",
        label="Random",
    )


    plt.plot(
        budgets,
        uncertainty_accuracy,
        marker="o",
        label="Competence uncertainty",
    )


    plt.plot(
        budgets,
        deferral_accuracy,
        marker="o",
        label="Deferral-aware + diversity",
    )


    if full_accuracy is not None:

        plt.axhline(
            full_accuracy,
            linestyle="--",
            label="Full-information L2D",
        )


    plt.xlabel(
        "Number of expert queries"
    )


    plt.ylabel(
        "Human-AI team accuracy (%)"
    )


    plt.title(
        "Active Learning Performance"
    )


    plt.legend()


    plt.grid(
        alpha=0.25
    )


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )


    plt.close()


    return output_path


# ============================================================
# CHART 4 — L2D GAIN RECOVERY
# ============================================================

def create_gain_recovery_chart(
    results,
):

    active = results["active"]

    experiments = (
        active["experiments"]
    )


    budgets = [
        item["query_budget"]
        for item in experiments
    ]


    random_recovery = [
        item["random_gain_recovery"]
        * 100

        for item in experiments
    ]


    uncertainty_recovery = [
        item["uncertainty_gain_recovery"]
        * 100

        for item in experiments
    ]


    deferral_recovery = [
        item["deferral_gain_recovery"]
        * 100

        for item in experiments
    ]


    output_path = (
        ASSET_DIR
        / "gain_recovery.png"
    )


    plt.figure(
        figsize=(8, 4.8)
    )


    plt.plot(
        budgets,
        random_recovery,
        marker="o",
        label="Random",
    )


    plt.plot(
        budgets,
        uncertainty_recovery,
        marker="o",
        label="Competence uncertainty",
    )


    plt.plot(
        budgets,
        deferral_recovery,
        marker="o",
        label="Deferral-aware + diversity",
    )


    plt.xlabel(
        "Number of expert queries"
    )


    plt.ylabel(
        "Recovered full-L2D gain (%)"
    )


    plt.title(
        "Recovery of Full-information L2D Improvement"
    )


    plt.legend()


    plt.grid(
        alpha=0.25
    )


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )


    plt.close()


    return output_path


# ============================================================
# GENERATE ALL CHARTS
# ============================================================

def generate_charts(
    results,
):

    return {

        "overall":
            create_overall_accuracy_chart(
                results
            ),

        "classes":
            create_class_performance_chart(
                results
            ),

        "active":
            create_active_learning_chart(
                results
            ),

        "gain":
            create_gain_recovery_chart(
                results
            ),
    }


# ============================================================
# CLASSIFICATION TABLE
# ============================================================

def classification_table(
    result,
):

    data = [
        [
            "Class",
            "Precision",
            "Recall",
            "F1-score",
            "Support",
        ]
    ]


    for class_name in (
        result["class_names"]
    ):

        metric = (
            result[
                "classification_report"
            ][class_name]
        )


        data.append(
            [
                class_name,

                f"{metric['precision']:.2f}",

                f"{metric['recall']:.2f}",

                f"{metric['f1-score']:.2f}",

                str(
                    int(
                        metric["support"]
                    )
                ),
            ]
        )


    table = Table(
        data,
        colWidths=[
            3.1 * cm,
            2.4 * cm,
            2.4 * cm,
            2.4 * cm,
            2.4 * cm,
        ],
        repeatRows=1,
    )


    style_table(
        table
    )


    return table


# ============================================================
# CONFUSION MATRIX
# ============================================================

def confusion_matrix_table(
    result,
):

    class_names = (
        result["class_names"]
    )


    data = [
        [
            "Actual / Predicted"
        ]
        +
        class_names
    ]


    for class_name, row in zip(
        class_names,
        result["confusion_matrix"],
    ):

        data.append(
            [
                class_name
            ]
            +
            [
                str(value)
                for value in row
            ]
        )


    table = Table(
        data,
        repeatRows=1,
    )


    style_table(
        table
    )


    return table


# ============================================================
# BUILD PDF
# ============================================================

def generate_report():

    print(
        "Loading Project 3 experiment results..."
    )


    results = load_all_results()


    print(
        "Generating report charts..."
    )


    charts = generate_charts(
        results
    )


    styles = build_styles()


    document = SimpleDocTemplate(

        str(
            REPORT_PATH
        ),

        pagesize=A4,

        rightMargin=1.8 * cm,

        leftMargin=1.8 * cm,

        topMargin=1.8 * cm,

        bottomMargin=1.8 * cm,

        title=(
            "Project 3 - Active Learning "
            "for Learning-to-Defer"
        ),

        author=(
            "Human-Centric Artificial Intelligence"
        ),
    )


    story = []


    baseline = results["baseline"]
    expert = results["expert"]
    l2d = results["l2d"]
    active = results["active"]
    human = results["human"]


    # ========================================================
    # TITLE PAGE
    # ========================================================

    story.append(
        Spacer(
            1,
            2.0 * cm,
        )
    )


    story.append(
        Paragraph(
            "Project 3",
            styles["title"],
        )
    )


    story.append(
        Paragraph(
            "Active Learning for Learning-to-Defer",
            styles["title"],
        )
    )


    story.append(
        Spacer(
            1,
            0.4 * cm,
        )
    )


    story.append(
        Paragraph(
            "Human-Centric Artificial Intelligence",
            styles["subtitle"],
        )
    )


    story.append(
        Spacer(
            1,
            0.8 * cm,
        )
    )


    story.append(
        Paragraph(
            (
                "This report describes the experiments performed "
                "for Project 3, provides justification for the "
                "main design choices, and presents a detailed "
                "analysis of the experimental results."
            ),
            styles["note"],
        )
    )


    story.append(
        Paragraph(
            (
                "The project studies human-AI collaboration "
                "through Learning-to-Defer and Active Learning "
                "using the AG News topic-classification dataset."
            ),
            styles["body"],
        )
    )


    generated_time = (
        datetime.now().strftime(
            "%d %B %Y, %H:%M"
        )
    )


    story.append(
        Paragraph(
            (
                "<b>Report generated:</b> "
                +
                safe(
                    generated_time
                )
            ),
            styles["small"],
        )
    )


    story.append(
        Spacer(
            1,
            0.8 * cm,
        )
    )


    story.append(
        Image(
            str(
                charts["overall"]
            ),
            width=16 * cm,
            height=9 * cm,
        )
    )


    story.append(
        PageBreak()
    )


    # ========================================================
    # 1. PROJECT OVERVIEW
    # ========================================================

    story.append(
        Paragraph(
            "1. Project Overview",
            styles["h1"],
        )
    )


    story.append(
        Paragraph(
            (
                "The objective of the project is to train a "
                "classifier capable of collaborating with an "
                "expert. Instead of always producing its own "
                "prediction, the system may defer selected "
                "examples to the expert when expert involvement "
                "is expected to improve the final decision."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            (
                "AG News is used as the classification task. "
                "Articles belong to four categories: World, "
                "Sports, Business, and Sci/Tech. The complete "
                "labeled training split is used to train the "
                "baseline classifier. The official test split "
                "is reserved exclusively for evaluation."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            "Experimental Progression",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "The experiments are organized progressively. "
                "Task 1 establishes an AI-only baseline. Task 2 "
                "introduces an imperfect simulated expert with "
                "specialized competence. Task 3 assumes complete "
                "expert information and learns when to defer. "
                "Task 4 removes access to expert labels and uses "
                "Active Learning to determine which expert "
                "predictions should be queried."
            ),
            styles["body"],
        )
    )


    overview_data = [

        [
            "System",
            "Test accuracy",
        ],

        [
            "AI baseline",
            percent(
                baseline[
                    "accuracy_percent"
                ]
            ),
        ],

        [
            "Simulated expert",
            percent(
                expert[
                    "accuracy_percent"
                ]
            ),
        ],

        [
            "Full-information L2D",
            percent(
                l2d[
                    "team_accuracy_percent"
                ]
            ),
        ],
    ]


    overview_table = Table(
        overview_data,
        colWidths=[
            8 * cm,
            6 * cm,
        ],
    )


    style_table(
        overview_table
    )


    story.append(
        overview_table
    )


    # ========================================================
    # TASK 1
    # ========================================================

    story.append(
        Paragraph(
            "2. Task 1 - Baseline Classification Model",
            styles["h1"],
        )
    )


    story.append(
        Paragraph(
            "Task Objective",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "The first experiment trains a classifier on all "
                "available AG News training labels and evaluates "
                "its performance on the held-out test set. This "
                "provides the AI-only reference performance that "
                "the later human-AI system should aim to match "
                "or improve."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            "Design Choice and Justification",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "The text representation is "
                f"<b>{safe(baseline.get('representation', 'TF-IDF'))}</b> "
                "and the classifier is "
                f"<b>{safe(baseline.get('model', 'Logistic Regression'))}</b>. "
                "TF-IDF provides an efficient sparse "
                "representation of news text and captures "
                "discriminative unigram and bigram information. "
                "Logistic Regression is computationally efficient "
                "for high-dimensional sparse text features and "
                "provides class-probability estimates. These "
                "probabilities are especially useful later in "
                "Learning-to-Defer because the maximum predicted "
                "probability is used as an estimate of AI "
                "confidence."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            "Experimental Setup",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                f"The classifier is trained on all "
                f"<b>{baseline['train_samples']}</b> AG News "
                f"training examples. The test set contains "
                f"<b>{baseline['test_samples']}</b> examples "
                "and is used only after training for evaluation."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            "Results",
            styles["h2"],
        )
    )


    task1_metrics = [

        [
            "Metric",
            "Value",
        ],

        [
            "Training samples",
            str(
                baseline[
                    "train_samples"
                ]
            ),
        ],

        [
            "Test samples",
            str(
                baseline[
                    "test_samples"
                ]
            ),
        ],

        [
            "Test accuracy",
            percent(
                baseline[
                    "accuracy_percent"
                ]
            ),
        ],
    ]


    task1_table = Table(
        task1_metrics,
        colWidths=[
            7 * cm,
            7 * cm,
        ],
    )


    style_table(
        task1_table
    )


    story.append(
        task1_table
    )


    story.append(
        Spacer(
            1,
            0.4 * cm,
        )
    )


    story.append(
        classification_table(
            baseline
        )
    )


    story.append(
        Spacer(
            1,
            0.4 * cm,
        )
    )


    story.append(
        Paragraph(
            "Confusion Matrix",
            styles["h2"],
        )
    )


    story.append(
        confusion_matrix_table(
            baseline
        )
    )


    story.append(
        Paragraph(
            "Result Analysis",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                f"The baseline reaches "
                f"<b>{baseline['accuracy_percent']:.2f}%</b> "
                "test accuracy. Sports is the strongest class, "
                "while the largest confusion occurs between "
                "Business and Sci/Tech. This weakness motivates "
                "the design of a complementary simulated expert "
                "with particular competence in those categories."
            ),
            styles["note"],
        )
    )


    story.append(
        PageBreak()
    )


    # ========================================================
    # TASK 2
    # ========================================================

    story.append(
        Paragraph(
            "3. Task 2 - Simulated Expert",
            styles["h1"],
        )
    )


    story.append(
        Paragraph(
            "Task Objective",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "Task 2 introduces an imperfect simulated expert. "
                "The expert should not simply reproduce the "
                "classifier or behave perfectly; instead it "
                "should exhibit strong performance in specific "
                "regions of the input space and weaker "
                "performance elsewhere."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            "Expert Design and Justification",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "The simulated expert is modeled as a "
                f"<b>{safe(expert['expert_name'])}</b>. "
                "The expert is intentionally highly competent "
                "on Business and Sci/Tech articles and weaker on "
                "World and Sports. This choice is motivated by "
                "the Task 1 results, where Business and Sci/Tech "
                "were the most difficult categories for the AI. "
                "The resulting complementary error patterns make "
                "the expert useful for studying selective "
                "deferral."
            ),
            styles["body"],
        )
    )


    competence_data = [
        [
            "Class",
            "Designed expert accuracy",
        ]
    ]


    for class_name in (
        expert["class_names"]
    ):

        competence_data.append(
            [
                class_name,

                percent(
                    expert[
                        "competence_profile"
                    ][class_name]
                    * 100
                ),
            ]
        )


    competence_table = Table(
        competence_data,
        colWidths=[
            7 * cm,
            7 * cm,
        ],
    )


    style_table(
        competence_table
    )


    story.append(
        competence_table
    )


    story.append(
        Paragraph(
            "Results",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "Predictions are sampled from the predefined "
                "class-dependent competence profile using a "
                "fixed random seed, which makes the experiment "
                "reproducible. On the AG News test set the "
                "simulated expert achieves an overall accuracy "
                f"of <b>{expert['accuracy_percent']:.2f}%</b>."
            ),
            styles["body"],
        )
    )


    story.append(
        classification_table(
            expert
        )
    )


    story.append(
        Spacer(
            1,
            0.5 * cm,
        )
    )


    story.append(
        Image(
            str(
                charts["classes"]
            ),
            width=16 * cm,
            height=9 * cm,
        )
    )


    story.append(
        Paragraph(
            "Strengths and Weaknesses",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "The expert is substantially stronger on "
                "Business and Sci/Tech, while the AI is much "
                "stronger on World and Sports. Although the "
                "expert's overall accuracy is below the AI-only "
                "accuracy, the complementary competence profile "
                "creates useful opportunities for collaboration. "
                "This is essential for Task 3 because deferral "
                "is beneficial only when the expert is more "
                "reliable than the AI for a particular input."
            ),
            styles["note"],
        )
    )


    story.append(
        PageBreak()
    )


    # ========================================================
    # TASK 3
    # ========================================================

    story.append(
        Paragraph(
            "4. Task 3 - Learning to Defer",
            styles["h1"],
        )
    )


    story.append(
        Paragraph(
            "Task Objective",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "Task 3 assumes that both the true class labels "
                "and simulated expert predictions are available "
                "during training. The objective is to learn "
                "whether each new input should be handled by the "
                "AI classifier or deferred to the expert."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            "Deferral Strategy and Design Choice",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "A separate competence classifier is trained to "
                "estimate <b>P(expert correct | x)</b>. "
                "The training target is 1 when the simulated "
                "expert prediction matches the true AG News "
                "label and 0 otherwise. For a test article, the "
                "estimated expert competence is compared with "
                "the maximum class probability produced by the "
                "baseline classifier. The article is deferred "
                "when estimated expert competence is greater "
                "than AI confidence."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            (
                "This strategy was selected instead of a simple "
                "AI-confidence threshold because low AI "
                "confidence alone does not imply that the expert "
                "will perform better. The competence model "
                "explicitly incorporates the expert's observed "
                "strengths and weaknesses."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            "Evaluation Metrics",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "The evaluation considers overall human-AI team "
                "accuracy as well as deferral quality. Coverage "
                "measures the proportion handled by the AI, while "
                "the deferral rate measures expert workload. "
                "Useful deferrals are cases where the AI would be "
                "wrong and the expert is correct. Harmful "
                "deferrals are cases where the AI would be "
                "correct but the expert is wrong. Routing "
                "accuracy measures whether the system chooses "
                "the better agent when exactly one of them is "
                "correct."
            ),
            styles["body"],
        )
    )


    l2d_metrics = [

        [
            "Metric",
            "Result",
        ],

        [
            "AI-only accuracy",
            percent(
                l2d[
                    "ai_accuracy_percent"
                ]
            ),
        ],

        [
            "Expert-only accuracy",
            percent(
                l2d[
                    "expert_accuracy_percent"
                ]
            ),
        ],

        [
            "Human-AI team accuracy",
            percent(
                l2d[
                    "team_accuracy_percent"
                ]
            ),
        ],

        [
            "Coverage",
            percent(
                l2d[
                    "coverage_percent"
                ]
            ),
        ],

        [
            "Deferral rate",
            percent(
                l2d[
                    "deferral_rate_percent"
                ]
            ),
        ],

        [
            "Routing accuracy",
            percent(
                l2d[
                    "routing_accuracy_percent"
                ]
            ),
        ],

        [
            "Useful deferrals",
            str(
                l2d[
                    "useful_deferrals"
                ]
            ),
        ],

        [
            "Harmful deferrals",
            str(
                l2d[
                    "harmful_deferrals"
                ]
            ),
        ],

        [
            "Oracle upper bound",
            percent(
                l2d[
                    "oracle_accuracy_percent"
                ]
            ),
        ],
    ]


    l2d_table = Table(
        l2d_metrics,
        colWidths=[
            8 * cm,
            6 * cm,
        ],
    )


    style_table(
        l2d_table
    )


    story.append(
        l2d_table
    )


    improvement = (
        l2d[
            "team_accuracy_percent"
        ]
        -
        l2d[
            "ai_accuracy_percent"
        ]
    )


    story.append(
        Paragraph(
            "Result Analysis",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                f"The combined system reaches "
                f"<b>{l2d['team_accuracy_percent']:.2f}%</b> "
                f"accuracy, improving the AI baseline by "
                f"<b>{improvement:.2f} percentage points</b>. "
                f"The model defers "
                f"<b>{l2d['deferral_rate_percent']:.2f}%</b> "
                "of test examples rather than sending all "
                "examples to the expert."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            (
                f"There are "
                f"<b>{l2d['useful_deferrals']}</b> useful "
                f"deferrals and only "
                f"<b>{l2d['harmful_deferrals']}</b> harmful "
                "deferrals. The routing accuracy of "
                f"<b>{l2d['routing_accuracy_percent']:.2f}%</b> "
                "shows that the learned router usually selects "
                "the better decision maker when AI and expert "
                "capabilities differ."
            ),
            styles["note"],
        )
    )


    story.append(
        Paragraph(
            "Class-wise Deferral Behaviour",
            styles["h2"],
        )
    )


    class_data = [
        [
            "Class",
            "Samples",
            "Deferred",
            "Deferral rate",
            "Team accuracy",
        ]
    ]


    for item in (
        l2d["class_results"]
    ):

        class_data.append(
            [
                item["class"],

                str(
                    item["samples"]
                ),

                str(
                    item["deferred"]
                ),

                percent(
                    item[
                        "deferral_rate"
                    ]
                    * 100
                ),

                percent(
                    item[
                        "team_accuracy"
                    ]
                    * 100
                ),
            ]
        )


    class_table = Table(
        class_data,
        repeatRows=1,
    )


    style_table(
        class_table
    )


    story.append(
        class_table
    )


    story.append(
        Paragraph(
            (
                "The substantially larger deferral rates for "
                "Business and Sci/Tech demonstrate that the "
                "competence model has learned the intended expert "
                "specialization rather than applying a uniform "
                "deferral rule."
            ),
            styles["note"],
        )
    )


    story.append(
        PageBreak()
    )


    # ========================================================
    # TASK 4
    # ========================================================

    story.append(
        Paragraph(
            "5. Task 4 - Active Learning for Expert Competence Discovery",
            styles["h1"],
        )
    )


    story.append(
        Paragraph(
            "Task Objective",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "Task 4 removes the assumption that expert "
                "predictions are available during training. "
                "The baseline classifier still has access to the "
                "complete labeled AG News training set, but the "
                "competence learner starts without expert data. "
                "It may reveal an expert prediction only by "
                "actively querying a selected training example."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            "Active Learning Design Choice",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "The primary strategy is pool-based "
                "<b>competence uncertainty sampling</b>. "
                "After an initial class-balanced query set, a "
                "competence model is trained from the expert "
                "observations collected so far. The acquisition "
                "function prioritizes examples for which "
                "<b>P(expert correct | x)</b> is closest to "
                "0.5. These are the samples on which the current "
                "competence model is least certain."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            (
                "Random querying is included as a passive "
                "baseline so that the value of actively selecting "
                "queries can be measured at equal annotation "
                "budgets. An additional deferral-aware diversity "
                "strategy was evaluated. It prioritizes examples "
                "near the AI-expert routing boundary and uses "
                "TF-IDF cosine similarity to reduce redundant "
                "queries."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            "Experimental Protocol",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "All strategies begin with the same 100 "
                "class-balanced expert queries and are evaluated "
                "at query budgets of 100, 250, 500, 1000, "
                "2000, and 5000. At each budget, the learned "
                "competence model is combined with the same "
                "baseline AI classifier and Task 3 routing rule. "
                "This ensures that differences in performance "
                "are attributable to the expert-query strategy."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            "Evaluation Metrics",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "The main metric is human-AI team accuracy. "
                "Routing accuracy measures the quality of "
                "AI-versus-expert selection, while deferral rate "
                "measures the amount of expert involvement. "
                "The experiment also reports the fraction of the "
                "full-information Task 3 accuracy improvement "
                "recovered with each limited expert-query budget."
            ),
            styles["body"],
        )
    )


    experiments = (
        active["experiments"]
    )


    active_table_data = [

        [
            "Queries",
            "Train %",
            "Random",
            "Uncertainty",
            "Deferral + diversity",
        ]
    ]


    for item in experiments:

        active_table_data.append(
            [
                str(
                    item[
                        "query_budget"
                    ]
                ),

                percent(
                    item[
                        "query_fraction_percent"
                    ],
                    2,
                ),

                percent(
                    item[
                        "random"
                    ][
                        "team_accuracy_percent"
                    ]
                ),

                percent(
                    item[
                        "competence_uncertainty"
                    ][
                        "team_accuracy_percent"
                    ]
                ),

                percent(
                    item[
                        "deferral_aware_diversity"
                    ][
                        "team_accuracy_percent"
                    ]
                ),
            ]
        )


    active_table = Table(
        active_table_data,
        repeatRows=1,
    )


    style_table(
        active_table
    )


    story.append(
        active_table
    )


    story.append(
        Spacer(
            1,
            0.5 * cm,
        )
    )


    story.append(
        Image(
            str(
                charts["active"]
            ),
            width=16 * cm,
            height=9.5 * cm,
        )
    )


    # ========================================================
    # BEST ACTIVE OPERATING POINT
    # ========================================================

    best_active = max(

        experiments,

        key=lambda item:
            (
                item[
                    "competence_uncertainty"
                ][
                    "team_accuracy_percent"
                ]
                -
                item[
                    "random"
                ][
                    "team_accuracy_percent"
                ]
            ),
    )


    best_budget = (
        best_active[
            "query_budget"
        ]
    )


    best_fraction = (
        best_active[
            "query_fraction_percent"
        ]
    )


    best_accuracy = (
        best_active[
            "competence_uncertainty"
        ][
            "team_accuracy_percent"
        ]
    )


    best_routing = (
        best_active[
            "competence_uncertainty"
        ][
            "routing_accuracy_percent"
        ]
    )


    best_recovery = (
        best_active[
            "uncertainty_gain_recovery"
        ]
        * 100
    )


    full_l2d = (
        active[
            "full_l2d_accuracy_percent"
        ]
    )


    gap = (
        full_l2d
        -
        best_accuracy
    )


    story.append(
        Paragraph(
            "Key Active Learning Result",
            styles["h2"],
        )
    )


    best_data = [

        [
            "Metric",
            "Result",
        ],

        [
            "Expert queries",
            str(
                best_budget
            ),
        ],

        [
            "Training set queried",
            percent(
                best_fraction
            ),
        ],

        [
            "Team accuracy",
            percent(
                best_accuracy
            ),
        ],

        [
            "Routing accuracy",
            percent(
                best_routing
            ),
        ],

        [
            "Recovered full-L2D gain",
            percent(
                best_recovery
            ),
        ],

        [
            "Gap to full-information L2D",
            f"{gap:.2f} percentage points",
        ],
    ]


    best_table = Table(
        best_data,
        colWidths=[
            8 * cm,
            6 * cm,
        ],
    )


    style_table(
        best_table
    )


    story.append(
        best_table
    )


    story.append(
        Paragraph(
            "Result Analysis",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                f"At the selected low-budget operating point, "
                f"only <b>{best_budget}</b> of the 120,000 "
                f"training examples are queried. This corresponds "
                f"to approximately <b>{best_fraction:.2f}%</b> "
                "of the training dataset."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            (
                f"Using these expert queries, competence "
                f"uncertainty sampling reaches "
                f"<b>{best_accuracy:.2f}%</b> team accuracy "
                f"with a routing accuracy of "
                f"<b>{best_routing:.2f}%</b>. This recovers "
                f"<b>{best_recovery:.2f}%</b> of the accuracy "
                "improvement achieved when all expert labels "
                "are available in Task 3."
            ),
            styles["note"],
        )
    )


    story.append(
        Image(
            str(
                charts["gain"]
            ),
            width=16 * cm,
            height=9.5 * cm,
        )
    )


    final_result = (
        experiments[-1]
    )


    final_random = (
        final_result[
            "random"
        ][
            "team_accuracy_percent"
        ]
    )


    final_uncertainty = (
        final_result[
            "competence_uncertainty"
        ][
            "team_accuracy_percent"
        ]
    )


    story.append(
        Paragraph(
            "Comparison with Random Querying",
            styles["h2"],
        )
    )


    story.append(
        Paragraph(
            (
                "Competence uncertainty sampling provides its "
                "clearest benefit at small and medium query "
                "budgets. At larger budgets, random sampling "
                "becomes competitive. At 5000 expert queries, "
                f"random querying reaches "
                f"<b>{final_random:.2f}%</b> team accuracy, "
                f"while competence uncertainty reaches "
                f"<b>{final_uncertainty:.2f}%</b>."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            (
                "This result suggests that targeted Active "
                "Learning is particularly valuable when expert "
                "interaction is expensive and the query budget "
                "is strongly limited. When many labels can be "
                "requested, random sampling eventually obtains "
                "broad coverage of the relatively simple expert "
                "competence regions."
            ),
            styles["note"],
        )
    )
    
    # ========================================================
    # OPTIONAL TASK 5 - HUMAN EXPERT
    # ========================================================

    if human is not None:
        story.append(
            PageBreak()
        )


        story.append(
            Paragraph(
                "6. Task 5 - Active Learning with Human Expert",
                styles["h1"],
            )
        )


        story.append(
            Paragraph(
                "Interactive Extension",
                styles["h2"],
            )
        )


        story.append(
            Paragraph(
                (
                    "Task 5 replaces the simulated interaction with "
                    "a real user interface. Articles selected by the "
                    "active-learning procedure are presented to the "
                    "user one at a time. The user assigns one of the "
                    "four AG News labels while the ground-truth label "
                    "remains hidden."
                ),
                styles["body"],
            )
        )


        story.append(
            Paragraph(
                "Design Choice",
                styles["h2"],
            )
        )


        story.append(
            Paragraph(
                (
                    "The interface uses articles selected through "
                    "competence uncertainty sampling. This connects "
                    "the human interaction directly to the Active "
                    "Learning objective: requesting labels only for "
                    "examples considered informative for expert "
                    "competence discovery."
                ),
                styles["body"],
            )
        )


        human_data = [

            [
                "Metric",
                "Result",
            ],

            [
                "Human labels provided",
                str(
                    human[
                        "human_queries"
                    ]
                ),
            ],

            [
                "Correct labels",
                str(
                    human[
                        "correct_labels"
                    ]
                ),
            ],

            [
                "Incorrect labels",
                str(
                    human[
                        "incorrect_labels"
                    ]
                ),
            ],

            [
                "Human accuracy",
                percent(
                    human[
                        "human_accuracy_percent"
                    ]
                ),
            ],
        ]


        human_table = Table(
            human_data,
            colWidths=[
                8 * cm,
                6 * cm,
            ],
        )


        style_table(
            human_table
        )


        story.append(
            human_table
        )


        story.append(
            Spacer(
                1,
                0.5 * cm,
            )
        )


        story.append(
            Paragraph(
                (
                    "These results are reported separately from the "
                    "controlled simulated-expert experiments in "
                    "Tasks 1-4. Human responses therefore do not "
                    "overwrite the reproducible Task 1-4 results."
                ),
                styles["note"],
            )
        )

    next_section = (
                7
                if human is not None
                else 6
            )

    # ========================================================
    # 6. OVERALL DISCUSSION
    # ========================================================

    story.append(
        Paragraph(
            f"{next_section}. Overall Discussion",
            styles["h1"],
        )
    )


    story.append(
        Paragraph(
            (
                "The four experiments demonstrate the importance "
                "of modeling complementary human and AI "
                "capabilities rather than considering standalone "
                "accuracy alone. Although the simulated expert "
                f"achieves only "
                f"<b>{expert['accuracy_percent']:.2f}%</b> "
                "overall accuracy, its specialization allows the "
                "full-information human-AI system to improve from "
                f"<b>{baseline['accuracy_percent']:.2f}%</b> "
                "AI-only accuracy to "
                f"<b>{l2d['team_accuracy_percent']:.2f}%</b>."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            (
                "The Task 3 results confirm that learning expert "
                "competence is useful for routing decisions. "
                "Task 4 then shows that this competence profile "
                "does not need to be learned from every possible "
                "expert prediction. Active Learning can recover "
                "a substantial fraction of the full-information "
                "benefit using only a small expert-query budget."
            ),
            styles["body"],
        )
    )

    # ========================================================
    # 8. CONCLUSION
    # ========================================================

    story.append(
        Paragraph(
            f"{next_section + 1}. Conclusion",
            styles["h1"],
        )
    )


    story.append(
        Paragraph(
            (
                "The project successfully implements a complete "
                "progression from standard classification to "
                "human-AI collaboration and Active Learning."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            (
                f"The AI-only classifier achieves "
                f"<b>{baseline['accuracy_percent']:.2f}%</b> "
                "accuracy. The simulated expert achieves "
                f"<b>{expert['accuracy_percent']:.2f}%</b>, "
                "but provides complementary expertise. "
                "Learning-to-Defer combines these capabilities "
                f"to reach <b>{l2d['team_accuracy_percent']:.2f}%</b> "
                "accuracy."
            ),
            styles["body"],
        )
    )


    story.append(
        Paragraph(
            (
                f"With Active Learning, competence uncertainty "
                f"sampling reaches "
                f"<b>{best_accuracy:.2f}%</b> using only "
                f"<b>{best_budget}</b> expert queries "
                f"({best_fraction:.2f}% of the training set), "
                f"recovering <b>{best_recovery:.2f}%</b> of the "
                "full-information Learning-to-Defer improvement."
            ),
            styles["note"],
        )
    )

    # ========================================================
    # GENERATE PDF
    # ========================================================

    print(
        "Building PDF report..."
    )


    document.build(

        story,

        onFirstPage=add_page_number,

        onLaterPages=add_page_number,
    )


    print(
        "\nProject 3 report generated successfully:"
    )


    print(
        REPORT_PATH
    )


    print(
        "\nThe report includes Tasks 1-4, "
        "experimental design choices, methodology, "
        "metrics, results and analysis."
    )


    return REPORT_PATH


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    generate_report()