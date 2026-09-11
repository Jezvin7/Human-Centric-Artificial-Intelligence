from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)


# ============================================================
# Paths
# ============================================================

APP_DIR = Path(__file__).resolve().parent

OUTPUT_PATH = (
    APP_DIR
    / "static"
    / "project4"
    / "reports"
    / "project4_report.pdf"
)


# ============================================================
# Project information
# ============================================================

PROJECT_TITLE = "Project 4: Preference Elicitation"
COURSE_NAME = "Human-Centric Artificial Intelligence"




# ============================================================
# Colours
# ============================================================

PURPLE = colors.HexColor("#2D1B69")
ACCENT = colors.HexColor("#5B3FB2")
LIGHT_PURPLE = colors.HexColor("#F1EEFA")
LIGHT_GREY = colors.HexColor("#F6F7FA")
MID_GREY = colors.HexColor("#D9DDE6")
TEXT_GREY = colors.HexColor("#4B5563")


# ============================================================
# Styles
# ============================================================

styles = getSampleStyleSheet()

styles.add(
    ParagraphStyle(
        name="ProjectTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=PURPLE,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
)

styles.add(
    ParagraphStyle(
        name="CourseTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        alignment=TA_CENTER,
        textColor=TEXT_GREY,
        spaceAfter=8,
    )
)

styles.add(
    ParagraphStyle(
        name="Author",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=TEXT_GREY,
    )
)

styles.add(
    ParagraphStyle(
        name="H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=PURPLE,
        spaceBefore=10,
        spaceAfter=8,
    )
)

styles.add(
    ParagraphStyle(
        name="H2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=ACCENT,
        spaceBefore=8,
        spaceAfter=5,
    )
)

styles.add(
    ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        spaceAfter=7,
    )
)

styles.add(
    ParagraphStyle(
        name="Formula",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.5,
        leading=13,
        leftIndent=8 * mm,
        rightIndent=8 * mm,
        backColor=LIGHT_GREY,
        borderColor=MID_GREY,
        borderWidth=0.5,
        borderPadding=7,
        spaceBefore=5,
        spaceAfter=8,
    )
)

styles.add(
    ParagraphStyle(
        name="Note",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=TEXT_GREY,
        backColor=LIGHT_PURPLE,
        borderPadding=7,
        spaceBefore=6,
        spaceAfter=8,
    )
)

styles.add(
    ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
    )
)


# ============================================================
# Helpers
# ============================================================

def P(text, style="BodyCustom"):
    """Create a paragraph."""
    return Paragraph(text, styles[style])


def section(title):
    return Paragraph(title, styles["H1Custom"])


def subsection(title):
    return Paragraph(title, styles["H2Custom"])


def make_table(data, widths=None):
    """
    Standard table style used throughout the report.
    """

    table = Table(
        data,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    PURPLE,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "LEADING",
                    (0, 0),
                    (-1, -1),
                    11,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#AEB4BF"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
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
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        LIGHT_GREY,
                    ],
                ),
            ]
        )
    )

    return table


# ============================================================
# Header / Footer
# ============================================================

def add_page_number(canvas, doc):
    canvas.saveState()

    canvas.setFont(
        "Helvetica",
        8
    )

    canvas.setFillColor(
        TEXT_GREY
    )

    page_number = canvas.getPageNumber()

    canvas.drawString(
        20 * mm,
        10 * mm,
        COURSE_NAME,
    )

    canvas.drawRightString(
        A4[0] - 20 * mm,
        10 * mm,
        f"Page {page_number}",
    )

    canvas.restoreState()


# ============================================================
# Main report
# ============================================================

def build_report():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=PROJECT_TITLE,
    )

    story = []

    # ========================================================
    # COVER PAGE
    # ========================================================

    story.append(
        Spacer(
            1,
            32 * mm,
        )
    )

    story.append(
        P(
            PROJECT_TITLE,
            "ProjectTitle",
        )
    )

    story.append(
        P(
            COURSE_NAME,
            "CourseTitle",
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    
    story.append(
        P(
            "<b>Objective</b>",
            "H2Custom",
        )
    )

    story.append(
        P(
            "The objective of this project is to design a rigorous "
            "user study comparing two methods for eliciting movie "
            "preferences: repeated pairwise choices and ranking sets "
            "of ten movies. The project also implements the complete "
            "participant-facing Django interface that could be used "
            "to conduct the proposed study."
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        P(
            "<b>Important:</b> The user study is designed as part of "
            "this project but is <b>not conducted</b>. Therefore, this "
            "report presents the experimental protocol and analysis "
            "plan but does not report empirical participant results.",
            "Note",
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # 1 INTRODUCTION
    # ========================================================

    story.append(
        section(
            "1. Introduction"
        )
    )

    story.append(
        P(
            "The purpose of preference elicitation is to infer what a "
            "user likes from a relatively small number of interactions. "
            "In this project, the application domain is movie "
            "recommendation. For a movie represented by a feature vector "
            "<i>x</i>, the user's utility is modeled as:"
        )
    )

    story.append(
        P(
            "U(x) = w^T x",
            "Formula",
        )
    )

    story.append(
        P(
            "The vector <i>w</i> represents the user's latent movie "
            "preferences. Positive components indicate characteristics "
            "that increase predicted utility, while negative components "
            "indicate characteristics that decrease predicted utility. "
            "The preference elicitation process therefore aims to estimate "
            "<i>w</i> from limited user feedback."
        )
    )

    # ========================================================
    # 2 DATASET
    # ========================================================

    story.append(
        section(
            "2. Dataset"
        )
    )

    story.append(
        P(
            "The project uses the IMDB 5000 Movie Dataset. The dataset "
            "contains metadata for approximately 5000 movies. Unlike "
            "traditional collaborative-filtering datasets, it does not "
            "contain ratings from individual users. Consequently, the "
            "system cannot directly learn a user's preferences from "
            "historical ratings. Instead, preference information must be "
            "obtained through interactions with the participant."
        )
    )

    story.append(
        P(
            "The implemented system uses movie metadata both for the "
            "numerical preference representation and for presenting "
            "understandable information to participants in the interface."
        )
    )

    # ========================================================
    # TASK 1
    # ========================================================

    story.append(
        section(
            "3. Task 1 - Movie Feature Representation"
        )
    )

    story.append(
        subsection(
            "3.1 Selected Features"
        )
    )

    story.append(
        P(
            "Each movie is represented using genre information, duration, "
            "release year, IMDb score and content rating. The objective is "
            "to capture several meaningful dimensions of movie preference "
            "while keeping the feature space sufficiently compact to "
            "estimate a user model from a limited number of interactions."
        )
    )

    feature_table = [
        [
            P("Feature", "Small"),
            P("Representation", "Small"),
            P("Justification", "Small"),
        ],
        [
            P("Genres", "Small"),
            P("Multi-hot encoding", "Small"),
            P(
                "A movie can belong to several genres and users frequently "
                "have genre-specific preferences.",
                "Small",
            ),
        ],
        [
            P("Duration", "Small"),
            P("Standardized numerical", "Small"),
            P(
                "Captures preferences for shorter or longer movies.",
                "Small",
            ),
        ],
        [
            P("Release year", "Small"),
            P("Standardized numerical", "Small"),
            P(
                "Captures preference for older or newer movies.",
                "Small",
            ),
        ],
        [
            P("IMDb score", "Small"),
            P("Standardized numerical", "Small"),
            P(
                "Provides information about the general reception of a movie.",
                "Small",
            ),
        ],
        [
            P("Content rating", "Small"),
            P("One-hot encoding", "Small"),
            P(
                "Represents categorical information about the intended "
                "audience/content category.",
                "Small",
            ),
        ],
    ]

    story.append(
        make_table(
            feature_table,
            [
                32 * mm,
                45 * mm,
                93 * mm,
            ],
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        P(
            "Director and actor identities are intentionally not included "
            "in the preference vector. These attributes have a very large "
            "number of distinct values and would therefore create a "
            "high-dimensional sparse representation. Because the purpose "
            "of the project is to estimate the latent preference vector "
            "from relatively few interactions, such a representation would "
            "make parameter estimation considerably more difficult. "
            "Director information may still be displayed to the participant "
            "without being included in the learned vector."
        )
    )

    story.append(
        subsection(
            "3.2 Feature Extraction Method"
        )
    )

    story.append(
        P(
            "The feature extraction procedure implemented in "
            "<font name='Courier'>feature_extraction.py</font> performs "
            "the following preprocessing steps:"
        )
    )

    extraction_steps = [
        [
            P("Step", "Small"),
            P("Operation", "Small"),
        ],
        [
            P("1", "Small"),
            P(
                "Load the IMDB CSV file with Pandas.",
                "Small",
            ),
        ],
        [
            P("2", "Small"),
            P(
                "Remove observations without movie titles and remove "
                "duplicate movie records.",
                "Small",
            ),
        ],
        [
            P("3", "Small"),
            P(
                "Split the genre string using the | separator and create "
                "one binary indicator for every genre. Since several "
                "indicators may equal one simultaneously, this is "
                "multi-hot encoding.",
                "Small",
            ),
        ],
        [
            P("4", "Small"),
            P(
                "Convert duration, release year and IMDb score to numeric "
                "values. Missing values are replaced by the median of the "
                "corresponding feature.",
                "Small",
            ),
        ],
        [
            P("5", "Small"),
            P(
                "Standardize the numerical features using "
                "z = (x - mean) / standard deviation.",
                "Small",
            ),
        ],
        [
            P("6", "Small"),
            P(
                "Replace missing content ratings with an Unknown category "
                "and one-hot encode the resulting categorical variable.",
                "Small",
            ),
        ],
        [
            P("7", "Small"),
            P(
                "Concatenate genre, numerical and content-rating features "
                "into the final numerical feature matrix X.",
                "Small",
            ),
        ],
    ]

    story.append(
        make_table(
            extraction_steps,
            [
                18 * mm,
                152 * mm,
            ],
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        P(
            "For movie i, the resulting row x_i of this feature matrix is "
            "used directly in the utility function U(x_i) = w^T x_i."
        )
    )

    # ========================================================
    # TASK 2
    # ========================================================

    story.append(
        section(
            "4. Task 2 - Preference Model"
        )
    )

    story.append(
        subsection(
            "4.1 Bradley-Terry Model for Pairwise Choices"
        )
    )

    story.append(
        P(
            "In the pairwise interface, the participant is shown two "
            "movies i and j and chooses the movie they would rather watch. "
            "Their utilities are U_i = w^T x_i and U_j = w^T x_j."
        )
    )

    story.append(
        P(
            "P(i > j | w) = exp(w^T x_i) / "
            "[exp(w^T x_i) + exp(w^T x_j)]",
            "Formula",
        )
    )

    story.append(
        P(
            "Equivalently, the probability may be written using the "
            "logistic function as:"
        )
    )

    story.append(
        P(
            "P(i > j | w) = sigmoid(w^T (x_i - x_j))",
            "Formula",
        )
    )

    story.append(
        P(
            "The parameter vector w is estimated by maximizing the "
            "likelihood of the participant's observed choices. In the "
            "implementation this is formulated as minimization of the "
            "negative log-likelihood with L2 regularization."
        )
    )

    story.append(
        subsection(
            "4.2 Extension to Rankings: Plackett-Luce Model"
        )
    )

    story.append(
        P(
            "The second interface does not provide a single pairwise "
            "comparison. Instead, the participant provides a complete "
            "ranking:"
        )
    )

    story.append(
        P(
            "i1 > i2 > ... > in",
            "Formula",
        )
    )

    story.append(
        P(
            "To model this information, the Bradley-Terry idea is extended "
            "using the Plackett-Luce model:"
        )
    )

    story.append(
        P(
            "P(i1 > i2 > ... > in | w) = "
            "Product_{k=1}^{n-1} "
            "{ exp(w^T x_ik) / "
            "Sum_{j=k}^{n} exp(w^T x_ij) }",
            "Formula",
        )
    )

    story.append(
        P(
            "The model interprets a ranking as a sequence of choices. "
            "First, i1 is selected as the most preferred movie from all "
            "n alternatives. Next, i2 is selected from the remaining "
            "n-1 alternatives. This continues until the ranking has been "
            "constructed."
        )
    )

    story.append(
        P(
            "This formulation is particularly appropriate because when "
            "only two movies are present, it reduces to the Bradley-Terry "
            "pairwise probability. Consequently, both elicitation "
            "interfaces can use the same movie representation, the same "
            "linear utility function and the same interpretation of the "
            "latent preference vector w."
        )
    )

    story.append(
        subsection(
            "4.3 Parameter Estimation"
        )
    )

    story.append(
        P(
            "Both models are estimated by maximum likelihood. The "
            "implementation minimizes the negative log-likelihood using "
            "numerical optimization and adds an L2 penalty to reduce "
            "overfitting when relatively few preference observations are "
            "available."
        )
    )

    # ========================================================
    # TASK 3
    # ========================================================

    story.append(
        section(
            "5. Task 3 - User Study Design"
        )
    )

    story.append(
        subsection(
            "5.1 Research Question"
        )
    )

    story.append(
        P(
            "<b>Research question:</b> Which preference elicitation "
            "method learns a new user's movie preferences more effectively "
            "and efficiently: repeated pairwise choices or rankings of ten "
            "movies?"
        )
    )

    story.append(
        subsection(
            "5.2 Hypotheses"
        )
    )

    story.append(
        P(
            "<b>Primary null hypothesis H0:</b> There is no difference "
            "between the pairwise and ranking elicitation methods in "
            "held-out preference prediction performance."
        )
    )

    story.append(
        P(
            "<b>Primary alternative hypothesis H1:</b> The ranking-based "
            "method produces lower held-out negative log loss than the "
            "pairwise method when both methods use the same movie exposure "
            "budget."
        )
    )

    story.append(
        P(
            "<b>Secondary hypothesis H2:</b> Ranking ten movies "
            "simultaneously is perceived as more mentally demanding than "
            "making pairwise movie choices."
        )
    )

    story.append(
        subsection(
            "5.3 Experimental Design"
        )
    )

    story.append(
        P(
            "A <b>within-subject design</b> is proposed. Every participant "
            "uses both preference elicitation interfaces. This design is "
            "appropriate because movie tastes differ considerably between "
            "individuals; evaluating both methods on the same participant "
            "reduces variability due to between-person differences."
        )
    )

    story.append(
        P(
            "A possible disadvantage of a within-subject study is the "
            "presence of learning, fatigue and order effects. To reduce "
            "these effects, the condition order is counterbalanced. "
            "Participants are randomly assigned with approximately equal "
            "probability to one of the following sequences:"
        )
    )

    condition_table = [
        [
            P("Group", "Small"),
            P("Condition order", "Small"),
        ],
        [
            P("A", "Small"),
            P(
                "Pairwise -> Ranking",
                "Small",
            ),
        ],
        [
            P("B", "Small"),
            P(
                "Ranking -> Pairwise",
                "Small",
            ),
        ],
    ]

    story.append(
        make_table(
            condition_table,
            [
                40 * mm,
                130 * mm,
            ],
        )
    )

    story.append(
        subsection(
            "5.4 Independent and Dependent Variables"
        )
    )

    variables_table = [
        [
            P("Type", "Small"),
            P("Variable", "Small"),
        ],
        [
            P("Independent variable", "Small"),
            P(
                "Preference elicitation method: Pairwise vs. Ranking.",
                "Small",
            ),
        ],
        [
            P("Primary dependent variable", "Small"),
            P(
                "Held-out negative log loss of predicted movie preferences.",
                "Small",
            ),
        ],
        [
            P("Secondary dependent variable", "Small"),
            P(
                "Held-out pairwise prediction accuracy.",
                "Small",
            ),
        ],
        [
            P("Secondary dependent variable", "Small"),
            P(
                "Time required to complete each elicitation condition.",
                "Small",
            ),
        ],
        [
            P("Subjective measure", "Small"),
            P(
                "Ease of expressing preferences.",
                "Small",
            ),
        ],
        [
            P("Subjective measure", "Small"),
            P(
                "Perceived mental demand.",
                "Small",
            ),
        ],
        [
            P("Subjective measure", "Small"),
            P(
                "Confidence that the responses represent the participant's "
                "actual preferences.",
                "Small",
            ),
        ],
    ]

    story.append(
        make_table(
            variables_table,
            [
                55 * mm,
                115 * mm,
            ],
        )
    )

    story.append(
        subsection(
            "5.5 Interaction Budget"
        )
    )

    story.append(
        P(
            "The two elicitation conditions are assigned an equal movie "
            "exposure budget. The pairwise condition contains 15 "
            "comparisons, exposing the participant to 30 movies. The "
            "ranking condition contains three ranking tasks with 10 movies "
            "each, also exposing the participant to 30 movies."
        )
    )

    budget_table = [
        [
            P("Stage", "Small"),
            P("Number of tasks", "Small"),
            P("Movie exposure", "Small"),
        ],
        [
            P("Pairwise elicitation", "Small"),
            P("15 pairwise choices", "Small"),
            P("30 movies", "Small"),
        ],
        [
            P("Ranking elicitation", "Small"),
            P("3 rankings x 10 movies", "Small"),
            P("30 movies", "Small"),
        ],
        [
            P("Held-out evaluation", "Small"),
            P("10 pairwise choices", "Small"),
            P("20 new movies", "Small"),
        ],
    ]

    story.append(
        make_table(
            budget_table,
            [
                55 * mm,
                65 * mm,
                50 * mm,
            ],
        )
    )

    story.append(
        subsection(
            "5.6 Movie Selection"
        )
    )

    story.append(
        P(
            "Movies are selected uniformly at random from the available "
            "dataset. For a participant, the implementation samples the "
            "movies used in the elicitation and evaluation stages without "
            "replacement. The 20 movies used in the ten held-out "
            "evaluation pairs are therefore not used in either training "
            "condition."
        )
    )

    story.append(
        P(
            "This separation is important because both learned models "
            "should be evaluated on preferences that were not directly "
            "observed during preference elicitation."
        )
    )

    story.append(
        P(
            "A possible future extension would replace uniform random "
            "sampling with an adaptive strategy that selects movies "
            "expected to provide more information about the user's "
            "preference vector."
        )
    )

    story.append(
        subsection(
            "5.7 Participants and Recruitment"
        )
    )

    story.append(
        P(
            "If the study were conducted, a practical target would be "
            "approximately 30-40 adult participants. This is proposed as "
            "a feasible sample for a course-scale within-subject study; "
            "a formal power analysis should be performed before a real "
            "scientific deployment if a specific minimum detectable "
            "effect is required."
        )
    )

    story.append(
        P(
            "Participants could be recruited through university mailing "
            "lists, student networks and online announcements. Participants "
            "should be at least 18 years old and sufficiently familiar with "
            "movies to express meaningful preferences. Machine-learning "
            "expertise is not required."
        )
    )

    story.append(
        subsection(
            "5.8 Complete Study Procedure"
        )
    )

    procedure = [
        [
            P("Step", "Small"),
            P("Procedure", "Small"),
        ],
        [
            P("1", "Small"),
            P(
                "Present participant information describing the purpose "
                "and procedure of the study.",
                "Small",
            ),
        ],
        [
            P("2", "Small"),
            P(
                "Obtain informed consent.",
                "Small",
            ),
        ],
        [
            P("3", "Small"),
            P(
                "Generate an anonymous participant identifier.",
                "Small",
            ),
        ],
        [
            P("4", "Small"),
            P(
                "Randomly assign the participant to Pairwise -> Ranking "
                "or Ranking -> Pairwise.",
                "Small",
            ),
        ],
        [
            P("5", "Small"),
            P(
                "Run the first elicitation condition and record responses "
                "and completion time.",
                "Small",
            ),
        ],
        [
            P("6", "Small"),
            P(
                "Administer a short post-condition questionnaire.",
                "Small",
            ),
        ],
        [
            P("7", "Small"),
            P(
                "Run the second elicitation condition and record responses "
                "and completion time.",
                "Small",
            ),
        ],
        [
            P("8", "Small"),
            P(
                "Administer the same questionnaire after the second "
                "condition.",
                "Small",
            ),
        ],
        [
            P("9", "Small"),
            P(
                "Present ten new held-out movie pairs and record which "
                "movie is preferred in each pair.",
                "Small",
            ),
        ],
        [
            P("10", "Small"),
            P(
                "Fit one preference vector using the pairwise responses "
                "and another using the ranking responses.",
                "Small",
            ),
        ],
        [
            P("11", "Small"),
            P(
                "Use both models to predict the same held-out choices.",
                "Small",
            ),
        ],
        [
            P("12", "Small"),
            P(
                "Compute objective performance metrics and compare the "
                "two conditions together with subjective questionnaire "
                "scores and completion times.",
                "Small",
            ),
        ],
        [
            P("13", "Small"),
            P(
                "Display the completion/debriefing screen.",
                "Small",
            ),
        ],
    ]

    story.append(
        make_table(
            procedure,
            [
                18 * mm,
                152 * mm,
            ],
        )
    )

    story.append(
        subsection(
            "5.9 Evaluation Metrics"
        )
    )

    story.append(
        P(
            "<b>Held-out accuracy:</b> the proportion of the ten unseen "
            "pairwise preferences that the learned model predicts "
            "correctly."
        )
    )

    story.append(
        P(
            "<b>Held-out negative log loss:</b> evaluates not only whether "
            "the prediction is correct but also whether the model assigns "
            "high probability to the participant's actual choice. This is "
            "used as the primary predictive metric."
        )
    )

    story.append(
        P(
            "<b>Completion time:</b> measures the time required to complete "
            "each elicitation method."
        )
    )

    story.append(
        P(
            "<b>Subjective evaluation:</b> participants rate ease of "
            "preference expression, mental demand and confidence using "
            "five-point response scales."
        )
    )

    story.append(
        subsection(
            "5.10 Statistical Analysis Plan"
        )
    )

    story.append(
        P(
            "Because every participant completes both interfaces, "
            "measurements from the two conditions are paired. For the "
            "primary analysis, held-out log loss from the ranking model "
            "is compared with held-out log loss from the pairwise model."
        )
    )

    story.append(
        P(
            "If the paired differences are approximately normally "
            "distributed, a paired t-test can be used. If this assumption "
            "is not appropriate, the Wilcoxon signed-rank test provides "
            "a non-parametric alternative."
        )
    )

    story.append(
        P(
            "The same paired-analysis logic can be used for completion "
            "time and questionnaire measures. Descriptive statistics "
            "should also be reported for all measures."
        )
    )

    story.append(
        subsection(
            "5.11 Ethical and Data-Protection Considerations"
        )
    )

    story.append(
        P(
            "Participants should receive sufficient information about the "
            "study before taking part and should explicitly provide "
            "informed consent. Participation should be voluntary and "
            "participants should be informed that they may stop the study "
            "at any time without penalty."
        )
    )

    story.append(
        P(
            "The implemented prototype does not request a participant's "
            "name or email address. Instead, a random identifier is stored "
            "with the interaction data. Collected information should be "
            "restricted to what is required for the study, including "
            "preference responses, completion times, condition order and "
            "questionnaire responses."
        )
    )

    story.append(
        P(
            "If the experiment were conducted in practice, participants "
            "should also be informed about the purpose of data collection, "
            "storage duration, access rights and the procedure for "
            "requesting deletion or withdrawal of their data."
        )
    )

    # ========================================================
    # TASK 4
    # ========================================================

    story.append(
        section(
            "6. Task 4 - Interactive User Study Interface"
        )
    )

    story.append(
        P(
            "The complete participant-facing study is implemented as a "
            "Django application. The interface is designed so that the "
            "experimental protocol described above could be run with real "
            "participants, although conducting the study itself is outside "
            "the scope of this project."
        )
    )

    interface_table = [
        [
            P("Page", "Small"),
            P("Purpose", "Small"),
        ],
        [
            P("Landing page", "Small"),
            P(
                "Provides access to this PDF report and the Start Study "
                "button.",
                "Small",
            ),
        ],
        [
            P("Study introduction", "Small"),
            P(
                "Displays participant information and obtains consent.",
                "Small",
            ),
        ],
        [
            P("Pairwise interface", "Small"),
            P(
                "Displays two movies and asks which movie the participant "
                "would rather watch.",
                "Small",
            ),
        ],
        [
            P("Ranking interface", "Small"),
            P(
                "Displays ten movies and allows the participant to reorder "
                "them from most preferred to least preferred.",
                "Small",
            ),
        ],
        [
            P("Questionnaire", "Small"),
            P(
                "Records subjective evaluation after each elicitation "
                "condition.",
                "Small",
            ),
        ],
        [
            P("Evaluation interface", "Small"),
            P(
                "Collects ten held-out pairwise choices using movies not "
                "used during elicitation.",
                "Small",
            ),
        ],
        [
            P("Completion page", "Small"),
            P(
                "Marks the end of the study and can display model "
                "evaluation information for demonstration purposes.",
                "Small",
            ),
        ],
    ]

    story.append(
        make_table(
            interface_table,
            [
                48 * mm,
                122 * mm,
            ],
        )
    )

    story.append(
        Spacer(
            1,
            6 * mm,
        )
    )

    story.append(
        P(
            "The application stores study state using an anonymous "
            "StudySession object. Pairwise responses, ranking responses, "
            "held-out choices, condition order, questionnaire responses "
            "and model estimates can therefore be associated with the same "
            "anonymous participant session."
        )
    )

    # ========================================================
    # CONCLUSION
    # ========================================================

    story.append(
        section(
            "7. Conclusion"
        )
    )

    story.append(
        P(
            "This project proposes and implements a complete protocol for "
            "comparing two preference elicitation strategies for movie "
            "recommendation. A compact metadata-based movie representation "
            "is shared by both methods. Pairwise preferences are modeled "
            "using Bradley-Terry, while complete rankings are modeled using "
            "the Plackett-Luce extension."
        )
    )

    story.append(
        P(
            "The proposed within-subject experiment compares both methods "
            "under an equal movie exposure budget and evaluates them on "
            "the same unseen preference choices. Counterbalancing, "
            "objective prediction metrics, completion time and subjective "
            "measures are included to provide a more complete comparison "
            "of effectiveness and user effort."
        )
    )

    story.append(
        P(
            "No real participants are required for the project submission; "
            "the implemented Django interface demonstrates how the proposed "
            "study could be conducted in practice."
        )
    )

    # ========================================================
    # Build PDF
    # ========================================================

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    print(
        "\nProject 4 report generated successfully."
    )

    print(
        f"Location: {OUTPUT_PATH}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    build_report()