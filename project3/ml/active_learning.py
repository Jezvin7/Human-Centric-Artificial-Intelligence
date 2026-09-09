from pathlib import Path
import json

import joblib
import numpy as np

from datasets import load_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


CLASS_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
]


QUERY_BUDGETS = [
    100,
    250,
    500,
    1000,
    2000,
    5000,
]


INITIAL_QUERIES = 100

BATCH_SIZE = 100

RANDOM_SEED = 42


# ------------------------------------------------------------
# Deferral-aware acquisition settings
# ------------------------------------------------------------

# We first keep examples near the AI/expert routing boundary.
CANDIDATE_MULTIPLIER = 10

# Maximum number of previous queried points used when
# calculating diversity.
MAX_REFERENCE_POINTS = 300

# Main priority remains the deferral boundary.
BOUNDARY_WEIGHT = 0.80

# Diversity prevents querying many almost-identical articles.
DIVERSITY_WEIGHT = 0.20


# ============================================================
# PATHS
# ============================================================

VECTORIZER_PATH = (
    MODEL_DIR
    / "tfidf_vectorizer.joblib"
)

BASELINE_MODEL_PATH = (
    MODEL_DIR
    / "baseline_classifier.joblib"
)

EXPERT_TRAIN_PATH = (
    MODEL_DIR
    / "expert_train_predictions.npy"
)

EXPERT_TEST_PATH = (
    MODEL_DIR
    / "expert_test_predictions.npy"
)

FULL_L2D_RESULTS_PATH = (
    MODEL_DIR
    / "l2d_results.json"
)

RESULTS_PATH = (
    MODEL_DIR
    / "active_learning_results.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading AG News dataset...")

dataset = load_dataset(
    "fancyzhx/ag_news"
)


train_texts = dataset["train"]["text"]

train_labels = np.array(
    dataset["train"]["label"]
)


test_texts = dataset["test"]["text"]

test_labels = np.array(
    dataset["test"]["label"]
)


print(
    f"Training samples: {len(train_labels)}"
)

print(
    f"Test samples: {len(test_labels)}"
)


# ============================================================
# LOAD TASK 1 MODEL
# ============================================================

print(
    "\nLoading baseline classifier and TF-IDF vectorizer..."
)


vectorizer = joblib.load(
    VECTORIZER_PATH
)

baseline_classifier = joblib.load(
    BASELINE_MODEL_PATH
)


X_train = vectorizer.transform(
    train_texts
)

X_test = vectorizer.transform(
    test_texts
)


# ============================================================
# LOAD HIDDEN EXPERT PREDICTIONS
# ============================================================

# IMPORTANT:
#
# During active learning these values represent hidden expert
# knowledge.
#
# They are revealed ONLY through query_expert().

hidden_expert_train = np.load(
    EXPERT_TRAIN_PATH
)

expert_test_predictions = np.load(
    EXPERT_TEST_PATH
)


# ============================================================
# BASELINE AI OUTPUT
# ============================================================

print(
    "Computing baseline AI confidence..."
)


ai_train_predictions = (
    baseline_classifier.predict(
        X_train
    )
)

ai_test_predictions = (
    baseline_classifier.predict(
        X_test
    )
)


ai_train_probabilities = (
    baseline_classifier.predict_proba(
        X_train
    )
)

ai_test_probabilities = (
    baseline_classifier.predict_proba(
        X_test
    )
)


ai_train_confidence = np.max(
    ai_train_probabilities,
    axis=1,
)

ai_test_confidence = np.max(
    ai_test_probabilities,
    axis=1,
)


ai_correct_test = (
    ai_test_predictions
    ==
    test_labels
)


expert_correct_test = (
    expert_test_predictions
    ==
    test_labels
)


ai_accuracy = accuracy_score(
    test_labels,
    ai_test_predictions,
)


# ============================================================
# LOAD FULL-INFORMATION TASK 3 RESULT
# ============================================================

full_l2d_accuracy = None


if FULL_L2D_RESULTS_PATH.exists():

    with open(
        FULL_L2D_RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        full_l2d_results = json.load(file)


    full_l2d_accuracy = (
        full_l2d_results["team_accuracy"]
    )


# ============================================================
# SIMULATED EXPERT QUERY
# ============================================================

def query_expert(sample_index):
    """
    Reveal the simulated expert prediction for exactly one
    queried training sample.
    """

    return int(
        hidden_expert_train[sample_index]
    )


# ============================================================
# REVEAL COMPETENCE LABEL
# ============================================================

def reveal_competence(indices):
    """
    Query the expert only for the selected indices.

    Competence label:
        1 = expert prediction correct
        0 = expert prediction incorrect
    """

    competence_labels = []


    for index in indices:

        expert_prediction = query_expert(
            int(index)
        )

        competence_labels.append(
            int(
                expert_prediction
                ==
                train_labels[index]
            )
        )


    return np.array(
        competence_labels,
        dtype=int,
    )


# ============================================================
# INITIAL QUERY SET
# ============================================================

def create_initial_query_set(
    labels,
    total_queries,
    seed,
):
    """
    Create a small class-balanced starting set.

    The AG News labels are known according to the project
    specification, so they may be used to ensure initial
    coverage of all four input regions.
    """

    rng = np.random.default_rng(
        seed
    )


    selected = []

    per_class = (
        total_queries
        //
        len(CLASS_NAMES)
    )


    for class_index in range(
        len(CLASS_NAMES)
    ):

        class_indices = np.where(
            labels == class_index
        )[0]


        chosen = rng.choice(
            class_indices,
            size=per_class,
            replace=False,
        )


        selected.extend(
            chosen.tolist()
        )


    return np.array(
        selected,
        dtype=int,
    )


# ============================================================
# TRAIN EXPERT-COMPETENCE MODEL
# ============================================================

def train_competence_model(
    queried_indices,
    competence_labels,
):
    """
    Estimate:

        P(expert correct | x)

    using only expert labels that have actually been queried.
    """

    if len(
        np.unique(
            competence_labels
        )
    ) < 2:

        raise ValueError(
            "The current query set contains only one "
            "competence class."
        )


    model = LogisticRegression(
        max_iter=500,
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )


    model.fit(
        X_train[queried_indices],
        competence_labels,
    )


    return model


# ============================================================
# AVAILABLE INDICES
# ============================================================

def get_available_indices(
    queried_indices,
):
    """
    Return training examples that have not yet been queried.
    """

    all_indices = np.arange(
        len(train_labels)
    )


    return np.setdiff1d(
        all_indices,
        queried_indices,
        assume_unique=False,
    )


# ============================================================
# STRATEGY 1 — RANDOM
# ============================================================

def select_random(
    available_indices,
    number_to_query,
    rng,
):
    return rng.choice(
        available_indices,
        size=number_to_query,
        replace=False,
    )


# ============================================================
# STRATEGY 2 — COMPETENCE UNCERTAINTY
# ============================================================

def select_competence_uncertainty(
    model,
    available_indices,
    number_to_query,
):
    """
    Standard uncertainty sampling.

    Highest uncertainty occurs at:

        P(expert correct | x) = 0.5
    """

    expert_probability = (
        model.predict_proba(
            X_train[available_indices]
        )[:, 1]
    )


    uncertainty = (
        1.0
        -
        np.abs(
            2.0
            * expert_probability
            -
            1.0
        )
    )


    number_to_query = min(
        number_to_query,
        len(available_indices),
    )


    if number_to_query == len(
        available_indices
    ):

        selected_positions = np.arange(
            len(available_indices)
        )

    else:

        selected_positions = np.argpartition(
            -uncertainty,
            number_to_query - 1,
        )[:number_to_query]


    return available_indices[
        selected_positions
    ]


# ============================================================
# STRATEGY 3 — DEFERRAL-AWARE + DIVERSITY
# ============================================================

def select_deferral_aware_diverse(
    model,
    available_indices,
    queried_indices,
    number_to_query,
):
    """
    Proposed Task 4 acquisition strategy.

    Step 1:
        Estimate expert competence.

    Step 2:
        Find articles near the routing boundary:

            P(expert correct | x)
                approximately equals
            P(AI correct | x)

    Step 3:
        From the best boundary candidates, select a diverse
        batch to avoid querying many near-duplicate examples.
    """

    number_to_query = min(
        number_to_query,
        len(available_indices),
    )


    # --------------------------------------------------------
    # Estimate expert competence on the available pool
    # --------------------------------------------------------

    expert_probability = (
        model.predict_proba(
            X_train[available_indices]
        )[:, 1]
    )


    ai_probability = (
        ai_train_confidence[
            available_indices
        ]
    )


    # --------------------------------------------------------
    # Deferral-boundary score
    #
    # Highest score when:
    #
    # P(expert correct) ~= P(AI correct)
    # --------------------------------------------------------

    boundary_distance = np.abs(
        expert_probability
        -
        ai_probability
    )


    boundary_score = (
        1.0
        -
        boundary_distance
    )


    # --------------------------------------------------------
    # Keep only strong boundary candidates before applying
    # diversity.
    # --------------------------------------------------------

    candidate_count = min(
        len(available_indices),

        max(
            number_to_query
            * CANDIDATE_MULTIPLIER,
            number_to_query,
        ),
    )


    if candidate_count == len(
        available_indices
    ):

        candidate_positions = np.arange(
            len(available_indices)
        )

    else:

        candidate_positions = np.argpartition(
            -boundary_score,
            candidate_count - 1,
        )[:candidate_count]


    candidate_indices = (
        available_indices[
            candidate_positions
        ]
    )


    candidate_boundary_score = (
        boundary_score[
            candidate_positions
        ]
    )


    candidate_features = (
        X_train[
            candidate_indices
        ]
    )


    # --------------------------------------------------------
    # Reference points from the already queried set
    #
    # We do not need every previous query for diversity.
    # A representative subset keeps the calculation efficient.
    # --------------------------------------------------------

    if len(queried_indices) > 0:

        if len(
            queried_indices
        ) > MAX_REFERENCE_POINTS:

            reference_positions = np.linspace(
                0,
                len(queried_indices) - 1,
                MAX_REFERENCE_POINTS,
                dtype=int,
            )

            reference_indices = (
                queried_indices[
                    reference_positions
                ]
            )

        else:

            reference_indices = (
                queried_indices
            )


        similarities = cosine_similarity(
            candidate_features,
            X_train[
                reference_indices
            ],
        )


        max_similarity = np.max(
            similarities,
            axis=1,
        )

    else:

        max_similarity = np.zeros(
            len(candidate_indices)
        )


    # --------------------------------------------------------
    # Greedy diverse batch selection
    # --------------------------------------------------------

    selected_positions = []

    already_selected = np.zeros(
        len(candidate_indices),
        dtype=bool,
    )


    for _ in range(
        number_to_query
    ):

        diversity_score = (
            1.0
            -
            max_similarity
        )


        combined_score = (
            BOUNDARY_WEIGHT
            * candidate_boundary_score

            +

            DIVERSITY_WEIGHT
            * diversity_score
        )


        combined_score[
            already_selected
        ] = -np.inf


        best_position = int(
            np.argmax(
                combined_score
            )
        )


        selected_positions.append(
            best_position
        )


        already_selected[
            best_position
        ] = True


        # ----------------------------------------------------
        # Update diversity with respect to the newly selected
        # point.
        # ----------------------------------------------------

        new_similarity = cosine_similarity(
            candidate_features,
            X_train[
                candidate_indices[
                    best_position
                ]
            ],
        ).reshape(-1)


        max_similarity = np.maximum(
            max_similarity,
            new_similarity,
        )


    return candidate_indices[
        selected_positions
    ]


# ============================================================
# ADD NEW QUERIES TO A STRATEGY
# ============================================================

def add_queries(
    queried_indices,
    competence_labels,
    new_indices,
):
    """
    Reveal only the newly selected expert labels and append
    them to the observed query set.
    """

    new_labels = reveal_competence(
        new_indices
    )


    updated_indices = np.concatenate(
        [
            queried_indices,
            new_indices,
        ]
    )


    updated_labels = np.concatenate(
        [
            competence_labels,
            new_labels,
        ]
    )


    return (
        updated_indices,
        updated_labels,
    )


# ============================================================
# EVALUATE DEFERRAL MODEL
# ============================================================

def evaluate_deferral(
    competence_model,
):
    """
    Same routing rule as Task 3:

        defer if
        estimated expert competence > AI confidence
    """

    expert_probability = (
        competence_model.predict_proba(
            X_test
        )[:, 1]
    )


    defer_mask = (
        expert_probability
        >
        ai_test_confidence
    )


    team_predictions = np.where(
        defer_mask,
        expert_test_predictions,
        ai_test_predictions,
    )


    team_accuracy = accuracy_score(
        test_labels,
        team_predictions,
    )


    deferral_rate = np.mean(
        defer_mask
    )


    coverage = (
        1.0
        -
        deferral_rate
    )


    # --------------------------------------------------------
    # Expert competence prediction accuracy
    # --------------------------------------------------------

    predicted_expert_correct = (
        expert_probability
        >=
        0.5
    )


    competence_accuracy = accuracy_score(
        expert_correct_test.astype(int),
        predicted_expert_correct.astype(int),
    )


    # --------------------------------------------------------
    # Useful and harmful deferrals
    # --------------------------------------------------------

    useful_deferrals = (
        defer_mask
        &
        (~ai_correct_test)
        &
        expert_correct_test
    )


    harmful_deferrals = (
        defer_mask
        &
        ai_correct_test
        &
        (~expert_correct_test)
    )


    useful_count = int(
        np.sum(
            useful_deferrals
        )
    )


    harmful_count = int(
        np.sum(
            harmful_deferrals
        )
    )


    # --------------------------------------------------------
    # Routing accuracy on decisive cases
    # --------------------------------------------------------

    expert_better = (
        expert_correct_test
        &
        (~ai_correct_test)
    )


    ai_better = (
        ai_correct_test
        &
        (~expert_correct_test)
    )


    decisive_cases = (
        expert_better
        |
        ai_better
    )


    correct_routing = (
        (
            defer_mask
            &
            expert_better
        )

        |

        (
            (~defer_mask)
            &
            ai_better
        )
    )


    if np.sum(
        decisive_cases
    ) > 0:

        routing_accuracy = np.mean(
            correct_routing[
                decisive_cases
            ]
        )

    else:

        routing_accuracy = 0.0


    # --------------------------------------------------------
    # Class-wise deferral
    # --------------------------------------------------------

    class_deferral = []


    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_mask = (
            test_labels
            ==
            class_index
        )


        class_rate = np.mean(
            defer_mask[
                class_mask
            ]
        )


        class_deferral.append(
            {
                "class":
                    class_name,

                "deferral_rate":
                    float(
                        class_rate
                    ),

                "deferral_rate_percent":
                    round(
                        float(
                            class_rate
                        )
                        * 100,
                        2,
                    ),
            }
        )


    return {

        "team_accuracy":
            float(
                team_accuracy
            ),

        "team_accuracy_percent":
            round(
                float(
                    team_accuracy
                )
                * 100,
                2,
            ),

        "competence_accuracy":
            float(
                competence_accuracy
            ),

        "competence_accuracy_percent":
            round(
                float(
                    competence_accuracy
                )
                * 100,
                2,
            ),

        "routing_accuracy":
            float(
                routing_accuracy
            ),

        "routing_accuracy_percent":
            round(
                float(
                    routing_accuracy
                )
                * 100,
                2,
            ),

        "coverage":
            float(
                coverage
            ),

        "coverage_percent":
            round(
                float(
                    coverage
                )
                * 100,
                2,
            ),

        "deferral_rate":
            float(
                deferral_rate
            ),

        "deferral_rate_percent":
            round(
                float(
                    deferral_rate
                )
                * 100,
                2,
            ),

        "useful_deferrals":
            useful_count,

        "harmful_deferrals":
            harmful_count,

        "class_deferral":
            class_deferral,
    }


# ============================================================
# GAIN RECOVERY
# ============================================================

def calculate_gain_recovery(
    team_accuracy,
):
    """
    Fraction of the full-information Task 3 improvement that
    has been recovered using only queried expert labels.
    """

    if (
        full_l2d_accuracy is None
        or
        full_l2d_accuracy
        <=
        ai_accuracy
    ):

        return None


    full_gain = (
        full_l2d_accuracy
        -
        ai_accuracy
    )


    active_gain = (
        team_accuracy
        -
        ai_accuracy
    )


    return (
        active_gain
        /
        full_gain
    )


# ============================================================
# INITIALIZATION
# ============================================================

print(
    "\n================================"
)

print(
    "TASK 4 — REFINED ACTIVE LEARNING"
)

print(
    "================================"
)


print(
    "\nStrategies:"
)

print(
    "1. Random querying"
)

print(
    "2. Competence uncertainty"
)

print(
    "3. Deferral-aware uncertainty + diversity"
)


initial_indices = create_initial_query_set(
    train_labels,
    INITIAL_QUERIES,
    RANDOM_SEED,
)


initial_competence = reveal_competence(
    initial_indices
)


print(
    f"\nInitial expert queries: "
    f"{len(initial_indices)}"
)

print(
    "Initial expert competence observations:"
)

print(
    "Expert correct:",
    int(
        np.sum(
            initial_competence
        )
    )
)

print(
    "Expert incorrect:",
    int(
        len(initial_competence)
        -
        np.sum(
            initial_competence
        )
    )
)


# ============================================================
# CREATE INDEPENDENT QUERY STATES
# ============================================================

random_indices = (
    initial_indices.copy()
)

random_targets = (
    initial_competence.copy()
)


uncertainty_indices = (
    initial_indices.copy()
)

uncertainty_targets = (
    initial_competence.copy()
)


deferral_indices = (
    initial_indices.copy()
)

deferral_targets = (
    initial_competence.copy()
)


random_rng = np.random.default_rng(
    RANDOM_SEED + 100
)


# ============================================================
# RUN EXPERIMENTS
# ============================================================

experiment_results = []


for budget in QUERY_BUDGETS:

    print(
        "\n--------------------------------"
    )

    print(
        f"Query budget: {budget}"
    )

    print(
        "--------------------------------"
    )


    # ========================================================
    # RANDOM
    # ========================================================

    while len(
        random_indices
    ) < budget:

        available = get_available_indices(
            random_indices
        )


        remaining = (
            budget
            -
            len(random_indices)
        )


        current_batch = min(
            BATCH_SIZE,
            remaining,
        )


        selected = select_random(
            available,
            current_batch,
            random_rng,
        )


        (
            random_indices,
            random_targets,
        ) = add_queries(
            random_indices,
            random_targets,
            selected,
        )


    random_model = train_competence_model(
        random_indices,
        random_targets,
    )


    random_metrics = evaluate_deferral(
        random_model
    )


    # ========================================================
    # STANDARD COMPETENCE UNCERTAINTY
    # ========================================================

    while len(
        uncertainty_indices
    ) < budget:

        uncertainty_model = (
            train_competence_model(
                uncertainty_indices,
                uncertainty_targets,
            )
        )


        available = get_available_indices(
            uncertainty_indices
        )


        remaining = (
            budget
            -
            len(uncertainty_indices)
        )


        current_batch = min(
            BATCH_SIZE,
            remaining,
        )


        selected = (
            select_competence_uncertainty(
                uncertainty_model,
                available,
                current_batch,
            )
        )


        (
            uncertainty_indices,
            uncertainty_targets,
        ) = add_queries(
            uncertainty_indices,
            uncertainty_targets,
            selected,
        )


    uncertainty_model = train_competence_model(
        uncertainty_indices,
        uncertainty_targets,
    )


    uncertainty_metrics = evaluate_deferral(
        uncertainty_model
    )


    # ========================================================
    # DEFERRAL-AWARE + DIVERSITY
    # ========================================================

    while len(
        deferral_indices
    ) < budget:

        deferral_model = (
            train_competence_model(
                deferral_indices,
                deferral_targets,
            )
        )


        available = get_available_indices(
            deferral_indices
        )


        remaining = (
            budget
            -
            len(deferral_indices)
        )


        current_batch = min(
            BATCH_SIZE,
            remaining,
        )


        selected = (
            select_deferral_aware_diverse(
                deferral_model,
                available,
                deferral_indices,
                current_batch,
            )
        )


        (
            deferral_indices,
            deferral_targets,
        ) = add_queries(
            deferral_indices,
            deferral_targets,
            selected,
        )


    deferral_model = train_competence_model(
        deferral_indices,
        deferral_targets,
    )


    deferral_metrics = evaluate_deferral(
        deferral_model
    )


    # ========================================================
    # GAIN RECOVERY
    # ========================================================

    random_recovery = (
        calculate_gain_recovery(
            random_metrics[
                "team_accuracy"
            ]
        )
    )


    uncertainty_recovery = (
        calculate_gain_recovery(
            uncertainty_metrics[
                "team_accuracy"
            ]
        )
    )


    deferral_recovery = (
        calculate_gain_recovery(
            deferral_metrics[
                "team_accuracy"
            ]
        )
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\nRandom querying:"
    )

    print(
        f"Team accuracy: "
        f"{random_metrics['team_accuracy_percent']:.2f}%"
    )

    print(
        f"Competence accuracy: "
        f"{random_metrics['competence_accuracy_percent']:.2f}%"
    )

    print(
        f"Routing accuracy: "
        f"{random_metrics['routing_accuracy_percent']:.2f}%"
    )

    print(
        f"Deferral rate: "
        f"{random_metrics['deferral_rate_percent']:.2f}%"
    )


    print(
        "\nCompetence uncertainty:"
    )

    print(
        f"Team accuracy: "
        f"{uncertainty_metrics['team_accuracy_percent']:.2f}%"
    )

    print(
        f"Competence accuracy: "
        f"{uncertainty_metrics['competence_accuracy_percent']:.2f}%"
    )

    print(
        f"Routing accuracy: "
        f"{uncertainty_metrics['routing_accuracy_percent']:.2f}%"
    )

    print(
        f"Deferral rate: "
        f"{uncertainty_metrics['deferral_rate_percent']:.2f}%"
    )


    print(
        "\nDeferral-aware + diversity:"
    )

    print(
        f"Team accuracy: "
        f"{deferral_metrics['team_accuracy_percent']:.2f}%"
    )

    print(
        f"Competence accuracy: "
        f"{deferral_metrics['competence_accuracy_percent']:.2f}%"
    )

    print(
        f"Routing accuracy: "
        f"{deferral_metrics['routing_accuracy_percent']:.2f}%"
    )

    print(
        f"Deferral rate: "
        f"{deferral_metrics['deferral_rate_percent']:.2f}%"
    )


    if (
        random_recovery is not None
        and
        uncertainty_recovery is not None
        and
        deferral_recovery is not None
    ):

        print(
            "\nRecovered full-L2D improvement:"
        )

        print(
            f"Random: "
            f"{random_recovery * 100:.2f}%"
        )

        print(
            f"Competence uncertainty: "
            f"{uncertainty_recovery * 100:.2f}%"
        )

        print(
            f"Deferral-aware + diversity: "
            f"{deferral_recovery * 100:.2f}%"
        )


    # ========================================================
    # STORE RESULT
    # ========================================================

    experiment_results.append(
        {
            "query_budget":
                int(
                    budget
                ),

            "query_fraction_percent":
                round(
                    (
                        budget
                        /
                        len(train_labels)
                    )
                    * 100,
                    4,
                ),

            "random":
                random_metrics,

            "competence_uncertainty":
                uncertainty_metrics,

            "deferral_aware_diversity":
                deferral_metrics,

            "random_gain_recovery":
                (
                    float(
                        random_recovery
                    )
                    if random_recovery is not None
                    else None
                ),

            "uncertainty_gain_recovery":
                (
                    float(
                        uncertainty_recovery
                    )
                    if uncertainty_recovery is not None
                    else None
                ),

            "deferral_gain_recovery":
                (
                    float(
                        deferral_recovery
                    )
                    if deferral_recovery is not None
                    else None
                ),
        }
    )


# ============================================================
# SAVE FINAL MODELS
# ============================================================

joblib.dump(
    random_model,
    MODEL_DIR
    / "active_random_competence_model.joblib",
)

joblib.dump(
    uncertainty_model,
    MODEL_DIR
    / "active_uncertainty_competence_model.joblib",
)

joblib.dump(
    deferral_model,
    MODEL_DIR
    / "active_deferral_competence_model.joblib",
)


np.save(
    MODEL_DIR
    / "random_queried_indices.npy",
    random_indices,
)

np.save(
    MODEL_DIR
    / "uncertainty_queried_indices.npy",
    uncertainty_indices,
)

np.save(
    MODEL_DIR
    / "deferral_aware_queried_indices.npy",
    deferral_indices,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

final_result = (
    experiment_results[-1]
)


print(
    "\n================================"
)

print(
    "TASK 4 — FINAL SUMMARY"
)

print(
    "================================"
)


print(
    f"\nAI baseline accuracy: "
    f"{ai_accuracy * 100:.2f}%"
)


if full_l2d_accuracy is not None:

    print(
        f"Full-information L2D accuracy: "
        f"{full_l2d_accuracy * 100:.2f}%"
    )


print(
    f"\nMaximum expert queries: "
    f"{QUERY_BUDGETS[-1]}"
)


print(
    f"Fraction of training set queried: "
    f"{QUERY_BUDGETS[-1] / len(train_labels) * 100:.2f}%"
)


print(
    "\nFinal team accuracies:"
)


print(
    "Random:",
    f"{final_result['random']['team_accuracy_percent']:.2f}%"
)


print(
    "Competence uncertainty:",
    f"{final_result['competence_uncertainty']['team_accuracy_percent']:.2f}%"
)


print(
    "Deferral-aware + diversity:",
    f"{final_result['deferral_aware_diversity']['team_accuracy_percent']:.2f}%"
)


print(
    "\nFinal routing accuracies:"
)


print(
    "Random:",
    f"{final_result['random']['routing_accuracy_percent']:.2f}%"
)


print(
    "Competence uncertainty:",
    f"{final_result['competence_uncertainty']['routing_accuracy_percent']:.2f}%"
)


print(
    "Deferral-aware + diversity:",
    f"{final_result['deferral_aware_diversity']['routing_accuracy_percent']:.2f}%"
)


# ============================================================
# SAVE JSON
# ============================================================

output = {

    "project_task":
        (
            "Active Learning for expert competence discovery"
        ),

    "proposed_strategy":
        (
            "Deferral-aware uncertainty combined with "
            "diversity sampling"
        ),

    "strategy_description":
        (
            "Queries articles close to the learned AI-expert "
            "routing boundary and applies cosine-similarity "
            "diversity to reduce redundant queries."
        ),

    "query_budgets":
        QUERY_BUDGETS,

    "initial_queries":
        INITIAL_QUERIES,

    "training_samples":
        int(
            len(train_labels)
        ),

    "ai_baseline_accuracy":
        float(
            ai_accuracy
        ),

    "ai_baseline_accuracy_percent":
        round(
            float(
                ai_accuracy
            )
            * 100,
            2,
        ),

    "full_l2d_accuracy":
        (
            float(
                full_l2d_accuracy
            )
            if full_l2d_accuracy is not None
            else None
        ),

    "full_l2d_accuracy_percent":
        (
            round(
                float(
                    full_l2d_accuracy
                )
                * 100,
                2,
            )
            if full_l2d_accuracy is not None
            else None
        ),

    "boundary_weight":
        BOUNDARY_WEIGHT,

    "diversity_weight":
        DIVERSITY_WEIGHT,

    "experiments":
        experiment_results,
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        indent=4,
    )


print(
    "\nSaved:"
)

print(
    RESULTS_PATH
)


print(
    "\nTask 4 refined experiment completed successfully."
)
# ============================================================
# TASK 5 - EXPORT HUMAN QUERY POOL
# ============================================================

HUMAN_QUERY_POOL_PATH = (
    MODEL_DIR
    / "human_query_pool.json"
)


def save_human_query_pool(
    selected_indices,
    train_texts,
    train_labels,
    max_queries=100,
):
    """
    Save articles selected by the active-learning strategy
    so that a real human can label them through the Django
    Task 5 interface.
    """

    query_pool = []


    for index in selected_indices[:max_queries]:

        index = int(index)

        query_pool.append(
            {
                "index":
                    index,

                "text":
                    str(
                        train_texts[index]
                    ),

                "true_label":
                    int(
                        train_labels[index]
                    ),
            }
        )


    with open(
        HUMAN_QUERY_POOL_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            query_pool,
            file,
            indent=2,
            ensure_ascii=False,
        )


    print(
        "\nHuman expert query pool saved:"
    )

    print(
        HUMAN_QUERY_POOL_PATH
    )


# ============================================================
# SELECT TASK 5 HUMAN QUERIES
# ============================================================

# The first INITIAL_QUERIES samples in uncertainty_indices
# belong to the class-balanced warm-up set.
#
# Therefore we skip those samples and use the next 100
# examples that were actually selected by competence
# uncertainty sampling.

task5_query_indices = (
    uncertainty_indices[
        INITIAL_QUERIES:
        INITIAL_QUERIES + 100
    ]
)


save_human_query_pool(
    task5_query_indices,
    train_texts,
    train_labels,
    max_queries=100,
)