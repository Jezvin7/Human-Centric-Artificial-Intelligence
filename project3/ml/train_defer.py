from pathlib import Path
import json

import joblib
import numpy as np

from datasets import load_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_models"

MODEL_DIR.mkdir(
    exist_ok=True
)


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

COMPETENCE_MODEL_PATH = (
    MODEL_DIR
    / "expert_competence_model.joblib"
)

DEFERRAL_MASK_PATH = (
    MODEL_DIR
    / "l2d_test_deferrals.npy"
)

COMPETENCE_SCORES_PATH = (
    MODEL_DIR
    / "expert_competence_scores.npy"
)

RESULTS_PATH = (
    MODEL_DIR
    / "l2d_results.json"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
]


# ============================================================
# 1. LOAD DATASET
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
# 2. LOAD TASK 1 MODEL
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


# Transform using the SAME TF-IDF representation as Task 1

X_train = vectorizer.transform(
    train_texts
)

X_test = vectorizer.transform(
    test_texts
)


# ============================================================
# 3. LOAD SIMULATED EXPERT LABELS FROM TASK 2
# ============================================================

print(
    "Loading simulated expert predictions..."
)


expert_train_predictions = np.load(
    EXPERT_TRAIN_PATH
)

expert_test_predictions = np.load(
    EXPERT_TEST_PATH
)


# ============================================================
# 4. BASELINE AI PREDICTIONS
# ============================================================

print(
    "Computing AI predictions and confidence..."
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


# Maximum class probability is used as the AI confidence.

ai_train_confidence = np.max(
    ai_train_probabilities,
    axis=1,
)

ai_test_confidence = np.max(
    ai_test_probabilities,
    axis=1,
)


# ============================================================
# 5. CREATE EXPERT-COMPETENCE LABELS
# ============================================================

# In Task 3 both ground-truth labels and expert labels are
# available during training.
#
# Target = 1:
# expert prediction was correct
#
# Target = 0:
# expert prediction was incorrect

expert_correct_train = (
    expert_train_predictions
    ==
    train_labels
).astype(int)


expert_correct_test = (
    expert_test_predictions
    ==
    test_labels
)


print(
    "\nExpert competence training labels:"
)

print(
    "Correct:",
    int(np.sum(expert_correct_train))
)

print(
    "Incorrect:",
    int(
        len(expert_correct_train)
        -
        np.sum(expert_correct_train)
    )
)


# ============================================================
# 6. TRAIN EXPERT COMPETENCE MODEL
# ============================================================

print(
    "\nTraining expert competence model..."
)


expert_competence_model = LogisticRegression(
    max_iter=500,
    solver="saga",
    class_weight="balanced",
    random_state=42,
)


expert_competence_model.fit(
    X_train,
    expert_correct_train,
)


joblib.dump(
    expert_competence_model,
    COMPETENCE_MODEL_PATH,
)


# ============================================================
# 7. ESTIMATE EXPERT COMPETENCE
# ============================================================

expert_competence_test = (
    expert_competence_model.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 8. LEARNING-TO-DEFER DECISION
# ============================================================

# Optional margin representing additional evidence required
# before querying the expert.
#
# margin = 0.0 means:
#
# defer whenever estimated expert correctness is greater
# than AI confidence.

DEFERRAL_MARGIN = 0.0


defer_mask = (
    expert_competence_test
    >
    (
        ai_test_confidence
        +
        DEFERRAL_MARGIN
    )
)


# ============================================================
# 9. HUMAN-AI TEAM PREDICTIONS
# ============================================================

team_predictions = np.where(
    defer_mask,
    expert_test_predictions,
    ai_test_predictions,
)


# ============================================================
# 10. BASIC ACCURACIES
# ============================================================

ai_accuracy = accuracy_score(
    test_labels,
    ai_test_predictions,
)

expert_accuracy = accuracy_score(
    test_labels,
    expert_test_predictions,
)

team_accuracy = accuracy_score(
    test_labels,
    team_predictions,
)


# ============================================================
# 11. COVERAGE AND DEFERRAL RATE
# ============================================================

# Lecture 5:
#
# Coverage = fraction of examples NOT deferred.

coverage = np.mean(
    ~defer_mask
)

deferral_rate = np.mean(
    defer_mask
)


# ============================================================
# 12. PERFORMANCE ON ROUTED SUBSETS
# ============================================================

num_deferred = int(
    np.sum(defer_mask)
)

num_ai_handled = int(
    np.sum(~defer_mask)
)


if num_ai_handled > 0:

    ai_covered_accuracy = accuracy_score(
        test_labels[~defer_mask],
        ai_test_predictions[~defer_mask],
    )

else:

    ai_covered_accuracy = None


if num_deferred > 0:

    expert_deferred_accuracy = accuracy_score(
        test_labels[defer_mask],
        expert_test_predictions[defer_mask],
    )

else:

    expert_deferred_accuracy = None


# ============================================================
# 13. DEFERRAL QUALITY
# ============================================================

ai_correct_test = (
    ai_test_predictions
    ==
    test_labels
)


# ------------------------------------------------------------
# Beneficial deferral
#
# AI would be wrong
# Expert is correct
# System defers
# ------------------------------------------------------------

useful_deferrals = (
    defer_mask
    &
    (~ai_correct_test)
    &
    expert_correct_test
)


# ------------------------------------------------------------
# Harmful deferral
#
# AI would be correct
# Expert is wrong
# System nevertheless defers
# ------------------------------------------------------------

harmful_deferrals = (
    defer_mask
    &
    ai_correct_test
    &
    (~expert_correct_test)
)


# ------------------------------------------------------------
# Both systems correct
# ------------------------------------------------------------

both_correct = (
    ai_correct_test
    &
    expert_correct_test
)


# ------------------------------------------------------------
# Both systems wrong
# ------------------------------------------------------------

both_wrong = (
    (~ai_correct_test)
    &
    (~expert_correct_test)
)


useful_deferral_count = int(
    np.sum(useful_deferrals)
)

harmful_deferral_count = int(
    np.sum(harmful_deferrals)
)


if num_deferred > 0:

    useful_deferral_rate = (
        useful_deferral_count
        /
        num_deferred
    )

    harmful_deferral_rate = (
        harmful_deferral_count
        /
        num_deferred
    )

else:

    useful_deferral_rate = 0.0
    harmful_deferral_rate = 0.0


# ============================================================
# 14. DEFERRAL DECISION QUALITY
# ============================================================

# We can directly assess the routing decision on examples
# where AI and expert differ in correctness.
#
# Case A:
# Expert correct, AI wrong
# Correct decision = DEFER
#
# Case B:
# AI correct, expert wrong
# Correct decision = KEEP WITH AI


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


num_decisive = int(
    np.sum(decisive_cases)
)


if num_decisive > 0:

    routing_accuracy = (
        np.sum(
            correct_routing
            &
            decisive_cases
        )
        /
        num_decisive
    )

else:

    routing_accuracy = 0.0


# ============================================================
# 15. ORACLE UPPER BOUND
# ============================================================

# An oracle knows whether AI or expert is correct.
#
# It succeeds whenever at least one of them is correct.

oracle_correct = (
    ai_correct_test
    |
    expert_correct_test
)


oracle_accuracy = np.mean(
    oracle_correct
)


# ============================================================
# 16. CLASS-WISE DEFERRAL STATISTICS
# ============================================================

class_results = []


for class_index, class_name in enumerate(CLASS_NAMES):

    class_mask = (
        test_labels
        ==
        class_index
    )

    class_total = int(
        np.sum(class_mask)
    )


    class_deferred = int(
        np.sum(
            defer_mask
            &
            class_mask
        )
    )


    class_deferral_rate = (
        class_deferred
        /
        class_total
    )


    class_team_accuracy = accuracy_score(
        test_labels[class_mask],
        team_predictions[class_mask],
    )


    class_results.append(
        {
            "class": class_name,

            "samples":
                class_total,

            "deferred":
                class_deferred,

            "deferral_rate":
                float(class_deferral_rate),

            "team_accuracy":
                float(class_team_accuracy),
        }
    )


# ============================================================
# 17. PRINT RESULTS
# ============================================================

print(
    "\n================================"
)

print(
    "TASK 3 — LEARNING TO DEFER"
)

print(
    "================================"
)


print(
    f"\nAI Accuracy: "
    f"{ai_accuracy * 100:.2f}%"
)

print(
    f"Expert Accuracy: "
    f"{expert_accuracy * 100:.2f}%"
)

print(
    f"Human-AI Team Accuracy: "
    f"{team_accuracy * 100:.2f}%"
)


print(
    f"\nCoverage: "
    f"{coverage * 100:.2f}%"
)

print(
    f"Deferral Rate: "
    f"{deferral_rate * 100:.2f}%"
)


print(
    f"\nAI-handled samples: "
    f"{num_ai_handled}"
)

print(
    f"Deferred samples: "
    f"{num_deferred}"
)


if ai_covered_accuracy is not None:

    print(
        f"AI accuracy on retained samples: "
        f"{ai_covered_accuracy * 100:.2f}%"
    )


if expert_deferred_accuracy is not None:

    print(
        f"Expert accuracy on deferred samples: "
        f"{expert_deferred_accuracy * 100:.2f}%"
    )


print(
    "\nDeferral quality:"
)

print(
    f"Useful deferrals: "
    f"{useful_deferral_count}"
)

print(
    f"Harmful deferrals: "
    f"{harmful_deferral_count}"
)

print(
    f"Useful deferral rate: "
    f"{useful_deferral_rate * 100:.2f}%"
)

print(
    f"Harmful deferral rate: "
    f"{harmful_deferral_rate * 100:.2f}%"
)


print(
    f"\nRouting accuracy on decisive cases: "
    f"{routing_accuracy * 100:.2f}%"
)


print(
    f"Oracle upper-bound accuracy: "
    f"{oracle_accuracy * 100:.2f}%"
)


print(
    "\nClass-wise results:"
)


for result in class_results:

    print(
        f"{result['class']}: "
        f"deferred "
        f"{result['deferred']}/"
        f"{result['samples']} "
        f"({result['deferral_rate'] * 100:.2f}%), "
        f"team accuracy "
        f"{result['team_accuracy'] * 100:.2f}%"
    )


# ============================================================
# 18. SAVE ARRAYS
# ============================================================

np.save(
    DEFERRAL_MASK_PATH,
    defer_mask,
)

np.save(
    COMPETENCE_SCORES_PATH,
    expert_competence_test,
)


# ============================================================
# 19. SAVE RESULTS
# ============================================================

results = {

    "strategy":
        (
            "Learned expert competence compared "
            "against AI confidence"
        ),

    "deferral_margin":
        DEFERRAL_MARGIN,

    "ai_accuracy":
        float(ai_accuracy),

    "ai_accuracy_percent":
        round(
            float(ai_accuracy) * 100,
            2,
        ),

    "expert_accuracy":
        float(expert_accuracy),

    "expert_accuracy_percent":
        round(
            float(expert_accuracy) * 100,
            2,
        ),

    "team_accuracy":
        float(team_accuracy),

    "team_accuracy_percent":
        round(
            float(team_accuracy) * 100,
            2,
        ),

    "coverage":
        float(coverage),

    "coverage_percent":
        round(
            float(coverage) * 100,
            2,
        ),

    "deferral_rate":
        float(deferral_rate),

    "deferral_rate_percent":
        round(
            float(deferral_rate) * 100,
            2,
        ),

    "num_test_samples":
        int(len(test_labels)),

    "num_ai_handled":
        num_ai_handled,

    "num_deferred":
        num_deferred,

    "ai_covered_accuracy":
        (
            float(ai_covered_accuracy)
            if ai_covered_accuracy is not None
            else None
        ),

    "expert_deferred_accuracy":
        (
            float(expert_deferred_accuracy)
            if expert_deferred_accuracy is not None
            else None
        ),

    "useful_deferrals":
        useful_deferral_count,

    "harmful_deferrals":
        harmful_deferral_count,

    "useful_deferral_rate":
        float(useful_deferral_rate),

    "harmful_deferral_rate":
        float(harmful_deferral_rate),

    "routing_accuracy":
        float(routing_accuracy),

    "routing_accuracy_percent":
        round(
            float(routing_accuracy) * 100,
            2,
        ),

    "oracle_accuracy":
        float(oracle_accuracy),

    "oracle_accuracy_percent":
        round(
            float(oracle_accuracy) * 100,
            2,
        ),

    "class_results":
        class_results,
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=4,
    )


# ============================================================
# 20. COMPLETE
# ============================================================

print(
    "\nSaved:"
)

print(
    COMPETENCE_MODEL_PATH
)

print(
    DEFERRAL_MASK_PATH
)

print(
    COMPETENCE_SCORES_PATH
)

print(
    RESULTS_PATH
)


print(
    "\nTask 3 Learning-to-Defer completed successfully."
)