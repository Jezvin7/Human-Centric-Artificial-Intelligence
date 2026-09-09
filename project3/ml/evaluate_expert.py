from pathlib import Path
import json

import numpy as np

from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from simulated_expert import (
    CLASS_NAMES,
    EXPERT_CONFUSION_PROFILE,
    generate_expert_predictions,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_models"

MODEL_DIR.mkdir(
    exist_ok=True
)


TRAIN_EXPERT_PATH = (
    MODEL_DIR
    / "expert_train_predictions.npy"
)

TEST_EXPERT_PATH = (
    MODEL_DIR
    / "expert_test_predictions.npy"
)

RESULTS_PATH = (
    MODEL_DIR
    / "expert_results.json"
)


# ============================================================
# 1. LOAD AG NEWS
# ============================================================

print(
    "Loading AG News dataset..."
)

dataset = load_dataset(
    "fancyzhx/ag_news"
)


train_labels = np.array(
    dataset["train"]["label"]
)

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
# 2. GENERATE SIMULATED EXPERT PREDICTIONS
# ============================================================

print(
    "\nGenerating simulated expert predictions..."
)


# Different seeds are deliberately used for the two dataset
# splits while keeping the overall simulation reproducible.

expert_train_predictions = (
    generate_expert_predictions(
        train_labels,
        seed=42,
    )
)

expert_test_predictions = (
    generate_expert_predictions(
        test_labels,
        seed=43,
    )
)


# ============================================================
# 3. TEST ACCURACY
# ============================================================

expert_accuracy = accuracy_score(
    test_labels,
    expert_test_predictions,
)


# ============================================================
# 4. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    expert_test_predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
)


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    expert_test_predictions,
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print(
    "\n================================"
)

print(
    "TASK 2 — SIMULATED EXPERT"
)

print(
    "================================"
)


print(
    "\nExpert type:"
)

print(
    "Business & Technology Specialist"
)


print(
    "\nDesigned competence profile:"
)


for index, class_name in enumerate(CLASS_NAMES):

    intended_accuracy = (
        EXPERT_CONFUSION_PROFILE[index][index]
    )

    print(
        f"{class_name}: "
        f"{intended_accuracy * 100:.1f}%"
    )


print(
    f"\nTest Accuracy: "
    f"{expert_accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{expert_accuracy * 100:.2f}%"
)


print(
    "\nClassification Report:"
)


print(
    classification_report(
        test_labels,
        expert_test_predictions,
        target_names=CLASS_NAMES,
    )
)


print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# 7. SAVE HIDDEN EXPERT PREDICTIONS
# ============================================================

np.save(
    TRAIN_EXPERT_PATH,
    expert_train_predictions,
)

np.save(
    TEST_EXPERT_PATH,
    expert_test_predictions,
)


# ============================================================
# 8. SAVE RESULTS FOR DJANGO / REPORT
# ============================================================

competence_profile = {}

for index, class_name in enumerate(CLASS_NAMES):

    competence_profile[class_name] = float(
        EXPERT_CONFUSION_PROFILE[index][index]
    )


results = {

    "expert_name":
        "Business & Technology Specialist",

    "description":
        (
            "A simulated domain expert specialized in "
            "Business and Sci/Tech news. The expert is "
            "deliberately less reliable on World and "
            "Sports articles."
        ),

    "accuracy":
        float(expert_accuracy),

    "accuracy_percent":
        round(
            float(expert_accuracy) * 100,
            2,
        ),

    "class_names":
        CLASS_NAMES,

    "competence_profile":
        competence_profile,

    "classification_report":
        report,

    "confusion_matrix":
        cm.tolist(),

    "train_samples":
        int(len(train_labels)),

    "test_samples":
        int(len(test_labels)),
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
# 9. COMPLETE
# ============================================================

print(
    "\nSaved:"
)

print(
    TRAIN_EXPERT_PATH
)

print(
    TEST_EXPERT_PATH
)

print(
    RESULTS_PATH
)


print(
    "\nTask 2 expert simulation completed successfully."
)