import numpy as np


CLASS_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
]


# ============================================================
# SIMULATED EXPERT PROFILE
# ============================================================
#
# Expert type:
# Business & Technology Specialist
#
# Rows:
# true class
#
# Columns:
# expert prediction
#
# Class order:
# 0 = World
# 1 = Sports
# 2 = Business
# 3 = Sci/Tech
#
# The expert is deliberately strong in Business and Sci/Tech
# and weaker in World and Sports.
# ============================================================

EXPERT_CONFUSION_PROFILE = np.array(
    [
        # Predicted:
        # World  Sports  Business  Sci/Tech

        [0.70,   0.05,   0.15,     0.10],   # Actual World

        [0.15,   0.65,   0.10,     0.10],   # Actual Sports

        [0.01,   0.01,   0.97,     0.01],   # Actual Business

        [0.01,   0.01,   0.02,     0.96],   # Actual Sci/Tech
    ]
)


# ============================================================
# GENERATE EXPERT PREDICTIONS
# ============================================================

def generate_expert_predictions(
    true_labels,
    seed=42,
):

    rng = np.random.default_rng(seed)
    expert_predictions = []
    for true_label in true_labels:
        probabilities = (
            EXPERT_CONFUSION_PROFILE[true_label]
        )
        prediction = rng.choice(
            len(CLASS_NAMES),
            p=probabilities,
        )
        expert_predictions.append(prediction)

    return np.array(expert_predictions)


# ============================================================
# QUERY EXPERT
# ============================================================

def query_expert(
    sample_index,
    stored_expert_predictions,
):
    

    return int(
        stored_expert_predictions[sample_index]
    )