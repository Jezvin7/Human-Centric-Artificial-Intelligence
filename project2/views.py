import os
import time
from django.conf import settings
from django.shortcuts import render

from .ml_utils import (
    train_candidate_models,
    select_best_model,
    save_tree_plot,
    get_logistic_coefficients,
)


def index(request):
    model_type = request.GET.get("model_type", "tree")

    if model_type not in ["tree", "logistic"]:
        model_type = "tree"

    try:
        lambda_value = float(request.GET.get("lambda_value", 0.0))
    except ValueError:
        lambda_value = 0.0

    candidates = train_candidate_models(model_type)
    selected_model = select_best_model(candidates, lambda_value)

    tree_image = None
    coefficient_rows = None
    coefficient_classes = None
    if model_type == "tree":
        tree_filename = save_tree_plot(
            selected_model["model"],
            os.path.join(settings.MEDIA_ROOT, "project2")
        )

        tree_image = f"{settings.MEDIA_URL}project2/{tree_filename}?v={int(time.time())}"

    elif model_type == "logistic":
        coefficient_rows, coefficient_classes = get_logistic_coefficients(
            selected_model["model"]
        )

    candidate_summary = []

    for candidate in candidates:
        candidate_summary.append({
            "parameter_value": candidate["parameter_value"],
            "accuracy": round(candidate["accuracy"], 3),
            "complexity": candidate["complexity"],
            "selection_score": round(
                candidate["accuracy"] - lambda_value * candidate["complexity"],
                3
            ),
        })

    context = {
        "model_type": model_type,
        "lambda_value": lambda_value,

        "accuracy": round(selected_model["accuracy"], 3),
        "complexity": selected_model["complexity"],
        "selection_score": round(selected_model["selection_score"], 3),

        "parameter_name": selected_model["parameter_name"],
        "parameter_value": selected_model["parameter_value"],

        "tree_image": tree_image,
        "coefficient_rows": coefficient_rows,
        "coefficient_classes": coefficient_classes,

        "candidate_summary": candidate_summary,
    }

    return render(request, "project2/index.html", context)