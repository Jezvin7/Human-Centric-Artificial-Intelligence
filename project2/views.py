import os
import time
from django.conf import settings
from django.shortcuts import render

from .ml_utils import (
    train_candidate_models,
    select_best_model,
    save_tree_plot,
    get_logistic_coefficients,

    get_train_test_data,
    generate_counterfactuals,
    compute_pdp,
    compute_ale,
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

        # ==========================================
    # ADDED FOR TASK 4 & 5: Backend Logic
    # ==========================================
    
    # Load dataset to use for tasks 4 and 5
    X_train, X_test, y_train, y_test, numerical_features, categorical_features = get_train_test_data()
    model = selected_model["model"]
    model_classes = list(model.classes_)

    # --- Task 4: Counterfactuals ---
    cf_index = request.GET.get("cf_index")
    target_class = request.GET.get("target_class")
    
    original_instance = None
    counterfactuals = []
    cf_same_as_prediction = False
    if cf_index is not None and target_class:
        try:
            cf_index = int(cf_index)
            # Using X_test to generate explanations for data the model hasn't trained on
            if 0 <= cf_index < len(X_test):
                x_instance = X_test.iloc[[cf_index]]
                original_instance = x_instance.to_dict('records')[0]
                
                # Get the current prediction
                original_pred = model.predict(x_instance)[0]
                original_instance['current_prediction'] = original_pred

                # Only generate if the target class is different from the current prediction
                if original_pred != target_class:
                    counterfactuals = generate_counterfactuals(
                        model=model,
                        x_instance=x_instance,
                        target_class=target_class,
                        X_train=X_train,
                        numerical_features=numerical_features,
                        categorical_features=categorical_features,
                        k=5,
                        N=1500
                    )
                else:
                    cf_same_as_prediction = True
        except ValueError:
            pass

    # --- Task 5: Feature Effect Plots (PDP & ALE) ---
    plot_feature = request.GET.get("plot_feature", numerical_features[0])
    pdp_data = {}
    ale_data = {}

    if plot_feature in numerical_features:
        for idx, cls_name in enumerate(model_classes):
            # Compute PDP
            pdp_x, pdp_y = compute_pdp(model, X_train, plot_feature, idx)
            pdp_data[cls_name] = {"x": pdp_x, "y": pdp_y}
            
            # Compute ALE
            ale_x, ale_y = compute_ale(model, X_train, plot_feature, idx)
            ale_data[cls_name] = {"x": ale_x, "y": ale_y}

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

        # ==========================================
        # ADDED FOR TASK 4 & 5: Context Variables
        # ==========================================
        "classes": model_classes,
        "numerical_features": numerical_features,
        "cf_index": cf_index,
        "target_class": target_class,
        "original_instance": original_instance,
        "counterfactuals": counterfactuals,
        "cf_same_as_prediction": cf_same_as_prediction,
        "plot_feature": plot_feature,
        "pdp_data": pdp_data,
        "ale_data": ale_data,
    }

    return render(request, "project2/index.html", context)