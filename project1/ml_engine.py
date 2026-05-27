from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_squared_error,
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def parse_values(raw_value, default_values, field_name):
    if raw_value is None or str(raw_value).strip() == "":
        return default_values

    try:
        values = [
            int(value.strip())
            for value in str(raw_value).split(",")
            if value.strip()
        ]
    except Exception:
        raise ValueError(
            f"{field_name} must contain comma-separated positive integers."
        )

    if not values:
        raise ValueError(f"{field_name} cannot be empty.")

    if any(value <= 0 for value in values):
        raise ValueError(
            f"{field_name} must contain only positive integers."
        )

    return values


def validate_test_size(test_size):
    try:
        test_size = float(test_size)
    except Exception:
        raise ValueError("Test size must be a number.")

    if test_size < 0.1 or test_size > 0.4:
        raise ValueError("Test size must be between 0.1 and 0.4.")

    return test_size


def prepare_data(df, target_col, task_type, test_size):
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError("Target column not found in dataset.")
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature column and one target column.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if X.empty:
        raise ValueError("No feature columns found.")
    if y.nunique() < 2 and task_type == "classification":
        raise ValueError("Classification requires at least two target classes.")
    for col in X.select_dtypes(include="object"):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if task_type == "classification":
        y = LabelEncoder().fit_transform(y.astype(str))
    test_size = validate_test_size(test_size)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y if task_type == "classification" else None,
    )


def evaluate_classification(y_test, preds):
    return {
        "accuracy": round(
            accuracy_score(y_test, preds),
            4,
        ),
        "precision": round(
            precision_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0,
            ),
            4,
        ),
        "recall": round(
            recall_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0,
            ),
            4,
        ),
        "f1_score": round(
            f1_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0,
            ),
            4,
        ),
    }


def evaluate_regression(y_test, preds):
    rmse = mean_squared_error(y_test, preds) ** 0.5

    return {
        "r2_score": round(
            r2_score(y_test, preds),
            4,
        ),
        "rmse": round(
            rmse,
            4,
        ),
    }


def filter_knn_values(knn_values, max_neighbors):
    valid_values = [
        value
        for value in knn_values
        if value <= max_neighbors
    ]

    if not valid_values:
        raise ValueError(
            f"KNN K values must be less than or equal to the number of training samples ({max_neighbors})."
        )

    return valid_values


def build_models(
    task_type,
    selected_models,
    hyperparameters,
    max_neighbors,
):
    models = []

    knn_values = []
    rf_values = []
    dt_values = []

    if "KNN" in selected_models:
        knn_values = parse_values(
            hyperparameters.get("knn_values"),
            [3, 5, 7],
            "KNN K values",
        )

        knn_values = filter_knn_values(
            knn_values,
            max_neighbors,
        )

    if "Random Forest" in selected_models:
        rf_values = parse_values(
            hyperparameters.get("rf_values"),
            [50, 100, 200],
            "Random Forest trees",
        )

    if "Decision Tree" in selected_models:
        dt_values = parse_values(
            hyperparameters.get("dt_depths"),
            [2, 4, 6],
            "Decision Tree depths",
        )

    if task_type == "classification":
        if "Logistic Regression" in selected_models:
            models.append(
                (
                    "Logistic Regression",
                    LogisticRegression(max_iter=2000),
                )
            )

        if "Random Forest" in selected_models:
            for n_trees in rf_values:
                models.append(
                    (
                        f"Random Forest (trees={n_trees})",
                        RandomForestClassifier(
                            n_estimators=n_trees,
                            random_state=42,
                        ),
                    )
                )

        if "KNN" in selected_models:
            for k_value in knn_values:
                models.append(
                    (
                        f"KNN (k={k_value})",
                        KNeighborsClassifier(
                            n_neighbors=k_value,
                        ),
                    )
                )

        if "SVM" in selected_models:
            models.append(
                (
                    "SVM",
                    SVC(),
                )
            )

        if "Decision Tree" in selected_models:
            for depth in dt_values:
                models.append(
                    (
                        f"Decision Tree (depth={depth})",
                        DecisionTreeClassifier(
                            max_depth=depth,
                            random_state=42,
                        ),
                    )
                )

    else:
        if "Linear Regression" in selected_models:
            models.append(
                (
                    "Linear Regression",
                    LinearRegression(),
                )
            )

        if "Random Forest" in selected_models:
            for n_trees in rf_values:
                models.append(
                    (
                        f"Random Forest (trees={n_trees})",
                        RandomForestRegressor(
                            n_estimators=n_trees,
                            random_state=42,
                        ),
                    )
                )

        if "KNN" in selected_models:
            for k_value in knn_values:
                models.append(
                    (
                        f"KNN (k={k_value})",
                        KNeighborsRegressor(
                            n_neighbors=k_value,
                        ),
                    )
                )

        if "SVM" in selected_models:
            models.append(
                (
                    "SVM",
                    SVR(),
                )
            )

        if "Decision Tree" in selected_models:
            for depth in dt_values:
                models.append(
                    (
                        f"Decision Tree (depth={depth})",
                        DecisionTreeRegressor(
                            max_depth=depth,
                            random_state=42,
                        ),
                    )
                )

    if not models:
        raise ValueError("No valid model configuration was created.")

    return models


def run_multiple_models(
    df,
    target_col,
    task_type,
    selected_models,
    test_size,
    hyperparameters,
):
    if not selected_models:
        raise ValueError("Please select at least one model.")

    X_train, X_test, y_train, y_test = prepare_data(
        df,
        target_col,
        task_type,
        test_size,
    )

    max_neighbors = len(X_train)

    models = build_models(
        task_type,
        selected_models,
        hyperparameters,
        max_neighbors,
    )

    results = []

    for model_name, model in models:
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if task_type == "classification":
                metrics = evaluate_classification(
                    y_test,
                    preds,
                )
            else:
                metrics = evaluate_regression(
                    y_test,
                    preds,
                )

            results.append(
                {
                    "model": model_name,
                    "metrics": metrics,
                }
            )

        except Exception as exc:
            results.append(
                {
                    "model": model_name,
                    "metrics": {
                        "error": str(exc),
                    },
                }
            )

    return results