import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def make_one_hot_encoder():
    """
    Compatibility helper because different sklearn versions use
    sparse_output or sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_penguin_data():
    """
    Loads the Palmer Penguins dataset.

    Target:
        species

    Features:
        island, sex, year,
        bill_length_mm, bill_depth_mm,
        flipper_length_mm, body_mass_g
    """
    df = load_penguins()
    df = df.dropna()

    target_col = "species"

    numerical_features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    categorical_features = [
        "island",
        "sex",
        "year",
    ]

    X = df[numerical_features + categorical_features]
    y = df[target_col]

    return df, X, y, numerical_features, categorical_features


def get_train_test_data():
    df, X, y, numerical_features, categorical_features = load_penguin_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, numerical_features, categorical_features


def build_tree_model(max_leaf_nodes):
    X_train, X_test, y_train, y_test, numerical_features, categorical_features = get_train_test_data()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", DecisionTreeClassifier(
                max_leaf_nodes=max_leaf_nodes,
                random_state=42,
            )),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    tree_model = model.named_steps["classifier"]
    leaves = tree_model.get_n_leaves()

    return {
        "model": model,
        "model_type": "tree",
        "accuracy": accuracy,
        "complexity": leaves,
        "leaves": leaves,
        "parameter_name": "max_leaf_nodes",
        "parameter_value": max_leaf_nodes if max_leaf_nodes is not None else "No limit",
    }


def build_logistic_model(C):
    X_train, X_test, y_train, y_test, numerical_features, categorical_features = get_train_test_data()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(
                penalty="l1",
                solver="saga",
                C=C,
                max_iter=5000,
                random_state=42,
            )),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logistic_model = model.named_steps["classifier"]

    # Complexity measure for logistic regression:
    # number of non-zero coefficients.
    non_zero_coefficients = int((logistic_model.coef_ != 0).sum())

    return {
        "model": model,
        "model_type": "logistic",
        "accuracy": accuracy,
        "complexity": non_zero_coefficients,
        "non_zero_coefficients": non_zero_coefficients,
        "parameter_name": "C",
        "parameter_value": C,
    }


def train_candidate_models(model_type):
    """
    Trains several models with different regularization strengths.

    Important:
    λ from the interface is NOT directly max_leaf_nodes or C.
    We first train several candidates, then select the model that maximizes:

        test_accuracy - λ * complexity
    """
    candidates = []

    if model_type == "tree":
        max_leaf_nodes_values = [2, 3, 4, 5, 6, 8, 10, 15, 20, None]

        for max_leaf_nodes in max_leaf_nodes_values:
            candidate = build_tree_model(max_leaf_nodes)
            candidates.append(candidate)

    elif model_type == "logistic":
        C_values = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

        for C in C_values:
            candidate = build_logistic_model(C)
            candidates.append(candidate)

    return candidates


def select_best_model(candidates, lambda_value):
    """
    Selects the model that maximizes:

        accuracy_test - lambda * complexity
    """
    best_candidate = max(
        candidates,
        key=lambda item: item["accuracy"] - lambda_value * item["complexity"]
    )

    best_candidate["selection_score"] = (
        best_candidate["accuracy"] - lambda_value * best_candidate["complexity"]
    )

    return best_candidate

def clean_feature_names(feature_names):
    cleaned_names = []

    for name in feature_names:
        name = name.replace("num__", "")
        name = name.replace("cat__", "")
        name = name.replace("_", " ")
        name = name.replace("island ", "island = ")
        name = name.replace("sex ", "sex = ")
        name = name.replace("year ", "year = ")

        cleaned_names.append(name)

    return cleaned_names


def save_tree_plot(model, media_root):
    os.makedirs(media_root, exist_ok=True)

    preprocessor = model.named_steps["preprocess"]
    tree_model = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    feature_names = clean_feature_names(feature_names)
    class_names = [str(class_name) for class_name in tree_model.classes_]
    filename = "decision_tree.png"
    filepath = os.path.join(media_root, filename)
    leaves = tree_model.get_n_leaves()
    depth = tree_model.get_depth()
    fig_width = max(8, leaves * 2.8)
    fig_height = max(5, depth * 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=12,
        impurity=False,
        proportion=False,
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(
        filepath,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    return filename


def get_logistic_coefficients(model):
    preprocessor = model.named_steps["preprocess"]
    logistic_model = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    class_names = logistic_model.classes_

    rows = []

    for feature_index, feature_name in enumerate(feature_names):
        row = {
            "feature": feature_name,
        }

        for class_index, class_name in enumerate(class_names):
            row[str(class_name)] = round(
                logistic_model.coef_[class_index][feature_index],
                4
            )

        rows.append(row)

    return rows, class_names