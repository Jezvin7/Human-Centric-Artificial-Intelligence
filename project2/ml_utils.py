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

import numpy as np
import pandas as pd


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

# ==========================================
# ADDED FOR TASK 4 & 5: Backend Logic
# ==========================================

def calculate_mad(series):
    """Helper function to calculate Median Absolute Deviation without SciPy."""
    median = series.median()
    mad = np.median(np.abs(series - median))
    return mad if mad > 0 else 1e-3  # Prevent division by zero

def generate_counterfactuals(model, x_instance, target_class, X_train, numerical_features, categorical_features, k=5, N=1000):
    """
    Task 4: Generates counterfactual explanations using MAD-weighted L1 distance.
    """
    # Calculate MAD for all numerical features based on the training set
    mad_values = {feat: calculate_mad(X_train[feat].dropna()) for feat in numerical_features}
    
    # 1. Randomly sample N points locally around x
    synth_data = pd.DataFrame(index=range(N), columns=X_train.columns)
    
    for feat in numerical_features:
        std = X_train[feat].std()
        # Inject Gaussian noise for continuous features
        base_val = x_instance[feat].values[0]
        synth_data[feat] = base_val + np.random.normal(0, std, N)
        
    for feat in categorical_features:
        # Randomly sample from available categories to noise categorical/binary data
        unique_vals = X_train[feat].dropna().unique()
        synth_data[feat] = np.random.choice(unique_vals, N)
        
    # 2. Check whether the prediction has the desired class
    predictions = model.predict(synth_data)
    valid_idx = np.where(predictions == target_class)[0]
    
    if len(valid_idx) == 0:
        return [] 
        
    valid_samples = synth_data.iloc[valid_idx].copy()
    
    # 3. Rank those that have the desired class by MAD-weighted L1-distance
    distances = np.zeros(len(valid_samples))
    
    for i, (_, row) in enumerate(valid_samples.iterrows()):
        dist = 0
        for feat in numerical_features:
            dist += abs(row[feat] - x_instance[feat].values[0]) / mad_values[feat]
        
        for feat in categorical_features:
            if row[feat] != x_instance[feat].values[0]:
                dist += 1.0  # L1 analog penalty for categorical shift
        distances[i] = dist
        
    valid_samples['distance'] = distances
    
    # 4. Return the best k counterfactuals
    best_cf = valid_samples.sort_values('distance').head(k)
    return best_cf.to_dict('records')


def compute_pdp(model, X_train, feature_name, class_index, grid_resolution=50):
    """
    Task 5: Global Model-Agnostic Methods - Partial Dependence Plot (PDP)
    """
    min_val = X_train[feature_name].min()
    max_val = X_train[feature_name].max()
    grid_vals = np.linspace(min_val, max_val, grid_resolution)
    
    pdp_values = []
    X_temp = X_train.copy()
    
    for val in grid_vals:
        X_temp[feature_name] = val
        # Predict probabilities for the entire dataset with the forced feature value
        probas = model.predict_proba(X_temp)
        # Average the probability for the specific species class
        avg_proba = np.mean(probas[:, class_index])
        pdp_values.append(avg_proba)
        
    return grid_vals.tolist(), pdp_values


def compute_ale(model, X_train, feature_name, class_index, bins=10):
    """
    Task 5: Accumulated Local Effects (ALE) using discretization.
    """
    # Discretize the feature into quantiles (bins)
    quantiles = np.linspace(0, 1, bins + 1)
    z_bounds = np.quantile(X_train[feature_name].dropna(), quantiles)
    z_bounds = np.unique(z_bounds) # Ensure bounds are strictly increasing
    
    ale_values = np.zeros(len(z_bounds) - 1)
    
    for k in range(1, len(z_bounds)):
        z_lower = z_bounds[k-1]
        z_upper = z_bounds[k]
        
        # Isolate samples that naturally fall into this bin
        if k == len(z_bounds) - 1:
            in_bin = (X_train[feature_name] >= z_lower) & (X_train[feature_name] <= z_upper)
        else:
            in_bin = (X_train[feature_name] >= z_lower) & (X_train[feature_name] < z_upper)
            
        X_bin = X_train[in_bin].copy()
        n_k = len(X_bin)
        
        if n_k > 0:
            # Replace the feature with the bin boundaries to compute the local effect
            X_upper = X_bin.copy()
            X_upper[feature_name] = z_upper
            X_lower = X_bin.copy()
            X_lower[feature_name] = z_lower
            
            probas_upper = model.predict_proba(X_upper)[:, class_index]
            probas_lower = model.predict_proba(X_lower)[:, class_index]
            
            ale_values[k-1] = np.mean(probas_upper - probas_lower)
            
    # Accumulate the effects
    ale_accumulated = np.cumsum(ale_values)
    
    # Center the plot
    ale_centered = ale_accumulated - np.mean(ale_accumulated)
    
    # Insert 0 at the start to align with the lower bound of the first bin
    final_ale = np.insert(ale_centered, 0, 0)
    
    return z_bounds.tolist(), final_ale.tolist()