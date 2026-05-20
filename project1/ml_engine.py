import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler


def run_ml_pipeline(df, target_col, model_name):

    df = df.copy()

    # ================= REMOVE MISSING VALUES =================

    df = df.dropna()

    # ================= FEATURES & TARGET =================

    X = df.drop(columns=[target_col])

    y = df[target_col]

    # ================= ENCODE CATEGORICAL FEATURES =================

    for col in X.select_dtypes(include=["object"]).columns:

        le = LabelEncoder()

        X[col] = le.fit_transform(X[col].astype(str))

    # ================= ENCODE TARGET =================

    if y.dtype == "object":

        y = LabelEncoder().fit_transform(y.astype(str))

    # ================= TRAIN TEST SPLIT =================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.3,

        random_state=42,

        stratify=y
    )

    # ================= FEATURE SCALING =================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # ================= MODEL SELECTION =================

    if model_name == "logistic_regression":

        model = LogisticRegression(max_iter=1000)

        model_label = "Logistic Regression"

    elif model_name == "decision_tree":

        model = DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )

        model_label = "Decision Tree"

    elif model_name == "random_forest":

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        model_label = "Random Forest"

    elif model_name == "knn":

        model = KNeighborsClassifier(
            n_neighbors=5
        )

        model_label = "KNN"

    else:

        return {

            "model": "Invalid Model",

            "accuracy": 0
        }

    # ================= TRAIN MODEL =================

    model.fit(X_train, y_train)

    # ================= PREDICTIONS =================

    predictions = model.predict(X_test)

    # ================= ACCURACY =================

    accuracy = accuracy_score(y_test, predictions)

    accuracy = round(accuracy * 100, 2)

    # ================= RETURN =================

    return {

        "model": model_label,

        "accuracy": accuracy
    }