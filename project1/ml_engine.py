import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def run_ml_pipeline(df, target_col, task_type):

    df = df.copy()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical feature columns
    for col in X.select_dtypes(include="object"):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode target if needed
    if pd.api.types.is_object_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if task_type == "classification" else None
    )

    if task_type == "classification":
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if task_type == "classification":
        score = accuracy_score(y_test, preds)
        metric = "accuracy"
    else:
        score = r2_score(y_test, preds)
        metric = "r2_score"

    return {
        "score": float(score),
        "metric": metric
    }