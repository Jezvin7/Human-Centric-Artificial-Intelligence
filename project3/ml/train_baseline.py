from pathlib import Path
import json
import joblib

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "baseline_classifier.joblib"
RESULTS_PATH = MODEL_DIR / "baseline_results.json"


# ---------------------------------------------------------
# 1. Load AG News
# ---------------------------------------------------------

print("Loading AG News dataset...")

dataset = load_dataset("fancyzhx/ag_news")

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

print(f"Training samples: {len(train_texts)}")
print(f"Test samples: {len(test_texts)}")


# ---------------------------------------------------------
# 2. TF-IDF representation
# ---------------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

print(f"Training feature shape: {X_train.shape}")
print(f"Test feature shape: {X_test.shape}")


# ---------------------------------------------------------
# 3. Logistic Regression
# ---------------------------------------------------------

print("\nTraining Logistic Regression classifier...")

classifier = LogisticRegression(
    max_iter=500,
    solver="saga",
    random_state=42,
)

classifier.fit(X_train, train_labels)


# ---------------------------------------------------------
# 4. Predictions
# ---------------------------------------------------------

print("\nEvaluating model...")

predictions = classifier.predict(X_test)

accuracy = accuracy_score(
    test_labels,
    predictions
)


# ---------------------------------------------------------
# 5. Detailed results
# ---------------------------------------------------------

class_names = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
]

report = classification_report(
    test_labels,
    predictions,
    target_names=class_names,
    output_dict=True,
)

cm = confusion_matrix(
    test_labels,
    predictions
)


# ---------------------------------------------------------
# 6. Display results
# ---------------------------------------------------------

print("\n================================")
print("TASK 1 — BASELINE RESULTS")
print("================================")

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        test_labels,
        predictions,
        target_names=class_names,
    )
)

print("\nConfusion Matrix:")
print(cm)


# ---------------------------------------------------------
# 7. Save model
# ---------------------------------------------------------

print("\nSaving trained model...")

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

joblib.dump(
    classifier,
    MODEL_PATH
)


# ---------------------------------------------------------
# 8. Save results for Django
# ---------------------------------------------------------

results = {
    "dataset": "AG News",
    "train_samples": len(train_texts),
    "test_samples": len(test_texts),
    "model": "Logistic Regression",
    "representation": "TF-IDF",
    "accuracy": float(accuracy),
    "accuracy_percent": round(float(accuracy) * 100, 2),
    "class_names": class_names,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
}

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        results,
        f,
        indent=4,
    )

print("\nSaved:")
print(VECTORIZER_PATH)
print(MODEL_PATH)
print(RESULTS_PATH)

print("\nTask 1 completed successfully.")