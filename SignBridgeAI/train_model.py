
from pathlib import Path
import csv
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

DATA_FILE = Path("data/isl_landmarks.csv")
MODEL_FILE = Path("models/isl_sign_model.pkl")


def main():
    if not DATA_FILE.exists():
        print(f"Dataset CSV not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        print("The CSV contains no usable samples.")
        return

    # Exclude label and image path from model features.
    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    if not feature_columns or "label" not in df.columns:
        print("CSV is missing labels or feature columns.")
        return

    df = df.dropna(subset=["label"] + feature_columns)
    df = df[df["label"] != "write"]
    counts = df["label"].value_counts()

    print("\nSamples per class:")
    print(counts.to_string())

    # Each class needs at least 2 samples for stratified splitting.
    if len(counts) < 2 or (counts < 2).any():
        print("\nSome classes have fewer than 2 samples.")
        print("Collect or recover more samples before training.")
        return

    X = df[feature_columns]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=True)
    )

    print(f"\nTraining on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n--- Evaluation ---")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nPer-class report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))
    print("\nConfusion Matrix:")
    labels = sorted(y.unique())

    cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
    )

    print("Label order:", labels)
    print(pd.DataFrame(
    cm,
    index=labels,
    columns=labels
    ).to_string())

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": feature_columns},
        MODEL_FILE
    )

    print(f"\nModel saved to: {MODEL_FILE.resolve()}")


if __name__ == "__main__":
    main()

