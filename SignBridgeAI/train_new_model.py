
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = "data/isl_landmarks_new.csv"
MODEL_PATH = "models/isl_sign_model_v2.pkl"

def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH, dtype={"label": str}, low_memory=False)

    feature_columns = [f"f{i}" for i in range(126)]
    X = df[feature_columns]
    y = df["label"].astype(str).str.strip()
    print("Dataset shape:", X.shape)
    print("Number of classes:", y.nunique())

    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining model. This may take a few minutes...")
    model = ExtraTreesClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight=None
    )

    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    predictions = model.predict(X_test)

    print("Test accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nNew model saved to:", MODEL_PATH)
    print("Your original model has not been changed.")

if __name__ == "__main__":
    main()