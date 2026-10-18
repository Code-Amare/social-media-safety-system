from pathlib import Path
import os

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "m.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "addiction_pipeline.pkl"
TEMP_MODEL_PATH = MODEL_DIR / "addiction_pipeline.pkl.new"

TARGET_COLUMN = "addicted_label"

# Must match build_addiction_features() in background_monitor.py.
FEATURE_COLUMNS = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "age",
    "gender",
    "work_study_hours",
    "sleep_hours",
    "stress_level",
    "academic_work_impact",
]

NUMERIC_FEATURES = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "age",
    "work_study_hours",
    "sleep_hours",
]

CATEGORICAL_FEATURES = [
    "gender",
    "stress_level",
    "academic_work_impact",
]


def main():
    print("=" * 60)
    print("EGATE AI — ADDICTION MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found:\n{DATASET_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    print(f"\nDataset: {DATASET_PATH}")
    print(f"Original rows: {len(df)}")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Identifiers and addiction_level are deliberately excluded.
    df = df[required_columns].copy()
    df = df.dropna(subset=[TARGET_COLUMN])

    # --------------------------------------------------------
    # 3. Normalize target labels to binary 0/1
    # --------------------------------------------------------

    raw_y = df[TARGET_COLUMN]

    if pd.api.types.is_numeric_dtype(raw_y):
        y = pd.to_numeric(raw_y, errors="coerce")
    else:
        label_map = {
            "0": 0,
            "false": 0,
            "no": 0,
            "not addicted": 0,
            "1": 1,
            "true": 1,
            "yes": 1,
            "addicted": 1,
        }

        y = raw_y.astype(str).str.strip().str.lower().map(label_map)

    if y.isna().any():
        raise ValueError(
            "Some addicted_label values are missing or unrecognized. "
            f"Observed labels: {raw_y.unique().tolist()}"
        )

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "addicted_label must contain only binary 0/1 labels. "
            f"Observed values: {sorted(y.unique().tolist())}"
        )

    y = y.astype(int)
    X = df[FEATURE_COLUMNS].copy()

    if y.nunique() != 2:
        raise ValueError(
            "Training requires both classes (0 and 1), but the "
            f"dataset contains: {y.value_counts().to_dict()}"
        )

    print("\nTarget distribution:")
    print(y.value_counts().sort_index().rename("count"))

    # --------------------------------------------------------
    # 4. Define preprocessing
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    # --------------------------------------------------------
    # 5. Evaluate on a held-out test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    evaluation_preprocessor = clone(preprocessor)

    X_train_encoded = evaluation_preprocessor.fit_transform(X_train)
    X_test_encoded = evaluation_preprocessor.transform(X_test)

    evaluation_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining evaluation model...")

    evaluation_model.fit(X_train_encoded, y_train)

    predictions = evaluation_model.predict(X_test_encoded)

    print(f"\nHeld-out accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # 6. Train the final model using all available data
    # --------------------------------------------------------

    print("Training final model on all labeled rows...")

    final_preprocessor = preprocessor
    X_encoded = final_preprocessor.fit_transform(X)

    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(X_encoded, y)

    # Keep the artifact structure expected by background_monitor.py.
    artifact = {
        "model": final_model,
        "preprocessor": final_preprocessor,
        "feature_columns": FEATURE_COLUMNS,
    }

    # --------------------------------------------------------
    # 7. Save to a temporary file and test before replacing
    # --------------------------------------------------------

    if TEMP_MODEL_PATH.exists():
        TEMP_MODEL_PATH.unlink()

    print(f"\nSaving candidate model: {TEMP_MODEL_PATH}")
    joblib.dump(artifact, TEMP_MODEL_PATH)

    print("Testing saved candidate model...")

    loaded_artifact = joblib.load(TEMP_MODEL_PATH)

    assert set(loaded_artifact.keys()) == {
        "model",
        "preprocessor",
        "feature_columns",
    }

    assert loaded_artifact["feature_columns"] == FEATURE_COLUMNS

    sample = X.iloc[[0]]
    sample_encoded = loaded_artifact["preprocessor"].transform(sample)

    probability = loaded_artifact["model"].predict_proba(sample_encoded)[0][1]

    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("Model returned an invalid probability.")

    print(f"Saved model test passed. Sample probability: {probability:.4f}")

    # Replace the broken artifact only after the candidate passes.
    os.replace(TEMP_MODEL_PATH, MODEL_PATH)

    print("\n" + "=" * 60)
    print("SUCCESS: ADDICTION MODEL REBUILT")
    print(f"Model saved to: {MODEL_PATH}")
    print("Artifact keys: model, preprocessor, feature_columns")
    print("=" * 60)

    print(
        "\nNote: test accuracy measures agreement with this dataset's "
        "labels; it does not establish clinical addiction diagnosis."
    )


if __name__ == "__main__":
    main()
