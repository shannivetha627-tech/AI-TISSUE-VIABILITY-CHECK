import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/tissue_viability.csv"
MODEL_PATH = "model/tissue_viability_model.pkl"
OUTPUT_PATH = "model/validation_predictions.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ============================================================
# HEADER
# ============================================================

print()
print("=" * 60)
print("       TISSUE VIABILITY MODEL VALIDATION")
print("=" * 60)
print()

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print()
print("Dataset loaded successfully.")
print()
print("Dataset Shape:")
print(df.shape)

# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()
print("=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df["Tissue_Viability"].value_counts())

print()
print("Target Percentages:")

print(
    df["Tissue_Viability"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ============================================================
# REQUIRED FEATURES
# ============================================================

FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "Diabetes",
    "Smoking",
    "Hypertension",
    "Heart_Rate",
    "Blood_Pressure_Sys",
    "SpO2",
    "Tissue_Temperature",
    "Blood_Flow",
    "Perfusion_Index",
    "Capillary_Refill_Time",
    "Surgery_Duration",
    "Flap_Type"
]

TARGET = "Tissue_Viability"

# ============================================================
# COLUMN VALIDATION
# ============================================================

print()
print("=" * 60)
print("COLUMN VALIDATION")
print("=" * 60)

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("Missing columns:")
    print(missing_columns)

    raise ValueError(
        "Required columns are missing from the dataset."
    )

print("All required columns are present.")

# Explicitly confirm Risk_Level is not used
if "Risk_Level" in FEATURES:

    raise ValueError(
        "Risk_Level must not be used as a model feature."
    )

print()
print("Risk_Level is NOT used as a model feature.")

# ============================================================
# MISSING VALUE CHECK
# ============================================================

print()
print("=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

missing_values = df[required_columns].isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

print()
print("Total Missing Values:", total_missing)

# ============================================================
# DUPLICATE CHECK
# ============================================================

print()
print("=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

duplicate_rows = df.duplicated().sum()

print("Duplicate Rows:", duplicate_rows)

# ============================================================
# PATIENT ID CHECK
# ============================================================

print()
print("=" * 60)
print("PATIENT ID CHECK")
print("=" * 60)

if "Patient_ID" in df.columns:

    duplicate_patient_ids = df["Patient_ID"].duplicated().sum()

    print(
        "Duplicate Patient IDs:",
        duplicate_patient_ids
    )

# ============================================================
# TARGET RELATIONSHIPS
# ============================================================

print()
print("=" * 60)
print("FEATURE RELATIONSHIPS WITH TARGET")
print("=" * 60)

categorical_features = [
    "Gender",
    "Diabetes",
    "Smoking",
    "Hypertension",
    "Blood_Flow",
    "Flap_Type"
]

for feature in categorical_features:

    print()
    print("-" * 40)
    print(feature)
    print("-" * 40)

    relationship = pd.crosstab(
        df[feature],
        df[TARGET],
        normalize="index"
    ).mul(100).round(2)

    print(relationship)

# ============================================================
# NUMERICAL STATISTICS
# ============================================================

numerical_features = [
    "Age",
    "BMI",
    "Heart_Rate",
    "Blood_Pressure_Sys",
    "SpO2",
    "Tissue_Temperature",
    "Perfusion_Index",
    "Capillary_Refill_Time",
    "Surgery_Duration"
]

print()
print("=" * 60)
print("NUMERICAL FEATURE STATISTICS BY TARGET")
print("=" * 60)

for feature in numerical_features:

    print()
    print("-" * 40)
    print(feature)
    print("-" * 40)

    statistics = (
        df.groupby(TARGET)[feature]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max"
            ]
        )
        .round(3)
    )

    print(statistics)

# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print()
print("=" * 60)
print("DATA SPLIT")
print("=" * 60)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print()
print("Loading trained model...")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print()
print("Model loaded successfully.")

print()
print("Model Type:")
print(type(model))

# ============================================================
# MODEL FEATURE CHECK
# ============================================================

print()
print("=" * 60)
print("MODEL FEATURE VALIDATION")
print("=" * 60)

print("Features supplied to model:")

for feature in FEATURES:

    print(" -", feature)

print()
print("Risk_Level:")
print(" - NOT USED")

# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("MODEL PERFORMANCE VALIDATION")
print("=" * 60)

print()
print("Generating predictions...")

y_pred = model.predict(X_test)

# ============================================================
# PREDICTION PROBABILITIES
# ============================================================

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(X_test)

    classes = model.classes_

    yes_index = list(classes).index("Yes")

    y_probability = probabilities[:, yes_index]

else:

    y_probability = None

# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("Accuracy:")
print(f"{accuracy * 100:.2f}%")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report:")
print()

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["No", "Yes"]
)

print()
print("Confusion Matrix:")
print()
print(cm)

# ============================================================
# ROC-AUC
# ============================================================

print()
print("ROC-AUC SCORE")
print("-" * 40)

if y_probability is not None:

    roc_auc = roc_auc_score(
        (y_test == "Yes").astype(int),
        y_probability
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

else:

    roc_auc = None

    print(
        "ROC-AUC could not be calculated."
    )

# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print()
print("=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

print(
    pd.Series(y_pred).value_counts()
)

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

print()
print("=" * 60)
print("ACTUAL VS PREDICTED")
print("=" * 60)

comparison = pd.crosstab(
    y_test,
    y_pred,
    rownames=["Actual"],
    colnames=["Predicted"]
)

print(comparison)

# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

print()
print("=" * 60)
print("SAVING VALIDATION RESULTS")
print("=" * 60)

validation_results = X_test.copy()

validation_results.insert(
    0,
    "Patient_ID",
    df.loc[X_test.index, "Patient_ID"]
)

validation_results["Actual_Tissue_Viability"] = y_test

validation_results["Predicted_Tissue_Viability"] = y_pred

if y_probability is not None:

    validation_results[
        "Prediction_Probability"
    ] = y_probability

validation_results.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print(
    "Validation predictions saved to:"
)
print(OUTPUT_PATH)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print(
    f"Dataset Size       : {len(df)}"
)

print(
    f"Training Records   : {len(X_train)}"
)

print(
    f"Testing Records    : {len(X_test)}"
)

print(
    f"Accuracy            : {accuracy * 100:.2f}%"
)

if roc_auc is not None:

    print(
        f"ROC-AUC             : {roc_auc:.4f}"
    )

print(
    f"Missing Values      : {total_missing}"
)

print(
    f"Duplicate Rows      : {duplicate_rows}"
)

print()
print(
    "Risk_Level was NOT used during validation."
)

print()
print(
    "Model validation completed successfully."
)

print("=" * 60)