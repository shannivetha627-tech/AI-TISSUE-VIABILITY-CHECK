import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("dataset/tissue_viability.csv")

print("\n========== FINAL MODEL TRAINING ==========\n")

print("Dataset Shape:", df.shape)


# =========================================================
# 2. REMOVE NON-ML COLUMNS
# =========================================================

# Patient_ID is used by the application to identify
# the patient, but it must NOT be an ML feature.

# Risk_Level is used by the application for display,
# but it must NOT be an ML feature because the current
# dataset directly associates Risk_Level with the target.

X = df.drop(
    columns=[
        "Patient_ID",
        "Risk_Level",
        "Tissue_Viability"
    ]
)

y = df["Tissue_Viability"]


# =========================================================
# 3. IDENTIFY COLUMN TYPES
# =========================================================

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_columns)

print("\nNumeric Features:")
print(numeric_columns)


# =========================================================
# 4. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# =========================================================
# 5. RANDOM FOREST
# =========================================================

random_forest = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 6. COMPLETE ML PIPELINE
# =========================================================

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", random_forest)
    ]
)


# =========================================================
# 7. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# =========================================================
# 8. TRAIN MODEL
# =========================================================

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)


# =========================================================
# 9. PREDICTION
# =========================================================

prediction = model.predict(X_test)


# =========================================================
# 10. EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    prediction
)

precision = precision_score(
    y_test,
    prediction,
    pos_label="Yes"
)

recall = recall_score(
    y_test,
    prediction,
    pos_label="Yes"
)

f1 = f1_score(
    y_test,
    prediction,
    pos_label="Yes"
)


print("\n========== FINAL EVALUATION ==========")

print(
    "Accuracy :",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(f1 * 100, 2),
    "%"
)


# =========================================================
# 11. CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        prediction,
        labels=["No", "Yes"]
    )
)


# =========================================================
# 12. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        prediction
    )
)


# =========================================================
# 13. SAVE COMPLETE PIPELINE
# =========================================================

joblib.dump(
    model,
    "saved_models/final_tissue_model.pkl"
)


print("\n========================================")
print("FINAL MODEL SAVED SUCCESSFULLY")
print("Location: saved_models/final_tissue_model.pkl")
print("========================================")