import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# TISSUE VIABILITY ML MODEL TRAINING
# =========================================================

print()
print("========================================")
print("   TISSUE VIABILITY ML MODEL TRAINING")
print("========================================")
print()


# =========================================================
# PATHS
# =========================================================

DATASET_PATH = "dataset/tissue_viability.csv"

MODEL_DIR = "model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tissue_viability_model.pkl"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    MODEL_DIR,
    "feature_importance.csv"
)


# =========================================================
# CREATE MODEL DIRECTORY
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    DATASET_PATH
)

print("Dataset Loaded Successfully!")
print()

print("First 5 Records:")
print(df.head())

print()

print("Dataset Shape:")
print(df.shape)


# =========================================================
# TARGET DISTRIBUTION
# =========================================================

print()
print("Tissue Viability Distribution:")

print(
    df["Tissue_Viability"].value_counts()
)


# =========================================================
# FEATURES
# =========================================================
#
# IMPORTANT:
# Risk_Level has intentionally been removed.
#
# The validation showed that Risk_Level directly mapped
# to Tissue_Viability:
#
# High   -> No
# Medium -> Yes
# Low    -> Yes
#
# Including Risk_Level would therefore create target leakage.
# =========================================================

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


print()
print("Features Used:")
print(FEATURES)

print()
print("Target:")
print(TARGET)


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print()
    print("ERROR: Missing required columns:")
    print(missing_columns)

    raise ValueError(
        "Dataset is missing required columns."
    )


# =========================================================
# PREPARE X AND y
# =========================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()


# =========================================================
# IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# =========================================================

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()


numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()


print()
print("Categorical Columns:")
print(categorical_columns)


print()
print("Numerical Columns:")
print(numerical_columns)


# =========================================================
# PREPROCESSING
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
# RANDOM FOREST MODEL
# =========================================================

classifier = RandomForestClassifier(

    n_estimators=200,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


# =========================================================
# COMPLETE PIPELINE
# =========================================================

model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            classifier
        )

    ]

)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

print()
print("========================================")
print("DATA SPLIT")
print("========================================")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "Training Records:",
    len(X_train)
)


print(
    "Testing Records:",
    len(X_test)
)


# =========================================================
# MODEL TRAINING
# =========================================================

print()
print("========================================")
print("TRAINING RANDOM FOREST MODEL")
print("========================================")
print()


model.fit(
    X_train,
    y_train
)


print(
    "Model training completed successfully!"
)


# =========================================================
# MODEL PREDICTION
# =========================================================

y_pred = model.predict(
    X_test
)


# =========================================================
# MODEL EVALUATION
# =========================================================

print()
print("========================================")
print("MODEL EVALUATION")
print("========================================")


accuracy = accuracy_score(
    y_test,
    y_pred
)


print()
print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print()
print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    MODEL_PATH
)


print()
print("========================================")
print("MODEL SAVED")
print("========================================")


print(
    "Model saved at:",
    MODEL_PATH
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print()
print("========================================")
print("TOP FEATURE IMPORTANCE")
print("========================================")


# Get the fitted preprocessing stage
fitted_preprocessor = model.named_steps[
    "preprocessor"
]


# Get the Random Forest classifier
fitted_classifier = model.named_steps[
    "classifier"
]


# Get transformed feature names
feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


# Get Random Forest feature importance
importances = (
    fitted_classifier
    .feature_importances_
)


feature_importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importances

})


# Sort by importance
feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


# Display top 15
print(
    feature_importance_df.head(15)
)


# =========================================================
# SAVE FEATURE IMPORTANCE
# =========================================================

feature_importance_df.to_csv(

    FEATURE_IMPORTANCE_PATH,

    index=False

)


print()
print(
    "Feature importance saved at:",
    FEATURE_IMPORTANCE_PATH
)


# =========================================================
# TRAINING SUMMARY
# =========================================================

print()
print("========================================")
print("       TRAINING COMPLETED")
print("========================================")

print()

print(
    "Dataset Size:",
    len(df)
)

print(
    "Training Records:",
    len(X_train)
)

print(
    "Testing Records:",
    len(X_test)
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print()

print(
    "Risk_Level was NOT used as a model feature."
)

print(
    "The trained model is ready for prediction."
)

print()

print(
    "Model:",
    MODEL_PATH
)

print(
    "Feature Importance:",
    FEATURE_IMPORTANCE_PATH
)

print()

print("========================================")
