import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# =========================================================
# 1. LOAD DATASET
# =========================================================

DATASET_PATH = "dataset/tissue_viability.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully!")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# =========================================================
# 2. REMOVE PATIENT ID
# =========================================================

df = df.drop(columns=["Patient_ID"])


# =========================================================
# 3. DEFINE INPUTS AND TARGET
# =========================================================

X = df.drop(columns=["Tissue_Viability"])

y = df["Tissue_Viability"]


print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())


# =========================================================
# 4. IDENTIFY FEATURE TYPES
# =========================================================

categorical_features = [
    "Gender",
    "Diabetes",
    "Smoking",
    "Hypertension",
    "Blood_Flow",
    "Flap_Type",
    "Risk_Level"
]

numeric_features = [
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


# =========================================================
# 5. PREPROCESSING
# =========================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# =========================================================
# 6. MACHINE LEARNING MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
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


print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# =========================================================
# 8. TRAIN MODEL
# =========================================================

print("\nTraining Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model Training Completed!")


# =========================================================
# 9. EVALUATE MODEL
# =========================================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================================================
# 10. SAVE MODEL
# =========================================================

MODEL_DIR = "ml_model"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tissue_viability_model.pkl"
)

joblib.dump(
    pipeline,
    MODEL_PATH
)


print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print(
    "Model Path:",
    MODEL_PATH
)