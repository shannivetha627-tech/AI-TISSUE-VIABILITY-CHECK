import pandas as pd
import joblib


# =========================================================
# 1. LOAD FINAL MODEL
# =========================================================

model = joblib.load(
    "saved_models/final_tissue_model.pkl"
)


# =========================================================
# 2. LOAD DATASET
# =========================================================

df = pd.read_csv(
    "dataset/tissue_viability.csv"
)


# =========================================================
# 3. SELECT PATIENT
# =========================================================

patient = df.iloc[[0]].copy()


# =========================================================
# 4. KEEP APPLICATION INFORMATION
# =========================================================

patient_id = patient["Patient_ID"].iloc[0]

risk_level = patient["Risk_Level"].iloc[0]


# =========================================================
# 5. PREPARE ML INPUT
# =========================================================

features = patient.drop(
    columns=[
        "Patient_ID",
        "Risk_Level",
        "Tissue_Viability"
    ]
)


# =========================================================
# 6. MAKE PREDICTION
# =========================================================

prediction = model.predict(
    features
)[0]


# =========================================================
# 7. GET PROBABILITY
# =========================================================

probabilities = model.predict_proba(
    features
)[0]


# =========================================================
# 8. DISPLAY RESULT
# =========================================================

print("\n======================================")
print("       TISSUE VIABILITY PREDICTION")
print("======================================")

print(
    "Patient ID :", 
    patient_id
)

print(
    "Risk Level :",
    risk_level
)

print(
    "Prediction :",
    prediction
)

print("\nPrediction Probability:")

for label, probability in zip(
    model.classes_,
    probabilities
):

    print(
        f"{label}: {probability * 100:.2f}%"
    )

print("======================================")