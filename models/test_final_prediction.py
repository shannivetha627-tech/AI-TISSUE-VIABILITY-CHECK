import pandas as pd
import joblib


# Load final model
model = joblib.load(
    "saved_models/final_tissue_model.pkl"
)


# Load dataset
df = pd.read_csv(
    "dataset/tissue_viability.csv"
)


# Select first patient
patient = df.iloc[[0]].copy()


# Application information
patient_id = patient["Patient_ID"].iloc[0]
risk_level = patient["Risk_Level"].iloc[0]


# Prepare ML features
features = patient.drop(
    columns=[
        "Patient_ID",
        "Risk_Level",
        "Tissue_Viability"
    ]
)


# Prediction
prediction = model.predict(features)[0]


# Prediction probability
probabilities = model.predict_proba(features)[0]


print("\n======================================")
print("       TISSUE VIABILITY PREDICTION")
print("======================================")

print("Patient ID :", patient_id)
print("Risk Level :", risk_level)
print("Prediction :", prediction)

print("\nPrediction Probability:")

for label, probability in zip(
    model.classes_,
    probabilities
):
    print(
        f"{label}: {probability * 100:.2f}%"
    )

print("======================================")