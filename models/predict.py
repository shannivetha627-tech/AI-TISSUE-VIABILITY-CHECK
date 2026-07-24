import joblib
import pandas as pd

# Load model
model = joblib.load("saved_models/random_forest.pkl")

# Example patient data
sample = {
    "Age": 45,
    "Gender": 1,
    "BMI": 25.4,
    "Diabetes": 0,
    "Smoking": 0,
    "Hypertension": 1,
    "Heart_Rate": 82,
    "Blood_Pressure_Sys": 130,
    "SpO2": 98,
    "Tissue_Temperature": 36.7,
    "Blood_Flow": 2,
    "Perfusion_Index": 4.5,
    "Capillary_Refill_Time": 2.0,
    "Surgery_Duration": 6.5,
    "Flap_Type": 0,
    "Risk_Level": 1
}
df = pd.DataFrame([sample])

prediction = model.predict(df)
prediction = model.predict(df)

if prediction[0] == 1:
    print("\nPrediction: Tissue is VIABLE ✅")
else:
    print("\nPrediction: Tissue is NOT VIABLE ❌")