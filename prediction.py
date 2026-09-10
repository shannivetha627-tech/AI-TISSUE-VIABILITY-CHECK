import os
import joblib
import pandas as pd


# =========================================================
# MODEL PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "tissue_viability_model.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_tissue_viability(patient):

    # =====================================================
    # MODEL INPUT FEATURES
    # =====================================================
    # IMPORTANT:
    # Risk_Level is intentionally NOT included.
    # The current model was trained without Risk_Level.

    data = {
        "Age": patient.age,
        "Gender": patient.gender,
        "BMI": patient.bmi,
        "Diabetes": patient.diabetes,
        "Smoking": patient.smoking,
        "Hypertension": patient.hypertension,
        "Heart_Rate": patient.heart_rate,
        "Blood_Pressure_Sys": patient.blood_pressure_sys,
        "SpO2": patient.spo2,
        "Tissue_Temperature": patient.tissue_temperature,
        "Blood_Flow": patient.blood_flow,
        "Perfusion_Index": patient.perfusion_index,
        "Capillary_Refill_Time": patient.capillary_refill_time,
        "Surgery_Duration": patient.surgery_duration,
        "Flap_Type": patient.flap_type
    }


    # =====================================================
    # CONVERT TO DATAFRAME
    # =====================================================

    patient_df = pd.DataFrame([data])


    # =====================================================
    # MAKE PREDICTION
    # =====================================================

    prediction = model.predict(
        patient_df
    )[0]


    # =====================================================
    # GET PREDICTION PROBABILITY
    # =====================================================

    probabilities = model.predict_proba(
        patient_df
    )[0]


    # =====================================================
    # GET CLASS NAMES AND PROBABILITIES
    # =====================================================

    classes = list(model.classes_)
    # Viable class index ('Yes')
    viable_index = classes.index("Yes") if "Yes" in classes else 1
    predicted_class_index = classes.index(prediction) if prediction in classes else 0

    viability_probability = float(probabilities[viable_index]) * 100
    predicted_class_probability = float(probabilities[predicted_class_index]) * 100


    # =====================================================
    # CONFIDENCE LEVEL (BASED ON PREDICTED CLASS CERTAINTY)
    # =====================================================

    if predicted_class_probability >= 80:

        confidence = "High"

    elif predicted_class_probability >= 60:

        confidence = "Moderate"

    else:

        confidence = "Low"


    # =====================================================
    # RECORDED TISSUE VIABILITY
    # =====================================================

    recorded_viability = patient.tissue_viability


    # =====================================================
    # AI VS RECORDED RESULT
    # =====================================================

    if prediction == recorded_viability:

        comparison = "Match"

    else:

        comparison = "Mismatch"


    # =====================================================
    # RETURN COMPLETE RESULT
    # =====================================================

    return {
        "prediction": prediction,

        "viability_probability": round(
            float(viability_probability),
            2
        ),

        "predicted_class_probability": round(
            float(predicted_class_probability),
            2
        ),

        "probability": round(
            float(viability_probability),
            2
        ),

        "confidence": confidence,

        "recorded_viability": recorded_viability,

        "comparison": comparison
    }