"""
NON-DESTRUCTIVE BATCH ML PERFORMANCE TEST

Purpose:
    Measure how fast the existing tissue viability model can process
    many patients in one batch.

IMPORTANT:
    - Does NOT modify Patient records.
    - Does NOT modify Prediction records.
    - Does NOT create PredictionHistory records.
    - Does NOT modify prediction.py.
    - Does NOT modify the ML model.
    - Uses the existing database.
    - Uses the existing trained model.
    - Uses the EXACT 15 features from prediction.py.

Tests:
    1. Load all patients.
    2. Convert patient data into one DataFrame.
    3. Run model.predict() ONCE for all patients.
    4. Run model.predict_proba() ONCE for all patients.
    5. Measure total inference time.
    6. Compare batch results against the existing
       predict_tissue_viability() function for a sample.
"""

import os
import sys
import time
import io
from contextlib import redirect_stdout
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
from flask import Flask

from database.database import db
from database.models import Patient
from prediction import predict_tissue_viability


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "instance",
    "tissue_viability.db"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "tissue_viability_model.pkl"
)

DATABASE_URI = "sqlite:///" + DATABASE_PATH


# ============================================================
# EXACT MODEL FEATURES
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


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def ist_string(dt):
    try:
        ist = ZoneInfo("Asia/Kolkata")
        return dt.astimezone(ist).strftime(
            "%d %b %Y, %I:%M:%S %p IST"
        )
    except Exception:
        return dt.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )


# ============================================================
# BUILD DATAFRAME
# ============================================================

def patient_to_dict(patient):

    return {
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


# ============================================================
# MAIN TEST
# ============================================================

def run_test():

    print()
    print("=" * 78)
    print("       TISSUE VIABILITY — BATCH 50,000 PATIENT TEST")
    print("=" * 78)
    print()

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    print("Database:")
    print(DATABASE_PATH)
    print()

    print("Model:")
    print(MODEL_PATH)
    print()

    if not os.path.exists(DATABASE_PATH):

        print("ERROR: Database file does not exist.")
        print(DATABASE_PATH)

        return 1

    if not os.path.exists(MODEL_PATH):

        print("ERROR: Model file does not exist.")
        print(MODEL_PATH)

        return 1

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    with app.app_context():

        try:

            patients = (
                Patient.query
                .order_by(Patient.patient_id.asc())
                .all()
            )

        except Exception as exc:

            print()
            print("ERROR: Could not load patients.")
            print(type(exc).__name__)
            print(str(exc))

            return 1

        total_patients = len(patients)

        print("Total patients found:")
        print(f"  {total_patients:,}")
        print()

        if total_patients == 0:

            print("ERROR: No patients found.")

            return 1

        # ----------------------------------------------------
        # BUILD DATAFRAME
        # ----------------------------------------------------

        print("Preparing batch feature DataFrame...")
        print()

        dataframe_start = time.perf_counter()

        rows = [
            patient_to_dict(patient)
            for patient in patients
        ]

        patient_df = pd.DataFrame(
            rows,
            columns=FEATURES
        )

        dataframe_time = (
            time.perf_counter()
            - dataframe_start
        )

        print(
            f"DataFrame created: "
            f"{len(patient_df):,} rows"
        )

        print(
            f"Feature count: "
            f"{len(patient_df.columns)}"
        )

        print(
            f"DataFrame preparation time: "
            f"{dataframe_time:.4f} seconds"
        )

        print()

        # ----------------------------------------------------
        # FEATURE VALIDATION
        # ----------------------------------------------------

        print("Validating features...")

        if list(patient_df.columns) != FEATURES:

            print()
            print("ERROR: Feature order mismatch.")

            print("Expected:")
            print(FEATURES)

            print("Actual:")
            print(list(patient_df.columns))

            return 1

        print("Feature validation: PASS")
        print()

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        print("Loading existing ML model...")

        try:

            model_load_start = time.perf_counter()

            model = joblib.load(MODEL_PATH)

            model_load_time = (
                time.perf_counter()
                - model_load_start
            )

        except Exception as exc:

            print()
            print("ERROR: Could not load model.")
            print(type(exc).__name__)
            print(str(exc))

            return 1

        print(
            f"Model loaded in "
            f"{model_load_time:.4f} seconds"
        )

        print()

        # ----------------------------------------------------
        # MODEL FEATURE CHECK
        # ----------------------------------------------------

        if hasattr(model, "feature_names_in_"):

            model_features = list(
                model.feature_names_in_
            )

            print("Model feature validation:")

            if model_features == FEATURES:

                print("  PASS")

            else:

                print("  WARNING: Model feature names differ.")

                print("  Model:")
                print(f"  {model_features}")

                print("  Expected:")
                print(f"  {FEATURES}")

        else:

            print(
                "Model feature_names_in_ "
                "not available."
            )

        print()

        # ----------------------------------------------------
        # BATCH PREDICTION
        # ----------------------------------------------------

        print("=" * 78)
        print("                    BATCH INFERENCE")
        print("=" * 78)
        print()

        print(
            f"Running model.predict() for "
            f"{total_patients:,} patients..."
        )

        batch_start = time.perf_counter()

        try:

            batch_predictions = model.predict(
                patient_df
            )

        except Exception as exc:

            print()
            print("ERROR during batch predict().")
            print(type(exc).__name__)
            print(str(exc))

            return 1

        predict_time = (
            time.perf_counter()
            - batch_start
        )

        print(
            f"Batch predict() completed in "
            f"{predict_time:.4f} seconds"
        )

        print()

        # ----------------------------------------------------
        # BATCH PROBABILITY
        # ----------------------------------------------------

        print(
            f"Running model.predict_proba() for "
            f"{total_patients:,} patients..."
        )

        probability_start = time.perf_counter()

        try:

            batch_probabilities = model.predict_proba(
                patient_df
            )

        except Exception as exc:

            print()
            print("ERROR during batch predict_proba().")
            print(type(exc).__name__)
            print(str(exc))

            return 1

        probability_time = (
            time.perf_counter()
            - probability_start
        )

        print(
            f"Batch predict_proba() completed in "
            f"{probability_time:.4f} seconds"
        )

        print()

        # ----------------------------------------------------
        # TOTAL INFERENCE TIME
        # ----------------------------------------------------

        total_inference_time = (
            predict_time
            + probability_time
        )

        rate = (
            total_patients / total_inference_time
            if total_inference_time > 0
            else 0
        )

        print("=" * 78)
        print("                 BATCH PERFORMANCE")
        print("=" * 78)
        print()

        print(
            f"Patients processed: "
            f"{total_patients:,}"
        )

        print(
            f"predict() time: "
            f"{predict_time:.4f} sec"
        )

        print(
            f"predict_proba() time: "
            f"{probability_time:.4f} sec"
        )

        print(
            f"TOTAL ML INFERENCE TIME: "
            f"{total_inference_time:.4f} sec"
        )

        print(
            f"Processing rate: "
            f"{rate:,.2f} patients/sec"
        )

        if total_patients > 0:

            average_ms = (
                total_inference_time
                / total_patients
                * 1000
            )

            print(
                f"Average ML time/patient: "
                f"{average_ms:.4f} ms"
            )

        print()

        # ----------------------------------------------------
        # RESULT DISTRIBUTION
        # ----------------------------------------------------

        prediction_counts = {}

        for prediction in batch_predictions:

            key = str(prediction)

            prediction_counts[key] = (
                prediction_counts.get(key, 0)
                + 1
            )

        print("Prediction distribution:")

        for key, value in sorted(
            prediction_counts.items()
        ):

            print(
                f"  {key}: {value:,}"
            )

        print()

        # ----------------------------------------------------
        # VERIFY BATCH AGAINST EXISTING FUNCTION
        # ----------------------------------------------------

        print("=" * 78)
        print("            INDIVIDUAL VS BATCH VALIDATION")
        print("=" * 78)
        print()

        sample_count = min(
            10,
            total_patients
        )

        print(
            f"Checking first "
            f"{sample_count} patients..."
        )

        mismatches = 0

        probability_mismatches = 0

        tolerance = 0.000001

        for index in range(sample_count):

            patient = patients[index]

            try:

                captured_output = io.StringIO()

                with redirect_stdout(
                    captured_output
                ):

                    individual_result = (
                        predict_tissue_viability(
                            patient
                        )
                    )

            except Exception as exc:

                print()
                print(
                    f"ERROR validating "
                    f"{patient.patient_id}"
                )

                print(
                    type(exc).__name__,
                    str(exc)
                )

                mismatches += 1

                continue

            individual_prediction = str(
                individual_result["prediction"]
            )

            batch_prediction = str(
                batch_predictions[index]
            )

            if (
                individual_prediction
                != batch_prediction
            ):

                mismatches += 1

                print(
                    f"Prediction mismatch: "
                    f"{patient.patient_id}"
                )

            # ----------------------------------------------
            # INDIVIDUAL PROBABILITY
            # ----------------------------------------------

            individual_probability = float(
                individual_result["probability"]
            )

            classes = model.classes_

            probability_dict = dict(
                zip(
                    classes,
                    batch_probabilities[index]
                )
            )

            batch_probability = (
                float(
                    probability_dict[
                        batch_predictions[index]
                    ]
                )
                * 100
            )

            if abs(
                individual_probability
                - batch_probability
            ) > tolerance:

                probability_mismatches += 1

                print(
                    f"Probability mismatch: "
                    f"{patient.patient_id} "
                    f"| individual="
                    f"{individual_probability:.6f}% "
                    f"| batch="
                    f"{batch_probability:.6f}%"
                )

        print()

        if mismatches == 0:

            print(
                "Prediction consistency: PASS"
            )

        else:

            print(
                f"Prediction consistency: "
                f"FAIL ({mismatches} mismatches)"
            )

        if probability_mismatches == 0:

            print(
                "Probability consistency: PASS"
            )

        else:

            print(
                f"Probability consistency: "
                f"FAIL "
                f"({probability_mismatches} mismatches)"
            )

        print()

        # ----------------------------------------------------
        # 60 SECOND TARGET
        # ----------------------------------------------------

        print("=" * 78)
        print("                    60-SECOND TARGET")
        print("=" * 78)
        print()

        if total_inference_time <= 60:

            print(
                "TARGET: PASS"
            )

            print()

            print(
                f"All {total_patients:,} patients "
                f"were processed by the ML model "
                f"in {total_inference_time:.2f} seconds."
            )

            print(
                "Batch ML inference is within "
                "the 60-second target."
            )

        else:

            print(
                "TARGET: FAIL"
            )

            print()

            print(
                f"Batch ML inference took "
                f"{total_inference_time:.2f} seconds."
            )

            print(
                "The current batch inference path "
                "does not meet the 60-second target."
            )

        print()

        # ----------------------------------------------------
        # PROJECTED FULL CYCLE
        # ----------------------------------------------------

        print("=" * 78)
        print("                  CYCLE ANALYSIS")
        print("=" * 78)
        print()

        print(
            "IMPORTANT:"
        )

        print(
            "This benchmark measures ML inference only."
        )

        print(
            "Database UPDATE/INSERT time is not included."
        )

        print(
            "PredictionHistory creation is not included."
        )

        print(
            "Scheduler overhead is not included."
        )

        print()

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        print("=" * 78)
        print("                     SAFETY CHECK")
        print("=" * 78)
        print()

        print(
            "Patient records modified: NO"
        )

        print(
            "Prediction records modified: NO"
        )

        print(
            "PredictionHistory modified: NO"
        )

        print(
            "ML model modified: NO"
        )

        print(
            "prediction.py modified: NO"
        )

        print(
            "Features modified: NO"
        )

        print()

        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        finish_time = utc_now()

        print("=" * 78)
        print("                     TEST COMPLETE")
        print("=" * 78)
        print()

        print(
            "Completed:"
        )

        print(
            ist_string(finish_time)
        )

        print()

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        exit_code = run_test()

        sys.exit(exit_code)

    except KeyboardInterrupt:

        print()
        print("Test interrupted by user.")

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 78)
        print("UNEXPECTED ERROR")
        print("=" * 78)
        print()

        print(
            type(exc).__name__
        )

        print(
            str(exc)
        )

        print()

        sys.exit(1)