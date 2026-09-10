"""Recalculate and timestamp one current prediction for every patient."""

from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import func

from app import app, db
from database.models import Patient, Prediction, PredictionHistory
from prediction import model


BATCH_SIZE = 1000


def prediction_timestamp_utc():
    """Return a naive UTC timestamp for the existing SQLite column."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def configure_model_workers():
    if hasattr(model, "named_steps"):
        steps = model.named_steps.values()
    else:
        steps = [model]
    for step in steps:
        if hasattr(step, "n_jobs"):
            step.n_jobs = 1


def patient_features(patient):
    """Use the same 15 feature names and values as prediction.py."""
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
        "Flap_Type": patient.flap_type,
    }


with app.app_context():
    database_uri = app.config["SQLALCHEMY_DATABASE_URI"]
    database_file = database_uri.removeprefix("sqlite:///")
    patients = Patient.query.order_by(Patient.patient_id.asc()).all()
    prediction_rows = defaultdict(list)
    configure_model_workers()

    for row in Prediction.query.order_by(Prediction.id.asc()).all():
        prediction_rows[row.patient_id].append(row)

    print("Database URI:", database_uri)
    print("Database file:", database_file)
    print("Total Patients:", len(patients))

    updated = 0
    created = 0
    history_created = 0
    failed = 0
    deleted_duplicates = 0

    for batch_start in range(0, len(patients), BATCH_SIZE):
        batch = patients[batch_start:batch_start + BATCH_SIZE]
        try:
            features = pd.DataFrame([patient_features(patient) for patient in batch])
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
            classes = list(model.classes_)
            viable_index = classes.index("Yes") if "Yes" in classes else 1

            for index, patient in enumerate(batch):
                prediction_label = predictions[index]
                probability = round(
                    float(probabilities[index][viable_index] * 100),
                    2,
                )
                rows = prediction_rows[patient.patient_id]

                if rows:
                    prediction = rows[0]
                    updated += 1
                    for duplicate in rows[1:]:
                        db.session.delete(duplicate)
                        deleted_duplicates += 1
                else:
                    prediction = Prediction(patient_id=patient.patient_id)
                    db.session.add(prediction)
                    prediction_rows[patient.patient_id].append(prediction)
                    created += 1

                recalculated_at = prediction_timestamp_utc()
                prediction.prediction = str(prediction_label)
                prediction.probability = probability
                prediction.created_at = recalculated_at
                db.session.add(PredictionHistory(
                    patient_id=patient.patient_id,
                    prediction=str(prediction_label),
                    probability=probability,
                    created_at=recalculated_at,
                    source="system_generated",
                ))
                history_created += 1

            db.session.commit()
        except Exception as error:
            db.session.rollback()
            failed += len(batch)
            print(f"Batch failed at patient {batch_start + 1}: {error}")

        processed = min(batch_start + BATCH_SIZE, len(patients))
        print(
            f"Processed: {processed}/{len(patients)} | "
            f"Updated: {updated} | Created: {created} | Failed: {failed}",
            flush=True,
        )

    total_records = Prediction.query.count()
    unique_patients = db.session.query(Prediction.patient_id).distinct().count()
    duplicate_groups = (
        db.session.query(Prediction.patient_id)
        .group_by(Prediction.patient_id)
        .having(func.count(Prediction.id) > 1)
        .count()
    )
    missing_timestamps = Prediction.query.filter(Prediction.created_at.is_(None)).count()

    print("Total Patients:", len(patients))
    print("Predictions Recalculated:", updated + created)
    print("Current Predictions Updated:", updated)
    print("Current Predictions Created:", created)
    print("History Events Created:", history_created)
    print("Missing Predictions:", len(patients) - unique_patients)
    print("Duplicate Rows Removed:", deleted_duplicates)
    print("Failed:", failed)
    print("Total Prediction Records:", total_records)
    print("Unique Patients With Predictions:", unique_patients)
    print("Duplicates:", duplicate_groups)
    print("Missing Prediction Timestamps:", missing_timestamps)

    if (
        len(patients) != 50000
        or total_records != len(patients)
        or unique_patients != len(patients)
        or duplicate_groups != 0
        or missing_timestamps != 0
        or failed != 0
    ):
        raise SystemExit("Prediction refresh validation failed.")
