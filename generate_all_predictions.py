from app import app, db, Patient, Prediction, PredictionHistory, prediction_timestamp_utc
from prediction import model
import pandas as pd
import time


print()
print("=" * 60)
print("BATCH AI PREDICTION GENERATION")
print("=" * 60)


with app.app_context():

    # ---------------------------------------------------------
    # GET ALL PATIENTS
    # ---------------------------------------------------------

    patients = Patient.query.all()

    total_patients = len(patients)

    print("Total Patients:", total_patients)


    # ---------------------------------------------------------
    # GET PATIENTS WHO ALREADY HAVE PREDICTIONS
    # ---------------------------------------------------------

    existing_ids = {
        patient_id
        for (patient_id,) in db.session.query(
            Prediction.patient_id
        ).distinct().all()
    }

    print(
        "Patients Already Having Predictions:",
        len(existing_ids)
    )


    # ---------------------------------------------------------
    # FIND PATIENTS WITHOUT PREDICTIONS
    # ---------------------------------------------------------

    missing_patients = [
        patient
        for patient in patients
        if patient.patient_id not in existing_ids
    ]

    print(
        "Patients Requiring Prediction:",
        len(missing_patients)
    )

    print("=" * 60)


    # ---------------------------------------------------------
    # CHECK MODEL
    # ---------------------------------------------------------

    print()
    print("Checking model configuration...")

    try:

        # Random Forest / estimator parallelism
        if hasattr(model, "named_steps"):

            for step_name, step in model.named_steps.items():

                if hasattr(step, "n_jobs"):

                    try:
                        step.n_jobs = 1
                        print(
                            f"Set n_jobs=1 for pipeline step: {step_name}"
                        )
                    except Exception:
                        pass

        elif hasattr(model, "n_jobs"):

            try:
                model.n_jobs = 1
                print("Set model n_jobs=1")
            except Exception:
                pass

    except Exception as e:

        print("Could not change n_jobs:", e)


    print("Model ready.")
    print("=" * 60)


    # ---------------------------------------------------------
    # PROCESS IN SMALL BATCHES
    # ---------------------------------------------------------

    batch_size = 100

    total_created = 0
    total_failed = 0

    start_time = time.time()


    for start in range(
        0,
        len(missing_patients),
        batch_size
    ):

        batch = missing_patients[
            start:start + batch_size
        ]


        # -----------------------------------------------------
        # CREATE DATAFRAME
        # -----------------------------------------------------

        rows = []

        valid_patients = []


        for patient in batch:

            rows.append({

                # IMPORTANT:
                # These names MUST exactly match prediction.py

                "Age": patient.age,

                "Gender": patient.gender,

                "BMI": patient.bmi,

                "Diabetes": patient.diabetes,

                "Smoking": patient.smoking,

                "Hypertension": patient.hypertension,

                "Heart_Rate": patient.heart_rate,

                "Blood_Pressure_Sys":
                    patient.blood_pressure_sys,

                "SpO2": patient.spo2,

                "Tissue_Temperature":
                    patient.tissue_temperature,

                "Blood_Flow":
                    patient.blood_flow,

                "Perfusion_Index":
                    patient.perfusion_index,

                "Capillary_Refill_Time":
                    patient.capillary_refill_time,

                "Surgery_Duration":
                    patient.surgery_duration,

                "Flap_Type":
                    patient.flap_type
            })

            valid_patients.append(patient)


        patient_df = pd.DataFrame(rows)


        # -----------------------------------------------------
        # RUN MODEL
        # -----------------------------------------------------

        try:

            predictions = model.predict(
                patient_df
            )

            probabilities = model.predict_proba(
                patient_df
            )

            classes = model.classes_


            # -------------------------------------------------
            # SAVE RESULTS
            # -------------------------------------------------

            for index, patient in enumerate(
                valid_patients
            ):

                prediction = predictions[index]


                probability_dict = dict(
                    zip(
                        classes,
                        probabilities[index]
                    )
                )


                prediction_probability = (
                    probability_dict[prediction] * 100
                )

                created_at = prediction_timestamp_utc()
                probability = round(
                    float(
                        prediction_probability
                    ),
                    2
                )

                new_prediction = Prediction(

                    patient_id=patient.patient_id,

                    prediction=str(
                        prediction
                    ),

                    probability=probability,
                    created_at=created_at
                )


                db.session.add(
                    new_prediction
                )
                db.session.add(PredictionHistory(
                    patient_id=patient.patient_id,
                    prediction=str(prediction),
                    probability=probability,
                    created_at=created_at,
                    source="system_generated",
                ))


                total_created += 1


            # -------------------------------------------------
            # COMMIT THIS BATCH
            # -------------------------------------------------

            db.session.commit()


            processed = min(
                start + batch_size,
                len(missing_patients)
            )


            elapsed = time.time() - start_time


            print(
                f"Processed "
                f"{processed}/"
                f"{len(missing_patients)} "
                f"| Created: {total_created} "
                f"| Failed: {total_failed} "
                f"| Time: {elapsed:.1f}s"
            )


        except Exception as e:

            db.session.rollback()

            total_failed += len(batch)

            print()
            print(
                "Batch Error:",
                e
            )

            print(
                "Batch starting at:",
                start
            )

            print()


    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("BATCH PREDICTION GENERATION COMPLETED")
    print("=" * 60)


    print(
        "Total Patients:",
        total_patients
    )


    print(
        "Already Had Predictions:",
        len(existing_ids)
    )


    print(
        "New Predictions Created:",
        total_created
    )


    print(
        "Failed:",
        total_failed
    )


    print(
        "Total Prediction Records:",
        Prediction.query.count()
    )


    print(
        "Unique Patients With Predictions:",
        db.session.query(
            Prediction.patient_id
        ).distinct().count()
    )


    remaining = (
        total_patients
        -
        db.session.query(
            Prediction.patient_id
        ).distinct().count()
    )


    print(
        "Patients Still Without Predictions:",
        remaining
    )


    print("=" * 60)