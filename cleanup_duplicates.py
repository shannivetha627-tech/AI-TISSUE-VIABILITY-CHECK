from app import app, db, Prediction
from sqlalchemy import func


with app.app_context():

    print()
    print("=" * 60)
    print("DUPLICATE PREDICTION CLEANUP")
    print("=" * 60)

    # ---------------------------------------------------------
    # SHOW DATABASE BEING USED
    # ---------------------------------------------------------

    try:
        print("Database URI:", app.config["SQLALCHEMY_DATABASE_URI"])
    except Exception:
        print("Database URI: Unable to determine")

    print("=" * 60)

    # ---------------------------------------------------------
    # FIND DUPLICATE PATIENT IDs
    # ---------------------------------------------------------

    duplicates = (
        db.session.query(
            Prediction.patient_id,
            func.count(Prediction.id).label("count")
        )
        .group_by(Prediction.patient_id)
        .having(func.count(Prediction.id) > 1)
        .all()
    )

    print()
    print("Patients with duplicates:", len(duplicates))

    deleted = 0

    # ---------------------------------------------------------
    # PROCESS EACH DUPLICATE PATIENT
    # ---------------------------------------------------------

    for patient_id, count in duplicates:

        rows = (
            Prediction.query
            .filter(
                Prediction.patient_id == patient_id
            )
            .order_by(
                Prediction.id.asc()
            )
            .all()
        )

        print()
        print("-" * 60)
        print("Patient:", patient_id)
        print("Prediction records:", count)

        # -----------------------------------------------------
        # KEEP FIRST / OLDEST RECORD
        # -----------------------------------------------------

        keep = rows[0]

        print()
        print("KEEPING:")
        print(
            "ID:",
            keep.id,
            "| Prediction:",
            keep.prediction,
            "| Probability:",
            keep.probability
        )

        # -----------------------------------------------------
        # DELETE EXTRA RECORDS
        # -----------------------------------------------------

        for row in rows[1:]:

            print("DELETING:")
            print(
                "ID:",
                row.id,
                "| Prediction:",
                row.prediction,
                "| Probability:",
                row.probability
            )

            db.session.delete(row)

            deleted += 1

    # ---------------------------------------------------------
    # COMMIT CHANGES
    # ---------------------------------------------------------

    if deleted > 0:

        db.session.commit()

        print()
        print("Database changes committed successfully.")

    else:

        print()
        print("No duplicate records found.")
        print("No database changes were required.")

    # ---------------------------------------------------------
    # VERIFY DATABASE
    # ---------------------------------------------------------

    total_predictions = (
        Prediction.query.count()
    )

    unique_patients = (
        db.session.query(
            Prediction.patient_id
        )
        .distinct()
        .count()
    )

    remaining_duplicates = (
        db.session.query(
            Prediction.patient_id,
            func.count(Prediction.id)
        )
        .group_by(
            Prediction.patient_id
        )
        .having(
            func.count(Prediction.id) > 1
        )
        .count()
    )

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("CLEANUP COMPLETED")
    print("=" * 60)

    print(
        "Duplicate records deleted:",
        deleted
    )

    print(
        "Total prediction records:",
        total_predictions
    )

    print(
        "Unique patients with predictions:",
        unique_patients
    )

    print(
        "Patients still having duplicates:",
        remaining_duplicates
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # FINAL STATUS
    # ---------------------------------------------------------

    if remaining_duplicates == 0:

        print()
        print("SUCCESS: No duplicate prediction records remain.")

    else:

        print()
        print(
            "WARNING: Some duplicate records still remain."
        )

    print()