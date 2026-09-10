from app import app, db, User, Patient


def create_patient_accounts():

    with app.app_context():

        patients = Patient.query.all()

        created = 0
        skipped = 0

        print()
        print("========================================")
        print("CREATING PATIENT ACCOUNTS")
        print("========================================")
        print("Total patients:", len(patients))
        print()

        for patient in patients:

            existing_user = User.query.filter_by(
                patient_id=patient.patient_id
            ).first()

            if existing_user:
                skipped += 1
                continue

            username = patient.patient_id

            # Example:
            # P001322 -> Patient@001322
            password = "Patient@" + patient.patient_id[1:]

            user = User(
                username=username,
                password=password,
                role="patient",
                patient_id=patient.patient_id
            )

            db.session.add(user)

            created += 1

            # Commit every 1000 accounts
            if created % 1000 == 0:

                db.session.commit()

                print(
                    "Created:",
                    created
                )

        db.session.commit()

        print()
        print("========================================")
        print("PATIENT ACCOUNT CREATION COMPLETED")
        print("========================================")
        print("Total Patients   :", len(patients))
        print("Accounts Created :", created)
        print("Already Existing :", skipped)
        print("========================================")


if __name__ == "__main__":

    create_patient_accounts()