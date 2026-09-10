import pandas as pd
from flask import Flask

from database.database import db
from database.models import Patient


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "instance",
    "tissue_viability.db"
)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + DATABASE_PATH
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)


# =========================================================
# IMPORT PATIENT DATA
# =========================================================

def import_patients():

    print("\n========================================")
    print("       PATIENT DATA IMPORT")
    print("========================================")

    # Load CSV
    df = pd.read_csv(
        "dataset/tissue_viability.csv"
    )

    print(
        "CSV Records Found:",
        len(df)
    )

    # Import inside Flask application context
    with app.app_context():

        # Check whether data already exists
        existing_count = Patient.query.count()

        if existing_count > 0:

            print(
                "\nDatabase already contains",
                existing_count,
                "patients."
            )

            print(
                "Import cancelled to avoid duplicates."
            )

            return

        # Insert patients
        for _, row in df.iterrows():

            patient = Patient(

                patient_id=row["Patient_ID"],

                age=row["Age"],

                gender=row["Gender"],

                bmi=row["BMI"],

                diabetes=row["Diabetes"],

                smoking=row["Smoking"],

                hypertension=row["Hypertension"],

                heart_rate=row["Heart_Rate"],

                blood_pressure_sys=row[
                    "Blood_Pressure_Sys"
                ],

                spo2=row["SpO2"],

                tissue_temperature=row[
                    "Tissue_Temperature"
                ],

                blood_flow=row["Blood_Flow"],

                perfusion_index=row[
                    "Perfusion_Index"
                ],

                capillary_refill_time=row[
                    "Capillary_Refill_Time"
                ],

                surgery_duration=row[
                    "Surgery_Duration"
                ],

                flap_type=row["Flap_Type"],

                risk_level=row["Risk_Level"],

                tissue_viability=row[
                    "Tissue_Viability"
                ]
            )

            db.session.add(patient)

        # Save all records
        db.session.commit()

        print(
            "\nSuccessfully imported:",
            len(df),
            "patients"
        )


# =========================================================
# CREATE DATABASE AND RUN IMPORT
# =========================================================

if __name__ == "__main__":

    with app.app_context():

        db.create_all()

    import_patients()

    print("\n========================================")
    print("       DATABASE READY")
    print("========================================")