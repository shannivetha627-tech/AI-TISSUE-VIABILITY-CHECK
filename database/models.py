from database.database import db


class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    username = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    role = db.Column(
        db.String(20),
        nullable=False
    )

    patient_id = db.Column(
        db.String(50),
        nullable=True,
        unique=True
    )

    face_embedding = db.Column(
        db.Text,
        nullable=True
    )


class Patient(db.Model):

    patient_id = db.Column(
        db.String(50),
        primary_key=True
    )

    age = db.Column(db.Integer)

    gender = db.Column(db.String(20))

    bmi = db.Column(db.Float)

    diabetes = db.Column(db.String(20))

    smoking = db.Column(db.String(20))

    hypertension = db.Column(db.String(20))

    heart_rate = db.Column(db.Integer)

    blood_pressure_sys = db.Column(db.Integer)

    spo2 = db.Column(db.Integer)

    tissue_temperature = db.Column(db.Float)

    blood_flow = db.Column(db.String(20))

    perfusion_index = db.Column(db.Float)

    capillary_refill_time = db.Column(db.Float)

    surgery_duration = db.Column(db.Float)

    flap_type = db.Column(db.String(100))

    risk_level = db.Column(db.String(20))

    tissue_viability = db.Column(db.String(20))


class Prediction(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    patient_id = db.Column(
        db.String(50),
        nullable=False
    )

    prediction = db.Column(
        db.String(20),
        nullable=False
    )

    probability = db.Column(
        db.Float
    )

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now()
    )

class PredictionHistory(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    patient_id = db.Column(
        db.String(50),
        nullable=False,
        index=True
    )

    prediction = db.Column(
        db.String(20),
        nullable=False
    )

    probability = db.Column(
        db.Float
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        index=True
    )

    source = db.Column(
        db.String(40),
        nullable=False,
        default="system_generated"
    )