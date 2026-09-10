"""Create and idempotently seed history from existing current predictions."""

from app import app, db
from database.models import Prediction, PredictionHistory


with app.app_context():
    db.create_all()
    migrated = 0
    for prediction in Prediction.query.all():
        if prediction.created_at is None:
            continue
        exists = PredictionHistory.query.filter_by(
            patient_id=prediction.patient_id,
            created_at=prediction.created_at,
            prediction=prediction.prediction,
            probability=prediction.probability,
        ).first()
        if exists is None:
            db.session.add(PredictionHistory(
                patient_id=prediction.patient_id,
                prediction=prediction.prediction,
                probability=prediction.probability,
                created_at=prediction.created_at,
                source="system_migrated",
            ))
            migrated += 1
    db.session.commit()
    print(f"Prediction history migration complete. Records added: {migrated}")