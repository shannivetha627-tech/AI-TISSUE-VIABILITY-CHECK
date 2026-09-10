import os
from database.database import db
from database.models import User, Patient

# Ensure app context
from app import app

with app.app_context():
    # Create doctor user
    if not User.query.filter_by(username='doc1').first():
        doc = User(username='doc1', password='password123', role='doctor')
        db.session.add(doc)
        print('Doctor user added')
    else:
        print('Doctor user already exists')

    # Create patient user and patient record
    if not User.query.filter_by(username='pat1').first():
        pat_user = User(username='pat1', password='password123', role='patient', patient_id='PAT001')
        db.session.add(pat_user)
        print('Patient user added')
    else:
        print('Patient user already exists')

    if not Patient.query.filter_by(patient_id='PAT001').first():
        pat = Patient(
            patient_id='PAT001', age=30, gender='Male', bmi=22.5,
            diabetes='No', smoking='No', hypertension='No', heart_rate=70,
            blood_pressure_sys=120, spo2=98, tissue_temperature=37.0,
            blood_flow='Normal', perfusion_index=3.5, capillary_refill_time=2.0,
            surgery_duration=2.0, flap_type='Skin', risk_level='Low', tissue_viability='Yes'
        )
        db.session.add(pat)
        print('Patient record added')
    else:
        print('Patient record already exists')

    db.session.commit()
    print('Committed changes')
