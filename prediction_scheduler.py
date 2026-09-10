import logging
import os
import threading
import time
from datetime import datetime, timezone

from database.database import db
from database.models import Patient, Prediction, PredictionHistory
from prediction import model as loaded_model
from prediction import predict_tissue_viability

# Configuration constants
DEFAULT_INTERVAL_SECONDS = 120
BATCH_SIZE = 1000  # Larger batch to reduce transaction overhead
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.5  # seconds

# Scheduler internal state
_scheduler_lock = threading.Lock()
_scheduler_thread = None
_stop_event = threading.Event()

def prediction_timestamp_utc():
    """Return the actual UTC generation time for the SQLite timestamp column."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

def configured_interval_seconds():
    try:
        return max(1, int(os.environ.get("PREDICTION_INTERVAL_SECONDS", str(DEFAULT_INTERVAL_SECONDS))))
    except ValueError:
        return DEFAULT_INTERVAL_SECONDS

def configured_patient_filter():
    if os.environ.get("PREDICTION_TEST_MODE", "false").lower() != "true":
        return None
    patient_id = os.environ.get(
        "TEST_PATIENT_ID",
        os.environ.get("PREDICTION_SCHEDULER_PATIENT_ID", ""),
    ).strip()
    return patient_id or None

def configure_model_workers():
    """Avoid nested joblib workers without changing the serialized model."""
    model_steps = getattr(loaded_model, "named_steps", {}).values()
    for step in model_steps:
        if hasattr(step, "n_jobs"):
            step.n_jobs = 1

def _load_patients(app):
    """Materialize all eligible patients with only the columns required for prediction.
    Returns a list of Patient objects (fully loaded) and a DataFrame ready for the model.
    """
    with app.app_context():
        patient_filter = configured_patient_filter()
        query = Patient.query
        if patient_filter:
            query = query.filter_by(patient_id=patient_filter)
        patients = query.order_by(Patient.patient_id.asc()).all()
    return patients

def _prepare_dataframe(patients):
    """Construct a pandas DataFrame containing the 15 required features for batch prediction."""
    import pandas as pd
    rows = []
    for p in patients:
        rows.append({
            "Age": p.age,
            "Gender": p.gender,
            "BMI": p.bmi,
            "Diabetes": p.diabetes,
            "Smoking": p.smoking,
            "Hypertension": p.hypertension,
            "Heart_Rate": p.heart_rate,
            "Blood_Pressure_Sys": p.blood_pressure_sys,
            "SpO2": p.spo2,
            "Tissue_Temperature": p.tissue_temperature,
            "Blood_Flow": p.blood_flow,
            "Perfusion_Index": p.perfusion_index,
            "Capillary_Refill_Time": p.capillary_refill_time,
            "Surgery_Duration": p.surgery_duration,
            "Flap_Type": p.flap_type,
        })
    return pd.DataFrame(rows)

def _bulk_upsert_predictions(patients, predictions, timestamp):
    """Create Prediction and PredictionHistory rows for a batch using high-performance bulk mapping inserts.
    *patients* and *predictions* are parallel lists.
    """
    pred_dicts = []
    hist_dicts = []
    for patient, pred in zip(patients, predictions):
        pred_dicts.append({
            "patient_id": patient.patient_id,
            "prediction": pred["prediction"],
            "probability": pred["probability"],
            "created_at": timestamp,
        })
        hist_dicts.append({
            "patient_id": patient.patient_id,
            "prediction": pred["prediction"],
            "probability": pred["probability"],
            "created_at": timestamp,
            "source": "system_generated",
        })
    patient_ids = [p.patient_id for p in patients]
    # Delete existing Prediction records for this batch using direct query and flush changes
    Prediction.query.filter(Prediction.patient_id.in_(patient_ids)).delete(synchronize_session=False)
    db.session.flush()
    db.session.bulk_insert_mappings(Prediction, pred_dicts)
    db.session.bulk_insert_mappings(PredictionHistory, hist_dicts)

def _run_batch(app, batch_patients):
    """Execute a single batch: ML inference and bulk DB writes with retry handling."""
    # 1. Prepare DataFrame
    df = _prepare_dataframe(batch_patients)
    # 2. Run model inference (both class and probability)
    ml_start = time.perf_counter()
    preds = loaded_model.predict(df)
    probs = loaded_model.predict_proba(df)
    ml_duration = time.perf_counter() - ml_start
    classes = list(loaded_model.classes_)
    viable_index = classes.index("Yes") if "Yes" in classes else 1
    results = []
    for pred_label, prob_vec in zip(preds, probs):
        viability_prob = round(float(prob_vec[viable_index]) * 100, 2)
        results.append({"prediction": str(pred_label), "probability": viability_prob})
    # 3. DB write with retries
    timestamp = prediction_timestamp_utc()
    for attempt in range(_MAX_RETRIES):
        try:
            _bulk_upsert_predictions(batch_patients, results, timestamp)
            db.session.commit()
            return len(batch_patients), ml_duration
        except Exception as exc:
            db.session.rollback()
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                app.logger.warning(
                    "[Prediction Scheduler] Batch write failed (attempt %s). Retrying in %.2f sec. Error: %s",
                    attempt + 1,
                    delay,
                    exc,
                )
                time.sleep(delay)
            else:
                app.logger.error(
                    "[Prediction Scheduler] Batch write failed after %s attempts. Skipping batch. Error: %s",
                    _MAX_RETRIES,
                    exc,
                )
                return 0, ml_duration

def run_prediction_cycle(app, patients=None):
    """Recalculate every eligible patient once and commit successful batches.
    Returns a dict of metrics for logging/validation.
    """
    with app.app_context():
        logger = app.logger
        cycle_started = time.monotonic()
        cycle_started_at = datetime.now(timezone.utc)
        metrics = {"db_read_seconds": 0.0, "ml_inference_seconds": 0.0, "db_write_seconds": 0.0}
        processed = 0
        errors = 0
        history_events = 0
        # Load patients in a single query (materialize)
        db_read_start = time.perf_counter()
        all_patients = _load_patients(app) if patients is None else patients
        total_patients = len(all_patients)
        metrics["db_read_seconds"] += time.perf_counter() - db_read_start
        logger.info("[Prediction Scheduler] Starting prediction cycle")
        logger.info("[Prediction Scheduler] Total patients: %s", total_patients)
        for batch_start in range(0, total_patients, BATCH_SIZE):
            batch = all_patients[batch_start : batch_start + BATCH_SIZE]
            batch_start_time = time.perf_counter()
            batch_processed, ml_dur = _run_batch(app, batch)
            batch_duration = time.perf_counter() - batch_start_time
            metrics["ml_inference_seconds"] += ml_dur
            metrics["db_write_seconds"] += (batch_duration - ml_dur)
            processed += batch_processed
            history_events += batch_processed
            if processed % 1000 == 0 or batch_start + len(batch) == total_patients:
                logger.info("[Prediction Scheduler] Processed %s / %s", processed, total_patients)
            if batch_processed == 0:
                errors += len(batch)
        duration_seconds = time.monotonic() - cycle_started
        cycle_ended_at = datetime.now(timezone.utc)
        logger.info("[Prediction Scheduler] Cycle started: %s", cycle_started_at.isoformat())
        logger.info("[Prediction Scheduler] Cycle completed: %s", cycle_ended_at.isoformat())
        logger.info("[Prediction Scheduler] Duration: %.2f seconds", duration_seconds)
        logger.info("[Prediction Scheduler] Patients processed: %s", processed)
        logger.info("[Prediction Scheduler] History events created: %s", history_events)
        if errors:
            logger.warning("[Prediction Scheduler] Failed patients: %s", errors)
        logger.info("[Prediction Scheduler] DB read time: %.2f seconds", metrics["db_read_seconds"])
        logger.info("[Prediction Scheduler] DB write time: %.2f seconds", metrics["db_write_seconds"])
        return {
            "cycle_started": cycle_started_at,
            "cycle_completed": cycle_ended_at,
            "duration_seconds": duration_seconds,
            "total_patients": total_patients,
            "processed": processed,
            "failed": errors,
            "history_events_created": history_events,
            **metrics,
        }

def _scheduler_loop(app, interval_seconds):
    logger = app.logger
    configure_model_workers()
    logger.info("[Prediction Scheduler] Started")
    while not _stop_event.is_set():
        logger.info("[Prediction Scheduler] Running prediction cycle")
        try:
            run_prediction_cycle(app)
            logger.info("[Prediction Scheduler] Cycle completed")
        except Exception:
            logger.exception("[Prediction Scheduler] Cycle failed")
        logger.info("[Prediction Scheduler] Next cycle in %s seconds", interval_seconds)
        _stop_event.wait(interval_seconds)

def start_prediction_scheduler(app, reloader_guard=False):
    """Start one daemon worker for this application process."""
    global _scheduler_thread
    app.logger.setLevel(logging.INFO)
    if os.environ.get("PREDICTION_SCHEDULER_ENABLED", "1") != "1":
        app.logger.info("[Prediction Scheduler] Disabled by configuration")
        return None
    if reloader_guard and os.environ.get("FLASK_USE_RELOADER", "true").lower() == "true" and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        app.logger.info("[Prediction Scheduler] Waiting for reloader child")
        return None
    with _scheduler_lock:
        if _scheduler_thread and _scheduler_thread.is_alive():
            return _scheduler_thread
        _stop_event.clear()
        interval_seconds = configured_interval_seconds()
        _scheduler_thread = threading.Thread(
            target=_scheduler_loop,
            args=(app, interval_seconds),
            name="prediction-scheduler",
            daemon=True,
        )
        _scheduler_thread.start()
        return _scheduler_thread

def stop_prediction_scheduler():
    _stop_event.set()