import os
import time
import json
import secrets
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    session,
    redirect,
    url_for
)

from database.database import db
from database.models import User, Patient, Prediction, PredictionHistory
# Face authentication imports are loaded lazily in production to avoid heavy dependencies

from sqlalchemy import inspect, text
from werkzeug.security import check_password_hash, generate_password_hash
from prediction import predict_tissue_viability


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Lazy import for face authentication (optional heavy dependencies)
# ---------------------------------------------------------------------------
def _load_face_auth():
    """Attempt to import face authentication utilities.
    Returns a tuple (extract_face_embedding, serialize_embedding, verify_face).
    If any import fails (e.g., missing mediapipe or opencv), returns (None, None, None).
    """
    try:
        from face_auth.face_verification import (
            extract_face_embedding,
            serialize_embedding,
            verify_face,
        )
        return extract_face_embedding, serialize_embedding, verify_face
    except Exception as e:
        app.logger.warning("Face authentication unavailable: %s", e)
        return None, None, None


# ============================================================
# SECRET KEY
# ============================================================

app.config["SECRET_KEY"] = os.environ.get(
    "TISSUE_VIABILITY_SECRET_KEY",
    secrets.token_hex(32)
)
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)
app.config["SESSION_TIMEOUT_SECONDS"] = int(os.environ.get("SESSION_TIMEOUT", "1800"))
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "0") == "1"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Mean absolute normalized-landmark distance; 0.16 is a prototype-only threshold.
app.config["FACE_MATCH_THRESHOLD"] = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.2"))

LOGIN_MAX_FAILURES = int(os.environ.get("DOCTOR_LOGIN_MAX_FAILURES", "5"))
LOGIN_COOLDOWN_SECONDS = int(os.environ.get("DOCTOR_LOGIN_COOLDOWN", "300"))
_login_failures = defaultdict(list)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "instance",
    "tissue_viability.db"
)

os.makedirs(
    os.path.dirname(DATABASE_PATH),
    exist_ok=True
)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + DATABASE_PATH
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "connect_args": {"timeout": 30, "check_same_thread": False},
    "isolation_level": "AUTOCOMMIT",
}

db.init_app(app)


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

with app.app_context():
    db.session.execute(text("PRAGMA journal_mode=WAL"))
    db.session.commit()


# ============================================================
# LOAD ML MODEL
# ============================================================

try:
    from prediction import model

    MODEL_AVAILABLE = True

    print("\n========================================")
    print("       ML MODEL LOADED SUCCESSFULLY")
    print("========================================")
    print(
        f"Model: {os.path.join(BASE_DIR, 'model', 'tissue_viability_model.pkl')}"
    )
    print("========================================\n")

except Exception as e:
    MODEL_AVAILABLE = False
    model = None

    print("\n========================================")
    print("          ML MODEL LOAD FAILED")
    print("========================================")
    print(f"Error: {e}")
    print("========================================\n")


# ============================================================
# LOGIN / SESSION HELPERS
# ============================================================

def is_logged_in():
    """Check whether a user is currently logged in."""
    return session.get("user_id") is not None


def current_role():
    """Return the currently logged-in user's role."""
    return session.get("role", "").lower()


def doctor_required():
    """Allow access only to logged-in doctors."""
    return (
        is_logged_in()
        and current_role() == "doctor"
        and session.get("face_verified") is True
        and session.get("liveness_verified") is True
    )


def pending_doctor_required():
    """Check for a doctor who passed the password step only."""
    return (
        session.get("pending_doctor_id") is not None
        and session.get("pending_doctor_role", "").lower() == "doctor"
        and time.time() - session.get("face_auth_started", 0) <= 120
    )


def audit_event(event, user_id=None, result="INFO"):
    """Write authentication metadata without credentials or biometric data."""
    app.logger.info(
        "security_audit event=%s user_id=%s result=%s",
        event,
        user_id if user_id is not None else "unknown",
        result,
    )


def _login_key(username):
    return f"{request.remote_addr or 'unknown'}:{username.lower()}"


def login_locked(username):
    now = time.time()
    failures = [
        stamp for stamp in _login_failures[_login_key(username)]
        if now - stamp < LOGIN_COOLDOWN_SECONDS
    ]
    _login_failures[_login_key(username)] = failures
    return len(failures) >= LOGIN_MAX_FAILURES


def record_login_failure(username):
    _login_failures[_login_key(username)].append(time.time())


def clear_login_failures(username):
    _login_failures.pop(_login_key(username), None)


def verify_doctor_password(user, password):
    """Accept old development passwords once, then migrate them to a hash."""
    if user.password.startswith(("pbkdf2:", "scrypt:")):
        return check_password_hash(user.password, password)
    if secrets.compare_digest(user.password, password):
        user.password = generate_password_hash(password)
        db.session.commit()
        return True
    return False


def patient_required():
    """Allow access only to logged-in patients."""
    return (
        is_logged_in()
        and current_role() == "patient"
    )


def prediction_timestamp_utc():
    """Return a naive UTC timestamp suitable for the existing SQLite column."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def format_prediction_timestamp(timestamp):
    """Convert the database's UTC timestamp to the application's IST display."""
    if timestamp is None:
        return "Time unavailable"
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(ZoneInfo("Asia/Kolkata")).strftime(
        "%d %b %Y, %I:%M %p IST"
    )

def save_prediction_result(patient_id, prediction_result, created_at=None, source="system_generated"):
    """Update the one current row and append one immutable history event."""
    timestamp = created_at or prediction_timestamp_utc()
    current_prediction = (
        Prediction.query
        .filter_by(patient_id=patient_id)
        .order_by(Prediction.id.asc())
        .first()
    )
    if current_prediction is None:
        current_prediction = Prediction(patient_id=patient_id)
        db.session.add(current_prediction)

    current_prediction.prediction = prediction_result["prediction"]
    current_prediction.probability = prediction_result["probability"]
    current_prediction.created_at = timestamp
    db.session.add(PredictionHistory(
        patient_id=patient_id,
        prediction=prediction_result["prediction"],
        probability=prediction_result["probability"],
        created_at=timestamp,
        source=source,
    ))
    return current_prediction


def authenticated_doctor_id():
    """Return the doctor ID from the current authenticated session only."""
    if current_role() == "doctor" and session.get("user_id") is not None:
        return session.get("user_id")
    if session.get("pending_doctor_role", "").lower() == "doctor":
        return session.get("pending_doctor_id")
    return None


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    # --------------------------------------------------------
    # Already logged in
    # --------------------------------------------------------

    if is_logged_in():

        if current_role() == "doctor":
            return redirect(
                url_for("doctor_dashboard")
            )

        if current_role() == "patient":
            return redirect(
                url_for("patient_dashboard")
            )


    # --------------------------------------------------------
    # LOGIN FORM SUBMISSION
    # --------------------------------------------------------

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        login_type = request.form.get(
            "login_type",
            ""
        ).strip().lower()

        if login_type == "doctor":
            audit_event("DOCTOR_LOGIN_ATTEMPT", result="STARTED")
            if login_locked(username):
                audit_event("DOCTOR_LOGIN_DENIED", result="LOCKED")
                return render_template(
                    "login.html",
                    error="Too many unsuccessful attempts. Please wait before trying again.",
                )


        # ----------------------------------------------------
        # Validate login type
        # ----------------------------------------------------

        if login_type not in ["doctor", "patient"]:

            return render_template(
                "login.html",
                error="Please select Doctor or Patient login."
            )


        # ----------------------------------------------------
        # Find user
        # ----------------------------------------------------

        user = User.query.filter_by(
            username=username
        ).first()


        # ----------------------------------------------------
        # Validate username
        # ----------------------------------------------------

        if not user or user.role.lower() != login_type:
            if login_type == "doctor":
                record_login_failure(username)
                audit_event("DOCTOR_PASSWORD_FAILURE", result="DENIED")

            return render_template(
                "login.html",
                error="Invalid username or password."
            )


        # ----------------------------------------------------
        # Validate password
        # ----------------------------------------------------

        password_valid = (
            verify_doctor_password(user, password)
            if login_type == "doctor"
            else user.password == password
        )
        if not password_valid:
            if login_type == "doctor":
                record_login_failure(username)
                audit_event("DOCTOR_PASSWORD_FAILURE", user.id, "DENIED")

            return render_template(
                "login.html",
                error="Invalid username or password."
            )


        # ----------------------------------------------------
        # Verify role
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Password success starts a fresh session. Doctors must
        # complete the separate camera verification step.
        # ----------------------------------------------------

        session.clear()

        if user.role.lower() == "doctor":
            clear_login_failures(username)
            audit_event("DOCTOR_PASSWORD_SUCCESS", user.id, "SUCCESS")
            session["pending_doctor_id"] = user.id
            session["pending_doctor_username"] = user.username
            session["pending_doctor_role"] = user.role
            session["face_auth_started"] = time.time()
            session["face_challenge"] = secrets.choice(("left", "right"))
            session.permanent = True
            return redirect(url_for("doctor_face_lock"))

        session["user_id"] = user.id
        session["username"] = user.username
        session["role"] = user.role
        if user.role.lower() == "patient":
            session["patient_id"] = user.patient_id


        # ----------------------------------------------------
        # Redirect based on role
        # ----------------------------------------------------

        if user.role.lower() == "patient":

            return redirect(
                url_for("patient_dashboard")
            )


        # ----------------------------------------------------
        # Invalid role fallback
        # ----------------------------------------------------

        session.clear()

        return render_template(
            "login.html",
            error="Invalid account role."
        )


    # ========================================================
    # GET LOGIN PAGE
    # ========================================================

    return render_template("login.html")


@app.route("/doctor/face-lock", methods=["GET"])
def doctor_face_lock():
    """Render the second authentication step after password success."""
    if not pending_doctor_required():
        return redirect(url_for("login"))
    return render_template(
        "face_lock.html",
        username=session.get("pending_doctor_username"),
        challenge=session.get("face_challenge", "left"),
    )


@app.route("/doctor/verify-face", methods=["POST"])
def verify_doctor_face():
    """Promote a pending doctor session after the camera check succeeds."""
    # Load face auth utilities lazily; if unavailable, return a clear error
    extract_face_embedding, serialize_embedding, verify_face = _load_face_auth()
    if verify_face is None:
        return jsonify({
            "verified": False,
            "success": False,
            "message": "Face verification service temporarily unavailable",

        }), 503

    if not pending_doctor_required():
        message = "Verification session expired."
        return jsonify({"verified": False, "success": False, "message": message, "error": message}), 403

    if login_locked(session.get("pending_doctor_username", "")):
        message = "Too many unsuccessful attempts. Please wait before trying again."
        audit_event("DOCTOR_LOGIN_DENIED", session.get("pending_doctor_id"), "LOCKED")
        return jsonify({"verified": False, "success": False, "message": message, "error": message}), 429

    doctor_id = session["pending_doctor_id"]
    user = db.session.get(User, doctor_id)
    image_data = request.form.getlist("image_data")
    if not image_data:
        image_data = request.form.getlist("image_data[]")
    challenge = session.get("face_challenge")
    audit_event("FACE_DETECTED", doctor_id, "RECEIVED")
    try:
        verified, message = verify_face(
            image_data,
            user.face_embedding if user else None,
            challenge=challenge,
            threshold=app.config["FACE_MATCH_THRESHOLD"],
        )
    except Exception:
        app.logger.exception("[FACE] Unexpected verification failure for user ID: %s", doctor_id)
        return jsonify({
            "verified": False,
            "success": False,
            "message": "Face verification failed. Please try again.",
            "error": "Face verification failed. Please try again.",
        }), 500
    app.logger.info("[FACE] Comparison result: %s", "MATCH" if verified else "NO MATCH")
    if not verified:
        audit_event("LIVENESS_FAILURE", doctor_id, "DENIED")
        audit_event("FACE_MATCH_FAILURE", doctor_id, "DENIED")
        record_login_failure(session.get("pending_doctor_username", ""))
        return jsonify({
            "verified": False,
            "success": False,
            "message": message,
            "error": message,
        }), 400

    session["user_id"] = session.pop("pending_doctor_id")
    session["username"] = session.pop("pending_doctor_username")
    session["role"] = session.pop("pending_doctor_role")
    session.pop("face_auth_started", None)
    session["face_verified"] = True
    session["liveness_verified"] = True
    session["last_activity"] = time.time()
    session.pop("face_challenge", None)
    audit_event("LIVENESS_SUCCESS", doctor_id, "SUCCESS")
    audit_event("FACE_MATCH_SUCCESS", doctor_id, "SUCCESS")
    audit_event("DOCTOR_LOGIN_SUCCESS", doctor_id, "SUCCESS")
    return jsonify({"verified": True, "redirect": url_for("doctor_dashboard")})


@app.route("/doctor/register-face", methods=["GET", "POST"])
def register_doctor_face():
    """Enroll one normalized face representation after password authentication."""
    # Lazy load face auth utilities; if unavailable, return a clear error
    extract_face_embedding, serialize_embedding, verify_face = _load_face_auth()
    if extract_face_embedding is None or serialize_embedding is None:
        return jsonify({
            "success": False,
            "message": "Face registration service temporarily unavailable",

        }), 503

    if not pending_doctor_required() and not doctor_required():
        if request.method == "POST":
            return jsonify({
                "success": False,
                "message": "Face registration requires an authenticated doctor.",
            }), 403
        return redirect(url_for("login"))

    if request.method == "POST":
        app.logger.info("[FACE] Registration request received")
        user_id = authenticated_doctor_id()
        app.logger.info(
            "[FACE] Authenticated user ID: %s; role: %s",
            user_id,
            current_role() or session.get("pending_doctor_role", ""),
        )
        user = db.session.get(User, user_id) if user_id is not None else None
        payload = request.get_json(silent=True) or {}
        # Try to get precomputed embedding from client
        embedding = payload.get("embedding")
        extraction_message = None
        if embedding is None:
            image_data = payload.get("image_data") or request.form.get("image_data", "")
            embedding, extraction_message = extract_face_embedding(image_data)
        if user is None or user.role.lower() != "doctor":
            return jsonify({
                "success": False,
                "message": "Authenticated doctor account not found.",
            }), 404
        if embedding is None:
            return jsonify({
                "success": False,
                "message": extraction_message or "Doctor photo could not be enrolled. Please provide a clear front-facing photo.",
            }), 400
        serialized_embedding = serialize_embedding(embedding)
        try:
            app.logger.info("[FACE] Saving representation for user ID: %s", user_id)
            user.face_embedding = serialized_embedding
            db.session.commit()
            app.logger.info("[FACE] Database commit successful")
            db.session.expire_all()
            stored_user = db.session.get(User, user_id)
            if stored_user is None or stored_user.face_embedding != serialized_embedding:
                db.session.rollback()
                app.logger.error("[FACE] Registration record verification failed for user ID: %s", user_id)
                return jsonify({
                    "success": False,
                    "message": "Face registration could not be saved.",
                }), 500
            app.logger.info("[FACE] Registration record verified for user ID: %s", user_id)
        except Exception:
            db.session.rollback()
            app.logger.exception("[FACE] Registration persistence failed for user ID: %s", user_id)
            return jsonify({
                "success": False,
                "message": "Face registration could not be saved.",
            }), 500
        return jsonify({
            "success": True,
            "message": "Face registered successfully",
            "redirect": url_for("doctor_face_lock"),
        })

    return render_template("face_register.html", username=session.get("pending_doctor_username", session.get("username")))


# ============================================================
# DOCTOR DASHBOARD
# ============================================================

# IMPORTANT:
# GET  -> Opens the dashboard
# POST -> Searches for a patient
#
# This fixes:
# POST /doctor -> 405 Method Not Allowed
# ============================================================

@app.route(
    "/doctor",
    methods=["GET", "POST"]
)
def doctor_dashboard():

    # --------------------------------------------------------
    # SECURITY CHECK
    # --------------------------------------------------------

    if not doctor_required():

        return (
            """
            <h1>Access Denied</h1>
            <p>
                You are not authorized to access the doctor dashboard.
            </p>
            """,
            403
        )


    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    patient = None
    prediction_result = None
    searched_id = ""
    all_predictions = Prediction.query.all()
    dashboard_summary = {
        "total_patients": Patient.query.count(),
        "stable": sum(
            prediction.prediction == "Yes"
            for prediction in all_predictions
        ),
        "at_risk": sum(
            prediction.prediction == "No"
            for prediction in all_predictions
        ),
        "recent_predictions": len(all_predictions)
    }
    patient_history = []


    # ========================================================
    # SEARCH PATIENT
    # ========================================================

    if request.method == "POST":

        searched_id = request.form.get(
            "patient_id",
            ""
        ).strip()


        # ----------------------------------------------------
        # Empty Patient ID
        # ----------------------------------------------------

        if not searched_id:

            return render_template(
                "doctor_dashboard.html",
                username=session.get("username"),
                patient=None,
                prediction_result=None,
                searched_id="",
                dashboard_summary=dashboard_summary,
                patient_history=patient_history
            )


        # ----------------------------------------------------
        # Find patient
        # ----------------------------------------------------

        patient = Patient.query.filter_by(
            patient_id=searched_id
        ).first()


        # ----------------------------------------------------
        # Patient not found
        # ----------------------------------------------------

        if patient is None:

            return render_template(
                "doctor_dashboard.html",
                username=session.get("username"),
                patient=None,
                prediction_result=None,
                searched_id=searched_id,
                dashboard_summary=dashboard_summary,
                patient_history=patient_history
            )


        # ====================================================
        # AI PREDICTION
        # ====================================================

        try:

            # ------------------------------------------------
            # Check model
            # ------------------------------------------------

            if not MODEL_AVAILABLE:

                prediction_result = {
                    "prediction": "Model unavailable",
                    "probability": None,
                    "confidence": None,
                    "recorded_viability": (
                        patient.tissue_viability
                    ),
                    "comparison": None
                }


            else:

                # --------------------------------------------
                # Generate AI prediction
                # --------------------------------------------

                prediction_result = (
                    predict_tissue_viability(patient)
                )


                # ============================================
                # SAVE / UPDATE PREDICTION
                # ============================================

                save_prediction_result(
                    patient.patient_id,
                    prediction_result,
                )


                db.session.commit()

                all_predictions = Prediction.query.all()
                dashboard_summary["stable"] = sum(
                    item.prediction == "Yes" for item in all_predictions
                )
                dashboard_summary["at_risk"] = sum(
                    item.prediction == "No" for item in all_predictions
                )
                dashboard_summary["recent_predictions"] = len(all_predictions)

            patient_history = (
                PredictionHistory.query
                .filter_by(patient_id=patient.patient_id)
                .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
                .limit(5)
                .all()
            )
            for prediction in patient_history:
                prediction.display_time = format_prediction_timestamp(
                    prediction.created_at
                )
            if patient_history:
                patient_history[0].is_latest = True


        except Exception as e:

            # -----------------------------------------------
            # Rollback database if something goes wrong
            # -----------------------------------------------

            db.session.rollback()


            prediction_result = {
                "prediction": "Error",
                "probability": None,
                "confidence": None,
                "recorded_viability": getattr(patient, "tissue_viability", "N/A"),
                "comparison": None,
            }

        return render_template(
        "doctor_dashboard.html",
        username=session.get("username"),

        patient=patient,

        prediction_result=prediction_result,
        searched_id=searched_id,
        dashboard_summary=dashboard_summary,
        patient_history=patient_history
    )

    return render_template(
        "doctor_dashboard.html",
        username=session.get("username"),
        patient=None,
        prediction_result=None,
        searched_id="",
        dashboard_summary=dashboard_summary,
        patient_history=patient_history,
    )


# ============================================================
# PATIENT DASHBOARD
# ============================================================

@app.route(
    "/patient",
    methods=["GET"]
)
def patient_dashboard():

    # --------------------------------------------------------
    # SECURITY CHECK
    # --------------------------------------------------------

    if not patient_required():

        return (
            """
            <h1>Access Denied</h1>
            <p>
                You are not authorized to access the patient dashboard.
            </p>
            """,
            403
        )


    # --------------------------------------------------------
    # Get patient ID from SESSION
    # --------------------------------------------------------

    patient_id = session.get(
        "patient_id"
    )


    if not patient_id:

        session.clear()

        return redirect(
            url_for("login")
        )


    # --------------------------------------------------------
    # Find patient
    # --------------------------------------------------------

    patient = Patient.query.filter_by(
        patient_id=patient_id
    ).first()


    if patient is None:

        session.clear()

        return (
            """
            <h1>Patient Record Not Found</h1>
            <p>
                Your patient record could not be found.
            </p>
            """,
            404
        )


    # --------------------------------------------------------
    # Get latest prediction
    # --------------------------------------------------------

    latest_prediction = (
        Prediction.query
        .filter_by(
            patient_id=patient.patient_id
        )
        .order_by(
            Prediction.created_at.desc()
        )
        .first()
    )


    # --------------------------------------------------------
    # Get prediction history
    # --------------------------------------------------------

    history_page = max(request.args.get("history_page", 1, type=int), 1)
    history_per_page = 5
    history_query = (
        PredictionHistory.query
        .filter_by(patient_id=patient.patient_id)
        .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
    )
    predictions = history_query.offset(
        (history_page - 1) * history_per_page
    ).limit(history_per_page).all()
    history_total = history_query.count()
    history_has_more = history_page * history_per_page < history_total


    # --------------------------------------------------------
    # Format timestamps
    # --------------------------------------------------------

    for prediction in predictions:
        prediction.display_time = format_prediction_timestamp(
            prediction.created_at
        )
        prediction.is_latest = prediction is predictions[0]

    if latest_prediction:
        latest_prediction.updated_display_time = format_prediction_timestamp(
            latest_prediction.created_at
        )


    # ========================================================
    # DISPLAY PATIENT DASHBOARD
    # ========================================================

    return render_template(

        "patient_dashboard.html",

        patient=patient,

        latest_prediction=latest_prediction,

        predictions=predictions,
        history_page=history_page,
        history_has_more=history_has_more,
        history_total=history_total
    )


@app.route(
    "/patient/history",
    methods=["GET"]
)
def patient_history_api():
    """Return only the authenticated patient's paginated prediction history."""
    if not patient_required():
        return jsonify({"error": "Patient authentication required."}), 403

    patient_id = session.get("patient_id")
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = min(max(request.args.get("per_page", 20, type=int), 1), 100)
    history_query = (
        PredictionHistory.query
        .filter_by(patient_id=patient_id)
        .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
    )
    records = history_query.offset((page - 1) * per_page).limit(per_page).all()
    response = jsonify({
        "patient_id": patient_id,
        "page": page,
        "per_page": per_page,
        "total": history_query.count(),
        "history": [
            {
                "id": item.id,
                "prediction": item.prediction,
                "probability": item.probability,
                "created_at": item.created_at.replace(tzinfo=timezone.utc).isoformat(),
                "display_time": format_prediction_timestamp(item.created_at),
                "source": item.source,
            }
            for item in records
        ],
    })
    response.headers["Cache-Control"] = "no-store"
    return response
@app.route(
    "/api/patient/predictions",
    methods=["GET"],
)
def patient_predictions_api():
    """Return the authenticated patient's latest prediction and paginated history.

    Includes latest prediction and recent history with cache control.
    """
    if not patient_required():
        return jsonify({"error": "Patient authentication required."}), 403

    patient_id = session.get("patient_id")
    # latest prediction
    latest = (
        Prediction.query
        .filter_by(patient_id=patient_id)
        .order_by(Prediction.created_at.desc())
        .first()
    )
    # paginated history
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = min(max(request.args.get("per_page", 20, type=int), 1), 100)
    history_query = (
        PredictionHistory.query
        .filter_by(patient_id=patient_id)
        .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
    )
    records = history_query.offset((page - 1) * per_page).limit(per_page).all()
    total = history_query.count()

    history_list = []
    for index, item in enumerate(records):
        history_list.append({
            "id": item.id,
            "prediction": item.prediction,
            "probability": item.probability,
            "created_at": item.created_at.replace(tzinfo=timezone.utc).isoformat(),
            "display_time": format_prediction_timestamp(item.created_at),
            "source": item.source,
            "is_latest": (page == 1 and index == 0),
        })

    payload = {
        "patient_id": patient_id,
        "latest": None,
        "history": history_list,
        "page": page,
        "per_page": per_page,
        "total": total,
    }
    if latest:
        payload["latest"] = {
            "id": latest.id,
            "prediction": latest.prediction,
            "probability": latest.probability,
            "created_at": latest.created_at.replace(tzinfo=timezone.utc).isoformat(),
            "display_time": format_prediction_timestamp(latest.created_at),
            "source": "system_generated",
        }
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store"
    return response


# ============================================================
# LOGOUT
# ============================================================

@app.route(
    "/logout",
    methods=["GET"]
)
def logout():
    if current_role() == "doctor":
        audit_event("DOCTOR_LOGOUT", session.get("user_id"), "SUCCESS")
    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# SECURITY HEADERS
# ============================================================

@app.before_request
def enforce_session_timeout():
    if not is_logged_in():
        return None
    last_activity = session.get("last_activity", time.time())
    if time.time() - last_activity > app.config["SESSION_TIMEOUT_SECONDS"]:
        if current_role() == "doctor":
            audit_event("SESSION_EXPIRED", session.get("user_id"), "DENIED")
        session.clear()
        return redirect(url_for("login"))
    session["last_activity"] = time.time()
    return None

@app.after_request
def add_security_headers(response):

    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, "
        "post-check=0, pre-check=0, max-age=0"
    )

    response.headers["Pragma"] = "no-cache"

    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"

    return response


# ============================================================
# APPLICATION START
# ============================================================

if not os.getenv("VERCEL"):
    from prediction_scheduler import start_prediction_scheduler
    start_prediction_scheduler(app, reloader_guard=True)

    print("\n========================================")
    print("   TISSUE VIABILITY PREDICTION SYSTEM")
    print("========================================")


    print("\n========================================")
    print("   TISSUE VIABILITY PREDICTION SYSTEM")
    print("========================================")

    print(
        f"Database: {DATABASE_PATH}"
    )

    print(
        f"Database exists: "
        f"{os.path.exists(DATABASE_PATH)}"
    )

    print(
        f"ML Model: "
        f"{os.path.join(BASE_DIR, 'model', 'tissue_viability_model.pkl')}"
    )

    print(
        f"ML Model available: "
        f"{MODEL_AVAILABLE}"
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print("========================================\n")

    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    use_reloader_option = os.environ.get("FLASK_USE_RELOADER", str(debug_mode)).lower() == "true"

    app.run(
        debug=debug_mode,
        use_reloader=use_reloader_option,
    )