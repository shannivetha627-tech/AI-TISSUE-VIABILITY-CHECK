from flask import Blueprint, render_template, request, session
from database.models import Patient

doctor_bp = Blueprint(
    "doctor",
    __name__,
    url_prefix="/doctor"
)


@doctor_bp.route("/", methods=["GET", "POST"])
def doctor_dashboard():

    if session.get("role", "").lower() != "doctor":
        return """
        <h1>Access Denied</h1>
        <p>Only doctors can access the doctor dashboard.</p>
        """, 403

    patient = None
    searched_id = ""

    if request.method == "POST":

        searched_id = request.form.get(
            "patient_id",
            ""
        ).strip()

        if searched_id:
            patient = Patient.query.filter_by(
                patient_id=searched_id
            ).first()

    return render_template(
        "doctor_dashboard.html",
        patient=patient,
        searched_id=searched_id
    )