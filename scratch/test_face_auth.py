# Automated backend tests for face registration and verification
import json
import os
import sys
import base64
from types import SimpleNamespace

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

from app import app, db
from database.models import User
import face_auth.face_verification as fv

# Monkey‑patch MediaPipe dependent functions to return a deterministic embedding
DUMMY_EMBEDDING = [float(i % 10) for i in range(468 * 3)]  # length 1404


def dummy_extract(image_data):
    # Ignore image_data, return the dummy embedding and a message
    return DUMMY_EMBEDDING, "dummy face"

import app as app_module
fv.extract_face_embedding = dummy_extract
app_module.extract_face_embedding = dummy_extract
# Disable liveness strictness for testing
fv.LIVENESS_MOVEMENT_THRESHOLD = 0.0
fv.LIVENESS_MIN_VALID_FRAMES = 5

def run_tests():
    with app.app_context():
        # Use an in‑memory SQLite DB for isolation
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        app.config["TESTING"] = True
        db.create_all()

        # Clean up existing test users if any
        User.query.filter(User.username.in_(["docA", "docB"])).delete(synchronize_session=False)
        db.session.commit()

        # Create two doctor users
        doc_a = User(username="docA", password="pwd", role="doctor")
        doc_b = User(username="docB", password="pwd", role="doctor")
        db.session.add_all([doc_a, doc_b])
        db.session.commit()

        # Helper to set session variables for pending doctor
        client = app.test_client()
        import time
        with client.session_transaction() as sess:
            sess["pending_doctor_id"] = doc_a.id
            sess["pending_doctor_username"] = doc_a.username
            sess["pending_doctor_role"] = doc_a.role
            sess["face_challenge"] = "left"
            sess["face_auth_started"] = time.time()
        # ---------- Registration ----------
        response = client.post(
            "/doctor/register-face",
            json={"image_data": "data:image/jpeg;base64," + base64.b64encode(b"dummy").decode()},
        )
        assert response.status_code == 200, f"Registration failed: {response.data}"
        data = response.get_json()
        assert data.get("success"), "Registration success flag missing"
        stored_user = db.session.get(User, doc_a.id)
        assert stored_user.face_embedding is not None, "Embedding not stored"
        # Validate JSON and numeric content without exposing values
        try:
            emb = json.loads(stored_user.face_embedding)
        except Exception as e:
            raise AssertionError("Stored embedding is not valid JSON") from e
        assert isinstance(emb, list) and len(emb) == 468 * 3, "Embedding length mismatch"
        assert all(isinstance(v, (int, float)) for v in emb), "Embedding contains non‑numeric values"

        # ---------- Verification (valid) ----------
        with client.session_transaction() as sess:
            sess["pending_doctor_id"] = doc_a.id
            sess["pending_doctor_username"] = doc_a.username
            sess["pending_doctor_role"] = doc_a.role
            sess["face_challenge"] = None
            sess["face_auth_started"] = time.time()
        # Provide 5 identical dummy frames
        frames = ["data:image/jpeg;base64," + base64.b64encode(b"frame") .decode()] * 5
        response = client.post(
            "/doctor/verify-face",
            data={"image_data": frames},
        )
        assert response.status_code == 200, f"Verification failed: {response.data}"
        result = response.get_json()
        assert result.get("verified"), "Verification should succeed with identical embedding"

        # ---------- Verification (cross‑doctor) ----------
        # Register a different embedding for doctor B
        client_b = app.test_client()
        with client_b.session_transaction() as sess:
            sess["pending_doctor_id"] = doc_b.id
            sess["pending_doctor_username"] = doc_b.username
            sess["pending_doctor_role"] = doc_b.role
            sess["face_challenge"] = None
            sess["face_auth_started"] = time.time()
        # Register B with a different dummy embedding (offset values)
        dummy_b = lambda img: ([v + 1.0 for v in DUMMY_EMBEDDING], "dummy B")
        fv.extract_face_embedding = dummy_b
        app_module.extract_face_embedding = dummy_b
        response = client_b.post(
            "/doctor/register-face",
            json={"image_data": "data:image/jpeg;base64," + base64.b64encode(b"dummyB").decode()},
        )
        assert response.status_code == 200, "Doctor B registration failed"
        # Restore original dummy for verification frames (A's embedding)
        fv.extract_face_embedding = dummy_extract
        app_module.extract_face_embedding = dummy_extract
        # Attempt to verify B using A's frames (should fail)
        with client_b.session_transaction() as sess:
            sess["pending_doctor_id"] = doc_b.id
            sess["pending_doctor_username"] = doc_b.username
            sess["pending_doctor_role"] = doc_b.role
            sess["face_challenge"] = None
            sess["face_auth_started"] = time.time()
        response = client_b.post(
            "/doctor/verify-face",
            data={"image_data": frames},
        )
        result = response.get_json()
        assert not result.get("verified"), "Cross‑doctor verification should fail"

        # ---------- Distance sanity check ----------
        # Directly call internal distance calculation using identical embedding
        stored_json = stored_user.face_embedding
        # Simulate verification path without liveness (already disabled)
        distances = []
        for _ in range(5):
            current, _ = dummy_extract("ignored")
            distances.append(
                sum(abs(c - s) for c, s in zip(current, json.loads(stored_json))) / len(current)
            )
        assert all(abs(d) < 1e-5 for d in distances), f"Identical embeddings must yield zero distance, got {distances}"
        # Verify threshold logic
        threshold = app.config["FACE_MATCH_THRESHOLD"]
        assert threshold >= 0, "Threshold must be non‑negative"
        assert distances[0] <= threshold, "Zero distance should pass threshold"

        print("AUTOMATED TESTS PASSED")

if __name__ == "__main__":
    run_tests()
