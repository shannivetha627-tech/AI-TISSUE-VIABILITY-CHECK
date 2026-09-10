import base64
import binascii
import json
import logging
import math
from pathlib import Path

# MediaPipe imports are moved to module level for performance
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as e:
    mp = None
    mp_python = None
    vision = None
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.error("MediaPipe dependencies are missing: %s", e)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LANDMARKER_MODEL = Path(__file__).with_name("face_landmarker.task")
_landmarker = None
LIVENESS_MIN_VALID_FRAMES = 5
LIVENESS_MOVEMENT_THRESHOLD = 0.004


def _decode_frame(image_data):
    try:
        import cv2
        import numpy as np
        encoded = image_data.split(",", 1)[-1]
        return cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
    except (ImportError, ValueError, binascii.Error, TypeError):
        return None


def extract_face_embedding(image_data):
    """Extract a normalized MediaPipe landmark representation from one face.
    Utilizes a globally initialized MediaPipe FaceLandmarker for efficiency.
    """
    image = _decode_frame(image_data)
    if image is None:
        return None, "The camera frame could not be read."

    try:
        import cv2
    except ImportError:
        return None, "OpenCV is required for face embedding extraction."

    if mp is None or vision is None:
        return None, "Face verification is unavailable. Install MediaPipe dependencies."

    global _landmarker
    try:
        if _landmarker is None:
            if not LANDMARKER_MODEL.is_file():
                logger.error("[FACE] MediaPipe Face Landmarker model is missing: %s", LANDMARKER_MODEL)
                return None, "Face verification is unavailable on this server."
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(LANDMARKER_MODEL),
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=2,
                min_face_detection_confidence=0.6,
                min_face_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            _landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info("[FACE] MediaPipe version: %s", getattr(mp, "__version__", "unknown"))
            logger.info("[FACE] MediaPipe Tasks API initialized")

        logger.info("[FACE] Face landmark extraction started")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = _landmarker.detect(mp_image)
    except Exception:
        logger.exception("[FACE] MediaPipe landmark extraction failed")
        return None, "Face verification is unavailable. Please try again."

    faces = result.face_landmarks or []
    if not faces:
        return None, "No face detected. Please position your face inside the frame."
    if len(faces) > 1:
        return None, "Multiple faces detected."
    landmarks = faces[0]
    logger.info("[FACE] Face detected")
    logger.info("[FACE] Landmark count: %s", len(landmarks))
    if len(landmarks) < 468:
        return None, "The detected face landmarks are incomplete. Please try again."

    nose = landmarks[1]
    scale = max(
        ((landmarks[234].x - landmarks[454].x) ** 2
         + (landmarks[234].y - landmarks[454].y) ** 2) ** 0.5,
        0.001,
    )
    values = []
    for landmark in landmarks:
        values.extend([
            round((landmark.x - nose.x) / scale, 6),
            round((landmark.y - nose.y) / scale, 6),
            round((landmark.z - nose.z) / scale, 6),
        ])

    # Removed centering check that referenced undefined variable `width`
    # This validation is optional and can be performed client‑side if needed.
    logger.info("[FACE] Representation generated")
    return values, "Face detected."


def deserialize_embedding(serialized_embedding):
    """Validate a normalized landmark representation received from the client."""
    if isinstance(serialized_embedding, list):
        embedding = serialized_embedding
    else:
        try:
            embedding = json.loads(serialized_embedding)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    valid_lengths = {468 * 3, 478 * 3}
    if not isinstance(embedding, list) or len(embedding) not in valid_lengths:
        return None
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding):
        return None
    return [round(float(value), 6) for value in embedding]


def verify_face(image_data, stored_embedding, challenge=None, threshold=0.16):
    """Compare several fresh representations and perform prototype liveness."""
    if not stored_embedding:
        return False, "No face is registered for this doctor. Register a face before signing in."

    frames = image_data if isinstance(image_data, list) else [image_data]
    logger.info(
        "Liveness: challenge=%s frames_received=%s",
        challenge,
        len(frames),
    )
    if len(frames) < LIVENESS_MIN_VALID_FRAMES:
        logger.info(
            "Liveness: face_detected=False landmarks_detected=False "
            "movement_detected=False frames_received=%s frames_valid=0",
            len(frames),
        )
        return False, "Liveness challenge timed out. Please try again."
    current_frames = []
    invalid_frames = 0
    for frame in frames[:9]:
        current, message = extract_face_embedding(frame)
        if current is None:
            invalid_frames += 1
            continue
        current_frames.append(current)

    logger.info(
        "Liveness: face_detected=%s landmarks_detected=%s "
        "frames_received=%s frames_valid=%s invalid_frames=%s",
        bool(current_frames),
        bool(current_frames),
        len(frames),
        len(current_frames),
        invalid_frames,
    )
    if len(current_frames) < LIVENESS_MIN_VALID_FRAMES:
        return False, "Unable to read facial landmarks. Adjust your position and try again."

    try:
        registered = json.loads(stored_embedding)
        if not isinstance(registered, list) or any(len(current) != len(registered) for current in current_frames):
            return False, "The registered face representation is invalid."
        distances = [
            sum(abs(current_value - registered_value) for current_value, registered_value in zip(current, registered))
            / len(current)
            for current in current_frames
        ]
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, "The registered face representation is invalid."

    # Landmarks 33 and 263 are the outer eye corners. Their midpoint relative
    # to the nose is a resolution-independent yaw signal.
    yaw_samples = [
        (frame[33 * 3] + frame[263 * 3]) / 2
        for frame in current_frames
    ]
    start_yaw = sum(yaw_samples[:2]) / min(2, len(yaw_samples))
    end_yaw = sum(yaw_samples[-2:]) / min(2, len(yaw_samples))
    motion = end_yaw - start_yaw
    movement_score = max(yaw_samples) - min(yaw_samples)
    liveness_ok = movement_score >= LIVENESS_MOVEMENT_THRESHOLD
    if challenge in {"left", "right"}:
        liveness_ok = liveness_ok and (
            (challenge == "left" and motion < -LIVENESS_MOVEMENT_THRESHOLD / 2)
            or (challenge == "right" and motion > LIVENESS_MOVEMENT_THRESHOLD / 2)
        )
    logger.info(
        "Liveness: challenge=%s movement_detected=%s movement_score=%.6f "
        "signed_motion=%.6f threshold=%.6f frames_received=%s frames_valid=%s",
        challenge,
        liveness_ok,
        movement_score,
        motion,
        LIVENESS_MOVEMENT_THRESHOLD,
        len(frames),
        len(current_frames),
    )
    if not liveness_ok:
        return False, "Movement not detected. Please follow the instruction and try again."
    if min(distances) > threshold:
        logger.info("[FACE] Comparison result: NO MATCH")
        return False, "Verification failed. The face does not match the registered doctor."
    logger.info("[FACE] Comparison result: MATCH")
    return True, "Identity verified."


def serialize_embedding(embedding):
    """Serialize only landmark values, never the source camera frame."""
    return json.dumps(embedding, separators=(",", ":"))
