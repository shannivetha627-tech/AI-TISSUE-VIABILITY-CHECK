import base64
import binascii


def detect_single_face(image_data):
    """Return a safe result for one centered face in a camera frame."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False, "Camera verification is unavailable on this server."

    try:
        encoded = image_data.split(",", 1)[-1]
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
    except (ValueError, binascii.Error, TypeError):
        return False, "The camera frame could not be read."

    if image is None:
        return False, "The camera frame could not be read."

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return False, "No face detected. Please position your face inside the frame."
    if len(faces) > 1:
        return False, "Multiple faces detected. Please ensure only one person is visible."

    x, _, width, _ = faces[0]
    center = x + width / 2
    if not image.shape[1] * 0.2 < center < image.shape[1] * 0.8:
        return False, "Face detected. Please center your face inside the frame."
    return True, "Face detected."
