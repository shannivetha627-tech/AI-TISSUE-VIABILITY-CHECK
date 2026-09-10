import json, os, sys, math
sys.path.append(r'C:/Users/Nivetha/OneDrive/Desktop/Tissue_Viability_Project')
from face_auth.face_verification import serialize_embedding, deserialize_embedding, verify_face, extract_face_embedding

# Dummy embedding (all zeros)
dummy = [0.0] * (468 * 3)
ser = serialize_embedding(dummy)
print('Serialized length', len(ser))
emb = deserialize_embedding(ser)
print('Deserialized ok', emb is not None and len(emb) == len(dummy))

# Monkey-patch extract_face_embedding to return dummy embedding for any frame
import face_auth.face_verification as fv

def mock_extract(image_data):
    return dummy, 'ok'

fv.extract_face_embedding = mock_extract

# Test self-comparison (should be verified)
frames = ['frame1', 'frame2', 'frame3', 'frame4', 'frame5']
verified, msg = verify_face(frames, ser, threshold=0.5)
print('Self-compare verified', verified, msg)

# Test different embedding (should fail)
other = [0.1] * (468 * 3)
ser_other = serialize_embedding(other)
verified2, msg2 = verify_face(frames, ser_other, threshold=0.05)
print('Different verify', verified2, msg2)
