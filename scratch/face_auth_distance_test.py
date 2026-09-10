import json, os, sys, math
sys.path.append(r'C:/Users/Nivetha/OneDrive/Desktop/Tissue_Viability_Project')
from face_auth.face_verification import serialize_embedding, deserialize_embedding, verify_face, extract_face_embedding

# Helper to compute mean absolute difference (same as verify_face)
def mean_abs_diff(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

# Create dummy embedding (all zeros)
embed_zero = [0.0] * (468 * 3)
ser_zero = serialize_embedding(embed_zero)
emb_zero = deserialize_embedding(ser_zero)
print('Zero embedding deserialized length', len(emb_zero))

# Create different embedding (all 0.1)
embed_one = [0.1] * (468 * 3)
ser_one = serialize_embedding(embed_one)
emb_one = deserialize_embedding(ser_one)
print('One embedding deserialized length', len(emb_one))

# Compute distance between identical embeddings
dist_self = mean_abs_diff(emb_zero, emb_zero)
print('Distance self (zero vs zero):', dist_self)

# Compute distance between different embeddings
dist_diff = mean_abs_diff(emb_zero, emb_one)
print('Distance diff (zero vs one):', dist_diff)

# Show that distance matches the threshold logic (default threshold from app config is 0.2)
threshold = 0.2
print('Self passes threshold?', dist_self <= threshold)
print('Diff passes threshold?', dist_diff <= threshold)
