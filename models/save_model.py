import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("dataset/tissue_viability.csv")

# ==========================
# Features and Target
# ==========================
X = df.drop(["Patient_ID", "Tissue_Viability"], axis=1)
y = df["Tissue_Viability"]

# ==========================
# Encode Categorical Columns
# ==========================
encoder = LabelEncoder()

categorical_columns = X.select_dtypes(include=["object", "string"]).columns

for col in categorical_columns:
    X[col] = encoder.fit_transform(X[col].astype(str))

# Encode Target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# ==========================
# Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Train Random Forest
# ==========================
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ==========================
# Create Folder if Missing
# ==========================
os.makedirs("saved_models", exist_ok=True)

# ==========================
# Save Model
# ==========================
joblib.dump(model, "saved_models/random_forest.pkl")
joblib.dump(target_encoder, "saved_models/target_encoder.pkl")

print("===================================")
print("Model Saved Successfully!")
print("Location : saved_models/")
print("===================================")