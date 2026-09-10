import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("dataset/tissue_viability.csv")


# ----------------------------
# Features and Target
# ----------------------------

# Patient_ID and Risk_Level are kept in the application,
# but are NOT used as ML features.

X = df.drop(
    ["Patient_ID", "Risk_Level", "Tissue_Viability"],
    axis=1
)

y = df["Tissue_Viability"]


# ----------------------------
# Encode Categorical Features
# ----------------------------

encoders = {}

for col in X.select_dtypes(
    include=["object", "string"]
).columns:

    encoder = LabelEncoder()

    X[col] = encoder.fit_transform(
        X[col].astype(str)
    )

    encoders[col] = encoder


# ----------------------------
# Encode Target
# ----------------------------

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)


# ----------------------------
# Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ----------------------------
# Final Tuned Random Forest
# ----------------------------

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)


# ----------------------------
# Train Model
# ----------------------------

model.fit(X_train, y_train)


# ----------------------------
# Save Model
# ----------------------------

saved_data = {
    "model": model,
    "feature_columns": X.columns.tolist(),
    "encoders": encoders,
    "target_encoder": target_encoder
}

joblib.dump(
    saved_data,
    "saved_models/random_forest.pkl"
)


print("\n======================================")
print("FINAL RANDOM FOREST MODEL SAVED")
print("======================================")

print("\nModel Parameters:")
print("n_estimators      :", 50)
print("max_depth         :", 10)
print("min_samples_split :", 2)

print("\nFeatures Used:")
for feature in X.columns:
    print("-", feature)

print("\nSaved Location:")
print("saved_models/random_forest.pkl")

print("\nModel saved successfully!")