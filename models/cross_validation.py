import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("dataset/tissue_viability.csv")

# Features and Target
X = df.drop("Tissue_Viability", axis=1)
y = df["Tissue_Viability"]

# Encode categorical columns
encoder = LabelEncoder()

for col in X.select_dtypes(include=["object", "string"]).columns:
    X[col] = encoder.fit_transform(X[col].astype(str))

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Random Forest Model
model = RandomForestClassifier(random_state=42)

# 5-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5)

print("\nCross Validation Scores:")
print(scores)

print("\nAverage Accuracy:", scores.mean())