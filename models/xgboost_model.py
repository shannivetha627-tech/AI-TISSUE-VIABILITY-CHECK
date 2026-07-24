import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv("dataset/tissue_viability.csv")

# Features and Target
X = df.drop("Tissue_Viability", axis=1)
y = df["Tissue_Viability"]

# Encode categorical columns
encoder = LabelEncoder()

for column in X.select_dtypes(include=["object", "string"]).columns:
    X[column] = encoder.fit_transform(X[column].astype(str))

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# XGBoost Model
model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nXGBoost Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, prediction))