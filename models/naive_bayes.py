import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("dataset/tissue_viability.csv")

# Features
X = df.drop("Tissue_Viability", axis=1)

# Target
y = df["Tissue_Viability"]

# Label Encoding
encoder = LabelEncoder()

for col in X.select_dtypes(include=["object","string"]).columns:
    X[col] = encoder.fit_transform(X[col].astype(str))

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Naive Bayes Model
model = GaussianNB()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nNaive Bayes Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, prediction))