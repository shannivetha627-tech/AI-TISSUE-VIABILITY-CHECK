import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import required models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("dataset/tissue_viability.csv")


# ----------------------------
# Features and Target
# ----------------------------
X = df.drop(
    ["Patient_ID", "Risk_Level", "Tissue_Viability"],
    axis=1
)
y = df["Tissue_Viability"]


# ----------------------------
# Encode Categorical Columns
# ----------------------------
encoder = LabelEncoder()

for col in X.select_dtypes(include=["object", "string"]).columns:
    X[col] = encoder.fit_transform(X[col].astype(str))


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
# Required Models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}


# ----------------------------
# Model Comparison
# ----------------------------
print("\n========== MODEL COMPARISON ==========\n")

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results.append([name, accuracy])


# ----------------------------
# Display Results
# ----------------------------
result_df = pd.DataFrame(
    results,
    columns=["Algorithm", "Accuracy"]
)

result_df["Accuracy"] = result_df["Accuracy"] * 100

print(result_df.to_string(index=False))


# ----------------------------
# Find Best Model
# ----------------------------
best_model = result_df.loc[
    result_df["Accuracy"].idxmax()
]

print("\n===================================")
print("Best Model :", best_model["Algorithm"])
print("Accuracy   :", round(best_model["Accuracy"], 2), "%")
print("===================================")