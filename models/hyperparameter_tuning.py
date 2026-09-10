import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("dataset/tissue_viability.csv")


# ----------------------------
# Features and Target
# ----------------------------
X = df.drop("Tissue_Viability", axis=1)
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
# Random Forest
# ----------------------------
rf = RandomForestClassifier(
    random_state=42
)


# ----------------------------
# Parameters for Tuning
# ----------------------------
parameters = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}


# ----------------------------
# Grid Search
# ----------------------------
grid = GridSearchCV(
    estimator=rf,
    param_grid=parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


# Train Grid Search
grid.fit(X_train, y_train)


# ----------------------------
# Results
# ----------------------------
print("\n========== HYPERPARAMETER TUNING ==========\n")

print("Best Parameters:")
print(grid.best_params_)

print("\nBest Cross Validation Accuracy:")
print(round(grid.best_score_ * 100, 2), "%")

print("\n============================================")
print("Tuned Random Forest Model Selected")
print("============================================")