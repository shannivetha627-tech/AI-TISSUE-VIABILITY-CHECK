import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ----------------------------
# Load Dataset
# ----------------------------
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

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------
# Models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", random_state=42)
}

print("\n========== MODEL COMPARISON ==========\n")

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results.append([name, accuracy])

# Display Results
result_df = pd.DataFrame(results, columns=["Algorithm", "Accuracy"])

print(result_df)

# Best Model
best_model = result_df.loc[result_df["Accuracy"].idxmax()]

print("\n===================================")
print("Best Model :", best_model["Algorithm"])
print("Accuracy   :", round(best_model["Accuracy"]*100,2),"%")
print("===================================")