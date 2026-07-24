import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature Scaling Completed Successfully!")

print("Training Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)