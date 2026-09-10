try:
    import seaborn as sns
except ImportError:
    sns = None

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/tissue_viability.csv")


# ==========================================
# 1. TISSUE VIABILITY DISTRIBUTION
# ==========================================

plt.figure(figsize=(7, 5))

df["Tissue_Viability"].value_counts().plot(kind="bar")

plt.title("Tissue Viability Distribution")
plt.xlabel("Tissue Viability")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ==========================================
# PREPARE DATA FOR RANDOM FOREST
# ==========================================

X = df.drop(
    ["Patient_ID", "Risk_Level", "Tissue_Viability"],
    axis=1
)

y = df["Tissue_Viability"]


# Encode categorical features
encoders = {}

for col in X.select_dtypes(
    include=["object", "string"]
).columns:

    encoder = LabelEncoder()

    X[col] = encoder.fit_transform(
        X[col].astype(str)
    )

    encoders[col] = encoder


# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 2. FEATURE IMPORTANCE
# ==========================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)


plt.figure(figsize=(9, 7))

importance.plot(kind="barh")

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Clinical Features")

plt.tight_layout()
plt.show()


# ==========================================
# 3. CONFUSION MATRIX
# ==========================================

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(6, 5))

if sns is not None:
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=target_encoder.classes_,
        yticklabels=target_encoder.classes_
    )
else:
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(target_encoder.classes_)), target_encoder.classes_)
    plt.yticks(range(len(target_encoder.classes_)), target_encoder.classes_)
    for i in range(len(target_encoder.classes_)):
        for j in range(len(target_encoder.classes_)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


print("\n======================================")
print("GRAPHS GENERATED SUCCESSFULLY")
print("======================================")

print("\nGenerated Graphs:")
print("1. Tissue Viability Distribution")
print("2. Random Forest Feature Importance")
print("3. Confusion Matrix")