import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Load Dataset
# ===========================
df = pd.read_csv("dataset/tissue_viability.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

# Shape
print("\nDataset Shape:")
print(df.shape)

# Columns
print("\nColumn Names:")
print(df.columns.tolist())

# Data Types
print("\nData Types:")
print(df.dtypes)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# First 5 Rows
print("\nFirst 5 Rows:")
print(df.head())

# ===========================
# Correlation Heatmap
# ===========================

numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()

# ===========================
# Risk Level Distribution
# ===========================

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Risk_Level")
plt.title("Risk Level Distribution")
plt.show()

# ===========================
# Tissue Viability Distribution
# ===========================

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Tissue_Viability")
plt.title("Tissue Viability Distribution")
plt.show()