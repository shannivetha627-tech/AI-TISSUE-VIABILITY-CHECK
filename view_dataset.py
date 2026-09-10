import pandas as pd

df = pd.read_csv("dataset/tissue_viability.csv")

print("\n========== DATASET INFORMATION ==========\n")

print("Number of Samples:", df.shape[0])
print("Number of Features:", df.shape[1] - 1)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTissue Viability Distribution:")
print(df["Tissue_Viability"].value_counts())

print("\nTissue Viability Percentage:")
print(df["Tissue_Viability"].value_counts(normalize=True) * 100)

print("\nFirst 5 Rows:")
print(df.head())

print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe(include="all").transpose())