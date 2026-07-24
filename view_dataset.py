import pandas as pd

# Load dataset
df = pd.read_csv("dataset/tissue_viability.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)