import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/tissue_viability.csv")

print("Dataset Loaded Successfully!")
print(df.head())

print("\nDataset Shape:")
print(df.shape)