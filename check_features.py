import pandas as pd

df = pd.read_csv("dataset/tissue_viability.csv")

features = [
    "SpO2",
    "Blood_Flow",
    "Capillary_Refill_Time",
    "Diabetes",
    "Smoking",
    "Tissue_Temperature",
    "Surgery_Duration",
    "Perfusion_Index"
]

print("\n========== FEATURE ANALYSIS ==========\n")

for feature in features:

    print("\n-----------------------------------")
    print("Feature:", feature)
    print("-----------------------------------")

    if df[feature].dtype in ["object", "string"]:
        print(pd.crosstab(
            df[feature],
            df["Tissue_Viability"]
        ))

    else:
        print(
            df.groupby("Tissue_Viability")[feature]
            .agg(["min", "mean", "max"])
        )