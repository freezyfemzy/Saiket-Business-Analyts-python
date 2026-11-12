import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("Telco_Customer_Churn_Cleaned.csv")

# 2. Standardize column names
cols = []

for c in df.columns:
    c = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", c.strip())
    cols.append(
        re.sub(r"[^a-zA-Z0-9]+", "_", c).strip("_").lower()
    )

df.columns = cols

# 3. Handle missing Total Charges
df["total_charges"] = pd.to_numeric(
    df["total_charges"].astype(str).str.strip().replace("", np.nan),
    errors="coerce"
)

df.loc[
    df["total_charges"].isna() & (df["tenure"] == 0),
    "total_charges"
] = 0

# 4. Handle other missing values
for c in df.columns:
    if df[c].isna().any():
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0])

# 5. Remove duplicate records
df = df.drop_duplicates()

# 6. Save cleaned dataset
df.to_csv("Telco_Customer_Churn_Cleaned.csv", index=False)

# 7. Inspect the dataset
print("FIRST 10 ROWS")
print(df.head(10))

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isna().sum())

# 8. Statistical analysis
print("\nMEAN AND MEDIAN")
print(
    df[["tenure", "monthly_charges", "total_charges"]]
    .agg(["mean", "median"])
)

# 9. Churn and retention percentages
print("\nCHURN PERCENTAGES")
print(df["churn"].value_counts(normalize=True) * 100)

# 10. Customer segmentation by tenure
df["tenure_category"] = pd.cut(
    df["tenure"],
    [-1, 12, 36, 1000],
    labels=["0-12 months", "13-36 months", "37+ months"]
)

print("\nCUSTOMER SEGMENTATION")
print(
    df.groupby("tenure_category", observed=False).agg(
        customers=("customer_id", "count"),
        avg_monthly_charges=("monthly_charges", "mean"),
        churn_rate=("churn", lambda s: (s == "Yes").mean() * 100)
    )
)

# 11. Churn analysis by customer characteristics
for c in ["gender", "senior_citizen", "payment_method", "contract"]:
    print(f"\nCHURN RATE BY {c.upper()}")
    print(
        df.groupby(c)["churn"].apply(
            lambda s: (s == "Yes").mean() * 100
        )
    )

print("\nANALYSIS COMPLETED SUCCESSFULLY!")
