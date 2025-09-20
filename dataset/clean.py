# clean_water_quality_data.py
import pandas as pd
import numpy as np

################ Step 1: Load CSV ################
# Replace this with the exact path to your file
exact_path = "c:/Users/kisha/OneDrive/Desktop/SIH NEW/dataset/water_potability.csv"
df = pd.read_csv(exact_path)

print("Before cleaning:")
print("Missing values:")
print(df.isnull().sum())
print("\nDataset shape:", df.shape)

################ Step 2: Handle Missing Values ################
print("\nFeature distributions:")
print(df[["ph", "TDS", "Hardness", "Turbidity"]].describe())

# Check for each feature and use appropriate imputation
for column in ["ph", "TDS", "Hardness", "Turbidity"]:
    if df[column].isnull().sum() > 0:
        # For normal distributions, use mean; for skewed, use median
        if abs(df[column].skew()) < 1:  # relatively symmetric
            df[column].fillna(df[column].mean(), inplace=True)
            print(f"Filled missing {column} with mean: {df[column].mean():.2f}")
        else:  # skewed distribution
            df[column].fillna(df[column].median(), inplace=True)
            print(f"Filled missing {column} with median: {df[column].median():.2f}")

################ Step 3: Create Label (Safe / Not Safe) ################
def classify_water(row):
    if (6.5 <= row["ph"] <= 8.5) and (row["TDS"] <= 500) and (row["Hardness"] <= 300) and (row["Turbidity"] <= 5):
        return 1   # Safe
    else:
        return 0   # Not Safe

df["Safe"] = df.apply(classify_water, axis=1)

print("\nAfter cleaning:")
print("Missing values:")
print(df.isnull().sum())
print("\nClass distribution:")
print(df["Safe"].value_counts())

################ Step 4: Save Cleaned Data ################
df.to_csv("cleaned_water_quality.csv", index=False)
print("\n✅ Cleaned data saved as 'cleaned_water_quality.csv'")