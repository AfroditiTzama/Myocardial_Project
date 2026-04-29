# eda.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "myocardial_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eda_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nStatistical summary:\n", df.describe())

# Missing values heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "missing_values_heatmap.png"), dpi=300, bbox_inches="tight")
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(14, 10))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300, bbox_inches="tight")
plt.show()

# Target distribution
target_col = "LET_IS"

if target_col in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df[target_col])
    plt.title(f"Distribution of {target_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{target_col}_distribution.png"), dpi=300, bbox_inches="tight")
    plt.show()

# Histograms of numerical features
numeric_df.hist(figsize=(18, 15), bins=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "numerical_histograms.png"), dpi=300, bbox_inches="tight")
plt.show()

# Boxplots for outlier detection
for col in numeric_df.columns[:10]:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{col}.png"), dpi=300, bbox_inches="tight")
    plt.show()

print("\nEDA completed successfully.")
print(f"Graphs saved in: {OUTPUT_DIR}")