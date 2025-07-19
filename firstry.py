# eda.py

import pandas as pd
import numpy as np

# 🔹 Διάβασε το CSV σου
df = pd.read_csv('Myocardial infarction complications Database.csv') 

# 🔍 Εμφάνισε βασικές πληροφορίες
print("🔍 Dataset shape:", df.shape)
print("\n📄 Columns:\n", df.columns)
print("\n🔎 First 5 rows:\n", df.head())

# ❓ Έλεγξε για missing values
print("\n❓ Missing values per column:\n", df.isnull().sum())

# ℹ️ Δες τύπους δεδομένων
print("\nℹ️ Data types:\n", df.dtypes)

# 📊 Στατιστική περίληψη
print("\n📊 Statistical summary:\n", df.describe())
