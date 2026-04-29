import os
import pickle
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "myocardial_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Φόρτωση δεδομένων
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nMissing values per column:\n", df.isna().sum())


# 2. Διαχείριση missing values
num_cols = df.select_dtypes(include=["float64", "int64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# 3. Ορισμός στόχου
target = "LET_IS"

print(f"\nTarget '{target}' value counts:\n")
print(df[target].value_counts())


# 4. Προεπεξεργασία
X = df.drop(columns=[target, "ID"])
y = df[target].copy()

# Merge rare classes
y = y.replace({
    2: 2,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    7: 2
})

X = pd.get_dummies(X)

feature_names = X.columns


# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. Τυποποίηση δεδομένων
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 7. SMOTE
smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nOriginal training set shape:")
print(Counter(y_train))

print("\nResampled training set shape:")
print(Counter(y_train_res))


# 8. Εκπαίδευση μοντέλου
model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y)),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train_res, y_train_res)


# 9. Αξιολόγηση μοντέλου
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n===== Model Evaluation =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

try:
    roc_auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="weighted"
    )
    print(f"ROC-AUC:   {roc_auc:.4f}")
except Exception as e:
    print("ROC-AUC could not be calculated:", e)

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred, zero_division=0))


# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()


# 11. Cross-validation
cv_scores = cross_val_score(
    model,
    X_train_res,
    y_train_res,
    cv=5,
    scoring="f1_weighted"
)

print("\n===== Cross-validation =====")
print("F1 weighted scores:", cv_scores)
print(f"Mean CV F1-score: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")


# 12. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Top 20 Feature Importances")

sns.barplot(
    x=importances[indices][:20],
    y=feature_names[indices][:20],
    palette="viridis"
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=300, bbox_inches="tight")
plt.show()


# 13. Αποθήκευση μοντέλου, scaler και feature names
with open(os.path.join(MODEL_DIR, "trained_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

pd.Series(feature_names).to_csv(
    os.path.join(MODEL_DIR, "feature_names.csv"),
    index=False
)

print("\n Model, scaler, feature names, confusion matrix and feature importance saved successfully.")