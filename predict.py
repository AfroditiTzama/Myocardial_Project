# predict_new_patient.py

import pandas as pd
import pickle

# Φόρτωση μοντέλου, scaler, feature names
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

feature_names = pd.read_csv('feature_names.csv').squeeze().tolist()

# Δημιουργία dictionary νέου ασθενούς με όλες τις στήλες
new_patient = {col: 0 for col in feature_names}

# ➡️ ➡️ ΕΔΩ ΣΥΜΠΛΗΡΩΝΕΙΣ ΤΑ ΠΡΑΓΜΑΤΙΚΑ ΔΕΔΟΜΕΝΑ ΤΟΥ ΑΣΘΕΝΗ
# Παράδειγμα:
new_patient['AGE'] = 75
new_patient['SEX'] = 1

# Αν υπάρχουν one-hot encoded στήλες όπως 'INF_ANAM_angina'
# new_patient['INF_ANAM_angina'] = 1

# Μετατροπή σε DataFrame
new_patient_df = pd.DataFrame([new_patient])

# Τυποποίηση
new_patient_scaled = scaler.transform(new_patient_df)

# Πρόβλεψη
prediction = model.predict(new_patient_scaled)
prediction_proba = model.predict_proba(new_patient_scaled)

print("🔮 Predicted complication class:", prediction[0])
print("📊 Prediction probabilities:", prediction_proba)
