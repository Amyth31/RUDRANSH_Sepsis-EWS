"""
Sepsis Risk Prediction Model - 6-Hour Early Warning System
Based on PhysioNet Sepsis Prediction Dataset (2019 Challenge)
Features: HR, MAP, Lactate, Creatinine, Platelets, Temperature, O2Sat, SBP
Engineered: 8 mean features, 8 trend features, 1 shock index = 17 total
"""

import numpy as np
import pandas as pd
import json
import pickle
import os
import glob
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV

np.random.seed(42)

# 1.  Load PhysioNet Dataset

DATASET_A = r"C:\Users\Amit\Pictures\kirk hammett\AIML_Project\Dataset.csv"

print("Loading real PhysioNet 2019 dataset ...")

files = glob.glob(os.path.join(DATASET_A, "*.psv"))
print(f"Found {len(files)} patient files ...")

all_records = []
skipped = 0

for fpath in files:
    try:
        pt = pd.read_csv(fpath, sep='|')

        cols_needed = ['HR', 'MAP', 'SBP', 'Temp', 'O2Sat',
                       'Lactate', 'Creatinine', 'Platelets', 'SepsisLabel']
        pt = pt[[c for c in cols_needed if c in pt.columns]].copy()

        pt = pt.ffill().bfill().fillna(pt.mean(numeric_only=True))

        if len(pt) < 6:
            skipped += 1
            continue

        window = pt.tail(6).reset_index(drop=True)
        t = np.arange(6)

        keys = ['HR', 'MAP', 'SBP', 'Temp', 'O2Sat',
                'Lactate', 'Creatinine', 'Platelets']

        row = {}
        valid = True
        for k in keys:
            if k not in window.columns:
                valid = False
                break
            s = window[k].values.astype(float)
            if np.any(np.isnan(s)):
                valid = False
                break
            row[f'mean_{k}']  = float(np.mean(s))
            row[f'trend_{k}'] = float(np.polyfit(t, s, 1)[0])

        if not valid:
            skipped += 1
            continue

        row['shock_index'] = row['mean_HR'] / max(row['mean_SBP'], 1)

        sepsis_flag  = int(pt['SepsisLabel'].max()) if 'SepsisLabel' in pt.columns else 0
        mean_lactate = row.get('mean_Lactate', 0)

        if sepsis_flag == 1:
            row['label'] = 2
        elif mean_lactate > 2.0:
            row['label'] = 1
        else:
            row['label'] = 0

        all_records.append(row)

    except Exception:
        skipped += 1
        continue

print(f"Loaded : {len(all_records)} patients")
print(f"Skipped: {skipped} patients (insufficient data)")

df = pd.DataFrame(all_records)
print(f"\nClass distribution:")
print(f"  Low Risk    (0): {int(np.sum(df['label']==0))}")
print(f"  Medium Risk (1): {int(np.sum(df['label']==1))}")
print(f"  High Risk   (2): {int(np.sum(df['label']==2))}")

# 2.  Train & Evaluate

feature_cols = [c for c in df.columns if c != 'label']
X = df[feature_cols].values
y = df['label'].values

print(f"\nDataset shape: {X.shape}  |  Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GradientBoostingClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=25,
    tol=1e-4
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("\nRunning 5-fold stratified cross-validation ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gb_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc_ovr', n_jobs=-1)
rf_scores = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='roc_auc_ovr', n_jobs=-1)

print(f"\nGradient Boosting  CV AUC (OvR): {gb_scores.mean():.4f} +/- {gb_scores.std():.4f}")
print(f"Random Forest      CV AUC (OvR): {rf_scores.mean():.4f} +/- {rf_scores.std():.4f}")

print("\nTraining final Gradient Boosting model ...")
sample_w = compute_sample_weight('balanced', y)
model.fit(X_scaled, y, sample_weight=sample_w)
rf_model.fit(X_scaled, y)

split = int(0.8 * len(X_scaled))
X_test, y_test = X_scaled[split:], y[split:]

y_prob_gb  = model.predict_proba(X_test)
y_prob_rf  = rf_model.predict_proba(X_test)
y_prob_ens = 0.6 * y_prob_gb + 0.4 * y_prob_rf
y_pred_ens = np.argmax(y_prob_ens, axis=1)

auc_gb  = roc_auc_score(y_test, y_prob_gb,  multi_class='ovr')
auc_rf  = roc_auc_score(y_test, y_prob_rf,  multi_class='ovr')
auc_ens = roc_auc_score(y_test, y_prob_ens, multi_class='ovr')

print(f"\n{'='*50}")
print(f"  Gradient Boosting  AUC (test): {auc_gb:.4f}")
print(f"  Random Forest      AUC (test): {auc_rf:.4f}")
print(f"  Ensemble           AUC (test): {auc_ens:.4f}  <- FINAL MODEL")
print(f"{'='*50}")

print("\nClassification Report (Ensemble):")
print(classification_report(y_test, y_pred_ens,
      target_names=['Low Risk', 'Medium Risk', 'High Risk']))

importances = model.feature_importances_
feat_imp = dict(zip(feature_cols, importances.tolist()))
feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

print("\nTop 10 Feature Importances:")
for k, v in list(feat_imp_sorted.items())[:10]:
    bar = "X" * int(v * 200)
    print(f"  {k:<22} {v:.4f}  {bar}")

y_bin = label_binarize(y_test, classes=[0, 1, 2])
per_class_auc = {}
class_names = ['Low Risk', 'Medium Risk', 'High Risk']
for i, name in enumerate(class_names):
    auc_i = roc_auc_score(y_bin[:, i], y_prob_ens[:, i])
    per_class_auc[name] = round(auc_i, 4)
    print(f"  AUC {name}: {auc_i:.4f}")

os.makedirs('model', exist_ok=True)

with open('model/gb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

metrics = {
    'auc_gb':          round(auc_gb, 4),
    'auc_rf':          round(auc_rf, 4),
    'auc_ensemble':    round(auc_ens, 4),
    'cv_gb_mean':      round(gb_scores.mean(), 4),
    'cv_gb_std':       round(gb_scores.std(), 4),
    'per_class_auc':   per_class_auc,
    'feature_importance': feat_imp_sorted,
    'n_train':         int(split),
    'n_test':          int(len(X_scaled) - split),
    'class_distribution': {
        'Low':    int(np.sum(y == 0)),
        'Medium': int(np.sum(y == 1)),
        'High':   int(np.sum(y == 2))
    }
}
with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n Model artifacts saved to model/")
print(f"\n Final Ensemble AUC: {auc_ens:.4f}")