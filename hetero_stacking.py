import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from models.PLS import PLSDA
from feature_engineering.feature_scaling import FeatureScaling
from data_ingestion.read_data import train_df, test_df

def fit_and_transform_features(train_df, test_df, encoding_list, label):
    if encoding_list == ['AAC', 'PAAC', 'APAAC']:
        scaler = FeatureScaling(encoding_list, 'none', 500)
    else:
        scaler = FeatureScaling(encoding_list, 'PCA', 500)

    # Fit on training
    raw_train = scaler.feature_encoder(train_df)
    X_train = pd.DataFrame(scaler.feature_reduction(raw_train, train_df[label], True))

    # Transform test using same scaler
    raw_test = scaler.feature_encoder(test_df)
    X_test = pd.DataFrame(scaler.feature_reduction(raw_test, test_df[label], False))

    return X_train, X_test, scaler

y_train, y_test = train_df['label'], test_df['label']

encodings = {
    'rf':  ['AAC', 'PAAC', 'APAAC'],
    'pls': ['DDC', 'DPC', 'TPC'],
    'svc': ['PCP', 'TPC'],
    'xgb': ['AAC', 'PAAC', 'APAAC', 'TPC']
}

X_train_sets, X_test_sets, scalers = {}, {}, {}

for name, enc_list in encodings.items():
    print(f"\n🔹 Extracting features for {name.upper()} using encodings {enc_list}...")
    X_train, X_test, scaler = fit_and_transform_features(train_df, test_df, enc_list, 'label')
    X_train_sets[name] = X_train
    X_test_sets[name] = X_test
    scalers[name] = scaler

base_models = {
    'rf': RandomForestClassifier(
        n_estimators=401, max_depth=20, min_samples_split=7,
        min_samples_leaf=1, max_features='log2', random_state=42
    ),
    'pls': PLSDA(n_components=15),
    'svc': SVC(
        C=0.001098881873199668, kernel='rbf', gamma=0.010951255895390475754,
        probability=True, random_state=42
    ),
    'xgb': XGBClassifier(
        n_estimators=500, max_depth=15, random_state=42,
        use_label_encoder=False, eval_metric='logloss'
    )
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_train = np.zeros((len(y_train), len(base_models)))
meta_test  = np.zeros((len(y_test), len(base_models)))

print("\n🔁 Performing Out-of-Fold training for base models...\n")

for i, (name, model) in enumerate(base_models.items()):
    print(f"➡️ Training {name.upper()} on its corresponding encoding features...")

    X_train = np.array(X_train_sets[name])
    X_test = np.array(X_test_sets[name])
    y = np.array(y_train)

    oof_pred = np.zeros(len(y_train))
    test_pred = np.zeros((len(y_test), kf.n_splits))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y), 1):
        m = clone(model)
        m.fit(X_train[train_idx], y[train_idx])
        oof_pred[val_idx] = m.predict_proba(X_train[val_idx])[:, 1]
        test_pred[:, fold - 1] = m.predict_proba(X_test)[:, 1]

        print(f"   Fold {fold} done.")

    meta_train[:, i] = oof_pred
    meta_test[:, i] = test_pred.mean(axis=1)

meta_model = LogisticRegression(
    C=0.01, solver='liblinear', max_iter=5000, random_state=42
)
meta_model.fit(meta_train, y_train)

y_prob = meta_model.predict_proba(meta_test)[:, 1]
y_pred = (y_prob > 0.5).astype(int)


acc = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
auc = roc_auc_score(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n📊 Final Heterogeneous Stacking Results")
print(f"Accuracy:     {acc:.4f}")
print(f"Sensitivity:  {sensitivity:.4f}")
print(f"Specificity:  {specificity:.4f}")
print(f"AUC:          {auc:.4f}")
print(f"MCC:          {mcc:.4f}")

results = pd.DataFrame({
    "Metric": ["Accuracy", "Sensitivity", "Specificity", "AUC", "MCC"],
    "Score": [acc, sensitivity, specificity, auc, mcc]
})
results.to_csv("results/ensemble/hetero_stacking.csv", index=False)

