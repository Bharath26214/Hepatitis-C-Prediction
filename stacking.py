from data_ingestion.read_data import train_df, test_df
from feature_engineering.feature_scaling import FeatureScaling

from models.PLS import PLSDA
from models.StackingClassifier import CustomStackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

train_df['peptide_name'] = train_df['peptide_name'].str.upper().replace("UOZB", "X", regex=True)
test_df['peptide_name'] = test_df['peptide_name'].str.upper().replace("UOZB", "X", regex=True)

scaler = FeatureScaling(['AAC', 'APAAC', 'PAAC', 'TPC'], 'PCA', 400)
raw_train = scaler.feature_encoder(train_df)
X_train = pd.DataFrame(scaler.feature_reduction(raw_train, train_df['label']))

raw_test = scaler.feature_encoder(test_df)
X_test = pd.DataFrame(scaler.feature_reduction(raw_test, test_df['label'], False))

y_train, y_test = train_df['label'], test_df['label']

base_models = [
    ('pls', PLSDA(n_components=20)), 
    ('log', LogisticRegression(
        C=0.0027059021490395217,
        max_iter=5000,
        penalty='l2',
        solver='liblinear',
        random_state=42
    )),
    ('xgb', XGBClassifier(
        n_estimators=500,
        max_depth=15,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )),
    ('knn', KNeighborsClassifier())
]

meta_model = LogisticRegression(
    C=0.01,
    solver='liblinear',
    random_state=42
)

model = CustomStackingClassifier(base_models, meta_model)

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

acc_test = accuracy_score(y_test, y_test_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
specificity_test = tn / (tn + fp)
sensitivity_test = tp / (tp + fn)
auc_test = roc_auc_score(y_test, y_test_prob)
mcc_test = matthews_corrcoef(y_test, y_test_pred)

results = {
    'Dataset': 'Test Set',
    'ACCURACY': round(acc_test, 4),
    'SENSITIVITY': round(sensitivity_test, 4),
    'SPECIFICITY': round(specificity_test, 4),
    'AUC SCORE': round(auc_test, 4),
    'MCC SCORE': round(mcc_test, 4)
}

print(f"\nTest Set -> ACC: {acc_test:.4f}, SEN: {sensitivity_test:.4f}, SPEC: {specificity_test:.4f}, AUC: {auc_test:.4f}, MCC: {mcc_test:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("results/ensemble/stacking.csv", index=False)
