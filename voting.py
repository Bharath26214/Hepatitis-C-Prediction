from data_ingestion.read_data import train_df, test_df
from feature_engineering.feature_scaling import FeatureScaling
from models.VotingClassifier import CustomVotingClassifier
from models.PLS import PLSDA

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd

rare_aas = 'UOZB'
train_df['peptide_name'] = train_df['peptide_name'].str.upper().replace("UOZB", "X", regex=True)
test_df['peptide_name'] = test_df['peptide_name'].str.upper().replace("UOZB", "X", regex=True)

scaler = FeatureScaling(['AAC', 'APAAC', 'PAAC', 'TPC'], 'PCA', 500)

raw_train = scaler.feature_encoder(train_df)
X_train = pd.DataFrame(scaler.feature_reduction(raw_train, train_df['label']))

raw_test = scaler.feature_encoder(test_df)
X_test = pd.DataFrame(scaler.feature_reduction(raw_test, test_df['label'], False))

y_train, y_test = train_df['label'], test_df['label']

model = CustomVotingClassifier(
    estimators=[
        ('pls', PLSDA(n_components=2)),
        ('log', LogisticRegression(
            C=0.0027059021490395217,
            max_iter=5000,
            penalty='l2',
            solver='liblinear',
            random_state=42
        )),
        ('svm', SVC(
            C=0.001098881873199668,
            kernel='rbf',
            gamma=0.010951255895390475754,
            probability=True,
            random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=401,
            max_depth=20,
            min_samples_split=7,
            min_samples_leaf=1,
            max_features='log2',
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
    ],
    voting='soft',
    weights=[1, 1, 1, 1, 1, 1]
)

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
results_df.to_csv("results/ensemble/voting.csv", index=False)
