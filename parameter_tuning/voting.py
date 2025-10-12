import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from models.VotingClassifier import CustomVotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from data_ingestion.read_data import train_df
from feature_engineering.feature_scaling import FeatureScaling
from models.PLS import PLSDA

import optuna

scaler = FeatureScaling(['AAC', 'APAAC', 'PAAC', 'TPC'], 'PCA', 500)
X = scaler.feature_encoder(train_df)
y = train_df['label'].values

def objective(trial):
    pls_n_components = trial.suggest_int("pls_n_components", 2, 20)

    log_C = trial.suggest_float("log_C", 1e-3, 1e3, log=True)
    log_penalty = trial.suggest_categorical("log_penalty", ["l2"])
    log_solver = trial.suggest_categorical("log_solver", ["lbfgs", "saga"])

    svm_C = trial.suggest_float("svm_C", 1e-3, 1e3, log=True)
    svm_kernel = trial.suggest_categorical("svm_kernel", ["linear", "rbf", "poly"])
    svm_gamma = trial.suggest_float("svm_gamma", 1e-4, 1e0, log=True)

    pls = PLSDA(n_components=pls_n_components)
    log = LogisticRegression(C=log_C, penalty=log_penalty, solver=log_solver, max_iter=5000)
    svm = SVC(C=svm_C, kernel=svm_kernel, gamma=svm_gamma, probability=True)

    clf = CustomVotingClassifier(
        estimators=[
            ('pls', pls),
            ('log', log),
            ('svm', svm),
        ],
        voting='soft'
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    return np.mean(scores)

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150, n_jobs=-1)

print("Best params:", study.best_params)
print("Best accuracy:", study.best_value)