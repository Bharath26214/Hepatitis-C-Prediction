from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from data_ingestion.read_data import train_df
from feature_engineering.feature_scaling import FeatureScaling
import optuna

scaler = FeatureScaling(['AAC', 'APAAC', 'PAAC', 'TPC'], 'mRMR', 500)
X = scaler.feature_encoder(train_df)
y = train_df['label'].values

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_uniform("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-4, 10.0),
        "use_label_encoder": False,
        "eval_metric": "logloss",  
        "random_state": 42,
        "n_jobs": -1,
    }

    clf = XGBClassifier(**params)

    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=-1)

print("Best params:", study.best_params)
print("Best accuracy:", study.best_value)