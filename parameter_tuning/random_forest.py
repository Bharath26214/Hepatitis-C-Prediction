import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from data_ingestion.read_data import train_df
from feature_engineering.feature_scaling import FeatureScaling
import optuna

scaler = FeatureScaling(['AAC', 'APAAC', 'PAAC', 'TPC'], 'mRMR', 500)
X = scaler.feature_encoder(train_df)
y = train_df['label'].values

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    return np.mean(scores)

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150, n_jobs=-1)

print("Best params:", study.best_params)
print("Best accuracy:", study.best_value)