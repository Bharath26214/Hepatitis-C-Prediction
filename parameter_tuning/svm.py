from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from data_ingestion.read_data import train_df
from feature_engineering.feature_scaling import FeatureScaling

scaler = FeatureScaling(['AAC', 'APAAC', 'PAAC', 'TPC'], 'mRMR', 500)
X = scaler.feature_encoder(train_df)
y = train_df['label'].values

svm = SVC()
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
}

svm_grid = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X, y)
print("Best SVM params:", svm_grid.best_params_)
print("Best SVM score:", svm_grid.best_score_)
