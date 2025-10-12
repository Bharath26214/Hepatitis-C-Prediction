from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, matthews_corrcoef, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from feature_engineering.feature_scaling import FeatureScaling
from models.PLS import PLSDA
from models.VotingClassifier import CustomVotingClassifier
from data_ingestion.read_data import train_df, test_df

import numpy as np

class ModelTraining:
    def __init__(self, feature_types, dr, n_features):
        self.feature_types = feature_types
        self.dr = dr
        self.n_features = n_features
        self.train_df = train_df
        self.test_df = test_df

        self.scaler = FeatureScaling(feature_types, dr, n_features)

        self.X_train = self.scaler.feature_encoder(self.train_df)
        self.y_train = self.train_df['label'].values

        self.X_test = self.scaler.feature_encoder(self.test_df)
        self.y_test = self.test_df['label'].values

        # Classifiers
        self.classifiers = {
            'random_forest': RandomForestClassifier(
                n_estimators=401,
                max_depth=23,
                min_samples_split=7,
                min_samples_leaf=1,
                max_features='log2',
                bootstrap=True
            ),
            'xg_boost': XGBClassifier(),
            'knn': KNeighborsClassifier(),
            'lgbm': LGBMClassifier(verbose=-1),
            'log_r': LogisticRegression(max_iter=5000),
            'svc': SVC(probability=True),
            'voting1': CustomVotingClassifier(
                estimators=[
                    ('pls', PLSDA(n_components=3)),
                    ('log', LogisticRegression(C=0.0027059021490395217,
                                               max_iter=5000,
                                               penalty='l2',
                                               solver='lbfgs')),
                    ('svm', SVC(C=0.020098881873199668,
                                kernel='poly',
                                gamma=0.001255895390475754,
                                probability=True)),
                ],
                voting='soft'
            ),
            'pls': PLSDA()
        }

    def cross_validation(self, model_name, model, n_splits=5):
        """Run CV on training set only (no test scaling)."""
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_scores = {"ACCURACY": [], "SENSITIVITY": [], "SPECIFICITY": [], "AUC SCORE": [], "MCC SCORE": []}

        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_tr, X_val = self.X_train[train_idx], self.X_train[val_idx]
            y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
            
            X_tr_scaled = self.scaler.feature_reduction(X_tr, y_tr)
            X_val_scaled = self.scaler.feature_reduction(X_val, y_val, fit=False)

            model.fit(X_tr_scaled, y_tr)
            y_pred = model.predict(X_val_scaled)
            y_prob = model.predict_proba(X_val_scaled)[:, 1]

            fold_scores["ACCURACY"].append(accuracy_score(y_val, y_pred))
            fold_scores["SENSITIVITY"].append(recall_score(y_val, y_pred))                
            fold_scores["SPECIFICITY"].append(recall_score(y_val, y_pred, pos_label=0))   
            fold_scores["AUC SCORE"].append(roc_auc_score(y_val, y_prob))
            fold_scores["MCC SCORE"].append(matthews_corrcoef(y_val, y_pred))

            avg_scores = {metric: np.mean(vals) for metric, vals in fold_scores.items()}

        return model_name, self.feature_types, avg_scores

    def apply_cross_validation(self):
        results = []
        for clf in self.classifiers:
            results.append(self.cross_validation(clf, self.classifiers[clf]))
        return results

    def train_classifiers(self):
        self.X_train = self.scaler.feature_reduction(self.X_train, self.y_train)
        self.X_test = self.scaler.feature_reduction(self.X_test, self.y_test, False)
        
        
        for clf_name in self.classifiers:
            self.classifiers[clf_name].fit(self.X_train, self.y_train)

        return self.classifiers, self.X_test, self.y_test, self.feature_types
