import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

class CustomStackingClassifier:
    def __init__(self, base_models, meta_model, n_folds=2, use_proba=True, random_state=42):
        self.base_models = base_models
        self.meta_model = clone(meta_model)
        self.n_folds = n_folds
        self.use_proba = use_proba
        self.random_state = random_state
        self.base_models_fitted = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Reset before fitting
        self.base_models_fitted = []

        # Meta features for training meta model
        meta_features = np.zeros((X.shape[0], len(self.base_models)))

        print("\nTraining base models with out-of-fold predictions for stacking:")
        for i, (name, model) in enumerate(self.base_models):
            oof_pred = np.zeros(X.shape[0])
            print(f"  -> Training {name}")

            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]

                m = clone(model)
                m.fit(X_train, y_train)

                if self.use_proba:
                    oof_pred[val_idx] = m.predict_proba(X_val)[:, 1]
                else:
                    oof_pred[val_idx] = m.predict(X_val)

            meta_features[:, i] = oof_pred

            # Refit the model on the entire training data for final use
            fitted_model = clone(model).fit(X, y)
            self.base_models_fitted.append((name, fitted_model))

        print("Training meta-model on stacked features...")
        self.meta_model.fit(meta_features, y)
        self.meta_train_features_ = meta_features
        return self

    def predict(self, X):
        meta_features = self._predict_base_features(X)
        return (self.meta_model.predict(meta_features) > 0.5).astype(int)

    def predict_proba(self, X):
        meta_features = self._predict_base_features(X)
        return self.meta_model.predict_proba(meta_features)

    def _predict_base_features(self, X):
        meta_features = np.zeros((X.shape[0], len(self.base_models_fitted)))
        for i, (name, model) in enumerate(self.base_models_fitted):
            if self.use_proba:
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = model.predict(X)
        return meta_features
