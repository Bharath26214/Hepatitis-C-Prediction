import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


class PLSDA(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        self.pls_ = PLSRegression(n_components=self.n_components)
        self.log_ = SVC(probability=True)

        Z = self.pls_.fit_transform(X, y)[0]   
        self.log_.fit(Z, y)
        return self

    def predict(self, X):
        Z = self.pls_.transform(X)
        Z = np.reshape(Z, (Z.shape[0], -1)) 
        return self.log_.predict(Z)

    def predict_proba(self, X):
        Z = self.pls_.transform(X)
        Z = np.reshape(Z, (Z.shape[0], -1))
        return self.log_.predict_proba(Z)

    def score(self, X, y):
        return self.log_.score(np.reshape(self.pls_.transform(X), (X.shape[0], -1)), y)
