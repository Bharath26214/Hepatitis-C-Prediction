from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
import numpy as np

class CustomVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, voting="hard", weights=None):
        """
        :param estimators: list of (name, estimator) tuples
        :param voting: "hard" or "soft"
        :param weights: list of weights for each estimator (same order as estimators)
        """
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.le_ = LabelEncoder()
        self.fitted_estimators_ = []

    def fit(self, X, y):
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        y_enc = self.le_.transform(y)

        self.fitted_estimators_ = []
        for name, est in self.estimators:
            cloned_est = clone(est)
            cloned_est.fit(X, y_enc)
            self.fitted_estimators_.append((name, cloned_est))
        return self

    def predict(self, X):
        if self.voting == "hard":
            preds = np.asarray([est.predict(X) for _, est in self.fitted_estimators_]).T
            if self.weights is None:
                maj = [np.bincount(p).argmax() for p in preds]
            else:
                # weighted hard voting
                maj = []
                for row in preds:
                    vote_counts = {}
                    for i, p in enumerate(row):
                        vote_counts[p] = vote_counts.get(p, 0) + self.weights[i]
                    maj.append(max(vote_counts, key=vote_counts.get))
            return self.le_.inverse_transform(maj)

        elif self.voting == "soft":
            prob_list = [est.predict_proba(X) for _, est in self.fitted_estimators_]
            if self.weights is not None:
                avg_proba = np.average(prob_list, axis=0, weights=self.weights)
            else:
                avg_proba = np.mean(prob_list, axis=0)
            return self.le_.inverse_transform(np.argmax(avg_proba, axis=1))

    def predict_proba(self, X):
        if self.voting != "soft":
            raise AttributeError("predict_proba only available when voting='soft'")
        prob_list = [est.predict_proba(X) for _, est in self.fitted_estimators_]
        if self.weights is not None:
            avg_proba = np.average(prob_list, axis=0, weights=self.weights)
        else:
            avg_proba = np.mean(prob_list, axis=0)
        return avg_proba

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
