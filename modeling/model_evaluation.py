from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from models.VotingClassifier import CustomVotingClassifier
import pandas as pd

class ModelEvaluator:
  def __init__(self, trained_classifiers, X_test, y_test, feature_types):
    self.trained_classifiers = trained_classifiers
    self.X_test = X_test
    self.y_test = y_test
    self.feature_types = feature_types
    self.test_results = []

  # Evaluate each model and save results
  def model_eval(self, model_name, model):
    y_test_pred = model.predict(self.X_test)

    if isinstance(model, CustomVotingClassifier):
        y_test_prob = y_test_pred
    elif hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(self.X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_test_prob = model.decision_function(self.X_test)
    else:
        y_test_prob = y_test_pred

    acc_test = accuracy_score(self.y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(self.y_test, y_test_pred).ravel()
    specificity_test = tn / (tn + fp)
    sensitivity_test = tp / (tp + fn)

    auc_test = roc_auc_score(self.y_test, y_test_prob)
    mcc_test = matthews_corrcoef(self.y_test, y_test_pred)

    self.test_results.append({
        'ENCODING TYPE': ', '.join(self.feature_types),
        'MODEL NAME': model_name,
        'ACCURACY': round(acc_test, 4),
        'SENSITIVITY': round(sensitivity_test, 4),
        'SPECIFICITY': round(specificity_test, 4),
        'AUC SCORE': round(auc_test, 4),
        'MCC SCORE': round(mcc_test, 4)
    })

  # Evaluating multiple classifiers
  def evaluate_classifiers(self):
    for model_name in self.trained_classifiers:
      self.model_eval(model_name, self.trained_classifiers[model_name])

    return pd.DataFrame(self.test_results)