import pandas as pd

from modeling.model_training import ModelTraining
from modeling.model_evaluation import ModelEvaluator

import warnings
warnings.filterwarnings('ignore')

class TradML:
    def __init__(self):
        self.single_encoding = [
                                'AAC',
                                'APAAC',
                                'PAAC',
                                'DDC',
                                'DPC',
                                'PCP',
                                'TPC',
                                'CKSAAP'
                            ]
        self.group_encoding = {
                    'local_encoding': ['AAC', 'PAAC', 'APAAC'],
                    'global_encoding': ['DDC', 'DPC', 'TPC'],
                    'other_group_1': ['AAC', 'PAAC', 'APAAC', 'TPC'],
                    'other_group_2': ['PCP', 'TPC']
                }
        
    def get_dr_and_features(self, feature_list, group_name=None):
        dr, n_features = None, None
        # For single features
        if any(f in ['TPC', 'CKSAAP'] for f in feature_list):
            dr, n_features = 'PCA', 500
        # For groups
        if group_name in ['global_encoding', 'other_group_1', 'other_group_2']:
            dr, n_features = 'PCA', 500
        return dr, n_features

    def run_cross_validation(self, feature_list, group_name=None):
        dr, n_features = self.get_dr_and_features(feature_list, group_name)
        trainer = ModelTraining(feature_list, dr, n_features)
        cross_val_results = trainer.apply_cross_validation()
        
        rows = []
        for model_name, encoding, metrics in cross_val_results:
            row = {
                "encoding": ', '.join(encoding),
                "model_name": model_name
                }
            row.update(metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def run_evaluation(self, feature_list, group_name=None):
        dr, n_features = self.get_dr_and_features(feature_list, group_name)
        trainer = ModelTraining(feature_list, dr, n_features)
        trained_classifiers, X_test, y_test, feature_types = trainer.train_classifiers()
        evaluator = ModelEvaluator(trained_classifiers, X_test, y_test, feature_types)
        return evaluator.evaluate_classifiers()
    
    def run(self):
        single_cv_df = pd.concat(
            [self.run_cross_validation([ft]) for ft in self.single_encoding],
            ignore_index=True
        ).round(4)
        
        single_test_results = pd.concat(
            [self.run_evaluation([ft]) for ft in self.single_encoding],
            ignore_index=True
        )
        
        group_cv_df = pd.concat(
            [self.run_cross_validation(enc, group) for group, enc in self.group_encoding.items()],
            ignore_index=True
        ).round(4)
        
        group_test_results = pd.concat(
            [self.run_evaluation(enc, group) for group, enc in self.group_encoding.items()],
            ignore_index=True
        )
        
        single_cv_df.to_csv('results/trad_ml/single_cross_val_results.csv', index=False)
        single_test_results.to_csv('results/trad_ml/single_feature_test_results.csv', index=False)
        
        
        group_test_results.to_csv('results/trad_ml/group_test_results.csv', index=False)
        group_cv_df.to_csv('results/trad_ml/group_cross_val_results.csv', index=False)
        
        
tradml = TradML()
tradml.run()