import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from feature_engineering.feature_development import FeatureDevelopment

from mrmr import mrmr_classif

class FeatureScaling:
    def __init__(self, feature_types, dr, n_features):
        self.feature_types = feature_types
        self.n_components = n_features
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.n_features = n_features
        self.dr = dr
        self.selected_features = None  

    # Standardization
    def normalization(self, features):
        epsilon = 1e-8
        features = np.array(features, dtype=float)
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return (features - mean) / (std + epsilon)

    # mRMR
    def mRMR(self, X, y, fit=True):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if fit:
            self.selected_features = mrmr_classif(X=X, y=y, K=self.n_features)
        return X[self.selected_features].values

    def feature_encoder(self, df):
        def encode_features(df):
            encoded_list = []
            for _, row in df.iterrows():
                feature_developer = FeatureDevelopment(row['peptide_name'])
                features = []
                if "APAAC" in self.feature_types:
                    features.append(feature_developer.APAAC())
                if "DDC" in self.feature_types:
                    features.append(feature_developer.DDC())
                if "CKSAAP" in self.feature_types:
                    features.append(feature_developer.CKSAAP())
                if "AAC" in self.feature_types:
                    features.append(feature_developer.AAC())
                if "PAAC" in self.feature_types:
                    features.append(feature_developer.PAAC())
                if "DPC" in self.feature_types:
                    features.append(feature_developer.DPC())
                if "PCP" in self.feature_types:
                    features.append(feature_developer.PCP())
                if "TPC" in self.feature_types:
                    features.append(feature_developer.TPC())

                features = [np.array(f, dtype=float) for f in features]
                encoded_list.append(np.concatenate(features))

            return np.array(encoded_list)

        return encode_features(df)

    def feature_reduction(self, features, y,  fit=True):
        features = self.normalization(features)
        if self.dr == 'PCA':
            if fit:
                features = self.pca.fit_transform(features)
            else:
                features = self.pca.transform(features)
                

        elif self.dr == 'mRMR':
            features = self.mRMR(features, y, fit=fit)
            features = self.normalization(features)
            
        return np.array(features)
            
            
