import numpy as np
import pandas as pd
from pyDeepInsight import ImageTransformer
from data_ingestion.read_data import train_df, test_df
from models.CNN import CNN

from feature_engineering.feature_scaling import FeatureScaling

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef

def features_to_images(X_train: np.ndarray, X_test: np.ndarray, pixels=(80, 80), method='pca'):
    it = ImageTransformer(feature_extractor=method, discretization='bin', pixels=pixels)

    X_train_img = it.fit_transform(X_train)
    X_test_img = it.transform(X_test)

    X_train_img = np.asarray(X_train_img, dtype=np.float32)
    X_test_img = np.asarray(X_test_img, dtype=np.float32)

    if X_train_img.ndim == 3:
        X_train_img = X_train_img[..., None]
    if X_test_img.ndim == 3:
        X_test_img = X_test_img[..., None]

    X_train_img -= X_train_img.min()
    if X_train_img.max() != 0:
        X_train_img /= X_train_img.max()

    X_test_img -= X_test_img.min()
    if X_test_img.max() != 0:
        X_test_img /= X_test_img.max()

    return X_train_img, X_test_img

scaler = FeatureScaling(['AAC', 'APAAC', 'PAAC', 'TPC'], 'none', 200)

raw_train = scaler.feature_encoder(train_df)
X_train = pd.DataFrame(scaler.feature_reduction(raw_train, train_df['label']))

raw_test = scaler.feature_encoder(test_df)
X_test = pd.DataFrame(scaler.feature_reduction(raw_test, test_df['label'], False))

y_train = train_df['label']
y_test = test_df['label']

X_train, X_test = features_to_images(np.array(X_train), np.array(X_test))

cnn = CNN((80, 80, 3), learning_rate=1e-5)

history = cnn.fit(X_train, y_train, X_test, y_test, 200)

y_pred = cnn.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_classes).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
auc = roc_auc_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred_classes)

print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUC: {auc:.4f}")
print(f"MCC: {mcc:.4f}")

results = {
    'Fold': 'Test',
    "Accuracy": round(accuracy, 4),
    "Sensitivity": round(sensitivity, 4),
    "Specificity": round(specificity, 4),
    "AUC": round(auc, 4),
    "MCC": round(mcc, 4)
}

cnn_results = pd.DataFrame([results]).to_csv('results/cnn/cnn_pca.csv', index=False)

