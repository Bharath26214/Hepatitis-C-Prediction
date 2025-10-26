from data_ingestion.read_data import train_df, test_df
from feature_engineering.feature_scaling import FeatureScaling

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from models.LSTM import AttLSTM
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np

rare_aas = 'UOZB' 
train_df['peptide_name'] = train_df['peptide_name'].str.upper().replace(
    "UOZB", "X", regex=True)

test_df['peptide_name'] = test_df['peptide_name'].str.upper().replace(
    'UOZB', "X", regex=True)

scaler = FeatureScaling(['AAC', 'APAAC', 'PAAC', 'TPC'], 'PCA', 500)

raw_train = scaler.feature_encoder(train_df)
X_train = pd.DataFrame(scaler.feature_reduction(raw_train, train_df['label']))

raw_test = scaler.feature_encoder(test_df)
X_test = pd.DataFrame(scaler.feature_reduction(raw_test, test_df['label'], False))

y_train, y_test = train_df['label'], test_df['label']
X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

input_shape = (777, 500)

model = AttLSTM(input_shape=input_shape, learning_rate=1e-4)

early_stopping = EarlyStopping(
        monitor='val_accuracy',     
        mode='max',                  
        patience=300,
        restore_best_weights=True,   
        verbose=1
    )

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test), 
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

y_pred = model.predict(X_test)
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

lstm_results = pd.DataFrame([results]).to_csv('results/lstm.csv', index=False)

