from data_ingestion.read_data import train_df, test_df
from feature_engineering.kmer_encoding import Kmer

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from models.LSTM import AttLSTM
from tensorflow.keras.callbacks import EarlyStopping
from models.CNN import CNN

from sklearn.preprocessing import MinMaxScaler
from gensim.models import KeyedVectors

import pandas as pd
import numpy as np

rare_aas = 'UOZB' 
train_df['peptide_name'] = train_df['peptide_name'].str.upper().replace(
    "UOZB", "X", regex=True)

test_df['peptide_name'] = test_df['peptide_name'].str.upper().replace(
    'UOZB', "X", regex=True)


X_train, X_test = Kmer(train_df, 3).encode_features(), Kmer(test_df, 3).encode_features()
y_train, y_test = train_df['label'], test_df['label']


model = KeyedVectors.load("models/protVec_100d_3grams.model", mmap='r')

def embed_protein_kmers(kmers_list, model, vector_size):
    zero_vec = np.zeros(vector_size, dtype=np.float32)
    embeddings = []

    for kmer in kmers_list:
        try:
            vec = model[kmer]  
        except KeyError:
            vec = zero_vec  
        embeddings.append(vec)

    embeddings = np.array(embeddings, dtype=np.float32)
    if embeddings.shape[0] == 0:
        return zero_vec  

    return np.array(embeddings).mean(axis=0)

X_train = np.array([embed_protein_kmers(seq, model, 100) for seq in X_train])
X_test = np.array([embed_protein_kmers(seq, model, 100) for seq in X_test])

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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

lstm_results = pd.DataFrame([results]).to_csv('results/lstm_kmer.csv', index=False)

