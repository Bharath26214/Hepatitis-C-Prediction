import pandas as pd
from data_ingestion.read_data import train_df, test_df

train_df = pd.concat(
    [pd.read_csv('data/protT5_features/train_embeddings.csv'),
     train_df['label'].reset_index(drop=True)],
    axis=1
)

test_df = pd.concat(
    [pd.read_csv('data/protT5_features/test_embeddings.csv'),
     test_df['label'].reset_index(drop=True)],
    axis=1
)