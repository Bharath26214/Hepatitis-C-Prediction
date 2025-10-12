import pandas as pd

train_df = pd.concat(
    [pd.read_csv('data/data_csv/TR_pos.csv'),
     pd.read_csv('data/data_csv/TR_neg.csv')],
    ignore_index=True
)

test_df = pd.concat(
    [pd.read_csv('data/data_csv/TS_pos.csv'),
     pd.read_csv('data/data_csv/TS_neg.csv')],
    ignore_index=True
)