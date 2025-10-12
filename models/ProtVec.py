import requests
import csv
import pandas as pd
from gensim.models import KeyedVectors

url = "https://raw.githubusercontent.com/ehsanasgari/Deep-Proteomics/master/protVec_100d_3grams.csv"

out_path = "models/protVec_100d_3grams.csv"

vector_size = 100
header = ["kmer"] + [f"word_2_vec_{i+1}" for i in range(vector_size)]

resp = requests.get(url, stream=True)
if resp.status_code == 200:
    with open(out_path, "w", newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        for line in resp.iter_lines(decode_unicode=True):
            if line:
                parts = line.strip().split("\t")
                parts = [p.strip().replace('"','') for p in parts]
                writer.writerow(parts)

else:
    print("Failed to download. Status code:", resp.status_code)
    
csv_path = out_path  
model_path = "models/protVec_100d_3grams.model"

df = pd.read_csv(csv_path, sep=",", index_col=0)  

df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

kv = KeyedVectors(vector_size=vector_size)
kv.add_vectors(list(df.index), df.values)
kv.save(model_path)