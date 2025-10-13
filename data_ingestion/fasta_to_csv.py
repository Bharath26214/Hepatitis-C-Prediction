import csv
import os
import pandas as pd

def convert_to_csv(input_file, output_file):
    data = []
    label = None
    with open(input_file) as f:
        name = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
                if "Positive" in name:
                    label = 1
                elif "Negative" in name:
                    label = 0
                else:
                    label = None 
            elif line and label is not None:
                data.append([line, label])

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["peptide_name", "label"])
        writer.writerows(data)

input_dir = 'data/data_fasta'
output_dir = 'data/data_csv'

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if not file.endswith(".txt"):
        continue  # skip non-txt files
    in_file = os.path.join(input_dir, file)
    out_file = os.path.join(output_dir, file.replace(".txt", ".csv"))
    convert_to_csv(in_file, out_file)
    print(f"Converted {file} -> {os.path.basename(out_file)}")


