import pandas as pd
import random
from sklearn.model_selection import train_test_split

# this script takes a dataset with heavy[SEP]light as input, splits the data on the [SEP] token, adds a 1 as label (paired) and generates random pairings of 
# heavy and light chains and adds a 0 for unpaired. In this way, the sequences are not shared between the training, validation and test files (run the script for each training, test and val file separately). 

# Load the data
file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids.txt"
with open(file_path, "r") as f:
    data = f.readlines()

# Process the data
paired_data = []
for line in data:
    heavy, light = line.strip().split("[SEP]")
    paired_data.append((heavy, light, 1))

# Generate unpaired data
heavy_chains = [item[0] for item in paired_data]
light_chains = [item[1] for item in paired_data]
unpaired_data = []
random.seed(42)
random.shuffle(light_chains)

for heavy, light in zip(heavy_chains, light_chains):
    unpaired_data.append((heavy, light, 0))

# Combine paired and unpaired data
combined_data = paired_data + unpaired_data
random.shuffle(combined_data)

# Convert to DataFrame
df = pd.DataFrame(combined_data, columns=["heavy", "light", "label"])

# save the dataset
df.to_csv("paired_full_seqs_sep_train_with_unpaired.csv", index=False)

print("Dataset saved successfully!")
