# !pip install biopython
from Bio.Align import substitution_matrices
import re
import pandas as pd


# heavy2light run name: save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1
#file_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/HEAVY2LIGHT_114312.o"
# output path: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/blosum_score_114416.o

# heavy2light without adapters run name: FULL_data_heavy2light_without_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1
file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/h2l_no_adaps_114294.o"

# IgBERT2IgBERT run name: FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1
#file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/b2b_adaps_114271.o"
# output path: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/blosum_score_114587.o

with open(file_path, 'r') as file:
    file_content = file.read()

# Use regex to extract the sequences
pattern = r"decoded light sequence:  ([A-Z ]+)\ntrue light sequence:  ([A-Z ]+)"
matches = re.findall(pattern, file_content)

df = pd.DataFrame(matches, columns=['Generated Sequence', 'True Sequence'])

#print(df)

def calculate_blosum_score(true_seq, generated_seq, matrix):
    score = 0
    matches = 0

    generated_light_seq = generated_seq.replace(" ", "")
    true_light_seq = true_seq.replace(" ", "")

    min_length = min(len(true_light_seq), len(generated_light_seq))

    for i in range(min_length):
        pair = (true_light_seq[i], generated_light_seq[i])
        if pair in matrix:
            score += matrix[pair]
        elif (pair[1], pair[0]) in matrix:
            score += matrix[(pair[1], pair[0])]
        if true_light_seq[i] == generated_light_seq[i]:
            matches += 1
    
    similarity_percentage = (matches / min_length) * 100
    
    return score, min_length, matches, similarity_percentage

# Load the BLOSUM62 matrix
blosum62 = substitution_matrices.load("BLOSUM62")

scores = []
similarities = []

for index, row in df.iterrows():
    score, min_length, matches, similarity_percentage = calculate_blosum_score(row['True Sequence'], row['Generated Sequence'], blosum62)
    scores.append(score)
    similarities.append(similarity_percentage)

    # Print results for each sequence pair
    print(f"\nSequence pair {index+1}:")
    print(f"True Sequence: {row['True Sequence']}")
    print(f"Generated Sequence: {row['Generated Sequence']}")
    print(f"BLOSUM Score: {score}")
    print(f"Minimum Length: {min_length}")
    print(f"Matches: {matches}")
    print(f"Similarity Percentage: {similarity_percentage}%")

df['BLOSUM Score'] = scores
df['Similarity Percentage'] = similarities

# Calculate the average BLOSUM score and average similarity percentage
average_blosum_score = sum(scores) / len(scores)
average_similarity_percentage = sum(similarities) / len(similarities)

print(f"\nAverage BLOSUM Score: {average_blosum_score}")
print(f"Average Similarity Percentage: {average_similarity_percentage}%")

# Save the dataframe to a CSV file (if needed)
#df.to_csv('sequences_with_scores.csv', index=False)

