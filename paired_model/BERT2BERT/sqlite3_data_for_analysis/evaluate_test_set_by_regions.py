# env: OAS_paired_env
from Bio import pairwise2
from Bio.Align import substitution_matrices
from crowelab_pyir import PyIR
import pandas as pd

# Load the BLOSUM62 matrix
blosum62 = substitution_matrices.load("BLOSUM62")


# Define the calculate_blosum_score function
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

# Sample data
true_sequences = ["D I Q V T Q S P S S L S A S I G D R V T I T C Q A S Q D I S D N L N W Y Q Q K P G K V P K L L I Y D A S N L Q T G V P S R F S G S G S G T Y F S V T I S S L Q P E D I A T Y Y C Q S Y G K F R P R T F G Q G T K L E I K"]
generated_sequences = ["D I Q M T Q S P S S L S A S V G D R V T I T C R T S Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F T L Q T G V P P R F S G S R S E T E F I L T V S N L R P E D F A T Y Y C L H H N S Y P Y T F G Q E S K L E I K"]

# Process sequences using PyIR to get regions
def get_regions(sequence):
    # Here you would use PyIR to identify the regions
    # This is a placeholder function
    return {
        "FWR1": "D I Q V T Q S P S S",
        "CDR1": "L S A S",
        "FWR2": "I G D R V T I T C",
        "CDR2": "Q A S Q",
        "FWR3": "D I S D N L N W Y Q Q K P G K V P K L L I Y D A S N",
        "CDR3": "L Q T G V P S R F S G S G S G T Y F S V T I",
        "FWR4": "S L Q P E D I A T Y Y C Q S Y G K F R P R T F G Q G T K L E I K"
    }

# Calculate metrics for each region
results = []
for true_seq, generated_seq in zip(true_sequences, generated_sequences):
    true_regions = get_regions(true_seq)
    generated_regions = get_regions(generated_seq)
    
    for region in true_regions.keys():
        score, min_length, matches, similarity_percentage = calculate_blosum_score(
            true_regions[region], generated_regions[region], blosum62
        )
        results.append({
            "Region": region,
            "True Sequence": true_regions[region],
            "Generated Sequence": generated_regions[region],
            "BLOSUM Score": score,
            "Similarity Percentage": similarity_percentage
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)

