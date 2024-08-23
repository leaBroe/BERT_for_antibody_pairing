import os
import csv

# alphafold_structure_predictions structure:
# ├── 10_best
# │   ├── 14594
# │   │   ├── gen_seq_14594_b_<unique_id>
# │   │   │   ├── gen_seq_14594_b_<unique_id>_unrelaxed_rank_003_alphafold2_ptm_model_1_seed_000.pdb
# │   │   └── true_seq_14594_b_<unique_id>
# │   │       ├── true_seq_14594_b_<unique_id>_unrelaxed_rank_003_alphafold2_ptm_model_1_seed_000.pdb
# │   └── ...
# ├── 10_middle
# │   └── ...
# └── 10_worst
#     └── ...

def collect_pdb_files(base_dir):
    pdb_pairs = {
        "b": [],
        "m": [],
        "w": []
    }

    # Define the categories and corresponding directory suffixes
    categories = {
        "10_best": "b",
        "10_middle": "m",
        "10_worst": "w"
    }

    # Loop through each category (best, middle, worst)
    for category_dir, suffix in categories.items():
        category_path = os.path.join(base_dir, category_dir)

        # Loop through each ID folder within the category
        for id_folder in os.listdir(category_path):
            id_folder_path = os.path.join(category_path, id_folder)

            if os.path.isdir(id_folder_path):
                gen_pdb_file = None
                true_pdb_file = None

                # Inside each ID folder, there are more folders (one for generated, one for true)
                for sub_folder in os.listdir(id_folder_path):
                    sub_folder_path = os.path.join(id_folder_path, sub_folder)

                    if os.path.isdir(sub_folder_path):
                        # Look for generated sequence PDB file
                        if sub_folder.startswith(f"gen_seq_{id_folder}_{suffix}"):
                            for file_name in os.listdir(sub_folder_path):
                                if "rank_001" in file_name and file_name.endswith(".pdb"):
                                    gen_pdb_file = os.path.join(sub_folder_path, file_name)

                        # Look for true sequence PDB file
                        if sub_folder.startswith(f"true_seq_{id_folder}_{suffix}"):
                            for file_name in os.listdir(sub_folder_path):
                                if "rank_001" in file_name and file_name.endswith(".pdb"):
                                    true_pdb_file = os.path.join(sub_folder_path, file_name)

                # Ensure both files are found before adding to the list
                if gen_pdb_file and true_pdb_file:
                    pdb_pairs[suffix].append((true_pdb_file, gen_pdb_file))
                else:
                    print(f"Warning: Could not find matching 'rank_001' PDB files for ID {id_folder} in {category_dir}")

    return pdb_pairs

def write_pdb_files_to_csv(pdb_pairs, output_dir):
    # Loop over each category in pdb_pairs dictionary
    for suffix, pairs in pdb_pairs.items():
        category_map = {"b": "best", "m": "middle", "w": "worst"}
        output_file = os.path.join(output_dir, f"pdb_files_{category_map[suffix]}.csv")

        # Write each pair to the corresponding category CSV file
        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write the header
            csvwriter.writerow(['true_sequence_path', 'generated_sequence_path'])

            # Write the data rows
            for true_pdb, gen_pdb in pairs:
                # Strip any surrounding single quotes (just in case)
                true_pdb = true_pdb.strip("'")
                gen_pdb = gen_pdb.strip("'")
                csvwriter.writerow([true_pdb, gen_pdb])

# Example usage
base_directory = "/Users/leabroennimann/Desktop/alphafold_structure_predictions"
output_directory = "/Users/leabroennimann/Desktop/pdb_files_output_categories"

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Collect the PDB file pairs
pdb_files = collect_pdb_files(base_directory)

# Write the PDB file pairs to separate CSV files for each category
write_pdb_files_to_csv(pdb_files, output_directory)
