import os

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
    pdb_pairs = []

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
                        elif sub_folder.startswith(f"true_seq_{id_folder}_{suffix}"):
                            for file_name in os.listdir(sub_folder_path):
                                if "rank_001" in file_name and file_name.endswith(".pdb"):
                                    true_pdb_file = os.path.join(sub_folder_path, file_name)

                # Ensure both files are found before adding to the list
                if gen_pdb_file and true_pdb_file:
                    pdb_pairs.append((true_pdb_file, gen_pdb_file))
                else:
                    print(f"Warning: Could not find matching 'rank_001' PDB files for ID {id_folder} in {category_dir}")

    return pdb_pairs

# Example usage
base_directory = "/Users/leabroennimann/Desktop/alphafold_structure_predictions"
output_image_base = "aligned_structure"

# Collect the PDB file pairs
pdb_files = collect_pdb_files(base_directory)

with open('pdb_files.txt', 'w') as f:
    for line in pdb_files:
        f.write(f"{line}\n")


