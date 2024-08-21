# used environment: pymol
import pymol
from pymol import cmd
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

# run this script inside the pymol terminal -> PyMOL > run /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/alphafold/pymol_superimposition.py


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


def align_and_plot(true_pdb, generated_pdb, output_image):
    # Load the true and generated PDB files
    true_name = "true_structure"
    generated_name = "generated_structure"

    cmd.load(true_pdb, true_name)
    cmd.load(generated_pdb, generated_name)

    # Align the generated protein to the true protein
    cmd.super(generated_name, true_name)

    # Make the structures transparent for better visualization
    cmd.show("cartoon", true_name)
    cmd.show("cartoon", generated_name)
    cmd.color("limegreen", true_name)
    cmd.color("orange", generated_name)

    # Group structures to manage visualization
    cmd.group("alignment", f"{true_name} {generated_name}")

    # Set the viewing parameters
    cmd.set_view((
        0.660292029, -0.262237906, 0.703738868,
        -0.351670712, 0.720008850, 0.598259807,
        -0.663583994, -0.642511070, 0.383194178,
        0.000000000, 0.000000000, -138.657791138,
        -0.371109009, -0.880016327, -0.868656635,
        109.318862915, 167.996719360, -20.000000000))

    # Render the image
    cmd.png(output_image, width=3200, height=2400, dpi=1200, ray=1)
    cmd.save(output_image)

    # Optionally, save the PyMOL session
    # cmd.save(output_image + ".pse")  # Uncomment if you want to save the session

    # Clear the session for the next pair
    cmd.delete("all")


# Example usage
base_directory = "/Users/leabroennimann/Desktop/alphafold_structure_predictions"
output_image_base = "aligned_structure"

# Collect the PDB file pairs
pdb_files = collect_pdb_files(base_directory)
output_dir = "/Users/leabroennimann/Downloads/pythonProject"

# Align and plot each pair individually
for i, (true_pdb, gen_pdb) in enumerate(pdb_files, 1):
    output_image = f"{output_dir}/{output_image_base}_{i}.png"
    align_and_plot(true_pdb, gen_pdb, output_image)