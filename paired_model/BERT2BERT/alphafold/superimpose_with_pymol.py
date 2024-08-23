
# run this script inside the pymol terminal -> PyMOL > run /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/alphafold/pymol_superimposition.py

import os
import csv
from pymol import cmd

output_image_base = "aligned_structure"


def align_and_plot(true_pdb, generated_pdb, output_image):
    # Load the true and generated PDB files
    true_name = "true_structure"
    generated_name = "generated_structure"

    cmd.load(true_pdb, true_name)
    cmd.load(generated_pdb, generated_name)

    # Align the generated protein to the true protein
    cmd.super(generated_name, true_name)

    # Set the colors for the structures
    cmd.color("green", true_name)
    cmd.color("magenta", generated_name)

    # Set the visualization style
    cmd.show("cartoon", true_name)
    cmd.show("cartoon", generated_name)

    # Apply consistent view settings
    cmd.set_view((
        0.660292029, -0.262237906, 0.703738868,
        -0.351670712, 0.720008850, 0.598259807,
        -0.663583994, -0.642511070, 0.383194178,
        0.000000000, 0.000000000, -138.657791138,
        -0.371109009, -0.880016327, -0.868656635,
        109.318862915, 167.996719360, -20.000000000))

    # Get the current view matrix (camera position)
    current_view = cmd.get_view()

    # Print the current view matrix
    print(f"View matrix for {output_image}:")
    for row in current_view:
        print(row)

    # Optionally, save the view matrix to a file
    view_file = output_image.replace(".png", "_view.txt")
    with open(view_file, "w") as vf:
        vf.write(f"View matrix for {output_image}:\n")
        for row in current_view:
            vf.write(f"{row}\n")

    # Render the image
    cmd.png(output_image, width=3200, height=2400, dpi=1200, ray=1)

    # Clear the session for the next pair
    cmd.delete("all")


# Function to process each category
def process_category(csv_file, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the PDB file pairs from the CSV file
    with open(csv_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        # Align and plot each pair individually
        for i, row in enumerate(reader, 1):
            true_pdb = row['true_sequence_path']
            generated_pdb = row['generated_sequence_path']

            output_image = f"{output_dir}/{output_image_base}_{i}.png"
            align_and_plot(true_pdb, generated_pdb, output_image)
            print(f"Generated image: {output_image}")


# Define the base directory for the output
output_base_dir = "/Users/leabroennimann/Downloads/pythonProject"

# Process the worst sequences
process_category("/Users/leabroennimann/Desktop/pdb_files_output_categories/pdb_files_worst.csv",
                 os.path.join(output_base_dir, "pymol_plots_worst"))

#Process the middle sequences
process_category("/Users/leabroennimann/Desktop/pdb_files_output_categories/pdb_files_middle.csv",
                 os.path.join(output_base_dir, "pymol_plots_middle"))

# Process the best sequences
process_category("/Users/leabroennimann/Desktop/pdb_files_output_categories/pdb_files_best.csv",
                 os.path.join(output_base_dir, "pymol_plots_best"))

