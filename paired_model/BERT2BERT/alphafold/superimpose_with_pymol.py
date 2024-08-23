
# run this script inside the pymol terminal -> PyMOL > run /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/alphafold/pymol_superimposition.py


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

    # Render the image
    cmd.png(output_image, width=3200, height=2400, dpi=1200, ray=1)

    # Clear the session for the next pair
    cmd.delete("all")


# Example usage
output_dir = "/Users/leabroennimann/Downloads/pythonProject"
output_image_base = "aligned_structure"

# Read the PDB file pairs from the txt file
with open("/Users/leabroennimann/Downloads/pythonProject/pdb_files.txt", "r") as pdb_files:
    content = pdb_files.readlines()

# Align and plot each pair individually
for i, line in enumerate(content, 1):
    # Clean up the path strings
    cleaned_line = line.strip()[2:-2]  # Remove the 2 first and last characters (which are "('" and "')"")
    true_pdb, gen_pdb = cleaned_line.split("', '")  # Split the line into the true and generated PDB paths
    output_image = f"{output_dir}/{output_image_base}_{i}.png"
    align_and_plot(true_pdb, gen_pdb, output_image)
    print(f"Generated image: {output_image}")

# No need to call cmd.quit() since PyMOL will be running