import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

def plot_images_in_grid(image_paths, grid_size=(3, 5), figsize=(15, 10)):
    fig, axes = plt.subplots(*grid_size, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if i < len(image_paths):
            img = imread(image_paths[i])
            ax.imshow(img)
            ax.axis('off')  # Hide the axes
        else:
            ax.axis('off')  # Hide empty subplots

    plt.tight_layout()
    plt.savefig("grid_plot.png")
    plt.show()

def get_image_paths(directory, num_images=5):
    # Get a sorted list of image files in the directory
    image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".png")])
    return image_files[:num_images]

# Define the directories where images are stored
worst_dir = "/Users/leabroennimann/Downloads/pythonProject/pymol_plots_worst"
middle_dir = "/Users/leabroennimann/Downloads/pythonProject/pymol_plots_middle"
best_dir = "/Users/leabroennimann/Downloads/pythonProject/pymol_plots_best"

# Get the paths for the first 5 images from each category
worst_images = get_image_paths(worst_dir, 5)
middle_images = get_image_paths(middle_dir, 5)
best_images = get_image_paths(best_dir, 5)

# Combine all image paths for the grid
all_images = worst_images + middle_images + best_images

# Plot the images in a 5x3 grid
plot_images_in_grid(all_images, grid_size=(3, 5))
