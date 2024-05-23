import os
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import our_tools.part1 as part1
import our_tools.part2 as part2

def get_full_file_name_starting_and_ending_with(base_folder, start_string, end_string):
    pattern = os.path.join(base_folder, start_string + "*" + end_string + ".nii")
    return glob.glob(pattern)[0]

def plot_slices(image, slices, colors=['black', 'gray', 'white', 'red', 'blue', 'yellow'], tissue_names=['Non-intracranial matter', 'Gray matter', 'White matter', 'Lesion', 'CSF', 'Unclassified intracranial matter']):
    # Define custom colormap
    custom_cmap = plt.cm.colors.ListedColormap(colors)

    # Create normalization for custom colormap
    norm = plt.cm.colors.Normalize(vmin=0, vmax=len(colors)-1)

    num_slices = len(slices)
    num_rows = int(num_slices ** 0.5)  # Calculate the number of rows for squared layout
    num_cols = (num_slices + num_rows - 1) // num_rows  # Calculate the number of columns
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for idx, slice_num in enumerate(slices):
        row = idx // num_cols
        col = idx % num_cols

        # Apply custom colormap explicitly
        colored_image = custom_cmap(image[:, :, slice_num])

        axes[row, col].imshow(colored_image, cmap=custom_cmap, norm=norm, interpolation='nearest')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Slice {slice_num}')

    # Create legend
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=tissue_names[i]) for i in range(len(colors))]
    fig.legend(handles=legend_elements, loc='center right')

    # Add a common title for the subplot grid
    fig.suptitle('Slices Visualization')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_basic_slices(data, min_slice, max_slice):

    slices = range(min_slice, max_slice)
    num_slices = len(slices)
    num_rows = int(num_slices**0.5)  # Calculate the number of rows for squared layout
    num_cols = (num_slices + num_rows - 1) // num_rows  # Calculate the number of columns
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for idx, slice_num in enumerate(slices):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].imshow(data[:, :, slice_num],interpolation='nearest', cmap = 'gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Slice {slice_num}')

    # Add a common title for the subplot grid
    fig.suptitle('Slices Visualization')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
    
def automatic_volumetric_and_tissue_features(base_folder, subfolder_structure, patient_numbers, patient_personal_summary_path_volumetric, patient_personal_summary_path_tissues, patient_global_summary_path, connectivity = 'face', min_volume = 8, threshold = 40, min_samples=10):

    part1.automatic_extraction_of_volumetric_features(base_folder, subfolder_structure, patient_numbers, patient_personal_summary_path_volumetric, patient_global_summary_path, connectivity, min_volume)

    part2.automatic_tissue_features(base_folder, subfolder_structure, patient_numbers, patient_personal_summary_path_tissues, threshold, min_samples)


