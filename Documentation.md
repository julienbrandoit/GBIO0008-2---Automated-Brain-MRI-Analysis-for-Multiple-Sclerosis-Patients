# Documentation

The purpose of this page is to present documentation on the various functions used in the code. A combination of these functions can be used to customize the pipeline.

For direct use of the automatic extraction without any intermediate customization other than defining the various thresholds, please refer to function `automatic_extraction_of_volumetric_features`, whose documentation is available [here](#direct-use1), for the automatic extraction of the volumetric features. To automatically extract the tissue features, please refer to function `automatic_tissue_features`, whose documentation is available [here](#direct-use2). Finally, for the complete extraction of both volumetric and tissue features, please refer to the function `automatic_volumetric_and_tissue_features`, whose documentation is available [here](#direct-use3).

# File utils.py


## get_full_file_name_starting_and_ending_with(base_folder, start_string, end_string)

Retrieve the full file path of a patient file based on a starting and an ending string within a specified base folder.
### Arguments:  
- base_folder (str): The base folder where the patient files are located.
- start_string (str): The starting string of the patient file name.
- end_string (str): The ending string of the patient file name.

### Returns:
- str: The full file path of the matching patient file.

## plot_slices(image, slices, colors=['black', 'gray', 'white', 'red', 'blue', 'yellow'], tissue_names=['Non-intracranial matter', 'Gray matter', 'White matter', 'Lesion', 'CSF', 'Unclassified intracranial matter'])

Plot slices of an image with different tissue types highlighted.

This function visualizes slices of an image with different tissue types highlighted using a custom colormap. Each tissue type is assigned a unique color, and a legend is added to the plot to indicate tissue types.

### Arguments:
- image (numpy.ndarray): The 3D image array to visualize.
- slices (list): A list of slice numbers to plot.
- colors (list): A list of color names or RGB values for each tissue type. Defaults to predefined colors.
- tissue_names (list): A list of names for each tissue type. Defaults to predefined names.

By default, the association between the colors and the tissues is the following: black for Non-intracranial matter, gray for Gray matter, white for White matter, red for Lesion, blue for CSF and yellow for Unclassified intracranial matter.

### Returns:
- None.

## plot_basic_slices(data, min_slice, max_slice):

Plot slices of 3D volumetric data.

This function visualizes slices of an image using a grey colormap

### Arguments:
- data (numpy.ndarray): 3D array containing the volumetric data.
- min_slice (int): Index of the first slice to be plotted.
- max_slice (int): Index of the last slice to be plotted.

### Returns:
- None.

<a name="direct-use3"></a>
## automatic_volumetric_and_tissue_features(base_folder, subfolder_structure, patient_numbers, patient_personnal_summary_path_volumetric, patient_personnal_summary_path_tissues, patient_global_summary_path, connectivity = 'face', min_volume = 8, threshold = 40, min_samples=10)

This function does the automatic extraction of the volumetric and tissue features for multiple patients based on their data folder structure. 

For the automatic extraction of volumetric features, this function does the segmentation of brain tissues and lesions, compute the lesion properties, and generate patient-specific and global summary reports.

For the automatic extraction of tissue features, this function retrieves quantitative values for lesions, computes statistical properties (mean, median, variance, modes),
and generates individual patient summary reports.

### Arguments:
- base_folder (str): The base folder containing patient data.
- subfolder_structure (str): The subfolder structure format string for locating patient data.
- patient_numbers (list): A list of patient numbers to process.
- patient_personnal_summary_path_volumetric (str): The file path template for saving patient-specific summary reports of the volumetric features.
- patient_personnal_summary_path_tissues (str): The file path template for saving patient-specific summary reports of the tissue features.
- patient_global_summary_path (str): The file path for saving the global summary report.
- connectivity (str, optional): The type of connectivity used for lesion segmentation. Defaults to 'face'. For more informations about connectivity, please refers to the documentation of *get_segmentation_and_properties* function.
- min_volume (float, optional): The minimum volume threshold for filtering out small lesions. Defaults to 8.
- min_samples (int, optional): The minimum number of samples required to form a cluster. Defaults to 10.
- threshold (int, optional): The threshold value for the data size to compute bimodality. Defaults to 40.

### Returns:
- None.


# File part1.py

## get_probability_maps(path_to_patient_folder)

Return the probability maps of different tissues extracted from patient files.

This function loads patient files from a specified folder and extracts probability maps of different tissues.
The tissue types are represented by different channels in the patient files. It loads the probability maps corresponding to each tissue type and returns them as an array, along with the affine transformation matrix.

The probability maps have to start with the prefix `wcx` with $x \in \{1, 2, 3, 4\}$

### Arguments:
- path_to_patient_folder (str): The path to the folder containing patient files.
### Returns:
- tuple: A tuple containing:    
    - numpy.ndarray: An array of probability maps where each map represents a tissue type.
    - numpy.ndarray: The affine transformation matrix of the probability maps.


## classify_voxels(probability_maps, threshold_1 = 0.5, threshold_2 = 0.2)

Classify the type of voxels based on probability maps of different tissues.

This function classifies each voxel based on the probability maps of different tissues provided as input.
It assigns a label to each voxel indicating the tissue type it most likely belongs to.

### Arguments:
- probability_maps (numpy.ndarray): An array containing probability maps of different tissues.
- threshold_1 (float): Threshold for considering a voxel as empty space. Defaults to 0.5. This threshold is named "intracranial confidence threshold" in the methods.
- threshold_2 (float): Threshold for considering a voxel as a specific tissue type. Defaults to 0.2. This threshold is named "matter confidence threshold" in the methods.

### Returns:
- numpy.ndarray: An array containing the classified labels for each voxel.

## get_intracranial_volume_in_voxels(brain, void_value = 0)

Compute the number of voxels belonging to the intracranial volume.

This function calculates the number of voxels belonging to the intracranial volume based on the classified brain image provided as input.

### Arguments:
- brain (numpy.ndarray): An array containing the classified labels for each voxel.
- void_value (int): The value representing empty space in the classified brain image. Defaults to 0.

### Returns:
- int: The number of voxels belonging to the intracranial volume.

## get_voxel_properties(voxel_matrix)

Compute the volume and surface of one voxel.

This function calculates the volume and surface area of a voxel based on its transformation matrix.

### Arguments:
- voxel_matrix (numpy.ndarray): The transformation matrix representing the voxel.

### Returns:
- tuple: A tuple containing:
    - float: The volume of the voxel.
    - float: The surface area of the voxel (the average area of one face).

## get_segmentation_and_properties(map, connectivity = 'face')

Segment lesions in a map and compute their properties.

This function segments lesions within a given map based on the specified connectivity and computes properties for each segmented lesion.

### Arguments:
- map (numpy.ndarray): The map containing lesion information.
- connectivity (str): The type of connectivity used for segmentation.
        Possibilities are *face* for a 6 neighbours connectivity, *edge* for a 18 neighbours connectivity and *corner* for a 26 neighbours connectivity. Default connectivity is *face*.

### Returns:
- tuple: A tuple containing:
    - numpy.ndarray: The segmented lesions labeled in the map.
    - list: A list of dictionaries, each containing properties of a segmented lesion.


## filter_lesion(segmented_map, properties, voxel_volume, min_volume = 8)

Remove lesions with a volume lower than a given value.

This function filters out lesions from a segmented map based on their volume.
Lesions with a volume lower than the specified minimum volume are removed.


### Arguments:
- segmented_map (numpy.ndarray): The segmented map containing labeled lesions.
- properties (list): A list of dictionaries containing properties of each lesion.
- voxel_volume (float): The volume of a single voxel.
- min_volume (float): The minimum volume threshold for lesions to be retained. Defaults to 8.

### Returns:
- tuple: A tuple containing:
    - numpy.ndarray: The cleaned segmented map with small lesions removed.
    - list: A list of dictionaries containing properties of retained lesions.


## transform_properties_into_metrics(properties, voxel_volume, voxel_surface)

Transform lesion properties from voxel size into real metrics.

This function converts lesion properties such as volume and surface area, which are originally measured in voxel size, into real metrics (e.g., mm³ and mm²) based on the provided voxel volume and surface.


### Arguments:
- properties (list): A list of dictionaries containing properties of each lesion.
- voxel_volume (float): The volume of a single voxel.
- voxel_surface (float): The surface area of a single voxel.

### Returns:
- pandas.DataFrame: A DataFrame containing the transformed properties of lesions, including volume, surface area, and compactness.

## get_lesion_map_and_probability_affine(patient_folder)

This functions computes the segmented lesion map, affine matrix, number of voxels belonging to the brain, properties of the lesions, volume and surface of a voxel of a given patient.

### Arguments:
- patient_folder (str): The full file path of the matching patient file.

### Returns:
- tuple: A tuple containing:
    - numpy.ndarray: The segmented map containing labeled lesions.
    - numpy.ndarray: The affine transformation matrix of the probability maps.
    - int: The number of voxels belonging to the intracranial volume.
    - list: A list of dictionaries containing properties of each lesion.
    - float: The volume of a single voxel.
    - float: The surface area of a single voxel.


## make_a_summary(ID, patients, brain_volumes)

From the informations on the lesions of the different patients, create a summary containing the following informations:

* Patient number
* Intracranial Volume (mm³)
* Number of lesions
* Total lesion load [%]
* Total lesion volume (mm³)
* Average lesion volume (mm³)
* Standard deviation of lesion volume (mm³)
* Total lesion surface (mm²)
* Average lesion surface (mm²)
* Standard deviation of lesion surface (mm²)
* Average lesion compactness
* Standard deviation of lesion compactness


### Arguments:
- ID (list): A list of patient IDs.
- patients (dict): A dictionary containing DataFrame objects of lesion properties for each patient.
- brain_volumes (dict): A dictionary containing intracranial volumes for each patient.

### Returns:

- pandas.DataFrame: A DataFrame containing the computed measurements for all patients.

<a name="direct-use1"></a>
## automatic_extraction_of_volumetric_features(base_folder, subfolder_structure, patient_numbers, patient_personnal_summary_path, patient_global_summary_path, connectivity = 'face', min_volume = 8) 

Automatically extract volumetric features from patient data.

This function automates the extraction of volumetric features from patient data, including segmentation of brain tissues and lesions, computation of lesion properties, and generation of patient-specific and global summary reports.

### Arguments:
- base_folder (str): The base folder containing patient data.
- subfolder_structure (str): The subfolder structure format string for locating patient data.
- patient_numbers (list): A list of patient numbers to process.
- patient_personnal_summary_path (str): The file path template for saving patient-specific summary reports.
- patient_global_summary_path (str): The file path for saving the global summary report.
- connectivity (str): The type of connectivity to use for lesion segmentation. Defaults to 'face'.
- min_volume (float): The minimum volume threshold for filtering out small lesions. Defaults to 8.

### Returns:
- None.

# File part2.py

## get_complete_data_and_affine(patient_folder, start_string, end_string)

Load complete data and affine information from multiple files.

This function loads complete data and affine information from multiple files specified by starting and ending patterns.


### Arguments:
- patient_folder (str): Path to the folder containing patient data.
- start_string (list of str): List of strings representing the starting pattern of filenames.
- end_string (list of str): List of strings representing the ending pattern of filenames.

### Returns:
- tuple: A tuple containing:
    - list: A list of 3D numpy arrays containing the complete data from each file.
    - list: A list of affine transformation matrices corresponding to each loaded data.

## get_lesion_list(patient_segmented_lesion_map)

Create a list of individual lesion masks from a segmented lesion map.

### Arguments:
- patient_segmented_lesion_map (numpy.ndarray): The segmented map containing labeled lesions.

### Returns:
- list: A list of individual lesion masks, each represented as a binary numpy array.

## interpolation(x_lesion, data_affine, data, probability_affine)

Perform interpolation to estimate voxel values from a different affine space.

This function performs interpolation to estimate voxel values from a different affine space.
It first transforms the coordinates of the lesion voxels from the space of the data to the space of the probability map. Then, it uses linear interpolation (`method='linear'`) to estimate the voxel values.

### Arguments:
- x_lesion (numpy.ndarray): The coordinates of the lesion voxels in the space of the data.
- data_affine (numpy.ndarray): The affine transformation matrix of the data space.
- data (numpy.ndarray): The original data containing voxel values.
- probability_affine (numpy.ndarray): The affine transformation matrix of the probability map space.

### Returns:
- numpy.ndarray: The interpolated voxel values estimated from the original data.

## get_quantitative_list(lesions_list, data, affine, probability_affine)

Calculate quantitative measures for each lesion in a list of lesion masks.

### Arguments:
- lesions_list (list): A list of individual lesion masks, each represented as a binary numpy array.
- data (list): A list containing the original data for each modality.
- affine (list): A list containing the affine transformation matrices for each modality.
- probability_affine (numpy.ndarray): The affine transformation matrix of the probability map space.

### Returns:
- list: A list containing quantitative measures for each lesion, where each element is an array of measures for each voxel within the lesion.

## get_automatic_quantitative_list(base_folder, subfolder_structure, patient_number)

Generate automatic quantitative measures for lesions in a patient's data.

### Arguments:
- base_folder (str): The base folder containing patient data.
- subfolder_structure (str): The folder structure template for patient data.
- patient_number (int): The patient number or identifier.

### Returns:
- list: A list containing quantitative measures for each lesion, where each element is an array of measures for each voxel within the lesion.

## automatic_quantitative_all_patients(base_folder, subfolder_structure, patient_numbers)

Generate automatic quantitative measures for lesions in data of multiple patients.

### Arguments:
- base_folder (str): The base folder containing patient data.
- subfolder_structure (str): The folder structure template for patient data.
- patient_numbers (list): A list of patient numbers or identifiers.

### Returns:
- list: A list containing quantitative measures for lesions in data of multiple patients.
  Each element of the list corresponds to a patient, and contains a list of quantitative measures
  for each lesion in that patient's data.

## lesion_quantities_histogram(lesion_data, lesion_number)

Plot histograms of quantitative measures for a specific lesion.

### Arguments:
- lesion_data (numpy.ndarray): The quantitative measures for the lesion, with each row representing a voxel and each column representing a dimension of measurement.
- lesion_number (int): The number or identifier of the lesion.

### Returns:
- None

## plot_all_lesions_histogram(quantitative_list)

Plot histograms of quantitative measures for all lesions in a list.

### Arguments:
- quantitative_list (list): A list containing quantitative measures for lesions.
  Each element of the list corresponds to a lesion, and contains a list of quantitative measures
  for each voxel within that lesion.

### Returns:
- None


## find_peaks_fct(lesion_data,lesion_number)
Display histograms and detect peaks for a given set of lesion data.

### Arguments:
- lesion_data (numpy.ndarray): An array containing lesion data.
- lesion_number (int): The number identifying the lesion.

### Returns:
- int: Returns 1 if the distribution is unimodal (has a single peak), otherwise returns 0.

## find_peaks_all_lesions(quantitative_list)
Perform unimodality testing with peak detection for multiple lesions.

### Arguments:
- quantitative_list (list): A list containing quantitative data arrays for each lesion.

### Returns:
- list: A list of test results (1 for unimodal, 0 for multimodal) for each lesion.

## get_modes(lesion_data,lesion_number, threshold = 40)
Identify modes (peaks) in the histograms of intensity distributions for a lesion.

### Arguments:
- lesion_data (ndarray): NumPy array containing the quantitative data for a lesion.
- lesion_number (int): the number identifying the lesion
- threshold (int, optional): The threshold value for the data size. If the size of the data is less than this threshold, the function returns 1, corresponding to unimodality and the mean of the data. Defaults to 40.

### Returns:
- list: A list of modes (peaks) identified in the intensity histograms.

## separate_modes(mode)
Convert modes to string representations, handling multiple modes in a single dimension.

### Arguments:
- modes (list): A list of modes (peaks) identified in the intensity histograms.

### Returns:
- list: A list of string representations of modes, with multiple modes in a single dimension represented as a comma-separated string.

## quantities_stats(lesion_data)

Calculate statistical measures for quantitative measures of a lesion.

This function calculates statistical measures for the quantitative measures of a lesion.
The statistical measures computed are the mean, median, and covariance matrix of the lesion data.

### Arguments:
- lesion_data (numpy.ndarray): The quantitative measures for the lesion, with each row representing a voxel and each column representing a dimension of measurement.

### Returns:
- tuple: A tuple containing:
    - numpy.ndarray: The mean of the quantitative measures for the lesion.
    - numpy.ndarray: The median of the quantitative measures for the lesion.
    - numpy.ndarray: The covariance matrix of the quantitative measures for the lesion.

## plot_stats(lesion_data, lesion_number)

Plot statistical measures for quantitative measures of a specific lesion.

### Arguments:
- lesion_data (numpy.ndarray): The quantitative measures for the lesion, with each row representing a voxel and each column representing a dimension of measurement.
- lesion_number (int): The number or identifier of the lesion.

### Returns:
- None

## get_properties(lesion_values, threshold=40, min_samples=10)
Compute statistical properties and modes for a list of lesion values.

### Arguments:
- lesions_values (list): A list containing values of lesions for each patient.
- min_samples (int, optional): The minimum number of samples required to form a cluster. Defaults to 10.
- threshold (int, optional): The threshold value for the data size. Defaults to 40.

### Returns:
- tuple: A tuple containing computed statistical properties and modes for each lesion:
   - numpy.ndarray: Array of mean values for each lesion.
   - numpy.ndarray: Array of median values for each lesion.
   - numpy.ndarray: Array of variance values for each lesion
   - list: List indicating unimodal (1) or multimodal (0) distribution for each lesion using peak test method.
   - numpy.ndarray: Array containing modes (peaks) for each lesion dimension using peak test method.
   - list: List indicating unimodal (1) or multimodal (0) distribution for each lesion using clustering method.
   - numpy.ndarray: Array containing modes for each lesion dimension using clustering method.
   - list: A matrix containing the bimodal separation between the clusters.

## data_frame(lesion_number, mean, median, variance, unimodality_pt, modes_pt, unimodality_clustering, modes_clustering, modes_separation)
 Create a pandas DataFrame containing statistical properties and modes for lesions.

### Arguments:
- lesion_number (int): Total number of lesions.
- mean (numpy.ndarray): Array of mean values for each lesion and variable.
- median (numpy.ndarray): Array of median values for each lesion and variable.
- variance (numpy.ndarray): Array of variance values for each lesion and variable.
- unimodality_pt (list): List indicating unimodal (1) or multimodal (0) distribution for each lesion using peak test method.
- modes_pt (numpy.ndarray): Array containing modes (peaks) for each lesion dimension using peak test method.
- unimodality_clustering (list): List indicating unimodal (1) or multimodal (0) distribution for each lesion using automatic clustering method.
- modes_clustering (numpy.ndarray): Array containing modes (peaks) for each lesion dimension using automatic clustering method.
- modes_separatio (list): A matrix containing the bimodal separation between the clusters.

### Returns:
- pandas.DataFrame: DataFrame containing statistical properties and modes for each lesion.

## modality_using_clustering(data, min_samples=10, threshold=40)
Perform clustering on the given data to identify modalities, returning the number of modalities and their centroids if the data size exceeds a certain threshold. If the data size is lower than a threshold, then data is considered as unimodal.

### Arguments:
- data (numpy.ndarray): The dataset to perform clustering on. Each row represents a data point.
- min_samples (int, optional): The minimum number of samples required to form a cluster. Defaults to 10.
- threshold (int, optional): The threshold value for the data size. If the size of the data is less than this threshold, the function returns 1, corresponding to unimodality and the mean of the data. Defaults to 40.

### Returns:
- tuple: A tuple containing:
    - int: The number of identified clusters, corresponding to number of modalities.
    - list: A list of centroids representing the mean values of each cluster.
    - list: A matrix containing the bimodal separation between the clusters.

<a name="direct-use2"></a>
## automatic_tissue_features(base_folder, subfolder_structure, patient_numbers, patient_personal_summary_path, threshold = 40, min_samples=10)
Perform automatic extraction of tissue features and generate summary reports for patients.

This function automates the extraction of tissue features for multiple patients based on their data folder structure.
It retrieves quantitative values for lesions, computes statistical properties (mean, median, variance, modes),
and generates individual patient summary reports in CSV format. Additionally, it saves a global summary file combining data from all patients.

### Arguments:
- base_folder (str): Base folder containing patient data.
- subfolder_structure (str): Structure of subfolders within the base folder.
- patient_numbers (list): List of patient numbers to process.
- patient_personal_summary_path (str): Path format for saving individual patient summary files.
- min_samples (int, optional): The minimum number of samples required to form a cluster. Defaults to 10.
- threshold (int, optional): The threshold value for the data size to compute bimodality. Defaults to 40.

### Returns:
- None