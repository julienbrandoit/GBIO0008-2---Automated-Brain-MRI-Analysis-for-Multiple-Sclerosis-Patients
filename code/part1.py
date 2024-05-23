import nibabel as nib
import numpy as np
import porespy as ps
import scipy.ndimage as spim
import pandas as pd

import our_tools.utils as utils

# Return the probability maps of the different tissues
def get_probability_maps(path_to_patient_folder):
    probability_maps = []
    probability_affine = None

    for c in [1, 2, 3, 4]:
        
        # Load patient file
        file_path = utils.get_full_file_name_starting_and_ending_with(path_to_patient_folder, f"wc{c}", "*")
        map = nib.load(file_path)  #Load patient file

        # Get the data array from the image
        data = map.get_fdata()
        probability_maps.append(np.array(data))

        if probability_affine is not None and not np.array_equal(np.array(map.affine), probability_affine):
            print("Error ! It seems that the probability maps don't match, check the contents of the file.")
        else:
            probability_affine = np.array(map.affine)
    
    # Compute map of voxels containing empty space
    c5 = 1 - np.sum(probability_maps, axis=0)
    probability_maps.append(c5)

    return np.array(probability_maps), probability_affine

# Classify the type of the voxels given the probability maps
def classify_voxels(probability_maps, threshold_1 = 0.5, threshold_2 = 0.2):

    # we put a void value (0) if sum(probability[0:4]) < threshold_1 i.e. if probability[4] > 1 - threshold_1
    # we put value 1 (2, 3 or 4) if probability[0] (1, 2 or 3) > threshold_2 and probability[0] (1, 2 or 3) is the maximum value between probability[0], probability[1], probability[2] and probability[3] and not void
    # else we put unknown value (5)
    classified_brain = np.zeros(probability_maps[0].shape)

    void_mask = probability_maps[-1] > 1 - threshold_1
    max_prob_class = np.argmax(probability_maps[:4], axis=0)
    max_prob_mask = np.max(probability_maps[:4], axis=0) > threshold_2

    classified_brain = np.where(void_mask, 0, np.where(max_prob_mask, max_prob_class + 1, 5))

    return classified_brain

# Compute the number of voxels belonging to the intracranial volume
def get_intracranial_volume_in_voxels(brain, void_value = 0):
    count_void = np.count_nonzero(brain == void_value)
    return brain.size - count_void

# Compute the volume and surface of one voxel
def get_voxel_properties(voxel_matrix):
    
    volume_voxel = abs(np.linalg.det(voxel_matrix))
    x = np.linalg.norm(voxel_matrix[:,0])
    y = np.linalg.norm(voxel_matrix[:,1])
    z = np.linalg.norm(voxel_matrix[:,2])
    surface_voxel = (2*x*y + 2*y*z + 2*x*z)/6 

    return volume_voxel, surface_voxel
    
def get_segmentation_and_properties(map, connectivity = 'face'):
    
    if connectivity == "face":
        s = spim.generate_binary_structure(3,1)
        
    elif connectivity == "edge":
        s = spim.generate_binary_structure(3,2)
        
    elif connectivity == "corner":
        s = spim.generate_binary_structure(3,3)
        
    else:
        print("The connectivity given does not exist. Possibilities are: face, edge and corner")
        exit(1)
    
    data_segm, _ = spim.label(map, structure = s)
    properties = ps.metrics.regionprops_3D(data_segm)

    return data_segm, properties

# Remove lesions with a volume lower than a given value
def filter_lesion(segmented_map, properties, voxel_volume, min_volume = 8):
    cleaned_properties = []
    cleaned_map = np.copy(segmented_map)

    clean_index = 1

    for l in range(len(properties)):
        property = properties[l]
        lesion_volume = property.volume * voxel_volume

        if lesion_volume < min_volume:
            cleaned_map[cleaned_map == l + 1] = 0
        else:
            cleaned_map[cleaned_map == l + 1] = clean_index
            clean_index += 1
            cleaned_properties.append(property)

    return cleaned_map, cleaned_properties

# Transform the lesions properties in voxel size into real metrics
def transform_properties_into_metrics(properties, voxel_volume, voxel_surface):
    lesion_num = len(properties)

    lesion_indices = []
    lesion_volumes = []
    lesion_areas = []
    lesion_compactnesses = []

    for l in range(lesion_num):
        property = properties[l]
        lesion_index = l + 1
        lesion_volume = property.volume * voxel_volume
        lesion_area = property.surface_area * voxel_surface
        lesion_compactness = 36*np.pi*lesion_volume**2/(lesion_area**3)

        lesion_indices.append(lesion_index)
        lesion_volumes.append(lesion_volume)
        lesion_areas.append(lesion_area)
        lesion_compactnesses.append(lesion_compactness)

    # Dataframe containing the lesions and their properties for a given patient
    patient = pd.DataFrame({
        'Volume (mm³)': lesion_volumes,
        'Surface (mm²)' : lesion_areas,
        'Compactness': lesion_compactnesses
    }, index=lesion_indices)
    patient.index.name = 'Lesion'

    return patient

def get_lesion_map_and_probability_affine(patient_folder):

    patient_maps, probability_affine = get_probability_maps(patient_folder)
    volume_voxel, surface_voxel = get_voxel_properties(probability_affine)
    patient_brain = classify_voxels(patient_maps)

    patient_brain_volume = get_intracranial_volume_in_voxels(patient_brain)
    patient_brain_volume *= volume_voxel

    patient_lesion_map = patient_brain == 3 #c3

    patient_segmented_lesion_map, patient_lesion_properties = get_segmentation_and_properties(patient_lesion_map)
    patient_segmented_lesion_map, patient_lesion_properties = filter_lesion(patient_segmented_lesion_map, patient_lesion_properties, volume_voxel)

    return patient_segmented_lesion_map, probability_affine, patient_brain_volume, patient_lesion_properties, volume_voxel, surface_voxel

# Compute different measurements of lesion properties for all patient
def make_a_summary(ID, patients, brain_volumes):
    patient_indices = []
    patient_nbrs = []
    patient_loads = []
    patient_total_volumes = []
    patient_avg_volumes = []
    patient_std_volumes = []
    patient_total_surfaces = []
    patient_avg_surfaces = []
    patient_std_surfaces = []
    patient_avg_compactnesses = []
    patient_std_compactnesses = []

    for p in ID:
        patient = patients[p]

        patient_index = p
        patient_nbr = len(patient)
        patient_total_volume = patient['Volume (mm³)'].sum()
        patient_load = patient_total_volume/brain_volumes[p] * 100
        patient_avg_volume = patient['Volume (mm³)'].mean()
        patient_std_volume = patient['Volume (mm³)'].std(ddof = 0)
        patient_total_surface = patient['Surface (mm²)'].sum()
        patient_avg_surface = patient['Surface (mm²)'].mean()
        patient_std_surface = patient['Surface (mm²)'].std(ddof = 0)
        patient_avg_compactness = patient['Compactness'].mean()
        patient_std_compactness = patient['Compactness'].std(ddof = 0)

        patient_indices.append(patient_index)
        patient_nbrs.append(patient_nbr)
        patient_total_volumes.append(patient_total_volume)
        patient_loads.append(patient_load)
        patient_avg_volumes.append(patient_avg_volume)
        patient_std_volumes.append(patient_std_volume)
        patient_total_surfaces.append(patient_total_surface)
        patient_avg_surfaces.append(patient_avg_surface)
        patient_std_surfaces.append(patient_std_surface)
        patient_avg_compactnesses.append(patient_avg_compactness)
        patient_std_compactnesses.append(patient_std_compactness)
     
    # Stock all measurements into a Dataframe
    summary = pd.DataFrame({
        'Patient': patient_indices,
        'Intracranial Volume (mm³)': brain_volumes,
        'Number of lesions': patient_nbrs,
        'Total lesion load [%]' : patient_loads,
        'Total lesion volume (mm³)': patient_total_volumes,
        'Average lesion volume (mm³)': patient_avg_volumes,
        'Std lesion volume (mm³)': patient_std_volumes,
        'Total lesion surface (mm²)' : patient_total_surfaces,
        'Average lesion surface (mm²)': patient_avg_surfaces,
        'Std lesion surface (mm²)': patient_std_surfaces,
        'Average lesion compactness': patient_avg_compactnesses,
        'Std lesion compactness': patient_std_compactnesses
    })

    return summary

def automatic_extraction_of_volumetric_features(base_folder, subfolder_structure, patient_numbers, patient_personnal_summary_path, patient_global_summary_path, connectivity = 'face', min_volume = 8):
    patient_informations = []

    for patient_number in patient_numbers:
        patient_folder = base_folder + subfolder_structure.format(patient_number)
        _, _, patient_brain_volume, patient_lesion_properties, volume_voxel, surface_voxel = get_lesion_map_and_probability_affine(patient_folder)

        patient_information = transform_properties_into_metrics(patient_lesion_properties, volume_voxel, surface_voxel)

        local_df = pd.DataFrame({   'Patient_Information': [patient_information],
                                    'Patient_Brain_Volume': [patient_brain_volume]},
                                    index=[patient_number])

        patient_informations.append(local_df)

    patient_informations = pd.concat(patient_informations)

    for ID, patient in patient_informations.iterrows():
        patient_number = ID
        patient_information = patient['Patient_Information']
        patient_information.to_csv(patient_personnal_summary_path.format(patient_number), index=True)

    all_summary = make_a_summary(patient_informations.index, patient_informations['Patient_Information'], patient_informations['Patient_Brain_Volume'])
    all_summary.to_csv(patient_global_summary_path, index=False)