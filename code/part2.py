import nibabel as nib
import numpy as np
from scipy.interpolate import interpn

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import pandas as pd


from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

import our_tools.utils as utils
import our_tools.part1 as part1


def get_complete_data_and_affine(patient_folder, start_string, end_string):

    data = []
    affine = []

    for i in range(len(end_string)):

        path = utils.get_full_file_name_starting_and_ending_with(patient_folder, start_string[i], end_string[i])
        Map = nib.load(path)
        data.append(Map.get_fdata())
        affine.append(Map.affine)

    return data, affine

def get_lesion_list(patient_segmented_lesion_map):
    lesions_list = []
    for i in range(np.max(patient_segmented_lesion_map)):
        local_map = (patient_segmented_lesion_map == i+1)
        lesions_list.append(local_map)
    
    return lesions_list


def interpolation(x_lesion, data_affine, data, probability_affine):
    
    x = np.dot(np.linalg.inv(data_affine), np.dot(probability_affine, x_lesion))
    points_data = (range(data.shape[0]), range(data.shape[1]), range(data.shape[2]))
    interp_data = interpn(points_data, data, x[:3], method='linear', bounds_error=False, fill_value=np.nan)
    
    return interp_data[0]


def get_quantitative_list(lesions_list, data, affine, probability_affine):

    quantitative_list = []
    for lesion_map in lesions_list:
        qt_lesion = []
        for i_lesion,j_lesion,k_lesion in np.argwhere(lesion_map != 0):
            local_vector = np.zeros(5, dtype=np.float32)
            x_lesion = np.array([i_lesion, j_lesion, k_lesion, 1])

            for i in range(len(data)):
                local_vector[i] = interpolation(x_lesion, affine[i], data[i], probability_affine)

            qt_lesion.append(local_vector)
            
        quantitative_list.append(np.array(qt_lesion))

    return quantitative_list
    
def get_automatic_quantitative_list(base_folder, subfolder_structure, patient_number):

    start_string = ["kwk", "wkt", "wkt", "wkt", "wkt"]
    end_string = ["*", "MT", "PD", "R1", "R2s_OLS"]

    patient_folder = base_folder + subfolder_structure.format(patient_number)

    patient_segmented_lesion_map, probability_affine, _, _, _, _ = part1.get_lesion_map_and_probability_affine(patient_folder)

    data, affine = get_complete_data_and_affine(patient_folder, start_string, end_string)

    lesions_list = get_lesion_list(patient_segmented_lesion_map)

    quantitative_list = get_quantitative_list(lesions_list, data, affine, probability_affine)

    return quantitative_list

def automatic_quantitative_all_patients(base_folder, subfolder_structure, patient_numbers):

    list = []

    for patient_number in patient_numbers:

        list.append(get_automatic_quantitative_list(base_folder, subfolder_structure, patient_number))
        
    return list


def lesion_quantities_histogram(lesion_data, lesion_number):

    plt.figure(figsize=(10, 6)) 
    images = ['FLAIR', 'MT', 'PD', 'R1', 'R2s']
    # Create marginal histograms
    for dim in range(lesion_data.shape[1]):
        plt.subplot( 2, 3, dim+1)
        sns.histplot(lesion_data[:, dim], bins='auto', kde=True)
        plt.title(f'Histogram for Dimension {images[dim]}')

    plt.suptitle(f'Histogram for lesion {lesion_number}', fontsize = 15)
    plt.tight_layout()  
    plt.show()
    
def plot_all_lesions_histogram(quantitative_list):
    
    for lesion_number in range(len(quantitative_list)):

        data = quantitative_list[lesion_number]
        lesion_quantities_histogram(data, lesion_number+1)


def find_peaks_fct(lesion_data, lesion_number):
    
    plt.figure(figsize=(10, 6))
    pic_number = []
    images = ['FLAIR', 'MT', 'PD', 'R1', 'R2s']
    for dim in range(1, lesion_data.shape[1]):
        plt.subplot(2, 2, dim)
        sns.histplot(lesion_data[:, dim], bins='auto', kde=True)

        ax = plt.gca() 
        line = ax.lines[0] 
        x_kde, y_kde = line.get_data()
        
        indices_pics, _ = find_peaks(y_kde, height=0.05 * np.max(y_kde),prominence=0.05) 
        pic_number.append(len(indices_pics))
        plt.title(f'Histogram for Dimension {images[dim]}')

        plt.plot(x_kde[indices_pics], y_kde[indices_pics], 'ro', markersize=8, label='Peaks')
    
    plt.suptitle(f'Histogram for Lesion {lesion_number}', fontsize=15)
    plt.tight_layout()
    plt.close()
    
    
    if np.sum(pic_number) > (lesion_data.shape[1]-1):
        return 0
    
    else:
        return 1
    
def find_peaks_all_lesions(quantitative_list, threshold = 40):
    
    unimodality_peak_test = []
    for lesion_number in range(len(quantitative_list)):
        data = quantitative_list[lesion_number]
        if len(data) < threshold:
            unimodality_peak_test.append(1)
        else:
            unimodality_peak_test.append(find_peaks_fct(data,lesion_number +1))
    
    return unimodality_peak_test
    
def get_modes(lesion_data, lesion_number, threshold = 40):
    plt.figure(figsize=(10, 6))
    pic_number = []
    images = ['FLAIR', 'MT', 'PD', 'R1', 'R2s']
    modes = []
    for dim in range(1, lesion_data.shape[1]):
        plt.subplot(2, 2, dim)
        sns.histplot(lesion_data[:, dim], bins='auto', kde=True)

        ax = plt.gca() 
        line = ax.lines[0] 
        x_kde, y_kde = line.get_data()
        
        if len(lesion_data) < threshold:
            indices_pics = [np.argmax(y_kde)]
        else:
            indices_pics, _ = find_peaks(y_kde, height=0.05 * np.max(y_kde),prominence=0.05)

        if(len(indices_pics) == 0):
            indices_pics =  [np.argmax(y_kde)]

        pic_number.append(len(indices_pics))
        modes.append(x_kde[indices_pics])
        
        plt.title(f'Histogram for Dimension {images[dim]}')

        plt.plot(x_kde[indices_pics], y_kde[indices_pics], 'ro', markersize=8, label='Peaks')
    
    plt.suptitle(f'Histogram for Lesion {lesion_number}', fontsize=15)
    plt.tight_layout()
    plt.close()
    
    flattened_modes = [value[0] if len(value) == 1 else value.tolist() for value in modes]
    
    return flattened_modes
       
def separate_modes(modes):
    final_mode = []
    
    for i in range(len(modes)):
        mode = modes[i]
        mode_string = []
        for j in range(len(mode)):
            
            if isinstance(mode[j], list): #Cela veut dire qu'il y a plus qu'un seul mode
                mode_string.append(','.join(map(str, mode[j])))
            else:
                mode_string.append(str(mode[j]))
                
        final_mode.append(mode_string)
    
    return final_mode
                     
def join_modes(input_string):
    # Split the input string by commas and strip any surrounding whitespace
    result_list = [item.strip() for item in input_string.split(',')]
    return result_list

def quantities_stats(lesion_data):

    # Calculate mean, median, and covariance matrix
    mean_data = np.mean(lesion_data, axis=0)
    median_data = np.median(lesion_data, axis=0)
    cov_matrix = np.cov(lesion_data, rowvar=False)
    variance = np.var(lesion_data, axis = 0)

    return mean_data, median_data, cov_matrix, variance

def plot_stats(lesion_data, lesion_number):

    mean_data, median_data, cov_matrix, _ = quantities_stats(lesion_data)
    labels = ['FLAIR', 'MT', 'PD', 'R1', 'R2s']
    x = range(len(labels))


    # Plot heatmaps
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    sns.heatmap(mean_data.reshape(1, -1), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
    plt.title('Mean')
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.ylabel('')

    plt.subplot(1, 3, 2)
    sns.heatmap(median_data.reshape(1, -1), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
    plt.title('Median')
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.ylabel('')

    plt.subplot(1, 3, 3)
    sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Covariance Matrix')
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.ylabel('')

    plt.suptitle(f'Statistics of lesion {lesion_number}', fontsize = 15)
    plt.tight_layout()
    plt.show()
    
def get_properties(lesions_values, threshold=40, min_samples=10):
    mean = []
    median = []
    variance = []
    modes_pt = []

    unimodality_c = []
    modes_c = []
    modes_separation_c =[]

    for i in range(len(lesions_values)):

        mean_lesion, median_lesion, covariance_lesion, variance_lesion = quantities_stats(lesions_values[i])
        mean.append(mean_lesion)
        median.append(median_lesion)
        variance.append(variance_lesion)
        modes_pt.append(get_modes(lesions_values[i],i, threshold))

        n_clusters, centroids, modes_separation = modality_using_clustering(lesions_values[i], min_samples, threshold)
        unimodality_c.append(1 if n_clusters == 1 else 0)
        modes_c.append(centroids)
        modes_separation_c.append(modes_separation)
    
    unimodality_pt = find_peaks_all_lesions(lesions_values, threshold)
    final_modes_pt = separate_modes(modes_pt)
    
    mean = np.array(mean)
    median = np.array(median)
    variance = np.array(variance)
    final_modes_pt = np.array(final_modes_pt)
    
    return mean, median, variance, unimodality_pt, final_modes_pt, unimodality_c, modes_c, modes_separation_c
  
def data_frame(lesion_number, mean, median, variance, unimodality_pt, modes_pt, unimodality_clustering, modes_clustering, modes_separation):
    nbr_modes = []
    for i in range(len(modes_pt)):
        nbr_modes.append([len(join_modes(modes_pt[i][j])) for j in range(4)])

    nbr_modes_clustering = []

    for i in range(len(modes_clustering)):
        nbr_modes_clustering.append(len(modes_clustering[i]))

    MT_modes_list = []
    PD_modes_list = []
    R1_modes_list = []
    R2s_modes_list = []

    for i in range(len(modes_clustering)):
        transposed_modes = np.array(modes_clustering[i]).T
        MT_modes = transposed_modes[0].flatten().tolist()
        PD_modes = transposed_modes[1].flatten().tolist()
        R1_modes = transposed_modes[2].flatten().tolist()
        R2s_modes = transposed_modes[3].flatten().tolist()

        MT_modes_list.append(MT_modes)
        PD_modes_list.append(PD_modes)
        R1_modes_list.append(R1_modes)
        R2s_modes_list.append(R2s_modes)

    data = { 
                'Lesion number': np.arange(1,lesion_number+1),
                'Mean FLAIR': mean[:, 0],
                'Mean MT': mean[:,1],
                'Mean PD': mean[:,2],
                'Mean R1': mean[:,3],
                'Mean R2s': mean[:,4],
                'Median FLAIR': median[:,0],
                'Median MT': median[:,1],
                'Median PD': median[:,2],
                'Median R1': median[:,3],
                'Median R2s': median[:,4],
                'Variance FLAIR': variance[:,0],
                'Variance MT': variance[:,1],
                'Variance PD': variance[:,2],
                'Variance R1': variance[:,3],
                'Variance R2s': variance[:,4],
                'Unimodal distribution peaks test': unimodality_pt,
                'Number of modes peaks test':nbr_modes,
                'Modes MT peaks test': modes_pt[:,0],
                'Modes PD peaks test': modes_pt[:,1],
                'Modes R1 peaks test': modes_pt[:,2],
                'Modes R2s peaks test':modes_pt[:,3],
                'Unimodal distribution clustering': unimodality_clustering,
                'Number of modes clustering': nbr_modes_clustering,
                'Modes MT clustering': MT_modes_list,
                'Modes PD clustering': PD_modes_list,
                'Modes R1 clustering': R1_modes_list,
                'Modes R2s clustering': R2s_modes_list,
                'Modes separation clustering': modes_separation
                }
        
    df = pd.DataFrame(data)
    
    return df
        

def automatic_tissue_features(base_folder, subfolder_structure, patient_numbers, patient_personal_summary_path, threshold = 40, min_samples=10):

    patient_informations = []

    for patient_number in patient_numbers:

        lesions_values = get_automatic_quantitative_list(base_folder, subfolder_structure, patient_number)

        mean, median, variance, unimodality_pt, modes_pt, unimodality_c, modes_c, modes_separation = get_properties(lesions_values, threshold, min_samples)
        
        df = data_frame(len(lesions_values),mean,median,variance,unimodality_pt,modes_pt,unimodality_c,modes_c, modes_separation)
        
        df.to_csv(patient_personal_summary_path.format(patient_number), index=False)
                    

def modality_using_clustering(data, min_samples=10, threshold=40):

    if len(data) < threshold:
        return 1, [np.mean(data, axis=0)], [0]

    X = StandardScaler().fit_transform(data[:, 1:])
    clustering = OPTICS(min_samples=min_samples).fit(X)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    centroids = []
    for cluster_id in range(n_clusters_):
        cluster_points = data[labels == cluster_id, 1:]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

    mode_separation = np.zeros((len(centroids),len(centroids)))

    for i in range(len(centroids)):
        for j in range(len(centroids)):

            mode_separation[i,j] = np.linalg.norm(centroids[i] - centroids[j])

    return n_clusters_, centroids, mode_separation.flatten().tolist()