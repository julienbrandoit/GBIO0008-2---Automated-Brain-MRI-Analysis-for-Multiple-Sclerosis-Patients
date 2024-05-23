# Automated Brain MRI Analysis for Multiple Sclerosis Patients -  MI2024_T06

## Description of the problem and context of the tool developed

[Multiple sclerosis](https://en.wikipedia.org/wiki/Multiple_sclerosis) is a chronic autoimmune disease that affects the central nervous system, including the brain and the spinal cord. The immune system mistakenly attacks the protective covering of nerve fibers called myelin, causing inflammation and damage. This damage disrupts the transmission of electrical impulses along the nerves, resulting in a range of symptoms and signs. Symptoms of this disease can vary and can include fatigue, muscle weakness, eye pain, blurred or double vision, problems with coordination and balance, ...

Multiple sclerosis is generally diagnosed through a combination of medical history, neurological examination, and various tests such as MRI to detect areas of damage in the brain and spinal cord. In this project, we received Magnetic Resonance Images from multiple sclerosis patients and the goal is to automatize the characterization and quantification of the lesions in the brain.

Characterizing and quantifying lesions are two important steps in patient management. They enable physicians to adapt treatments and monitor disease progression. 

Although very important, these two tasks can be very time-consuming and subjective if they have to be performed by a human, which is why we are proposing an automatic characterization and quantification pipeline. In addition to being much faster than manual work, this tool enables standardized image processing and offers the possibility of defining parameters that have a direct bearing on the *specificity* and *sensitivity* required by the medical staff.

In this project, we are asked to extract several types of features from the MRI images : 
- *The volumetric features* include the number of lesions, their volume and volume compactness, their surface in each patient's brain, as well as global measures such as the total volume of the lesion, the total load lesion, ... 

- *The tissue features* provide valuable insights into the nature and pathology of the lesions. They include the mean, median and variance of the signal in each image, but also the distribution of intensities (unimodal, bimodal or multimodal), all of which allow for the study of tissue homogeneity within the lesions. 

## Installation


### Python Librairies 
The pipeline is developed in python and relies on several external libraries that need to be installed in the python environment in order to use the tool. These libraries are mainly used for image processing and image analysis.

To install the required libraries, follow the steps below:

1. **Nibabel**:
   - Nibabel is used for reading and writing neuroimaging data files.
   - Install it using pip:
     ```bash
     pip install nibabel
     ```

2. **Porespy**:
   - Porespy contains a collection of image analysis functions used to extract information from 3D images.
   - To install Porespy, use pip:
     ```bash
     pip install porespy
     ```

3. **Scipy.ndimage**:
   - Scipy.ndimage provides various image processing functions.
   - Scipy is a prerequisite for this, so if you haven't installed it yet, you can install both with pip:
     ```bash
     pip install scipy
     ```
4. **Pandas**, **Matplotlib**, **Numpy**:
    - These libraries are used to format data and perform certain numerical calculations.
      ```bash
      pip install pandas matplotlib numpy
      ```
5. **sklearn**:
   - sklearn is used for the automatic clustering method
     ```bash
     pip install sklearn
     ```
One can also run this single command to install all the libraries:
```bash
pip install scipy porespy nibabel pandas matplotlib numpy sklearn
```     

## Usage

The entire pipeline is currently gathered in the [`utils.py, part1.py and part2.py`](code) files and can be directly downloaded and imported into projects as it is done in the [demo notebooks](demo). You only have to download the [code](code) folder and name it as you want to give a name to the module (we have named it  `our_tools` in the notebooks).

### Data format
The images received are all in the NIfTI format. 

For each patients, 3 types of images are provided : 
- a FLAIR MR image showing lesions as bright spots. 
- 4 posterior probability maps `c1`,`c2`,`c3`,`c4` for respectively the grey matter, the white matter, the lesion and the cerebro-spinal fluid. The maps have been derived from the FLAIR images and are useful to segment the lesions.
- 4 quantitative MR images (`MT`,`PD`,`R1`,`R2s`) estimating physical properties of the tissue sample in each voxel.

The data organization must be done as follows in order to run the pipeline correctly: a base folder named data containing subfolders corresponding to each patient. The folder containing the images relative to one patient takes the following pattern : sub-MSPA{xxx}_warped where xxx is the patient number. 

For each patient, the probability maps must start with the prefix `wcx`where x can be either 1,2,3,4, indicating the type of map. The FLAIR MR image must start with the prefix `kwk`. The 4 quantitative MR images must start with the prefix `wkt`and must end with the suffix `MT`,`PD`,`R1` or `R2s_OLS`, depending on the type of images. 

All of these default path names and structures can be modified by passing arguments to the pipeline [see the [documentation page](Documentation.md)]. 

### Methods
A detailed presentation of the methods used is available on the following pages: [Methods for extracting volumetric features](Volumetric_features_methods.md) and [Methods for extracting tissue features](Tissue_features_methods.md). These pages also present a proof of concept detailing how the results extracted by our pipeline from five test patients, for example, can be used.
### Documentation and get started

Detailed documentation of all pipeline functions is available on the [documentation page](Documentation.md).

#### Extracting the volumetric features
For easy use of the pipeline, the *automatic_extraction_of_volumetric_features* function can be used directly. Below is an extract from the documentation explaining how to use it:
```python
automatic_extraction_of_volumetric_features(base_folder,
                                            subfolder_structure,
                                            patient_numbers, 
                                            patient_personnal_summary_path, 
                                            patient_global_summary_path):
    """
    Perform automatic extraction of volumetric features from patients data. Saving format is 'csv'.

    Args:
        base_folder (str): Path to the base folder containing patient data.
        subfolder_structure (str): Format string for subfolder structure containing patient data.
        patient_numbers (list): List of patient numbers.
        patient_personnal_summary_path (str): Path pattern for saving individual patient summary files.
        patient_global_summary_path (str): Path for saving global summary file.

    Returns:
        None
    """
  ```

For a concrete example, take a look at the demo notebook: [volumetric_features.ipynb](demo/volumetric_features.ipynb).

#### Extracting the tissue features
```python
automatic_tissue_features(base_folder, subfolder_structure patient_numbers,patient_personnal_summary_path):
    """
    Perform automatic extraction of tissue features and generate summary reports for patients. Saving format is 'csv'.

    Args:

        base_folder (str): Base folder containing patient data.
        subfolder_structure (str): Structure of subfolders within the base folder.
        patient_numbers (list): List of patient numbers to process.
        patient_personal_summary_path (str): Path format for saving individual patient summary files.

    Returns:
        None

    
    """
  ```
For a concrete example, take a look at the demo notebook: [tissue_features.ipynb](demo/tissue_features.ipynb).

#### Extracting everythings automatically

For easy use of the complete pipeline, the *automatic_volumetric_and_tissue_features* function can be used directly. Below is an extract from the documentation explaining how to use it:
```python
automatic_volumetric_and_tissue_features(base_folder,
                                            subfolder_structure,
                                            patient_numbers, 
                                            patient_personnal_summary_path_volumetric,
                                            patient_personnal_summary_path_tissues, 
                                            patient_global_summary_path):
    """
    Perform automatic extraction of volumetric annd tissue features from patients data. Saving format is 'csv'.

    Args:
        base_folder (str): Path to the base folder containing patient data.
        subfolder_structure (str): Format string for subfolder structure containing patient data.
        patient_numbers (list): List of patient numbers.
        patient_personnal_summary_path_volumetric (str): Path pattern for saving individual patient summary files for volumetric features.
        patient_personnal_summary_path_tissue (str): Path pattern for saving individual patient summary files for tissue features.
        patient_global_summary_path (str): Path for saving global summary file for volumetric features.

    Returns:
        None
    """
  ```

For a concrete example, take a look at the demo notebook: [global.ipynb](demo/global.ipynb).


## Authors and Acknowledgment

This project was carried out by 3 authors who contributed equally :

- Julien Brandoit
- Lucas Stordeur
- Anais Ledent
