import json
import math
from pathlib import Path 
import os
import numpy as np
import pandas as pd 
from tqdm import tqdm


import rasterio as rio
from skimage import io, color, feature, exposure
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

from skimage import img_as_ubyte
from skimage.filters import (
    threshold_otsu, threshold_yen, threshold_isodata, threshold_li,
    threshold_mean, threshold_minimum, threshold_triangle, threshold_local
)

import seaborn as sns



# Set Seaborn style for a more professional look

def set_style():
    sns.set(style="whitegrid", context="notebook")

    plt.rcParams['font.family'] = 'STIXGeneral'
    # Set the DPI for the plots
    plt.rcParams['figure.dpi'] = 500
    
    fontSize = 13
    # Update Matplotlib rcParams for font size
    plt.rcParams.update({
        'font.size': fontSize,
        'axes.titlesize': fontSize,
        'axes.labelsize': fontSize,
        'xtick.labelsize': fontSize,
        'ytick.labelsize': fontSize,
        'legend.fontsize': fontSize,
        'figure.titlesize': fontSize
    })

    # Update Seaborn context with font size settings
    sns.set_context("paper", rc={
        "font.size": fontSize,
        "axes.titlesize": fontSize,
        "axes.labelsize": fontSize,
        "xtick.labelsize": fontSize,
        "ytick.labelsize": fontSize,
        "legend.fontsize": fontSize,
        "figure.titlesize": fontSize
    })

def read_tif(file_path, band_indices):
    """
    Reads specified bands from a TIFF file.

    Parameters:
    - file_path (str): Path to the .tif file.
    - band_indices (list of int): Indices of the bands to read.

    Returns:
    - dict: A dictionary where keys are band indices and values are the corresponding band data.
    """
    data = {}
    with rio.open(file_path) as src:
        crs = src.crs
        transform = src.transform
        
        for index in band_indices:
            data[index] = src.read(index)
    return data, crs, transform


def get_coco(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data


def convert_to_dict(tuple_list):
    """
    Converts a list of tuples into a dictionary where the key is 254 and the value is a list of all the associated lists (boxes).

    Parameters:
    tuple_list (list): A list of tuples, where each tuple contains a key and a list of values.

    Returns:
    dict: A dictionary with the key 254 and the value as a list of all the associated lists (boxes).
    """
    result_dict = {}
    
    for key, value in tuple_list:
        if key not in result_dict:
            result_dict[key] = []
        result_dict[key].append(value)
    
    return result_dict


def list_files(folder):
    f = Path(folder)
    tmp = [x for x in f.glob('**/*.npy')]
    return tmp


def get_band_files(band_num):
    files = list_files(f'/Data_large/marine/PythonProjects/MMDET/studies/crops/B{band_num}')
    return [x.as_posix() for x in files] 


def get_corresponding_B1_files(band_num):
    B1_files = get_band_files(band_num=1)
    corresponding = [x.replace('B1',f'B{band_num}') for x in B1_files]
    
    if band_num == 1:
        return B1_files
    else:
        return corresponding


def compute_hog_features(image_array, threshold=0.0):
    """
    Compute Histogram of Oriented Gradients (HOG) features for an image and 
    apply a threshold to filter out low-intensity gradients in the resulting HOG image.
    
    Parameters:
    image_array (ndarray): Input image as a NumPy array.
    threshold (float): Minimum intensity threshold for HOG features.

    Returns:
    tuple: 
        - hog_features (ndarray): The HOG features of the image.
        - hog_image (ndarray): The HOG visualization image with the applied threshold.
    """
    # Compute HOG features and the HOG image
    hog_features, hog_image = feature.hog(image_array, pixels_per_cell=(2, 2), 
                                          cells_per_block=(1, 1), visualize=True, 
                                          block_norm='L2-Hys')
    
    # Apply threshold to HOG features
    hog_features[hog_features < threshold] = 0
    
    return hog_features, hog_image


def compute_lbp_features(image_array, radius=1, n_points=None, method='uniform'):
    """
    Compute Local Binary Pattern (LBP) features for an image. LBP is a visual descriptor used for texture classification.
    
    Parameters:
    image_array (ndarray): Input image as a NumPy array.
    radius (int): Radius of the circle used to compute the LBP. Default is 1.
    n_points (int or None): Number of points to consider in the circular neighborhood. If None, it defaults to 8 * radius.
    method (str): Method to determine the pattern ('default', 'ror', 'uniform', 'var'). Default is 'uniform'.

    Returns:
    ndarray: The LBP feature image with the same dimensions as the input image, where each pixel value represents the LBP value of the corresponding pixel in the input image.
    """
    # If n_points is not specified, set it to 8 * radius
    if n_points is None:
        n_points = 8 * radius
    
    # Compute the LBP features
    lbp_image = feature.local_binary_pattern(image_array, n_points, radius, method=method)
    
    return lbp_image


def count_hog_features(image_array, threshold=0.0):
    """
    Count the number of significant HOG features in an image with a given threshold.
    
    Parameters:
    image_array (ndarray): Input image as a NumPy array.
    threshold (float): Minimum intensity threshold for HOG features.
    
    Returns:
    int: Number of significant HOG features.
    """
    # Compute HOG features with threshold
    hog_features, _ = compute_hog_features(image_array, threshold)
    # Count non-zero entries in the HOG feature vector
    num_features = np.count_nonzero(hog_features)
    return num_features


def visualize_img(image_array, size=2):
    """
    Visualize an image.
    
    Parameters:
    image_array (ndarray): Input image as a NumPy array.
    threshold (float): Minimum intensity threshold for HOG features.
    """
    fig, (ax1) = plt.subplots(1, 1, figsize=(size, size), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image_array, cmap=plt.cm.jet)

    plt.show()
    

def visualize_hog(image_array, threshold=0.0):
    """
    Visualize HOG features for an image with a given threshold.
    
    Parameters:
    image_array (ndarray): Input image as a NumPy array.
    threshold (float): Minimum intensity threshold for HOG features.
    """
    # Compute HOG features and image with threshold
    hog_features, hog_image = compute_hog_features(image_array, threshold)
    # Rescale the histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    # Plot the original and HOG images
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    
    ax1.axis('off')
    ax1.imshow(image_array, cmap=plt.cm.jet)
    ax1.set_title('Original Image')

    ax2.axis('off')
    ax2.imshow(hog_image, cmap=plt.cm.gray)
    ax2.set_title('HOG Features')

    plt.show()
    

def get_hog_and_features(input_img, threshold_value=0.85):
    """
    Get HOG image and number of significant HOG features for an input image.

    Parameters:
    input_img (ndarray): Input image as a NumPy array.
    threshold_value (float): Minimum intensity threshold for HOG features.

    Returns:
    tuple: HOG image, number of significant HOG features.
    """
    num_features = count_hog_features(input_img, threshold=threshold_value)
    hog_features, hog_image = compute_hog_features(input_img, threshold=threshold_value)
    return hog_image, num_features


def plot_bands_and_hog(array_in_band, threshold_value=0.85):
    """
    Plot original images and their HOG (Histogram of Oriented Gradients) features for each band in a 2-row layout.
    Utilizes Seaborn for enhanced aesthetics.

    Parameters:
    array_in_band (dict): Dictionary where keys are band indices (1-12) and values are 2D numpy arrays representing images.
    threshold_value (float): Minimum intensity threshold for HOG features, defaults to 0.85.

    Returns:
    None: The function displays a plot with the original images and their corresponding HOG features.
    """


    # Create a 2-row subplot layout
    fig, axes = plt.subplots(2, 12, figsize=(20, 2), sharex=True, sharey=True, dpi=300)
    
    for band in range(1, 13):
        # Get the image and its corresponding HOG features
        image_array = array_in_band[band]
        hog_image, num_features = get_hog_and_features(image_array, threshold_value)
        
        # Plot the original image in the first row
        axes[0, band-1].imshow(image_array, cmap='jet')
        axes[0, band-1].axis('off')
        axes[0, band-1].set_title(f'Band {band}', fontsize=10, fontweight='bold')
        
        # Plot the HOG features in the second row
        axes[1, band-1].imshow(hog_image, cmap='gray')
        axes[1, band-1].axis('off')
        axes[1, band-1].set_title(f'N. feats: ({num_features})', fontsize=9, fontweight='bold')
    
    # Adjust layout and spacing
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    plt.show()
    
    
def process_images(B, threshold_value=0.75):
    """
    Process images to compute HOG features for multiple bands.

    Parameters:
    B (dict): A dictionary where the keys are band numbers (1 to 12) and the values are lists of file paths.

    Returns:
    tuple: Two dictionaries:
        - band_num_hog: A dictionary where keys are band numbers (1 to 12) and values are lists of HOG feature counts.
        - stemToHog: A dictionary where keys are file stems and values are HOG feature counts.
    """
    
    band_num_hog = {i: [] for i in range(1, 13)}
    stemToHog = {}

    for i in tqdm(range(3826), desc="Processing images"):
        for sel_band in range(1, 13):
            filename = B[sel_band][i]
            try:
                image_array = np.load(filename)
                num_features = count_hog_features(image_array, threshold_value)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
            
            band_num_hog[sel_band].append(num_features)
            
            stem = Path(filename).stem
            stemToHog[stem] = num_features
            
    return band_num_hog, stemToHog


def plot_hog_features(band_num_hog):
    """
    Plot histograms of HOG features for each band and a bar plot of the average values.

    Parameters:
    band_num_hog (dict): A dictionary where keys are band numbers (1 to 12) and values are lists of HOG feature counts.
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3), dpi=300, sharex=True)


    # Extract HOG features for each band
    hog_features_by_band = [band_num_hog[band] for band in range(1, 13)]


    # Combine all HOG features from all bands into a single array
    all_hog_features = np.concatenate(hog_features_by_band)

    # Apply min-max normalization across all bands
    min_val = np.min(all_hog_features)
    max_val = np.max(all_hog_features)
    normalized_all_hog_features = (all_hog_features - min_val) / (max_val - min_val)

    # Split the normalized data back into the respective bands
    split_indices = np.cumsum([len(band) for band in hog_features_by_band[:-1]])
    normalized_hog_features_by_band = np.split(normalized_all_hog_features, split_indices)

    # Calculate mean and standard deviation for each band after normalization
    means = [np.mean(band) for band in normalized_hog_features_by_band]
    stds = [np.std(band) for band in normalized_hog_features_by_band]
    
    # Plot HOG features as a box plot
    # ax1.boxplot(normalized_hog_features_by_band, patch_artist=True, boxprops=dict(facecolor="lightblue"), showfliers=False)


    # Customizing the plot
    ax1.set_title('Distribution of HOG Features by Band')
    ax1.set_xlabel('Band Number')
    ax1.set_ylabel('HOG Features')
    
    ax1.set_xticks(range(0, 12, 1))
    ax1.set_xticklabels([f'B{i}' for i in range(1, 13)], rotation=0)
    
    ax1.plot(range(0, 12), means, linestyle='-', color='b') 
    ax1.errorbar(range(0, 12), means, yerr=stds, fmt='o', ecolor='b', capsize=4, capthick=1, color='b', markersize=6)
    
    
    
    # AX2:
    # data = pd.read_pickle('mAPs.pkl')
    # data_50 = pd.read_pickle('mAPs_50.pkl')
    # data_75 = pd.read_pickle('mAPs_75.pkl')

    def custom_sort_key(key):
        """
        Extracts the numeric part of the key for sorting purposes.
        
        Parameters:
        key (str): The dictionary key to be processed.
        
        Returns:
        int: The numeric part of the key.
        """
        return int(key.split('_b')[-1])

    # sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: custom_sort_key(item[0]))}
    # sorted_data_50 = {k: v for k, v in sorted(data_50.items(), key=lambda item: custom_sort_key(item[0]))}
    # sorted_data_75 = {k: v for k, v in sorted(data_75.items(), key=lambda item: custom_sort_key(item[0]))}


    # # Extract keys and values maintaining the order
    # for s in [sorted_data, sorted_data_50, sorted_data_75]:
    #     labels = list(s.keys())
    #     values = list(s.values())
    #     ax2.plot(labels, values, marker='o', linestyle='-')
    
    # ax2.set_title('Line Plot of Given Data')
    # ax2.set_xlabel('Band Number')
    # ax2.set_ylabel('$mAP$')
    # ax2.legend(['$mAP$', '$mAP_{50}$','$mAP_{75}$'])

    # # Rotate x-axis labels for better readability
    # # Setting the x-ticks
    # ax2.set_xticks(range(0, 12))
    # ax2.set_xticklabels([f'B{i}' for i in range(1, 13)], rotation=0)
    # ax2.set_ylim([0.,1])
    
    plt.tight_layout()
    plt.show()
    

def plot_hog_n_noise(band_num_hog):
    """
    Plot histograms of HOG features for each band and a bar plot of the average values.

    Parameters:
    band_num_hog (dict): A dictionary where keys are band numbers (1 to 12) and values are lists of HOG feature counts.
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3), dpi=300, sharex=True)


    # Extract HOG features for each band
    hog_features_by_band = [band_num_hog[band] for band in range(1, 13)]


    # Combine all HOG features from all bands into a single array
    all_hog_features = np.concatenate(hog_features_by_band)

    # Apply min-max normalization across all bands
    min_val = np.min(all_hog_features)
    max_val = np.max(all_hog_features)
    normalized_all_hog_features = (all_hog_features - min_val) / (max_val - min_val)

    # Split the normalized data back into the respective bands
    split_indices = np.cumsum([len(band) for band in hog_features_by_band[:-1]])
    normalized_hog_features_by_band = np.split(normalized_all_hog_features, split_indices)

    # Calculate mean and standard deviation for each band after normalization
    means = [np.mean(band) for band in normalized_hog_features_by_band]
    stds = [np.std(band) for band in normalized_hog_features_by_band]
    
    # Plot HOG features as a box plot
    # ax1.boxplot(normalized_hog_features_by_band, patch_artist=True, boxprops=dict(facecolor="lightblue"), showfliers=False)


    # Customizing the plot
    ax1.set_title('Distribution of HOG Features by Band')
    ax1.set_xlabel('Band Number')
    ax1.set_ylabel('HOG Features')
    
    ax1.set_xticks(range(0, 12, 1))
    ax1.set_xticklabels([f'B{i}' for i in range(1, 13)], rotation=0)
    
    ax1.plot(range(0, 12), means, linestyle='-', color='b') 
    ax1.errorbar(range(0, 12), means, yerr=None, fmt='o', ecolor='b', capsize=4, capthick=1, color='b', markersize=6)
    
    
    
    # AX2:
    # data = pd.read_pickle('mAPs.pkl')
    # data_50 = pd.read_pickle('mAPs_50.pkl')
    # data_75 = pd.read_pickle('mAPs_75.pkl')

    def custom_sort_key(key):
        """
        Extracts the numeric part of the key for sorting purposes.
        
        Parameters:
        key (str): The dictionary key to be processed.
        
        Returns:
        int: The numeric part of the key.
        """
        return int(key.split('_b')[-1])

    # sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: custom_sort_key(item[0]))}
    # sorted_data_50 = {k: v for k, v in sorted(data_50.items(), key=lambda item: custom_sort_key(item[0]))}
    # sorted_data_75 = {k: v for k, v in sorted(data_75.items(), key=lambda item: custom_sort_key(item[0]))}


    # # Extract keys and values maintaining the order
    # for s in [sorted_data, sorted_data_50, sorted_data_75]:
    #     labels = list(s.keys())
    #     values = list(s.values())
    #     ax2.plot(labels, values, marker='o', linestyle='-')
    
    # ax2.set_title('Line Plot of Given Data')
    # ax2.set_xlabel('Band Number')
    # ax2.set_ylabel('$mAP$')
    # ax2.legend(['$mAP$', '$mAP_{50}$','$mAP_{75}$'])

    # # Rotate x-axis labels for better readability
    # # Setting the x-ticks
    # ax2.set_xticks(range(0, 12))
    # ax2.set_xticklabels([f'B{i}' for i in range(1, 13)], rotation=0)
    # ax2.set_ylim([0.,1])
    
    plt.tight_layout()
    plt.show()
    


def threshold(grayscale_image, method: str = 'otsu', block_size: int = 35, offset: float = 10) -> None:
    """
    Applies a specified thresholding method to separate the foreground and background of an image.

    Parameters:
    grayscale_image (ndarray): The input grayscale image.
    method (str): The thresholding method to use ('otsu', 'yen', 'isodata', 'li', 'mean', 
                  'minimum', 'triangle', 'local'). Default is 'otsu'.
    block_size (int): The size of the neighborhood (must be odd) used for local thresholding. 
                      Used only if method is 'local'.
    offset (float): Constant subtracted from the mean or weighted mean. Used only if method 
                    is 'local'.

    Returns:
    ndarray: The binary image resulting from the thresholding process, as an 8-bit image.
    """
    if method == 'otsu':
        thresh_value = threshold_otsu(grayscale_image)
        binary_image = grayscale_image > thresh_value
    elif method == 'yen':
        thresh_value = threshold_yen(grayscale_image)
        binary_image = grayscale_image > thresh_value
    elif method == 'isodata':
        thresh_value = threshold_isodata(grayscale_image)
        binary_image = grayscale_image > thresh_value
    elif method == 'li':
        thresh_value = threshold_li(grayscale_image)
        binary_image = grayscale_image > thresh_value
    elif method == 'mean':
        thresh_value = threshold_mean(grayscale_image)
        binary_image = grayscale_image > thresh_value
    elif method == 'minimum':
        thresh_value = threshold_minimum(grayscale_image)
        binary_image = grayscale_image > thresh_value
    elif method == 'triangle':
        thresh_value = threshold_triangle(grayscale_image)
        binary_image = grayscale_image > thresh_value
    elif method == 'local':
        local_thresh = threshold_local(grayscale_image, block_size=block_size, offset=offset)
        binary_image = grayscale_image > local_thresh
    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods: 'otsu', 'yen', 'isodata', 'li', "
                         "'mean', 'minimum', 'triangle', 'local'.")

    # Convert the binary image to an 8-bit image for visualization
    binary_image_ubyte = img_as_ubyte(binary_image)
    return binary_image_ubyte


def calculate_fit_distances(mask):
    """
    Calculate the distances to fit the foreground (True values) to the
    top, left, right, and bottom of a binary mask.

    Parameters:
    mask (numpy.ndarray): Binary mask where True represents the foreground.

    Returns:
    dict: Distances to the top, left, right, and bottom boundaries.
    """
    # Get indices of the foreground
    rows, cols = np.where(mask)
    
    if len(rows) == 0 or len(cols) == 0:
        print("Error. Keeping original bounding box.")
        return {
            'top': 0,
            'left': 0,
            'right': 0,
            'bottom': 0
        }
    else:
        # Calculate distances
        top_dist = min(rows)
        bottom_dist = mask.shape[0] - max(rows) - 1
        left_dist = min(cols)
        right_dist = mask.shape[1] - max(cols) - 1
        
        return {
            'top': top_dist,
            'left': left_dist,
            'right': right_dist,
            'bottom': bottom_dist
        }


def update_bbox_with_dist(bbox, dist):
    """
    Aggiorna un bounding box nel formato COCO utilizzando le correzioni da dist.

    Args:
    bbox (list): Bounding box nel formato COCO [x, y, width, height].
    dist (dict): Distanze dai bordi {'top': int, 'left': int, 'right': int, 'bottom': int}.

    Returns:
    list: Bounding box aggiornato [x, y, width, height].
    """
    x, y, width, height = bbox

    # Applicare le correzioni
    new_x = x + dist['left']
    new_y = y + dist['top']
    new_width = width - dist['left'] - dist['right']
    new_height = height - dist['top'] - dist['bottom']

    return [new_x, new_y, new_width, new_height]


def convert_np_to_native(obj):
    """
    Recursively convert numpy data types to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_np_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_native(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_to_json(data, file_path):
    """
    Saves the given data to a JSON file specified by file_path.

    Parameters:
    data (dict): The data to be saved. This should be a dictionary.
    file_path (str): The file path where the JSON file will be saved.

    Returns:
    None
    """
    try:
        # Convert any non-serializable types to native Python types
        data = convert_np_to_native(data)
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)  # Using indent for better readability
        print("Data successfully saved to", file_path)
    except Exception as e:
        print("Failed to save data:", e)