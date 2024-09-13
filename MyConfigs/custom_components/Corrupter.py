# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

import mmcv
import mmengine.fileio as fileio

import rasterio as rio
import numpy as np

import numpy as np
from scipy.ndimage import convolve
from skimage import util

from .aux import OpticalSystem


def radiometric_spike_noise(array: np.ndarray, severity=.1):
    """
    Randomly amplifies or reduces the values in each column of a 2D or 3D array.

    This function iterates through each channel (2D slice) of the input array and modifies 
    its columns by applying a random amplification or reduction factor. The degree of 
    amplification or reduction is controlled by the `severity` parameter.

    Parameters:
    ----------
    array : numpy.ndarray
        A 2D or 3D numpy array. If the array is 3D, the first dimension is considered as 
        the channel dimension.
    severity : float, optional
        A positive float that controls the severity of the amplification/reduction. 
        A higher value will result in a greater range of possible changes. Default is 1.0.

    Returns:
    -------
    numpy.ndarray
        The modified array with amplified or reduced column values.
    """
    # Ensure severity is positive
    if severity <= 0:
        raise ValueError("Severity must be a positive number.")
    
    # If the input is 2D, convert it to 3D with a single channel
    if array.ndim == 2:
        array = np.expand_dims(array, axis=0)
    
    # Apply random amplification/reduction to each channel and column
    for channel in range(array.shape[0]):
        for col in range(array.shape[2]):
            # Generate a random factor for amplification/reduction
            factor = 1 + (np.random.rand() * severity * np.random.choice([-1, 1]))
            # Apply the factor to the column
            array[channel, :, col] *= factor
    
    # If the input was originally 2D, return to 2D
    if array.shape[0] == 1:
        return np.squeeze(array, axis=0)
    
    return array


def add_gaussian_noise(image: np.ndarray, mean: float = 0, stddev: float = 1.0) -> np.ndarray:
    """
    Adds Gaussian noise to a np.float32 image.

    Parameters:
    -----------
    image : np.ndarray
        The input image to be corrupted. Must be of type np.float32.
    stddev : float, optional
        The standard deviation of the Gaussian noise. This controls the severity of the noise.
        Default is 1.0.

    Returns:
    --------
    np.ndarray
        The corrupted image, maintaining the same np.float32 data type.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise ValueError("Input image must be a numpy array with dtype np.float32")

    # Generate Gaussian noise
    gaussian_noise = np.random.normal(loc=mean, scale=stddev, size=image.shape).astype(np.float32)

    # Add the Gaussian noise to the original image
    if stddev == 0:
        noisy_image = image
    else:
        noisy_image = image + gaussian_noise

    return noisy_image


def corrupt_image(image: np.ndarray, noise_type: str, severity: float) -> np.ndarray:
    """
    Corrupts a np.float32 image with a specified type and severity of noise using skimage.

    Parameters:
    -----------
    image : np.ndarray
        The input image to be corrupted. Must be of type np.float32.
    noise_type : str
        The type of noise to apply. Options include 'gaussian', 'salt', 'pepper', 
        's&p' (salt and pepper), 'speckle', 'poisson'.
    severity : float
        The severity of the noise. For most noise types, this is a value between 0 and 1, 
        where higher values correspond to more severe noise.

    Returns:
    --------
    np.ndarray
        The corrupted image, maintaining the same np.float32 data type.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise ValueError("Input image must be a numpy array with dtype np.float32")

    if noise_type == 'gaussian':
        noisy_image = add_gaussian_noise(image, mean=0, stddev=severity)
    elif noise_type == 'spike':
        noisy_image = radiometric_spike_noise(image, severity=severity)
    else:
        raise ValueError("Unsupported noise type. Choose from 'gaussian', 'spike'.")

    # Convert the noisy image back to np.float32 if needed
    return noisy_image.astype(np.float32)



@TRANSFORMS.register_module()
class ImageCorruption(BaseTransform):
    """Corruption augmentation for images.
    This class implements corruption transforms based on `skimage corruptions`.
    Attributes:
        sensor (str): The type of sensor, either 'sentinel' or 'venus'.
        SNR (float): Signal-to-noise ratio, must be a positive number.
        mtf_at_fe (float): Modulation transfer function at the Nyquist frequency.
        - img (np.uint8): The image to be corrupted.
        - img (np.uint8): The corrupted image.
        sensor (str): The type of sensor, either 'sentinel' or 'venus'.
        SNR (float, optional): Signal-to-noise ratio, default is 10.
        mtf_at_fe (float, optional): Modulation transfer function at the Nyquist frequency, default is 0.2.
    Raises:
        AssertionError: If SNR is not a positive number.
        AssertionError: If sensor is not 'sentinel' or 'venus'.
        RuntimeError: If `imagecorruptions` is not installed.
    Methods:
        transform(results: dict) -> dict:
            Applies corruption to the image in the results dictionary.
        __repr__() -> str:
            Returns a string representation of the ImageCorruption object.
    """

    def __init__(self, sensor: str, SNR: float = 10, mtf_at_fe: float = 0.2) -> None:
        self.sensor = sensor
        self.SNR = SNR

        self.mtf_at_fe = mtf_at_fe
        assert SNR >= 0, 'SNR must be a positive number'
        assert sensor in ['sentinel', 'venus'], 'sensor must be either sentinel or venus'
        
        self.optical_system = OpticalSystem(sensor, mtf_at_fe, SNR)



    def transform(self, results: dict) -> dict:
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """

        if self.corruption is None:
            raise RuntimeError('imagecorruptions is not installed')
        results['img'] = self.optical_system.apply(results['img'])
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(corruption={self.corruption}, '
        repr_str += f'severity={self.severity})'
        return repr_str



# @TRANSFORMS.register_module()
# class ImageCorruption(BaseTransform):
#     """Corruption augmentation.

#     Corruption transforms implemented based on
#     `skimage corruptions`.

#     Required Keys:

#     - img (np.uint8)


#     Modified Keys:

#     - img (np.uint8)


#     Args:
#         corruption (str): Corruption name.
#         severity (int): The severity of corruption. Defaults to 1.
#     """

#     def __init__(self, corruption: str, severity: int = 1) -> None:
#         self.corruption = corruption
#         self.severity = severity

#     def transform(self, results: dict) -> dict:
#         """Call function to corrupt image.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Result dict with images corrupted.
#         """

#         if self.corruption is None:
#             raise RuntimeError('imagecorruptions is not installed')
#         results['img'] = corrupt_image(
#                                 results['img'],
#                                 noise_type=self.corruption,
#                                 severity=self.severity)
#         return results

#     def __repr__(self) -> str:
#         repr_str = self.__class__.__name__
#         repr_str += f'(corruption={self.corruption}, '
#         repr_str += f'severity={self.severity})'
#         return repr_str