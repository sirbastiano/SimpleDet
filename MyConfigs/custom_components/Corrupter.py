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
from skimage import util

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
    elif noise_type == 'salt':
        noisy_image = util.random_noise(image, mode='salt', amount=severity)
    elif noise_type == 'pepper':
        noisy_image = util.random_noise(image, mode='pepper', amount=severity)
    elif noise_type == 's&p':
        noisy_image = util.random_noise(image, mode='s&p', amount=severity)
    elif noise_type == 'speckle':
        noisy_image = util.random_noise(image, mode='speckle', var=severity**2)
    elif noise_type == 'poisson':
        noisy_image = util.random_noise(image, mode='poisson')
    else:
        raise ValueError("Unsupported noise type. Choose from 'gaussian', 'salt', 'pepper', 's&p', 'speckle', 'poisson'.")

    # Convert the noisy image back to np.float32 if needed
    return noisy_image.astype(np.float32)

@TRANSFORMS.register_module()
class ImageCorruption(BaseTransform):
    """Corruption augmentation.

    Corruption transforms implemented based on
    `skimage corruptions`.

    Required Keys:

    - img (np.uint8)


    Modified Keys:

    - img (np.uint8)


    Args:
        corruption (str): Corruption name.
        severity (int): The severity of corruption. Defaults to 1.
    """

    def __init__(self, corruption: str, severity: int = 1) -> None:
        self.corruption = corruption
        self.severity = severity

    def transform(self, results: dict) -> dict:
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """

        if self.corruption is None:
            raise RuntimeError('imagecorruptions is not installed')
        results['img'] = corrupt_image(
            results['img'],
            noise_type=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(corruption={self.corruption}, '
        repr_str += f'severity={self.severity})'
        return repr_str