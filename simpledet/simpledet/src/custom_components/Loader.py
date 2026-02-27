# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

import mmcv
import mmengine.fileio as fileio

import rasterio as rio
import numpy as np
from PIL import Image
import cv2

# Required by loader
def read_tif(file_path, band_indices):
    """
    Reads specified bands from a TIFF file.

    Parameters:
    - file_path (str): Path to the .tif file.
    - band_indices (list of int): Indices of the bands to read.

    Returns:
    - numpy.ndarray: A numpy array containing the stacked band data.
    """
    data = []
    with rio.open(file_path) as src:
        for index in band_indices:
            data.append(src.read(index))
    stacked_data = np.stack(data, axis=0)
    return np.transpose(stacked_data, (1, 2, 0))


@TRANSFORMS.register_module()
class SelBandLoader(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img: The loaded image data.
    - img_shape: The shape of the loaded image.
    - ori_shape: The original shape of the loaded image.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        bands_list (list): Indices of the bands to read from the TIFF file.
            Defaults to [1].
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the file
            backend. It may contain a 'backend' key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
    """

    def __init__(self,
                 to_float32: bool = False,
                 bands_list: list = [1],
                 ignore_empty: bool = False,
                 decode_backend = 'cv2',
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.bands_list = bands_list
        # Check if bands_list is valid
        if len(self.bands_list) == 0:
            raise ValueError("bands_list cannot be empty.")
        if len(self.bands_list) == 1:
            self.color_type = 'grayscale'
        if len(self.bands_list) == 3:
            self.color_type = 'color'
        if len(self.bands_list) > 3:
            raise ValueError("bands_list cannot have more than 3 bands for png files.")

        self.backend_args: Optional[dict] = None
        self.imdecode_backend = decode_backend
        self.file_client_args = None
        
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        img = self.load_img(filename)

        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        
        # print(f"Loaded image shape: {img.shape}")
        # print("Original shape: ", results['ori_shape'])
        # print("Image shape: ", results['img_shape'])
        
        return results


    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"bands_list={self.bands_list}, "
                    f'backend_args={self.backend_args})')

        return repr_str
    
    
    def load_img(self, filename):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            try:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
                img = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.imdecode_backend)

            except Exception as e:
                if self.ignore_empty:
                    return None
                else:
                    raise f"Exception raised: {e}"
        
        
        elif filename.lower().endswith(('.tif', '.tiff')):
            try:
                img = read_tif(filename, self.bands_list)
            except Exception as e:
                if self.ignore_empty:
                    return None
                else:
                    raise f"Exception raised: {e}"
        else:
            raise ValueError(f"Unsupported image format: {filename}")
        
        return img

        