import os, sys
import torch
import numpy as np
import pandas as pd 
import random
import json
import re

from pathlib import Path

# To set deterministic behaviour:
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'
sys.path.append('/Data_large/marine/PythonProjects/MMDET/MyConfigs')

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.evaluation import DumpDetResults

from mmdet.utils import setup_cache_size_limit_of_dynamo
import argparse


def set_seed(seed):
    # Set the seed for generating random numbers in PyTorch
    torch.manual_seed(seed)
    # If using GPUs, ensure that the random numbers are generated the same way
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Set the seed for generating random numbers in Python
    random.seed(seed)
    
    # Set the seed for generating random numbers in numpy
    np.random.seed(seed)
    
    # Ensure deterministic behavior by setting the flag
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optionally, set environment variables to ensure reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

def init_cfg(folder):
    cfg = Config.fromfile(f'{folder}/vfnet_r18.py')
    return cfg

def extract_first_band(path):
    # Regular expression to extract the first number after the first 'b'
    match = re.search(r'perfect_b(\d+)', path)

    # Extracted value
    if match:
        first_b_number = match.group(1)
        return int(first_b_number)

def list_weights(folder):
    return [x for x in Path(folder).rglob('*.pth') if 'best' not in str(x)]

def get_weight_b(band_sel):
    weigths = list_weights(f'/Data_large/marine/PythonProjects/MMDET/checkpoints/VENuS/Single/perfect_b{band_sel}')
    assert len(weigths) > 0, 'No weights found'
    weight_path = weigths[0]
    return weight_path


def set_NoiseTesting_config(weight_path, sensor: str, SNR: float, mtf_at_fe: float, Seed: int = 18):
    """
    Set the configuration for noise testing.
    Args:
        weight_path (str): The path to the weight file.
        sensor (str): The sensor: sentinel or venus
        SNR (float): The signal to noise ratio.
        mtf_at_fe (float): The modulation transfer function at the Nyquist frequency.
        Seed (int) : The seed for the random number generator.
    Returns:
        Config: The modified configuration object.
    """
    IMG_SIZE = 2048 # Default
    
    if isinstance(weight_path, str) :
        cfgDir = Path(weight_path).parent
    else:
        cfgDir = weight_path.parent
    
    
    cfg = init_cfg(folder=f'{cfgDir}')
    band_sel = extract_first_band(weight_path.as_posix())
    assert band_sel is not None, f'Band not found in the weight path: {weight_path}'
    assert band_sel in [x for x in range(1,13,1)], f'Band not in the range [1,12]: {band_sel}'
    
    # Modify Config
    CHECKPOINT = weight_path
    cfg.load_from = f"{CHECKPOINT}"
    print(f'Loading from: {CHECKPOINT}')
    cfg.test_dataloader.dataset.pipeline = [{'type': 'SelBandLoader', 'to_float32': True, 'bands_list': [band_sel]},
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(keep_ratio=False, scale=(IMG_SIZE,IMG_SIZE,), type='Resize'),
                        dict(type='ImageCorruption', sensor=sensor, SNR=SNR, mtf_at_fe=mtf_at_fe), # 'gaussian', 
                        dict(
                            meta_keys=('img_path', 'img_id', 'seg_map_path', 
                                    'height', 'width', 'instances', 'sample_idx', 
                                    'img', 'img_shape', 'ori_shape', 'scale', 'scale_factor', 
                                    'keep_ratio', 'homography_matrix', 'gt_bboxes', 'gt_ignore_flags', 
                                    'gt_bboxes_labels'),
                            type='PackDetInputs'),
                    ]

    work_dir = '/Data_large/marine/PythonProjects/MMDET/studies/tta_study'
    cfg.work_dir = work_dir
    
    # Hook for using the SIoU intestead of the IoU:
    cfg.test_evaluator.type = 'SIoUCocoMetric'
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Run noise testing study.')
    parser.add_argument('--sensor', type=str, required=True, choices=['sentinel', 'venus'], help='The sensor type.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    
    Results = {'Band':[], 'SNR':[], 'MTF':[], 'Precision':[], 'Recall':[], 'F1':[]}

    SENSOR = args.sensor
    BAND = 10 if SENSOR == 'venus' else 8
    mtf_tgt = 0.2 if SENSOR == 'venus' else 0.3

    # SNR = 174 # Nominal SNR --> for S-2 at B8: 174 | for VENµS: 100. 
    SNR_levels = np.geomspace(174, 1, 20)

    MTFS = [0.2, 0.1, 0.05, 0.01, 0.001] if SENSOR == 'venus' else [0.3, 0.2, 0.1, 0.05, 0.01]

    for mtf_tgt in MTFS:
        for SNR in SNR_levels:
            # TODO: Update weight path according to band_selected.
            weight_path = get_weight_b(BAND) if SENSOR == 'venus' else '/Data_large/marine/PythonProjects/MMDET/checkpoints/Sentinel/Special/BS_4/LR_0.0005/IMG_2048/BANDS__b8/42_Optim_SGD/epoch_130.pth'
            cfg = set_NoiseTesting_config(weight_path, sensor=SENSOR, SNR=SNR, mtf_at_fe = mtf_tgt, Seed=18)

            # Run the test
            runner = RUNNERS.build(cfg)
            work_dir = '/Data_large/marine/PythonProjects/MMDET/studies/tta_study'
            runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=f'{work_dir}/test_result/test.pkl'))
            output_test_data = runner.test()
            
            ####### Save the results:
            P = output_test_data['coco/bbox_mAP']
            R = output_test_data['coco/bbox_AR@100']
            
            Results['Band'].append(BAND)
            Results['SNR'].append(SNR)
            Results['MTF'].append(mtf_tgt)
            
            Results['Precision'].append(P)
            Results['Recall'].append(R)
            try:
                F1 = 2 * (P * R) / (P + R)
            except ZeroDivisionError:
                F1 = 0
            Results['F1'].append(F1)
            

    pd.DataFrame(Results).to_pickle(f'{SENSOR}_SNR_study_b{BAND}.pkl')
    print('Done')