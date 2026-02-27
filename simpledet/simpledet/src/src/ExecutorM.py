import argparse
import logging
import os
import torch
import numpy as np
import random
import json

# To set deterministic behaviour:
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.evaluation import DumpDetResults

from mmdet.utils import setup_cache_size_limit_of_dynamo

import logging

def set_logger(workdir):
    """
    Sets up a logger that logs exclusively to a file.

    Parameters:
    -----------
    workdir : str
        The directory where the log file ('executor.log') will be saved.

    Returns:
    --------
    logging.Logger
        Configured logger that writes logs to a file.
    """
    # Set up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    logger.handlers = []

    # Create a file handler that logs to 'executor.log'
    file_handler = logging.FileHandler(f'{workdir}/executor.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def init_cfg():
    base_folder = '/Data_large/marine/PythonProjects/MMDET/MyConfigs'
    cfg = Config.fromfile(f'{base_folder}/Venus_b5/vfnet_r18.py')
    return cfg


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

     
def parse_args():
    def parse_band_list(band_str):
        """Parse a comma-separated string into a list of integers."""
        return [int(band) for band in band_str.split(',')]

    parser = argparse.ArgumentParser(description="Trainer")
    
    parser.add_argument('--band', type=parse_band_list, help='The band list num to train(list)', default='1,2,3')
    parser.add_argument('--seed', type=int, help='The seed', default=41)
    parser.add_argument('--batch_size', type=int, help='The batch size', default=3)
    parser.add_argument('--learning_rate', type=float, help='The learning rate', default=0.001)
    parser.add_argument('--resize', type=int, help='The resize dimension', default=2048)
    parser.add_argument('--random_crop', type=int, help='The resize dimension', default=None)
    
    args = parser.parse_args()
    return args


def main(args):
    setup_cache_size_limit_of_dynamo()
    
    cfg = init_cfg()
    
    MAX_EPOCHS = 30
    BAND_SEL = args.band # Selecting the Band list for Venus
    AMP = False
    # Normalization:
    MEANS=[158.69588,124.42161,109.27108,105.380424,88.40926,98.93067,88.819916,94.20678,103.540764,111.64337,122.92817,79.31501]
    STD=[34.95446,46.282494,56.252197,55.741932,64.54027,59.59095,69.65824,68.40028,77.930405,103.4634,105.30468,65.8369]
    
    MEAN_VALS = [MEANS[x-1] for x in BAND_SEL]
    STD_VALS = [STD[x-1] for x in BAND_SEL]
    # Resizing:
    IMG_SIZE = args.resize

    # Training:
    BS = args.batch_size
    LR = args.learning_rate

    # Annotations:
    ann_file = {'Train': f'/Data_large/marine/Datasets/VENuS/annotations/perfect/train__band_{BAND_SEL[0]}.json',
        'Val': f'/Data_large/marine/Datasets/VENuS/annotations/perfect/val__band_{BAND_SEL[0]}.json',
        'Test': f'/Data_large/marine/Datasets/VENuS/annotations/perfect/test__band_{BAND_SEL[0]}.json',
        }

    ## Testing:
    data_root = '/Data_large/marine/Datasets/VENuS/ds_L0/'
    data_prefix = f'perfect/'

    # Deterministic Behaviour setting:
    SEED = args.seed
    set_seed(SEED)
    cfg.randomness = dict(
        seed = SEED, # 41 72 18
        diff_rank_seed=True,
        # deterministic=True
    )


    optimizers =  {'SGD':{'type': 'OptimWrapper', 'optimizer': {'type': 'SGD', 'lr': LR, 'momentum': 0.9, 'weight_decay': 0.0001}},
                'Adam':{'type': 'OptimWrapper', 'optimizer': {'type': 'Adam', 'lr': LR, 'weight_decay': 0.0001}},
                'AdamW':{'type': 'OptimWrapper', 'optimizer': {'type': 'Adam', 'lr': LR, 'weight_decay': 0.0001}},}
    selOpt = 'SGD'
    
    # Savedir
    bandsNames = ''.join([f'_b{x}' for x in BAND_SEL])
    singleMulti = 'Multi' if len(BAND_SEL) > 1 else 'Single'
    #### WORKDIR
    workdir = f'/Data_large/marine/PythonProjects/MMDET/checkpoints/VENuS/{singleMulti}/perfect{bandsNames}/{SEED}_BS_{BS}_LR_{LR}_ME_{MAX_EPOCHS}_OPT_{selOpt}'
    cfg.work_dir = workdir

    #### AMP
    # enable automatic-mixed-precision training
    if AMP is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # Dataloader:
    # cfg.model.data_preprocessor.mean = [float(x) for x in MEAN_VALS]
    # cfg.model.data_preprocessor.std = [float(x) for x in STD_VALS]
    cfg.model.data_preprocessor = dict(
        mean=[float(x) for x in MEAN_VALS],
        pad_size_divisor=1,
        std=[float(x) for x in STD_VALS],
        type='MyPrePro')

    # Model Inputs:
    cfg.model.backbone.in_channels = len(BAND_SEL)

    # Annotation file:
    cfg.train_dataloader.dataset.ann_file = ann_file['Train']
    cfg.train_dataloader.dataset.data_prefix = {'img':data_prefix}
    cfg.train_dataloader.dataset.data_root = data_root
    
    cfg.val_dataloader.dataset.ann_file = ann_file['Val']
    cfg.val_dataloader.dataset.data_prefix = {'img':data_prefix}
    cfg.val_dataloader.dataset.data_root = data_root
    
    #       Evaluators:
    cfg.val_evaluator = dict(
        ann_file=ann_file['Val'],
        backend_args=None,
        format_only=False,
        metric='bbox',
        type='CocoMetric')
    #      Pipeline:
    #             Hook for custom loader:
    cfg.train_dataloader.dataset.pipeline[0] = {'type': 'SelBandLoader', 'to_float32': True, 'bands_list': BAND_SEL}
    cfg.val_dataloader.dataset.pipeline[0] = {'type': 'SelBandLoader', 'to_float32': True, 'bands_list': BAND_SEL}
    
    cfg.train_dataloader.dataset.pipeline[3] = {'type': 'Resize', 'scale': (IMG_SIZE, IMG_SIZE), 'keep_ratio': False}
    cfg.val_dataloader.dataset.pipeline[2] = {'type': 'Resize', 'scale': (IMG_SIZE, IMG_SIZE), 'keep_ratio': False}
    
    # Adding random crop to the pipeline. TODO: training with decreasing size
    if args.random_crop is not None:
        assert isinstance(args.random_crop, int), 'RandomCrop Error: single dimension must be specified. E.g. 224'
        # insert random crop: 
        rc = dict(type='RandomCrop', crop_size=(args.random_crop, args.random_crop))
        cfg.train_dataloader.dataset.pipeline.insert(3, rc)
        cfg.val_dataloader.dataset.pipeline.insert(2, rc)
        

    # Training params:
    cfg.train_dataloader.batch_size = BS
    cfg.train_cfg = {'type': 'EpochBasedTrainLoop', 'max_epochs': MAX_EPOCHS, 'val_interval': 1}

    # TODO: implement stages as in: https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/configs/rtmdet/rtmdet_x_p6_4xb8-300e_coco.py#L78
    # lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

    cfg.optim_wrapper = optimizers[selOpt]

    cfg.param_scheduler = [{'type': 'LinearLR',
                            'start_factor': 0.001,
                            'by_epoch': True,
                            'begin': 0,
                            'end': MAX_EPOCHS//5},
                            {'type': 'MultiStepLR',
                            'begin': 0,
                            'end': MAX_EPOCHS//5,
                            'by_epoch': True,
                            'milestones': [MAX_EPOCHS//4, MAX_EPOCHS//3, MAX_EPOCHS//2],
                            'gamma': 0.75}, 
                            {# use cosine lr scheduler
                            'type':'CosineAnnealingLR',
                            'eta_min':LR * 0.05,
                            'begin':MAX_EPOCHS//2,
                            'end':MAX_EPOCHS,
                            'T_max':MAX_EPOCHS//1.5,
                            'by_epoch':True,
                            'convert_to_iter_based':True,}
                            ]

    #### Test Config hooks:
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = False

    cfg.test_dataloader = dict(
                batch_size=1,
                dataset=dict(
                    ann_file=ann_file['Test'],
                    data_root=data_root,
                    data_prefix=dict(img=data_prefix),
                    filter_cfg=dict(filter_empty_gt=True),
                    metainfo=dict(classes=('Vessel', ), palette=[
                        (
                            220,
                            20,
                            60,
                        ),
                    ]),
                    pipeline=[{'type': 'SelBandLoader', 'to_float32': True, 'bands_list': BAND_SEL},
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(keep_ratio=False, scale=(IMG_SIZE,IMG_SIZE,), type='Resize'),
                        dict(
                            meta_keys=('img_path', 'img_id', 'seg_map_path', 
                                    'height', 'width', 'instances', 'sample_idx', 
                                    'img', 'img_shape', 'ori_shape', 'scale', 'scale_factor', 
                                    'keep_ratio', 'homography_matrix', 'gt_bboxes', 'gt_ignore_flags', 
                                    'gt_bboxes_labels'),
                            type='PackDetInputs'),
                    ],
                    test_mode=True,
                    type='CocoDataset'),
                drop_last=False,
                num_workers=2,
                persistent_workers=True,
                sampler=dict(shuffle=False, type='DefaultSampler'))

    cfg.test_evaluator = dict(
                type='CocoMetric',
                metric='bbox',
                format_only=False,
                ann_file=ann_file['Test'],
                outfile_prefix=f'{workdir}/test_results')


    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    
    logger = set_logger(workdir)
    logger.info(f'Configuration: {cfg}')
    logger.info(f'Workdir: {workdir}')
    
    if RUN:
        ########## TRAINING:
        logger.info('\n\nStarting training...')
        runner.train()
        logger.info('\n\nTraining finished.')
        ########## TESTING:
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=f'{workdir}/test_result/test.pkl'))
        # start testing
        logger.info('\n\nStarting testing...')
        output_test_data =runner.test()
        logger.info('\n\nTesting finished.')

        # Specify the file name
        file_name = f'{workdir}/test_result/coco_metrics.json'# Specify the filepath
        # Write the dictionary to a JSON file
        with open(file_name, 'w') as json_file:
            json.dump(output_test_data, json_file, indent=4)
        logger.info(f"Data has been saved to {file_name}")
    

RUN = True

if __name__ == '__main__':
    args = parse_args()
    main(args=args)