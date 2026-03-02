import argparse
import logging
import os
import gc
import torch
import numpy as np
import random
import json

# To set deterministic behaviour:
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'

from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.evaluation import DumpDetResults

import timm 

from mmdet.utils import setup_cache_size_limit_of_dynamo



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


def init_cfg(file_config):
    """
    Initialize the configuration from a file.

    Args:
        file_config (str): The path to the configuration file.

    Returns:
        Config: The initialized configuration object.
    """
    cfg = Config.fromfile(file_config)
    return cfg


def set_seed(seed):
    """
    Set the seed for generating random numbers in PyTorch, Python, and numpy to ensure reproducibility.
    Args:
        seed (int): The seed value to set.
    Returns:
        None
    """
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
    
    parser.add_argument('--band', type=parse_band_list, help='The band list num to train(list)', default='5')
    parser.add_argument('--seed', type=int, help='The seed', default=41)
    parser.add_argument('--batch_size', type=int, help='The batch size', default=1)
    parser.add_argument('--learning_rate', type=float, help='The learning rate', default=0.001)
    parser.add_argument('--resize', type=int, help='The resize dimension', default=2048)
    parser.add_argument('--random_crop', type=int, help='The resize dimension', default=None)
    parser.add_argument('--sensor', type=str, help='The MSI sensor (SENTINEL, VENUS)', default='VENUS')
    parser.add_argument('--max_epochs', type=int, help='The maximum epochs', default=50)
    parser.add_argument('--optimizer', type=str, help='The optimizer type (SGD, Adam, AdamW)', default='SGD')
    parser.add_argument('--encoder_name', type=str, help='The encoder from timm', default='resnet18.fb_ssl_yfcc100m_ft_in1k')
    
    args = parser.parse_args()
    return args


def main(args):
    global logger, workdir
    setup_cache_size_limit_of_dynamo()
    
    SENSOR = args.sensor # 'VENUS' or 'SENTINEL'
    BAND_SEL = args.band # Selecting the Band for Venus or Sentinel
    resize = args.resize
    BS = args.batch_size
    LR = args.learning_rate
    random_crop = args.random_crop
    MAX_EPOCHS = args.max_epochs #
    selOpt = args.optimizer
    model_name = args.encoder_name
    AMP = False
    SEED = args.seed
    
    init_cfg_file = {'SENTINEL': '/Data_large/marine/PythonProjects/MMDET/MyConfigs/Sentinel_b2/vfnet_r18.py', 'VENUS': '/Data_large/marine/PythonProjects/MMDET/MyConfigs/Venus_b5/vfnet_r18.py'}
    cfg = init_cfg(init_cfg_file[args.sensor])
    # Deterministic Behaviour setting:
    set_seed(SEED)
    cfg.randomness = dict(
        seed = SEED, # 41 72 18
        diff_rank_seed=True,
        # deterministic=True
    )
    

    # Normalization:
    MEANS=[200,154,92,63] if SENSOR == 'SENTINEL' else [158.69588,124.42161,109.27108,105.380424,88.40926,98.93067,88.819916,94.20678,103.540764,111.64337,122.92817,79.31501] 
    STD=[22,24,22,60] if SENSOR == 'SENTINEL' else [34.95446,46.282494,56.252197,55.741932,64.54027,59.59095,69.65824,68.40028,77.930405,103.4634,105.30468,65.8369]

    indexCorrect = {2:0, 3:1, 4:2, 8:3} if SENSOR == 'SENTINEL' else {i:i-1 for i in range(1, 13, 1)}
    MEAN_VALS = [MEANS[indexCorrect[x]] for x in BAND_SEL]
    STD_VALS = [STD[indexCorrect[x]] for x in BAND_SEL]
    # Resizing:
    IMG_SIZE = resize

    # Annotations:
    base_annot = '/Data_large/marine/Datasets/VDS2Raw/annotations' if SENSOR == 'SENTINEL' else '/Data_large/marine/Datasets/VENuS/annotations/perfect'
    ann_file = {'Train': f'{base_annot}/train__band_{BAND_SEL[0]}.json',
                'Val': f'{base_annot}/val__band_{BAND_SEL[0]}.json',
                'Test': f'{base_annot}/test__band_{BAND_SEL[0]}.json',}

    ## Dataloading Directories:
    data_root = '/Data_large/marine/Datasets/VDS2Raw/' if SENSOR == 'SENTINEL' else '/Data_large/marine/Datasets/VENuS' # where the images are stored
    data_prefix = 'imgs/' if SENSOR == 'SENTINEL' else 'ds_L0/perfect/'


    optimizers =  {'SGD':{'type': 'OptimWrapper', 'optimizer': {'type': 'SGD', 'lr': LR, 'momentum': 0.9, 'weight_decay': 0.0001}},
                'Adam':{'type': 'OptimWrapper', 'optimizer': {'type': 'Adam', 'lr': LR, 'weight_decay': 0.0001}},
                'AdamW':{'type': 'OptimWrapper', 'optimizer': {'type': 'Adam', 'lr': LR, 'weight_decay': 0.0001}},}


    ## Savedir:
    bandsNames = ''.join([f'_b{x}' for x in BAND_SEL])
    singleMulti = 'Multi' if len(BAND_SEL) > 1 else 'Single'
    # WORKDIR:
    kMode = {'SENTINEL':'Sentinel', 'VENUS':'VENuS'}
    workdir = f'/Data_large/marine/PythonProjects/MMDET/checkpoints/{kMode[SENSOR]}/MyNet/{singleMulti}/{model_name}/perfect{bandsNames}/{SEED}_BS_{BS}_LR_{LR}_ME_{MAX_EPOCHS}_OPT_{selOpt}'
    cfg.work_dir = workdir


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

    # Pipeline:
    # Hook for custom loader:
    loadCorrect = {2:1, 3:2, 4:3, 8:4} if SENSOR == 'SENTINEL' else {i:i for i in range(1, 13, 1)}
    BAND_SEL_LOAD = [loadCorrect[x] for x in BAND_SEL]

    cfg.train_dataloader.dataset.pipeline[0] = {'type': 'SelBandLoader', 'to_float32': True, 'bands_list': BAND_SEL_LOAD}
    cfg.val_dataloader.dataset.pipeline[0] = {'type': 'SelBandLoader', 'to_float32': True, 'bands_list': BAND_SEL_LOAD}

    cfg.train_dataloader.dataset.pipeline[3] = {'type': 'Resize', 'scale': (IMG_SIZE, IMG_SIZE), 'keep_ratio': False}
    cfg.val_dataloader.dataset.pipeline[2] = {'type': 'Resize', 'scale': (IMG_SIZE, IMG_SIZE), 'keep_ratio': False}

    # Metainfo
    classes = ('vessel',) if SENSOR == 'SENTINEL' else ('Vessel',)
    cfg.train_dataloader.dataset.metainfo = {'classes': classes, 'palette': [(220, 20, 60)]}
    cfg.val_dataloader.dataset.metainfo = {'classes': classes, 'palette': [(220, 20, 60)]}


    # Adding random crop to the pipeline. TODO: training with decreasing size
    if random_crop is not None:
        assert isinstance(random_crop, int), 'RandomCrop Error: single dimension must be specified. E.g. 224'
        # insert random crop: 
        rc = dict(type='RandomCrop', crop_size=(random_crop, random_crop))
        cfg.train_dataloader.dataset.pipeline.insert(4, rc)
        cfg.val_dataloader.dataset.pipeline.insert(2, rc)
        

    # Training params:
    cfg.train_dataloader.batch_size = BS
    cfg.train_cfg = {'type': 'EpochBasedTrainLoop', 'max_epochs': MAX_EPOCHS, 'val_interval': 1}

    # TODO: implement stages as in: https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/configs/rtmdet/rtmdet_x_p6_4xb8-300e_coco.py#L78
    # lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

    cfg.optim_wrapper = optimizers[selOpt]

    # TODO: reset correct scheduler
    cfg.param_scheduler = [{'type': 'LinearLR',
                            'start_factor': 0.001,
                            'by_epoch': True,
                            'begin': 0,
                            'convert_to_iter_based': True,
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
                    metainfo=dict(classes=classes, palette=[
                        (
                            220,
                            20,
                            60,
                        ),
                    ]),
                    pipeline=[{'type': 'SelBandLoader', 'to_float32': True, 'bands_list': BAND_SEL_LOAD},
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
    
    
    
    ###### MODEL CUSTOMIZATION ######
    ###### MODEL CUSTOMIZATION ######
    features_only = True
    pretrained = True
    in_chans = len(BAND_SEL)

    cfg.model.backbone = {
        'type': 'TimmEncoder',
        'model_name': model_name, 
        'features_only': features_only, 
        'pretrained': pretrained, 
        'in_chans': in_chans,
        'frozen_stages': 1,
    }
    
    m = timm.create_model(model_name, features_only=features_only, pretrained=pretrained, in_chans=in_chans)
    t = torch.randn(1, 1, 224, 224)

    out = m(t)
    in_channels_neck = []
    for item in out:
        in_channels_neck.append(item.shape[1])

    del m, t, out
    gc.collect()
        
    cfg.model.neck = {
        'type': 'FPN',
        'in_channels': in_channels_neck,
        'out_channels': 256,
        'num_outs': len(in_channels_neck) + 1,
        'start_level': 1, # TODO; check if this is correct
        'end_level': -1,
        'add_extra_convs': 'on_output',
        'relu_before_extra_convs': True,
        'no_norm_on_lateral': False,
        'conv_cfg': None,
        'norm_cfg': None,
        'act_cfg': None,
        'upsample_cfg': dict(mode='nearest'),
        'init_cfg': dict(type='Xavier', layer='Conv2d', distribution='uniform')
    }
    ###### END MODEL CUSTOMIZATION ######
    
        # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
        
    logger = set_logger(workdir)
    logger.info(f'Config: {cfg.pretty_text}')
    logger.info(f'Workdir: {workdir}')
    logger.info(f'Random Seed: {args.seed}')
    logger.info(f'Band Selection: {BAND_SEL}')
    logger.info(f'Band Load Selection: {BAND_SEL_LOAD}')
    logger.info(f'Image Size: {IMG_SIZE}')
    logger.info(f'Batch Size: {BS}')
    logger.info(f'Learning Rate: {LR}')
    logger.info(f'Max Epochs: {MAX_EPOCHS}')
    logger.info(f'Optimizer: {selOpt}')
    logger.info(f'AMP: {AMP}')
    logger.info(f'Random Crop: {random_crop}')

    logger.info('*** Start training ***')
    
    return runner
    
    
if __name__ == '__main__':
    runner = main(parse_args())
    runner.train()
    logger.info('*** Training finished ***')
    runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=f'{workdir}/test_result/test.pkl'))
    # start testing
    output_test_data =runner.test()
    # Specify the file name
    file_name = f'{workdir}/test_result/coco_metrics.json'# Specify the filepath
    # Write the dictionary to a JSON file
    with open(file_name, 'w') as json_file:
        json.dump(output_test_data, json_file, indent=4)

    logger.info(f"Data has been saved to {file_name}")
    
    

