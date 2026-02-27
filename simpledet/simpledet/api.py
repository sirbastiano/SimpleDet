import numpy as np
import torch
import os, sys
import torch
import random
import json
import logging
import ast
import time
from typing import Optional

_OPENMMLAB_ERROR: Optional[Exception] = None
try:
    from mmengine.config import Config, DictAction
    from mmengine.logging import MMLogger
    from mmengine.model import revert_sync_batchnorm
    from mmengine.runner import Runner
    from mmengine.utils import digit_version
    from mmengine.logging import print_log
    from mmengine.registry import RUNNERS, init_default_scope, MODELS
    from mmdet.evaluation import DumpDetResults
    from mmdet.utils import setup_cache_size_limit_of_dynamo
except Exception as _exc:  # pragma: no cover - optional dependency path
    _OPENMMLAB_ERROR = _exc
    Config = DictAction = MMLogger = revert_sync_batchnorm = None
    Runner = print_log = None
    RUNNERS = init_default_scope = MODELS = None
    DumpDetResults = None
    setup_cache_size_limit_of_dynamo = None


def _ensure_openmmlab() -> None:
    """Raise a clearer dependency error when OpenMMLab modules are unavailable."""
    if _OPENMMLAB_ERROR is not None:
        raise ModuleNotFoundError(
            "simpledet.api requires mmengine and mmdet runtime dependencies. "
            "Install them (for example via scripts/install_simpledet_env.sh) before "
            "running API features."
        ) from _OPENMMLAB_ERROR


if setup_cache_size_limit_of_dynamo is not None:
    setup_cache_size_limit_of_dynamo()


# ---------- Helpers ----------
model = dict(
    type='FOVEA',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[105.380424],
        std=[55.741932],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='Backbone',
        in_channels=3,
        base_channels=64,
        depths=[4, 4, 12, 4],
        num_angles=90,
        drop_path_rate=0.1,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        ),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='FoveaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        base_edge_list=[16, 32, 64, 128, 256],
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        sigma=0.4,
        with_deform=False,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.50,
            alpha=0.4,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=1000,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))


class ObjectDetectionPipeline:
    
    def __init__(self, 
                seed=71,
                model_cfg=None, 
                resize=768, 
                batch_size=2, 
                learning_rate=0.001,
                random_crop=None, 
                max_epochs=1,
                data_folder=None, # where the data is stored
                result_folder=None, # where the results are stored
                config_folder="./src", # where the config files are stored.
                data_prefix='imgs/', # prefix for the data files where images are stored.
                mean_vals=[0.485, 0.485, 0.485], # mean values for normalization expressed as list for each channel.
                std_vals=[1.0, 1.0, 1.0], # mean values for normalization expressed as list for each channel.
                in_channels=1, # number of channels in input
                tif_channels_to_load=None, # list of channels to load from the tif file, in form of list
                annot_file_train=None, # annotation file for training
                annot_file_val=None, # annotation file for validation
                annot_file_test=None, # annotation file for testing
                categories = ('wake',), # categories to detect in form of tuple
                amp=True, # Automatic Mixed Precision
                optimizer_choice='SGD', # Optimizer choice: SGD, Adam, AdamW
                scheduler_choice=None, # Scheduler choice: MultiStepLR, CosineAnnealingLR
                val_interval=1, # Validation interval
                skip_cfg=False, # Skip the original configuration of the model (useful when using custom models)
                segmentation=False, # whether to use segmentation or not for the OD model.
        ):
        # Training configuration        
        self.seed = seed
        self.model_cfg = model_cfg
        assert self.model_cfg is not None, 'Model configuration must be specified as a dict.'
        self.resize = resize
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_crop = random_crop
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.amp = amp
        self.optimizer_choice = optimizer_choice
        self.scheduler_choice = scheduler_choice
        self.skip_cfg = skip_cfg # Freeze configuration of the model when it's not a standard model.

        # Additional asserts
        assert isinstance(self.model_cfg, dict), 'Model config must be an dict.'
        assert isinstance(self.seed, int), 'Seed must be an integer.'
        assert isinstance(self.resize, int), 'Resize must be an integer.'
        assert isinstance(self.batch_size, int), 'Batch size must be an integer.'
        assert isinstance(self.learning_rate, float), 'Learning rate must be a float.'
        assert self.random_crop is None or isinstance(self.random_crop, int), 'Random crop must be None or an integer.'
        assert isinstance(self.max_epochs, int), 'Max epochs must be an integer.'
        assert isinstance(self.val_interval, int), 'Validation interval must be an integer.'
        assert isinstance(self.amp, bool), 'AMP must be a boolean.'
        assert isinstance(self.optimizer_choice, str), 'Optimizer choice must be a string.'
        assert self.scheduler_choice is None or isinstance(self.scheduler_choice, str), 'Scheduler choice must be None or a string.'


        # Data configuration:
        self.base_folder = "."
        self.config_folder = config_folder
        self.data_root = data_folder
        self.annot_file_train = annot_file_train
        self.annot_file_val = annot_file_val
        self.annot_file_test = annot_file_test
        self.result_folder = result_folder
        assert self.data_root is not None, 'Data folder must be specified.'
        assert self.annot_file_train is not None, 'Annotation file for training must be specified.'
        assert self.annot_file_val is not None, 'Annotation file for validation must be specified.'
        assert self.annot_file_test is not None, 'Annotation file for testing must be specified.'
        
        # Loading configuration:        
        self.data_prefix = data_prefix
        self.in_channels = in_channels
        self.mean_vals = mean_vals
        self.std_vals = std_vals
        self.tif_channels_to_load = tif_channels_to_load
        assert self.tif_channels_to_load is not None, 'Channels to load from the tif file must be specified.'
        assert type(self.tif_channels_to_load) == list, 'Channels to load from the tif file must be specified as a list.'
        assert len(tif_channels_to_load) == self.in_channels, 'Number of channels to load from the tif file must be equal to the number of input channels.'
        self.categories = categories
        assert self.categories is not None, 'Categories must be specified.'
        self.segmentation = segmentation

        _ensure_openmmlab()

        # Outputs:
        self.workdir = self._construct_workdir()        
        self.outfile_results = f"{self.workdir}/testSet" 
        self.outfile_test = f"{self.workdir}/tests.pkl"
        self.outfile_metric = f"{self.workdir}/metrics.pkl"
        self.optimizers = self._get_optimizers()
        
        # Starting the pipeline:
        self.cfg = self._init_cfg()
        self.logger = self._set_logger()
        self._set_seed()
        self._configure_pipeline()
        self.logger.info(
            f'Pipeline configured with the following parameters: '
            f'Seed {self.seed}, '
            f'Batch Size: {self.batch_size}, '
            f'Learning Rate: {self.learning_rate}, '
            f'Random Crop: {self.random_crop}, '
            f'Max Epochs: {self.max_epochs}, '
            f'AMP: {self.amp}, '
            f'Optimizer: {self.optimizer_choice}'
        )

    def _init_cfg(self):
        _ensure_openmmlab()
        print("Reading base config file:")
        print(f'{self.config_folder}/base_config.py')
        cfg = Config.fromfile(f'{self.config_folder}/base_config.py')
        
        
        preprocessor = cfg.model["data_preprocessor"]
        self.model_cfg["data_preprocessor"] = preprocessor # Overwrite the preprocessor with the one from the base config.
        cfg.model = self.model_cfg
        
        cfg.work_dir = self.workdir
        
        if self.segmentation:
            
            train_pipeline = [
                    dict(type='LoadImageFromFile',to_float32=True, color_type='grayscale', imdecode_backend='tifffile', backend_args=None),
                    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                    dict(type='RandomFlip', prob=0.4),
                    dict(type='Resize', scale=(768,768), keep_ratio=False),
                    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
                    dict(
                        type='PackDetInputs',
                        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                'scale_factor'))
                ]

            test_pipeline = [
                    dict(type='LoadImageFromFile',to_float32=True, color_type='grayscale', imdecode_backend='tifffile', backend_args=None),
                    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                    dict(type='Resize', scale=(768,768), keep_ratio=False),
                    dict(
                        type='PackDetInputs',
                        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                'scale_factor'))
                ]
            # Update with segmentation pipeline    
            cfg.train_dataloader.dataset.pipeline = train_pipeline
            cfg.val_dataloader.dataset.pipeline = test_pipeline
            cfg.test_dataloader.dataset.pipeline = test_pipeline
                
        return cfg

    def _set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(self.seed)

    def _construct_workdir(self):
        workdir = (f'{self.result_folder}/'
                   f'BS_{self.batch_size}_'
                   f'LR_{self.learning_rate}_IMG_{self.resize}_'
                   f'{self.seed}_Optim_{self.optimizer_choice}')
        return workdir

    def _get_optimizers(self):
        return {
            'SGD': {'type': 'OptimWrapper', 'optimizer': {'type': 'SGD', 'lr': self.learning_rate, 'momentum': 0.9, 'weight_decay': 0.0001}},
            'Adam': {'type': 'OptimWrapper', 'optimizer': {'type': 'Adam', 'lr': self.learning_rate, 'weight_decay': 0.0001}},
            'AdamW': {'type': 'OptimWrapper', 'optimizer': {'type': 'AdamW', 'lr': self.learning_rate, 'weight_decay': 0.0001}},
        }

    def _set_logger(self):
        # This function would configure and return a logger based on the workdir
        os.makedirs(self.workdir, exist_ok=True)
        logger = logging.getLogger(self.workdir)
        handler = logging.FileHandler(f'{self.workdir}/pipeline.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _configure_pipeline(self):        
        _ensure_openmmlab()
        # Configure cfg with preprocessor
        self.cfg.model.data_preprocessor = dict(
            mean=[float(x) for x in self.mean_vals],
            pad_size_divisor=1,
            std=[float(x) for x in self.std_vals],
            type='MyPrePro')


        # Freeze configuration of the model when it's not a standard model.
        if not self.skip_cfg:
            self.cfg.model.backbone.in_channels = self.in_channels
            self.cfg.model.bbox_head.num_classes = len(self.categories)

        # Dataloader directories and annotation files
        self.cfg.train_dataloader.dataset.ann_file = self.annot_file_train
        self.cfg.train_dataloader.dataset.data_prefix = {'img': self.data_prefix}
        self.cfg.train_dataloader.dataset.data_root = self.data_root

        self.cfg.val_dataloader.dataset.ann_file = self.annot_file_val
        self.cfg.val_dataloader.dataset.data_prefix = {'img': self.data_prefix}
        self.cfg.val_dataloader.dataset.data_root = self.data_root

        # Evaluators
        self.cfg.val_evaluator = dict(
            ann_file=self.annot_file_val,
            backend_args=None,
            format_only=False,
            metric='bbox',
            type='CocoMetric')

        # Pipeline configuration

        band_sel_load = [int(x) for x in self.tif_channels_to_load]
        # self.cfg.train_dataloader.dataset.pipeline[0] = {'type':'LoadImageFromFile', "to_float32":True, "color_type":"grayscale", "imdecode_backend":"cv2", "backend_args":None, }
        self.cfg.train_dataloader.dataset.pipeline[0] = {'type': 'SelBandLoader', 'to_float32': True, 'bands_list': band_sel_load}
        
        self.cfg.val_dataloader.dataset.pipeline[0] = {'type': 'SelBandLoader', 'to_float32': True, 'bands_list': band_sel_load}

        self.cfg.train_dataloader.dataset.pipeline[3] = {'type': 'Resize', 'scale': (self.resize, self.resize), 'keep_ratio': False}
        self.cfg.val_dataloader.dataset.pipeline[2] = {'type': 'Resize', 'scale': (self.resize, self.resize), 'keep_ratio': False}

        # Metainfo
        self.cfg.train_dataloader.dataset.metainfo = {'classes': self.categories, 'palette': [(220, 20, 60)]}
        self.cfg.val_dataloader.dataset.metainfo = {'classes': self.categories, 'palette': [(220, 20, 60)]}

        # Random crop (if applicable)
        if self.random_crop is not None:
            assert isinstance(self.random_crop, int), 'RandomCrop Error: single dimension must be specified. E.g. 224'
            rc = dict(type='RandomCrop', crop_size=(self.random_crop, self.random_crop))
            self.cfg.train_dataloader.dataset.pipeline.insert(4, rc)
            self.cfg.val_dataloader.dataset.pipeline.insert(2, rc)

        
        # Training parameters (mandatory)
        self.cfg.train_dataloader.batch_size = self.batch_size
        self.cfg.train_cfg = {'type': 'EpochBasedTrainLoop', 'max_epochs': self.max_epochs, 'val_interval': self.val_interval}
        self.cfg.optim_wrapper = self.optimizers[self.optimizer_choice]

        
        # Learning rate scheduler (optional)
        if self.scheduler_choice is not None:
            self.cfg.param_scheduler = [
                {'type': 'LinearLR', 'start_factor': 0.001, 'by_epoch': True, 'begin': 0, 'convert_to_iter_based': True, 'end': self.max_epochs // 5},
                {'type': 'MultiStepLR', 'begin': 0, 'end': self.max_epochs // 5, 'by_epoch': True, 'milestones': [self.max_epochs // 4, self.max_epochs // 3, self.max_epochs // 2], 'gamma': 0.75},
                {'type': 'CosineAnnealingLR', 'eta_min': self.learning_rate * 0.05, 'begin': self.max_epochs // 2, 'end': self.max_epochs, 'T_max': self.max_epochs // 1.5, 'by_epoch': True, 'convert_to_iter_based': True}
            ]

        # Test configuration
        self.cfg.test_dataloader = dict(
            batch_size=1,
            dataset=dict(
                ann_file=self.annot_file_test,
                data_root=self.data_root,
                data_prefix=dict(img=self.data_prefix),
                filter_cfg=dict(filter_empty_gt=True),
                metainfo=dict(classes=self.categories, palette=[(220, 20, 60)]),
                pipeline=[
                    {'type': 'SelBandLoader', 'to_float32': True, 'bands_list': band_sel_load},
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(keep_ratio=False, scale=(self.resize, self.resize), type='Resize'),
                    dict(type='PackDetInputs', meta_keys=('img_path', 'img_id', 'seg_map_path', 'height', 'width', 'instances', 'sample_idx', 'img', 'img_shape', 'ori_shape', 'scale', 'scale_factor', 'keep_ratio', 'homography_matrix', 'gt_bboxes', 'gt_ignore_flags', 'gt_bboxes_labels'))
                ],
                test_mode=True,
                type='CocoDataset'),
            drop_last=False,
            num_workers=2,
            persistent_workers=True,
            sampler=dict(shuffle=False, type='DefaultSampler')
        )

        self.cfg.test_evaluator = dict(
            type='CocoMetric',
            metric='bbox',
            format_only=False,
            ann_file=self.annot_file_test,
            outfile_prefix=self.outfile_results)

        # Enable AMP if needed
        if self.amp:
            optim_wrapper = self.cfg.optim_wrapper.type
            if optim_wrapper == 'AmpOptimWrapper':
                print_log('AMP training is already enabled in your config.', logger='current', level=logging.WARNING)
            else:
                assert optim_wrapper == 'OptimWrapper', f'`--amp` is only supported when the optimizer wrapper type is `OptimWrapper` but got {optim_wrapper}.'
                self.cfg.optim_wrapper.type = 'AmpOptimWrapper'
                self.cfg.optim_wrapper.loss_scale = 'dynamic'

    def get_pth(self):
        """
        Given a folder, this function returns a list of .pth file paths.

        Args:
        folder (str): The folder to search for .pth files.

        Returns:
        list: A list of file paths to .pth files.
        """
        folder = self.workdir
        pth_filepaths = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.pth'):
                    pth_filepaths.append(os.path.join(root, file))
                    
        pth_filepaths = [x for x in pth_filepaths if "best" in x]
        assert len(pth_filepaths) == 1, 'There should be only one .pth file in the folder.'
        return pth_filepaths[0]

    def build(self):
        _ensure_openmmlab()
        # Check if custom runner is defined
        if 'runner_type' not in self.cfg:
            runner = Runner.from_cfg(self.cfg)
        else:
            runner = RUNNERS.build(self.cfg)
        
        self.runner = runner
        self.logger.info(f'Pipeline built successfully.')

    def train(self):
        _ensure_openmmlab()
        self.runner.train()
        
    def test(self):
        _ensure_openmmlab()
        self.runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=self.outfile_test))
        # start testing
        output_test_data =self.runner.test()

        # Specify the file name
        file_name = self.outfile_metric # Specify the filepath
        # Write the dictionary to a JSON file
        with open(file_name, 'w') as json_file:
            json.dump(output_test_data, json_file, indent=4)

        self.logger.info(f"Data has been saved to {file_name}")
        self.logger.info(f'Testing completed successfully.')
        self.logger.info(f'output_test_data: {output_test_data}')
        
        
        

# Example usage:
if __name__ == "__main__":
    # ------ Project Configuration ------
    # This section is used to configure the project parameters, such as project name, model name,
    PROJECT_NAME = 'project_name'
    ModelName = 'model_name'
    SEGMENTATION = False
    # End of Project Configuration

    # ------ Data Configuration ------
    # This section is used to configure the data parameters, such as data root, image prefix
    Band_num = 1

    DATAROOT = "/path/to/data"
    IMG_PREFIX = f"{DATAROOT}/B{Band_num}/imgs/"
    ANNOT_TRAIN = f"{DATAROOT}/B{Band_num}/Annotations/train_annotations.json"
    ANNOT_VAL = f"{DATAROOT}/B{Band_num}/Annotations/val_annotations.json"
    ANNOT_TEST = f"{DATAROOT}/B{Band_num}/Annotations/test_annotations.json"
    CATEGORIES = ('category1',)
    BASE_CONFIG_PATH = "/path/to/configs"
    # End of Data Configuration

    # ------ Model Configuration ------
    # This section is used to configure the model train parameters:
    INPUT_SIZE = 512
    BATCH_SIZE = 4
    EPOCHS = 30
    CHANNELS_LOADED = [1,1,1] # 1-indexed for rasterio
    INPUT_CHANNELS = len(CHANNELS_LOADED)
    # End of Model Configuration


    MODEL_CFG = model
    # now time:
    now = time.strftime("%Y%m%d-%H%M%S")

    RESULT_FOLDER = f"/Data_large/marine/PythonProjects/MMDET/Results/{PROJECT_NAME}/{ModelName}/{now}"

    pipeline = ObjectDetectionPipeline(
                    seed=71, 
                    model_cfg=MODEL_CFG,
                    resize=INPUT_SIZE, 
                    batch_size=BATCH_SIZE, 
                    learning_rate=0.001,
                    in_channels=INPUT_CHANNELS, # number of channels in input
                    random_crop=None, 
                    max_epochs=EPOCHS,
                    result_folder=RESULT_FOLDER, # where the results are stored
                    config_folder=BASE_CONFIG_PATH, # where the config files are stored.
                    data_folder=DATAROOT, # where the data is stored
                    data_prefix=IMG_PREFIX, # prefix for the data files where images are stored.
                    mean_vals=[0.485, 0.456, 0.406], # mean values for normalization expressed as list for each channel.
                    std_vals=[1.0, 1.0, 1.0], # mean values for normalization expressed as list for each channel.
                    tif_channels_to_load=CHANNELS_LOADED, # list of channels to load from the tif file, in form of list
                    annot_file_train=ANNOT_TRAIN, # annotation file for training
                    annot_file_val=ANNOT_VAL, # annotation file for validation
                    annot_file_test=ANNOT_TEST, # annotation file for testing
                    categories = CATEGORIES, # categories to detect in form of tuple
                    amp=False, # Automatic Mixed Precision
                    optimizer_choice='SGD', # Optimizer choice: SGD, Adam, AdamW
                    scheduler_choice=None, # Scheduler choice: MultiStepLR, CosineAnnealingLR
                    val_interval=1, # Validation interval
                    skip_cfg=True,
                    segmentation=SEGMENTATION,
                )
    
    pipeline.build() # Builds the pipeline
    pipeline.train() # Trains the model
    pipeline.test()  # Tests the model
    
