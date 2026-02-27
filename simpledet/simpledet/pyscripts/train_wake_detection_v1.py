"""
Wake Detection Training Script

This script provides a command-line interface for training wake detection models
with configurable parameters including model depths and other hyperparameters.
"""

import argparse
import ast
import json
import logging
import os
import random
import sys
sys.path.append('/Data_large/marine/PythonProjects/MMDET/MyConfigs')
import time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from dotenv import load_dotenv

from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.logging import print_log
from mmengine.registry import RUNNERS, init_default_scope, MODELS
from mmdet.evaluation import DumpDetResults
from mmdet.utils import setup_cache_size_limit_of_dynamo


class ObjectDetectionPipeline:
    """
    Object Detection Pipeline for wake detection training and testing.
    
    This class handles the complete pipeline for object detection including
    data loading, model training, validation, and testing.
    """
    
    def __init__(self, 
                seed: int = 13,
                model_cfg: Optional[Dict[str, Any]] = None, 
                resize: int = 768, 
                batch_size: int = 2, 
                learning_rate: float = 0.01,
                random_crop: Optional[int] = None, 
                max_epochs: int = 1,
                data_folder: Optional[str] = None,
                result_folder: Optional[str] = None,
                config_folder: str = "./MyConfigs",
                data_prefix: str = 'imgs/',
                mean_vals: List[float] = [0.485, 0.456, 0.406],
                std_vals: List[float] = [1.0, 1.0, 1.0],
                in_channels: int = 1,
                tif_channels_to_load: Optional[List[int]] = None,
                annot_file_train: Optional[str] = None,
                annot_file_val: Optional[str] = None,
                annot_file_test: Optional[str] = None,
                categories: Tuple[str, ...] = ('wake',),
                amp: bool = True,
                optimizer_choice: str = 'SGD',
                scheduler_choice: Optional[str] = None,
                val_interval: int = 1,
                skip_cfg: bool = False,
                segmentation: bool = False,
        ):
        """
        Initialize the ObjectDetectionPipeline.
        
        Args:
            seed: Random seed for reproducibility
            model_cfg: Model configuration dictionary
            resize: Image resize dimension
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            random_crop: Random crop size (optional)
            max_epochs: Maximum training epochs
            data_folder: Path to data folder
            result_folder: Path to results folder
            config_folder: Path to config files
            data_prefix: Prefix for data files
            mean_vals: Normalization mean values
            std_vals: Normalization std values
            in_channels: Number of input channels
            tif_channels_to_load: List of channels to load from TIF files
            annot_file_train: Training annotation file path
            annot_file_val: Validation annotation file path
            annot_file_test: Testing annotation file path
            categories: Detection categories
            amp: Enable Automatic Mixed Precision
            optimizer_choice: Optimizer type (SGD, Adam, AdamW)
            scheduler_choice: Learning rate scheduler type
            val_interval: Validation interval
            skip_cfg: Skip configuration freeze
            segmentation: Enable segmentation mode
        """
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
        self.skip_cfg = skip_cfg

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

    def _init_cfg(self) -> Config:
        """Initialize configuration from base config file."""
        print("Reading base config file:")
        print(f'{self.config_folder}/base_config.py')
        cfg = Config.fromfile(f'{self.config_folder}/base_config.py')
        
        preprocessor = cfg.model["data_preprocessor"]
        self.model_cfg["data_preprocessor"] = preprocessor
        cfg.model = self.model_cfg
        
        cfg.work_dir = self.workdir
        
        if self.segmentation:
            train_pipeline = [
                dict(type='LoadImageFromFile',to_float32=True, color_type='grayscale', imdecode_backend='tifffile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(type='RandomFlip', prob=0.5),
                dict(type='Resize', scale=(self.resize, self.resize), keep_ratio=False),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
                dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
            ]

            test_pipeline = [
                dict(type='LoadImageFromFile',to_float32=True, color_type='grayscale', imdecode_backend='tifffile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(type='Resize', scale=(self.resize, self.resize), keep_ratio=False),
                dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
            ]
            
            cfg.train_dataloader.dataset.pipeline = train_pipeline
            cfg.val_dataloader.dataset.pipeline = test_pipeline
            cfg.test_dataloader.dataset.pipeline = test_pipeline
                
        return cfg

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(self.seed)

    def _construct_workdir(self) -> str:
        """Construct work directory path."""
        workdir = (f'{self.result_folder}/'
                   f'BS_{self.batch_size}_'
                   f'LR_{self.learning_rate}_IMG_{self.resize}_'
                   f'{self.seed}_Optim_{self.optimizer_choice}')
        return workdir

    def _get_optimizers(self) -> Dict[str, Dict[str, Any]]:
        """Get optimizer configurations."""
        return {
            'SGD': {'type': 'OptimWrapper', 'optimizer': {'type': 'SGD', 'lr': self.learning_rate, 'momentum': 0.9, 'weight_decay': 0.0001}},
            'Adam': {'type': 'OptimWrapper', 'optimizer': {'type': 'Adam', 'lr': self.learning_rate, 'weight_decay': 0.0001}},
            'AdamW': {'type': 'OptimWrapper', 'optimizer': {'type': 'AdamW', 'lr': self.learning_rate, 'weight_decay': 0.0001}},
        }

    def _set_logger(self) -> logging.Logger:
        """Set up logger for the pipeline."""
        os.makedirs(self.workdir, exist_ok=True)
        logger = logging.getLogger(self.workdir)
        handler = logging.FileHandler(f'{self.workdir}/pipeline.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _configure_pipeline(self) -> None:
        """Configure the pipeline with all necessary parameters."""
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
                {'type': 'MultiStepLR', 'begin': 0, 'end': self.max_epochs, 'by_epoch': True, 'milestones': [self.max_epochs // 2, self.max_epochs ], 'gamma': 0.75},
                {'type': 'CosineAnnealingLR', 'eta_min': self.learning_rate * 0.05, 'begin': self.max_epochs // 3, 'end': self.max_epochs, 'T_max': self.max_epochs // 1.5, 'by_epoch': True, 'convert_to_iter_based': True}
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

    def get_pth(self) -> str:
        """
        Get the path to the best model checkpoint.
        
        Returns:
            Path to the best .pth file
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

    def build(self) -> None:
        """Build the training runner."""
        if 'runner_type' not in self.cfg:
            runner = Runner.from_cfg(self.cfg)
        else:
            runner = RUNNERS.build(self.cfg)
        
        self.runner = runner
        self.logger.info(f'Pipeline built successfully.')

    def train(self) -> None:
        """Train the model."""
        self.runner.train()
        
    def test(self) -> None:
        """Test the model and save results."""
        self.runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=self.outfile_test))
        output_test_data = self.runner.test()

        with open(self.outfile_metric, 'w') as json_file:
            json.dump(output_test_data, json_file, indent=4)

        self.logger.info(f"Data has been saved to {self.outfile_metric}")
        self.logger.info(f'Testing completed successfully.')
        self.logger.info(f'output_test_data: {output_test_data}')


def create_model_config(args) -> Dict[str, Any]:
    """
    Create model configuration dictionary with specified args.
    
    Args:
        
    Returns:
        Model configuration dictionary
    

    """
    
    neck_channels = args.neck_channels
    neck_channels = ast.literal_eval(neck_channels) if isinstance(neck_channels, str) else neck_channels
    assert isinstance(neck_channels, list), 'Neck channels must be a list.'
    
    model = dict(
        type='FOVEA',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[0],
            std=[1],
            bgr_to_rgb=True,
            pad_size_divisor=1),
        backbone=dict(
            type='Backbone',
            pretrained=True,
            features_only=True,
            encoder_name=args.encoder_name,
            ),
        neck=dict(
            type='FPN',
            in_channels=neck_channels,
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
        train_cfg=dict(),
        test_cfg=dict(
            nms_pre=1000,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
    
    return model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Wake Detection Training Script')
    
    # Model configuration


    parser.add_argument('--project_name', type=str, default='scieottiche',
                       help='Project name for organizing results')
    parser.add_argument('--model_name', type=str, default='MyCoolModel',
                       help='Model name for organizing results')
    
    parser.add_argument('--encoder_name', type=str, default='MyCoolModel',
                       help='Model name for organizing results')

    parser.add_argument('--neck_channels', type=str, default='[96, 192, 384, 768]',
                       help='Neck channels for the model')

    # Training parameters
    parser.add_argument('--input_size', type=int, default=768,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=13,
                       help='Random seed')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'],
                       help='Optimizer choice')
    parser.add_argument('--amp', action='store_true',
                       help='Enable Automatic Mixed Precision')
    
    parser.add_argument('--scheduler_choice', type=str, default=None, choices=['MultiStepLR', 'CosineAnnealingLR'],
                       help='Learning rate scheduler choice')
    
    
    # Data configuration
    parser.add_argument('--band_num', type=int, default=8,
                       help='Band number for data loading')
    parser.add_argument('--channels_loaded', type=str, default='[1,1,1]',
                       help='Channels to load as string representation of list')
    parser.add_argument('--data_root', type=str, 
                       default='/Data_large/marine/PythonProjects/MMDET/Data/Multispectral_Wakes',
                       help='Root directory of the dataset')
    parser.add_argument('--config_path', type=str,
                       default='/Data_large/marine/PythonProjects/MMDET/MyConfigs',
                       help='Path to configuration files')
    parser.add_argument('--result_root', type=str,
                       default='/Data_large/marine/PythonProjects/MMDET/Results',
                       help='Root directory for results')
    
    # Execution modes
    parser.add_argument('--train_only', action='store_true',
                       help='Only train the model, skip testing')
    parser.add_argument('--test_only', action='store_true',
                       help='Only test the model, skip training')
    parser.add_argument('--segmentation', action='store_true',
                       help='Enable segmentation mode')
    
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    # Load environment variables
    load_dotenv('/Data_large/marine/PythonProjects/MMDET/.env')
    print("Environment variables loaded from .env file.")
    # Setup MMDet
    setup_cache_size_limit_of_dynamo()
    
    # Parse arguments
    args = parse_args()
    

    # Parse channels loaded from string
    try:
        channels_loaded = ast.literal_eval(args.channels_loaded)
        assert isinstance(channels_loaded, list), "Channels loaded must be a list"
        assert all(isinstance(c, int) for c in channels_loaded), "All channels must be integers"
    except (ValueError, SyntaxError, AssertionError) as e:
        raise ValueError(f"Invalid channels_loaded format: {args.channels_loaded}. Expected list of integers like '[1,1,1]'") from e
    
    # Create model configuration
    model_cfg = create_model_config(args)
    
    # Setup data paths
    band_num = args.band_num
    data_root = args.data_root
    img_prefix = f"{data_root}/B{band_num}/imgs/"
    annot_train = f"{data_root}/B{band_num}/Annotations/LAST_SWD_Train.json"
    annot_val = f"{data_root}/B{band_num}/Annotations/LAST_SWD_Test.json"
    annot_test = f"{data_root}/B{band_num}/Annotations/LAST_SWD_Test.json"
    
    # Create result folder with timestamp
    now = time.strftime("%Y%m%d-%H%M%S")
    result_folder = f"{args.result_root}/{args.project_name}/{args.model_name}/{now}"
    
    # Create pipeline
    pipeline = ObjectDetectionPipeline(
        seed=args.seed,
        model_cfg=model_cfg,
        resize=args.input_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        in_channels=len(channels_loaded),
        random_crop=None,
        max_epochs=args.epochs,
        result_folder=result_folder,
        config_folder=args.config_path,
        data_folder=data_root,
        data_prefix=img_prefix,
        mean_vals=[129, 129, 129],
        std_vals=[70, 70, 70],
        tif_channels_to_load=channels_loaded,
        annot_file_train=annot_train,
        annot_file_val=annot_val,
        annot_file_test=annot_test,
        categories=('wake',),
        amp=args.amp,
        optimizer_choice=args.optimizer,
        scheduler_choice=args.scheduler_choice,
        val_interval=1,
        skip_cfg=True,
        segmentation=args.segmentation,
    )
    
    # Build pipeline
    pipeline.build()
    
    # Execute based on mode
    if not args.test_only:
        print("Starting training...")
        pipeline.train()
        print("Training completed!")
    
    if not args.train_only:
        print("Starting testing...")
        pipeline.test()
        print("Testing completed!")
    
    print(f"Results saved to: {result_folder}")


if __name__ == '__main__':
    main()
