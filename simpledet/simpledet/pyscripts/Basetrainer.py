import logging
import os, sys
import torch
import numpy as np
import random
import json
import logging

# To set deterministic behaviour:
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.evaluation import DumpDetResults

from mmdet.utils import setup_cache_size_limit_of_dynamo
setup_cache_size_limit_of_dynamo()


class ObjectDetectionPipeline:
    def __init__(self, seed=71, band=[0], resize=1024, batch_size=2, learning_rate=0.001,
                 random_crop=None, max_epochs=30, amp=False,
                 optimizer_choice='SGD'):
        
        self.seed = seed
        self.band = band
        self.resize = resize
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_crop = random_crop
        self.max_epochs = max_epochs
        self.amp = amp
        self.optimizer_choice = optimizer_choice

        self.base_folder = '/Data_large/marine/PythonProjects/MMDET/MyConfigs'
        self.data_root = '/Data_large/marine/PythonProjects/MMDET/Data/Multispectral_Wakes' 
        self.data_prefix = 'B8/imgs' 
        self.workdir = self._construct_workdir()
        self.optimizers = self._get_optimizers()

        self.cfg = self._init_cfg()
        self.logger = self._set_logger()
        self._set_seed()
        self._configure_pipeline()
        self.logger.info(f'Pipeline configured with the following parameters: Seed {self.seed}, Batch Size: {self.batch_size}, Learning Rate: {self.learning_rate}, Random Crop: {self.random_crop}, Sensor: {self.sensor}, Special: {self.special}, Max Epochs: {self.max_epochs}, AMP: {self.amp}, Optimizer: {self.optimizer_choice}')

    def _init_cfg(self):
        cfg = Config.fromfile(f'{self.base_folder}/base_config.py')
        cfg.work_dir = self.workdir
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
        bands_names = ''.join([f'_b{x}' for x in self.band])
        single_multi = 'Multi' if len(self.band) > 1 else 'Single'
        k_mode = {'SENTINEL': 'Sentinel', 'VENUS': 'VENuS'}
        workdir = (f'/Data_large/marine/PythonProjects/MMDET/Deploy_out/'
                   f'{k_mode[self.sensor]}/Export/BS_{self.batch_size}/'
                   f'LR_{self.learning_rate}/IMG_{self.resize}/BANDS_{bands_names}/'
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
        # Normalization
        means = [200, 154, 92, 63] if self.sensor == 'SENTINEL' else [158.69588, 124.42161, 109.27108, 105.380424, 88.40926, 98.93067, 88.819916, 94.20678, 103.540764, 111.64337, 122.92817, 79.31501]
        stds = [22, 24, 22, 60] if self.sensor == 'SENTINEL' else [34.95446, 46.282494, 56.252197, 55.741932, 64.54027, 59.59095, 69.65824, 68.40028, 77.930405, 103.4634, 105.30468, 65.8369]
        mean_vals = [means[self.index_correct[x]] for x in self.band]
        std_vals = [stds[self.index_correct[x]] for x in self.band]

        # Configure cfg with preprocessor
        self.cfg.model.data_preprocessor = dict(
            mean=[float(x) for x in mean_vals],
            pad_size_divisor=1,
            std=[float(x) for x in std_vals],
            type='MyPrePro')

        # Model Inputs
        self.cfg.model.backbone.in_channels = len(self.band)

        # Dataloader directories and annotation files
        special_case = '_' if not self.special else 'special_'
        base_annot = '/Data_large/marine/Datasets/VDS2Raw/annotations' if self.sensor == 'SENTINEL' else '/Data_large/marine/Datasets/VENuS/annotations/perfect'
        ann_file = {mode: f'{base_annot}/{mode.lower()}_{special_case}band_{self.band[0]}.json' for mode in ['Train', 'Val', 'Test']}

        self.cfg.train_dataloader.dataset.ann_file = ann_file['Train']
        self.cfg.train_dataloader.dataset.data_prefix = {'img': self.data_prefix}
        self.cfg.train_dataloader.dataset.data_root = self.data_root

        self.cfg.val_dataloader.dataset.ann_file = ann_file['Val']
        self.cfg.val_dataloader.dataset.data_prefix = {'img': self.data_prefix}
        self.cfg.val_dataloader.dataset.data_root = self.data_root

        # Evaluators
        self.cfg.val_evaluator = dict(
            ann_file=ann_file['Val'],
            backend_args=None,
            format_only=False,
            metric='bbox',
            type='CocoMetric')

        # Pipeline configuration
        load_correct = {2: 1, 3: 2, 4: 3, 8: 4} if self.sensor == 'SENTINEL' else {i: i for i in range(1, 13, 1)}
        band_sel_load = [load_correct[x] for x in self.band]

        self.cfg.train_dataloader.dataset.pipeline[0] = {'type': 'SelBandLoader', 'to_float32': True, 'bands_list': band_sel_load}
        self.cfg.val_dataloader.dataset.pipeline[0] = {'type': 'SelBandLoader', 'to_float32': True, 'bands_list': band_sel_load}

        self.cfg.train_dataloader.dataset.pipeline[3] = {'type': 'Resize', 'scale': (self.resize, self.resize), 'keep_ratio': False}
        self.cfg.val_dataloader.dataset.pipeline[2] = {'type': 'Resize', 'scale': (self.resize, self.resize), 'keep_ratio': False}

        # Metainfo
        classes = ('vessel',) if self.sensor == 'SENTINEL' else ('Vessel',)
        self.cfg.train_dataloader.dataset.metainfo = {'classes': classes, 'palette': [(220, 20, 60)]}
        self.cfg.val_dataloader.dataset.metainfo = {'classes': classes, 'palette': [(220, 20, 60)]}

        # Random crop (if applicable)
        if self.random_crop is not None:
            assert isinstance(self.random_crop, int), 'RandomCrop Error: single dimension must be specified. E.g. 224'
            rc = dict(type='RandomCrop', crop_size=(self.random_crop, self.random_crop))
            self.cfg.train_dataloader.dataset.pipeline.insert(4, rc)
            self.cfg.val_dataloader.dataset.pipeline.insert(2, rc)

        # Training parameters
        self.cfg.train_dataloader.batch_size = self.batch_size
        self.cfg.train_cfg = {'type': 'EpochBasedTrainLoop', 'max_epochs': self.max_epochs, 'val_interval': 1}
        self.cfg.optim_wrapper = self.optimizers[self.optimizer_choice]

        # Learning rate scheduler
        self.cfg.param_scheduler = [
            {'type': 'LinearLR', 'start_factor': 0.001, 'by_epoch': True, 'begin': 0, 'convert_to_iter_based': True, 'end': self.max_epochs // 5},
            {'type': 'MultiStepLR', 'begin': 0, 'end': self.max_epochs // 5, 'by_epoch': True, 'milestones': [self.max_epochs // 4, self.max_epochs // 3, self.max_epochs // 2], 'gamma': 0.75},
            {'type': 'CosineAnnealingLR', 'eta_min': self.learning_rate * 0.05, 'begin': self.max_epochs // 2, 'end': self.max_epochs, 'T_max': self.max_epochs // 1.5, 'by_epoch': True, 'convert_to_iter_based': True}
        ]

        # Test configuration
        self.cfg.test_dataloader = dict(
            batch_size=1,
            dataset=dict(
                ann_file=ann_file['Test'],
                data_root=self.data_root,
                data_prefix=dict(img=self.data_prefix),
                filter_cfg=dict(filter_empty_gt=True),
                metainfo=dict(classes=classes, palette=[(220, 20, 60)]),
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
            ann_file=ann_file['Test'],
            outfile_prefix=f'{self.workdir}/test_results')

        # Enable AMP if needed
        if self.amp:
            optim_wrapper = self.cfg.optim_wrapper.type
            if optim_wrapper == 'AmpOptimWrapper':
                print_log('AMP training is already enabled in your config.', logger='current', level=logging.WARNING)
            else:
                assert optim_wrapper == 'OptimWrapper', f'`--amp` is only supported when the optimizer wrapper type is `OptimWrapper` but got {optim_wrapper}.'
                self.cfg.optim_wrapper.type = 'AmpOptimWrapper'
                self.cfg.optim_wrapper.loss_scale = 'dynamic'

    def build(self):
        # Check if custom runner is defined
        if 'runner_type' not in self.cfg:
            runner = Runner.from_cfg(self.cfg)
        else:
            runner = RUNNERS.build(self.cfg)
        
        self.runner = runner
        self.logger.info(f'Pipeline built successfully.')

    def train(self):
        self.runner.train()
        
    def test(self):
        self.runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=f'{self.workdir}/test_result/test.pkl'))
        # start testing
        output_test_data =self.runner.test()

        # Specify the file name
        file_name = f'{self.workdir}/test_result/coco_metrics.json'# Specify the filepath
        # Write the dictionary to a JSON file
        with open(file_name, 'w') as json_file:
            json.dump(output_test_data, json_file, indent=4)

        self.logger.info(f"Data has been saved to {file_name}")
        self.logger.info(f'Testing completed successfully.')
        self.logger.info(f'output_test_data: {output_test_data}')



if __name__ == '__main__':
    # Usage:
    for seed in [42, 71, 18, 53, 89]:
        pipeline = ObjectDetectionPipeline(batch_size=2, band=[5], seed=seed, learning_rate=0.001, max_epochs=30)
        pipeline.build() # Builds the pipeline

        pipeline.train()
        pipeline.test()