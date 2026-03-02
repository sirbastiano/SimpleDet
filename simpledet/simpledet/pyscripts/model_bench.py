import numpy as np
import torch
import os
import sys
import random
import json
import logging
import time
 
# ruff: noqa: F403,F405,F601


sys.path.append('/Data_large/marine/PythonProjects/MMDET/notebooks/Tools')
sys.path.append('/Data_large/marine/PythonProjects/MMDET/MyConfigs/custom_components')
sys.path.append('/Data_large/marine/PythonProjects/MMDET/MyConfigs')
# To set deterministic behaviour:
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'

from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmdet.evaluation import DumpDetResults


from MyConfigs.models import *
from runscripts.flops_n_latency.latency import my_inference_benchmark
from runscripts.flops_n_latency.flops import benchmark_flops

from mmdet.utils import setup_cache_size_limit_of_dynamo
import argparse
setup_cache_size_limit_of_dynamo()

# Set up logging
logging.basicConfig(level=logging.INFO, filename='bench_log.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


model_dict = {
    'TridentFasterRCNN_r50_caffe': TridentFasterRCNN_r50_caffe,
    'VFNet_r101_fpn': VFNet_r101_fpn,
    'VFNet_r18_fpn': VFNet_r18_fpn,
    'VFNet_r34_fpn': VFNet_r34_fpn,
    'VFNet_r50_fpn': VFNet_r50_fpn,
    'cascade_mask_rcnn_r101_fpn': cascade_mask_rcnn_r101_fpn,
    'cascade_mask_rcnn_r18_fpn': cascade_mask_rcnn_r18_fpn,
    'cascade_mask_rcnn_r34_fpn': cascade_mask_rcnn_r34_fpn,
    'cascade_mask_rcnn_r50_fpn': cascade_mask_rcnn_r50_fpn,
    'cascade_rcnn_r50_fpn': cascade_rcnn_r50_fpn,
    'deformable_detr_r50': deformable_detr_r50,
    'detr_r50': detr_r50,
    'fcos_r50_fpn': fcos_r50_fpn,
    'fcos_r18_fpn': fcos_r18_fpn,
    'fcos_r101_fpn': fcos_r101_fpn,
    'fcos_r34_fpn': fcos_r34_fpn,
    'faster_rcnn_r101_fpn': faster_rcnn_r101_fpn,
    'faster_rcnn_r18_fpn': faster_rcnn_r18_fpn,
    'faster_rcnn_r34_fpn': faster_rcnn_r34_fpn,
    'faster_rcnn_r50_fpn': faster_rcnn_r50_fpn,
    'faster_rcnn_r50_fpn_caffe_c4': faster_rcnn_r50_fpn_caffe_c4,
    'fovea_r101': fovea_r101,
    'fovea_r18': fovea_r18,
    'fovea_r34': fovea_r34,
    'fovea_r50': fovea_r50,
    'mask_rcnn_r101_fpn': mask_rcnn_r101_fpn,
    'mask_rcnn_r18_fpn': mask_rcnn_r18_fpn,
    'mask_rcnn_r34_fpn': mask_rcnn_r34_fpn,
    'mask_rcnn_r50_fpn': mask_rcnn_r50_fpn,
    'retina_swin': retina_swin,
    'retinanet_r101_fpn': retinanet_r101_fpn,
    'retinanet_r18_fpn': retinanet_r18_fpn,
    'retinanet_r34_fpn': retinanet_r34_fpn,
    'retinanet_r50_fpn': retinanet_r50_fpn,
    'ssd300': ssd300,
    'htc_r50_fpn': htc_r50_fpn,
    'htc_r101_fpn': htc_r101_fpn,
    'yolof_r50_caffe': yolof_r50_caffe,
    'centernet_50': centernet_update_r50_fpn_8xb8_amp_lsj_200e_coco,
    'centernet_18': centernet_update_r18_fpn_8xb8_amp_lsj_200e_coco,
    'fsaf_r50_fpn': fsaf_r50_fpn,
    'SparseRCNN_r50': SparseRCNN_r50,
    'retina_effb3': retina_effb3,
    'cascade_rcnn_hrnetv2p_w40': cascade_rcnn_hrnetv2p_w40,
    'cascade_rcnn_hrnetv2p_w32_20e_coco': cascade_rcnn_hrnetv2p_w32_20e_coco,
}


class ObjectDetectionPipeline:
    def __init__(self, 
                seed=71,
                model_cfg=None, 
                resize=512, 
                batch_size=2, 
                learning_rate=0.001,
                random_crop=None, 
                max_epochs=1,
                data_folder=None, # where the data is stored
                result_folder=None, # where the results are stored
                config_folder="./MyConfigs", # where the config files are stored.
                data_prefix='imgs/', # prefix for the data files where images are stored.
                mean_vals=[0.485, 0.456, 0.406], # mean values for normalization expressed as list for each channel.
                std_vals=[1.0, 1.0, 1.0], # mean values for normalization expressed as list for each channel.
                in_channels=1, # number of channels in input
                tif_channels_to_load=None, # list of channels to load from the tif file, in form of list
                annot_file_train=None, # annotation file for training
                annot_file_val=None, # annotation file for validation
                annot_file_test=None, # annotation file for testing
                categories = ('vessel',), # categories to detect in form of tuple
                amp=True, # Automatic Mixed Precision
                optimizer_choice='SGD', # Optimizer choice: SGD, Adam, AdamW
                scheduler_choice=None, # Scheduler choice: MultiStepLR, CosineAnnealingLR
                val_interval=1, # Validation interval
                skip_cfg=False,
                segmentation=False,
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
        assert isinstance(self.tif_channels_to_load, list), 'Channels to load from the tif file must be specified as a list.'
        assert len(tif_channels_to_load) == self.in_channels, 'Number of channels to load from the tif file must be equal to the number of input channels.'
        self.categories = categories
        assert self.categories is not None, 'Categories must be specified.'
        self.segmentation = segmentation

        # Outputs:
        self.workdir = self._construct_workdir()        
        print(f"Workdir: {self.workdir}")
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
        # Check if custom runner is defined
        if 'runner_type' not in self.cfg:
            runner = Runner.from_cfg(self.cfg)
        else:
            runner = RUNNERS.build(self.cfg)
        
        self.runner = runner
        self.logger.info('Pipeline built successfully.')

    def train(self):
        self.runner.train()
        
    def test(self):
        self.runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=self.outfile_test))
        # start testing
        output_test_data =self.runner.test()

        # Specify the file name
        file_name = self.outfile_metric # Specify the filepath
        # Write the dictionary to a JSON file
        with open(file_name, 'w') as json_file:
            json.dump(output_test_data, json_file, indent=4)

        self.logger.info(f"Data has been saved to {file_name}")
        self.logger.info('Testing completed successfully.')
        self.logger.info(f'output_test_data: {output_test_data}')
        
        
        
def model_bench(model, args=None):
    PROJECT_NAME = 'scieottiche'
    SEGMENTATION = False
    if model.startswith('SSS'):
        SEGMENTATION = True
        model = model.replace('SSS','')

    logging.info(f"Model: {model}")
    ModelName = model

    DATAROOT = "/Data_large/marine/PythonProjects/MMDET/Data/Multispectral_Wakes"
    IMG_PREFIX = f"{DATAROOT}/B4/imgs/"
    ANNOT_TRAIN = "/Data_large/marine/PythonProjects/MMDET/Data/Multispectral_Wakes/B4/Annotations/LAST_SWD_Train.json"
    ANNOT_VAL = "/Data_large/marine/PythonProjects/MMDET/Data/Multispectral_Wakes/B4/Annotations/LAST_SWD_Test.json"
    ANNOT_TEST = "/Data_large/marine/PythonProjects/MMDET/Data/Multispectral_Wakes/B4/Annotations/LAST_SWD_Test.json"
    CATEGORIES = ('wake',)
    BASE_CONFIG_PATH = "/Data_large/marine/PythonProjects/MMDET/MyConfigs"

    INPUT_SIZE = 768
    BATCH_SIZE = 4
    EPOCHS = args.epochs
    CHANNELS_LOADED = [1,1,1] # 1-indexed for rasterio
    INPUT_CHANNELS = len(CHANNELS_LOADED)
    INPUT_CHANNELS = 3

    MODEL_CFG = model_dict[ModelName]
    # now time:
    now = time.strftime("%Y%m%d-%H%M%S")

    RESULT_FOLDER = f"/Data_large/marine/PythonProjects/MMDET/Results/{PROJECT_NAME}/{ModelName}/{now}"

    # Check if the model has already been benchmarked
    benchmark_dir = "/Data_large/marine/PythonProjects/MMDET/BENCH/"
    for file in os.listdir(benchmark_dir):
        if file.endswith("_benchmark_results.txt") and ModelName == file:
            print(f"Model {ModelName} already benchmarked!")
            logging.info(f"Model {ModelName} already benchmarked!")
            return None
    else:
        logging.info(f"Model {ModelName} benchmarking started.")
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

        logging.info(f"Model {ModelName} pipeline configured.")
        pipeline.build() # Builds the pipeline
        logging.info(f"Model {ModelName} pipeline built.")
        pipeline.train() 
        logging.info(f"Model {ModelName} pipeline trained.")
        pipeline.test() # Test the pipeline
        logging.info(f"Model {ModelName} pipeline tested.")
        
        try:
            logging.info(f"Model {ModelName} benchmarking started.")
            logger = MMLogger.get_instance(name='MMLogger')
            # if detr is selected: pipeline.cfg.model.bbox_head.loss_cls.class_weight = float(1.0)
            
            if model == 'detr_r50':
                #bugfix mmdet
                pipeline.cfg.model.bbox_head.loss_cls.class_weight = float(1.0)
            
            benchmark = my_inference_benchmark(cfg=pipeline.cfg.copy(), logger=logger, checkpoint=pipeline.get_pth()) 
            
            res = benchmark.run(3)
            flops = benchmark_flops(cfg=pipeline.cfg.copy(), logger=logger)
            # Write the model name and results to a .txt file
            with open(f"/Data_large/marine/PythonProjects/MMDET/BENCH/{ModelName}_benchmark_results.txt", "w") as file:
                file.write(f"Model: {ModelName}\n")
                file.write(f"Results: {res}\n")
                file.write(f"FLOPs: {flops}\n")
            logging.info(f"Model {ModelName} benchmarked successfully.")
            logging.info(f"Results: {res}")
            return res
        except Exception as e:
            print(f"Error: {e}")
            logging.error(f"Error: {e}")
            return None


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark object detection models.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to benchmark.')
    parser.add_argument('--epochs', type=int, required=False, default=4, help='Number of epochs.')
    return parser.parse_args()


if __name__ == "__main__":
    print("Starting the benchmarking process...")
    args = parse_args()
    model_bench(args.model, args=args)
    print("Benchmarking process completed.")
