
custom_imports = dict(imports=['custom_components.Loader',
                               'custom_components.PreProcessor', 
                               'custom_components.Corrupter',
                               'custom_components.Head',
                               'custom_components.Encoder',
                               'mmpretrain.models'
                               ], 
                    allow_failed_imports=False)


############## MODEL CONFIGS ##############

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

############## DATALOADER CONFIGS ##############


# dataset settings
data_root = "./Data"
dataset_type = 'CocoDataset'

# # Modify dataset related settings
metainfo = {
    'classes': ('wake', ),
    'palette': [
        (220, 20, 60),
    ]
}

backend_args = None
IMG_SCALE = (1280, 1280)
reader = 'tifffile'
color_type='grayscale' # 'grayscale' or 'color'

train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True, color_type=color_type, imdecode_backend=reader, backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.4),
    dict(type='Resize', scale=IMG_SCALE, keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True, color_type=color_type, imdecode_backend=reader, backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=IMG_SCALE, keep_ratio=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='imgs/imgs_B2/'),
        filter_cfg=dict(filter_empty_gt=True), # , min_size=32
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='imgs/imgs_B2/'),
        test_mode=True,
        pipeline=test_pipeline,
        filter_cfg=dict(filter_empty_gt=True), # , min_size=32
        backend_args=backend_args))

# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + 'annotations/test.json',
        data_prefix=dict(img='imgs/imgs_B2/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True), # , min_size=32
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + 'annotations/test.json',
    outfile_prefix='./work_dirs/test_Sentinel')


############## SCHEDULE CONFIGS ##############

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=120,
        by_epoch=True,
        milestones=[50, 80, 90],
        gamma=0.7)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=2)


############## RUNTIME CONFIGS ##############
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    early_stopping=dict(type="EarlyStoppingHook", monitor="coco/bbox_mAP_50", patience=25, min_delta=0.005),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best=r'coco/bbox_mAP_50', max_keep_ckpts=1, save_last=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


vis_backends = [dict(type='LocalVisBackend'),]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False