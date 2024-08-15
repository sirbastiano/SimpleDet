IMG_SCALE = (
    2304,
    2304,
)
auto_scale_lr = dict(base_batch_size=2, enable=False)
backend_args = None
color_type = 'grayscale'
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'custom_components.Loader',
        'custom_components.PreProcessor',
        'custom_components.Corrupter',
    ])
data_root = '/Data_large/marine/Datasets/VENuS/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=1,
        save_best='coco/bbox_mAP_50',
        save_last=False,
        type='CheckpointHook'),
    early_stopping=dict(
        min_delta=0.005,
        monitor='coco/bbox_mAP_50',
        patience=10,
        type='EarlyStoppingHook'),
    logger=dict(interval=1, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=False, type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = '/Data_large/marine/PythonProjects/MMDET/checkpoints/VENuS/Single/perfect_b10/53_BS_2_LR_0.0008_ME_30_OPT_SGD/epoch_30.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=('Vessel', ), palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    backbone=dict(
        depth=18,
        frozen_stages=1,
        in_channels=1,
        init_cfg=dict(checkpoint='torchvision://resnet18', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        center_sampling=False,
        dcn_on_last_conv=False,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.5, type='GIoULoss'),
        loss_bbox_refine=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0,
            type='VarifocalLoss',
            use_sigmoid=True),
        num_classes=1,
        stacked_convs=3,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='VFNetHead',
        use_atss=True,
        use_vfl=True),
    data_preprocessor=dict(
        mean=[
            111.64337,
        ],
        pad_size_divisor=1,
        std=[
            103.4634,
        ],
        type='MyPrePro'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            64,
            128,
            256,
            512,
        ],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=9, type='ATSSAssigner'),
        debug=False,
        pos_weight=-1),
    type='VFNet')
optim_wrapper = dict(
    optimizer=dict(lr=0.0008, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=6, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=6,
        gamma=0.75,
        milestones=[
            7,
            10,
            15,
        ],
        type='MultiStepLR'),
    dict(
        T_max=20.0,
        begin=15,
        by_epoch=True,
        convert_to_iter_based=True,
        end=30,
        eta_min=4e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(diff_rank_seed=True, seed=53)
reader = 'tifffile'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/Data_large/marine/Datasets/VENuS/annotations/perfect/test__band_10.json',
        data_prefix=dict(img='perfect/'),
        data_root='/Data_large/marine/Datasets/VENuS/ds_L0/',
        filter_cfg=dict(filter_empty_gt=True),
        metainfo=dict(classes=('Vessel', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(bands_list=[
                10,
            ], to_float32=True, type='SelBandLoader'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                2048,
                2048,
            ), type='Resize'),
            dict(corruption='gaussian', severity=2.0, type='ImageCorruption'),
            dict(
                meta_keys=(
                    'img_path',
                    'img_id',
                    'seg_map_path',
                    'height',
                    'width',
                    'instances',
                    'sample_idx',
                    'img',
                    'img_shape',
                    'ori_shape',
                    'scale',
                    'scale_factor',
                    'keep_ratio',
                    'homography_matrix',
                    'gt_bboxes',
                    'gt_ignore_flags',
                    'gt_bboxes_labels',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/Data_large/marine/Datasets/VENuS/annotations/perfect/test__band_10.json',
    format_only=False,
    metric='bbox',
    outfile_prefix=
    '/Data_large/marine/PythonProjects/MMDET/checkpoints/VENuS/Single/perfect_b10/53_BS_2_LR_0.0008_ME_30_OPT_SGD/test_results',
    type='CocoMetric')
test_pipeline = [
    dict(
        backend_args=None,
        color_type='grayscale',
        imdecode_backend='tifffile',
        to_float32=False,
        type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        2304,
        2304,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=30, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file=
        '/Data_large/marine/Datasets/VENuS/annotations/perfect/train__band_10.json',
        backend_args=None,
        data_prefix=dict(img='perfect/'),
        data_root='/Data_large/marine/Datasets/VENuS/ds_L0/',
        filter_cfg=dict(filter_empty_gt=True),
        metainfo=dict(classes=('Vessel', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(bands_list=[
                10,
            ], to_float32=True, type='SelBandLoader'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.4, type='RandomFlip'),
            dict(keep_ratio=False, scale=(
                2048,
                2048,
            ), type='Resize'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        color_type='grayscale',
        imdecode_backend='tifffile',
        to_float32=False,
        type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.4, type='RandomFlip'),
    dict(keep_ratio=True, scale=(
        2304,
        2304,
    ), type='Resize'),
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        '/Data_large/marine/Datasets/VENuS/annotations/perfect/val__band_10.json',
        backend_args=None,
        data_prefix=dict(img='perfect/'),
        data_root='/Data_large/marine/Datasets/VENuS/ds_L0/',
        filter_cfg=dict(filter_empty_gt=True),
        metainfo=dict(classes=('Vessel', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(bands_list=[
                10,
            ], to_float32=True, type='SelBandLoader'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                2048,
                2048,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/Data_large/marine/Datasets/VENuS/annotations/perfect/val__band_10.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/Data_large/marine/PythonProjects/MMDET/studies/tta_study'
