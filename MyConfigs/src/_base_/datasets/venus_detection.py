# dataset settings
dataset_type = 'CocoDataset'
data_root = '/Data_large/marine/Datasets/VENuS/'

# Modify dataset related settings
metainfo = {
    'classes': ('Vessel', ),
    'palette': [
        (220, 20, 60),
    ]
}

backend_args = None
IMG_SCALE = (2304, 2304)
reader = 'tifffile'
color_type='color' # 'grayscale' or 'color'

train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=False, color_type=color_type, imdecode_backend=reader, backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.4),
    dict(type='Resize', scale=IMG_SCALE, keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile',to_float32=False, color_type=color_type, imdecode_backend=reader, backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=IMG_SCALE, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/coarse/train.json',
        data_prefix=dict(img='ds_L0/coarse3/'),
        filter_cfg=dict(filter_empty_gt=True), # , min_size=32
        pipeline=train_pipeline,
        backend_args=backend_args))

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type = 'RepeatDataset',
#         times = 2,
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             metainfo=metainfo,
#             ann_file='annotations/coarse/train.json',
#             data_prefix=dict(img='ds_L0/coarse3/'),
#             filter_cfg=dict(filter_empty_gt=True), # , min_size=32
#             pipeline=train_pipeline,
#             backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/coarse/val.json',
        data_prefix=dict(img='ds_L0/coarse3/'),
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
        ann_file=data_root + 'annotations/coarse/test.json',
        data_prefix=dict(img='ds_L0/coarse3/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True), # , min_size=32
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/coarse/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + 'annotations/coarse/test.json',
    outfile_prefix='./work_dirs/test_Venus')