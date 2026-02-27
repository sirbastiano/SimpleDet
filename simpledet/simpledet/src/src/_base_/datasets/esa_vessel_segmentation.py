# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/roberto/PythonProjects/S2RAWVessel/mmdetection/data/S2ESA/'

# Modify dataset related settings
metainfo = {
    'classes': ('Vessel', ),
    'palette': [
        (220, 20, 60),
    ]
}


backend_args = None
IMG_SCALE = (1024, 1024)
reader = 'tifffile'

train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=False, color_type='color', imdecode_backend=reader, backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=IMG_SCALE, keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=False, color_type='color', imdecode_backend=reader, backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
        type = 'RepeatDataset',
        times = 1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/train.json',
            data_prefix=dict(img='imgs/'),
            filter_cfg=dict(filter_empty_gt=True), # , min_size=32
            pipeline=train_pipeline,
            backend_args=backend_args)))

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
        ann_file='annotations/val.json',
        data_prefix=dict(img='imgs/'),
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
        data_prefix=dict(img='imgs/'),
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
    outfile_prefix='./work_dirs/esa_v2_detection/test_results/')