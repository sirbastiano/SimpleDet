_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/venus_detection.py',
    '../_base_/schedules/schedule_120.py', '../_base_/esa_runtime.py'
]

model = dict(
    data_preprocessor=dict(
        # The mean and std are used in PyCls when training RegNets
        mean=[200, 154, 116], # [123.675, 116.28, 103.53]
        std=[22, 24, 27],
        bgr_to_rgb=False),
    backbone=dict(
        _delete_=True,
        type='RegNet',
        arch='regnetx_3.2gf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_3.2gf')),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 432, 1008],
        out_channels=256,
        num_outs=5))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00005),
    clip_grad=dict(max_norm=35, norm_type=2))
