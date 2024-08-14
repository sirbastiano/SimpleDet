_base_ = [
    '../_base_/datasets/venus_detection.py',
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/schedules/schedule_120.py', 
    '../_base_/esa_runtime.py'
]


# model
model = dict(
    data_preprocessor=dict(
        bgr_to_rgb=False,
        pad_size_divisor=1),
    backbone=dict(
        in_channels=3, ##### MODIFIED #####
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))

