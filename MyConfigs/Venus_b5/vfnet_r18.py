_base_ = [
    './vfnet_r50.py',
]

custom_imports = dict(imports=['custom_components.Loader',
                               'custom_components.PreProcessor', 
                               'custom_components.Corrupter',
                               'custom_components.Encoder',
                               ], 
                    allow_failed_imports=False)

# model
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))