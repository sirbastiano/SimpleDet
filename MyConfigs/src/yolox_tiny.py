_base_ = './yolox_s.py'

# model settings
model = dict(
    # data_preprocessor=dict(batch_augments=[
    #     dict(
    #         type='BatchSyncRandomResize',
    #         random_size_range=(1600, 2024),
    #         size_divisor=32,
    #         interval=10)
    # ]),
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(in_channels=96, feat_channels=96))