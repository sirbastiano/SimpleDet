# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1048, val_interval=1)
val_cfg = dict(type='ValLoop', val_interval=1)
test_cfg = dict(type='TestLoop', val_interval=1)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-05, by_epoch=True, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=0,
        end=1048,
        T_max=150,
        by_epoch=True,
        convert_to_iter_based=True)
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
