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


# wanb_cfg = dict(type='WandbVisBackend',
#                 init_kwargs={
#                     'project': 'ESA_S2L1C',
#                     'group': 'generic_name',
#                 })

vis_backends = [dict(type='LocalVisBackend'),]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False