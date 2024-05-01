_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/cityscapes_1024x1024_repeat.aug.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='OthersEncoderDecoder',
    pretrained='pretrained/segformer.b5.1024x1024.city.160k.replace.pth',
    backbone=dict(
        type='mit_b5',
        style='pytorch',
        drop_rate=0.1,
        drop_path_rate=0.1,
        pet_cls="Adapter",
        adapt_blocks=[0, 1, 2, 3],
        aux_classifier=False),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024,1024), stride=(768,768)),
    freeze_backbone=True,
    freeze_decode_head=True)

# data
data = dict(samples_per_gpu=2)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=500, metric='mIoU')

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.01)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

