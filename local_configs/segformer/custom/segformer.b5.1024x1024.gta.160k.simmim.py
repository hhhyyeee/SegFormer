_base_ = [
    '../../_base_/models/segformer.py',
    # '../../_base_/datasets/gta_1024x1024_repeat.py',
    '../../_base_/datasets/gta_1024x1024_repeat.gtatest.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='SimMIMEncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(
        type='mit_b5_cvp_simmim',
        style='pytorch',
        img_size=1024,

        pet_cls='Adapter',
        # adapt_blocks=[0, 1, 2, 3],
        aux_classifier=False,

        scale_factor=4,
        input_type="fft",
        freq_nums=0.25,
        prompt_type="highpass",
        # tuning_stage=1234,
        tuning_stage=None,

        handcrafted_tune=False,
        embedding_tune=False,
        conv_tune=False, #!DEBUG

        adaptor="adaptor"
    ),
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
    recon_neck=dict(
        type='SimMIMSegFormerDecoder',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],

        # type='SimMIMLinearDecoder',
        # in_channels=128 * 2**3,
        decoder_params=dict(embed_dim=768),
        encoder_stride=4 #32
    ),
    recon_head=dict(
        type='SimMIMHead',
        patch_size=4,
        loss=dict(type='PixelReconstructionLoss', criterion='L1', channel=3)
    ),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'),
    test_cfg=dict(mode='slide', crop_size=(1024,1024), stride=(768,768)),
    mask_cfg = dict(mask_ratio=0.3, test_input_size=(1024, 1024)))

# data
data = dict(samples_per_gpu=2)
evaluation = dict(interval=1, metric='mIoU')
# evaluation = dict(interval=1000, metric='mIoU')
# evaluation = dict(interval=4000, metric='mIoU')

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


