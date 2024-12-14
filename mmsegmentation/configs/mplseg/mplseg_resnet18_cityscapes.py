_base_ = [
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_125k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MPLSeg',
        backbone_channels=(64, 128, 256, 512),
        mid_channels=(256, 384, 512, 512),
        fuse_channels=(128, 192, 256, 256),
        out_indices=(0, 1, 2, 3),
        backbone_cfg=dict(
            type='ResNetV1c',
            in_channels=3,
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 1, 1),
            strides=(1, 2, 2, 2),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet18_v1c')),
        norm_cfg=norm_cfg,
        init_cfg=None),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=0,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            ]),
    auxiliary_head=[
dict(
            type='FCNHead',
            in_channels=192,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.1),
                dict(
                type='PSLLoss', use_sigmoid=False, loss_weight=0.75)
                ]),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.1),
                dict(
                type='PSLLoss', use_sigmoid=False, loss_weight=0.5)
                ]),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=128,
            num_convs=1,
            num_classes=19,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.1),
                dict(
                type='PSLLoss', use_sigmoid=False, loss_weight=0.25)
                ]),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    )

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,
)
