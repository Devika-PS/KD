_base_ = [
    '../_base_/datasets/cifar10_bs128.py'
]

# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './Student_withnograd/'

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='step', step=[60, 120, 150], gamma=0.2)
runner = dict(type='EpochBasedRunner', max_epochs=200)

# model settings
model = dict(
    type='KDClassifier',
    kd_loss=dict(type='DistillationLoss', Temperature=4.0),
    at_loss=dict(type='atloss'),
    train_cfg=dict(alpha=0.9, beta=1e+3,
                   teacher_checkpoint='/home/mushti/mmlab/mmclassification/Teacher/epoch_200.pth'), # Input your teacher checkpoint
    backbone=dict(
        # return_tuple=False,
        student=dict(
            type='ResNet_CIFAR',
            depth=16,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        teacher=dict(
            type='ResNet_CIFAR',
            depth=34,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch')
    ),
    neck=dict(
        student=dict(type='GlobalAveragePooling'),
        teacher=dict(type='GlobalAveragePooling')
    ),
    head=dict(
        student=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        ),
        teacher=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        )
    ))