_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    # pretrained='https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar', # for old version of mmdetection
    backbone=dict(
        type='poolformer_s36_feat',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', 
            checkpoint=\
                'https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar', 
            ),
        ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

