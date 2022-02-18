_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/road.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

# model settings
model = dict(
#    pretrained='models/road_v1.pth'
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 18, 3]),
    decode_head=dict(num_classes=5, norm_cfg=norm_cfg, in_channels=[64, 128, 320, 512]),
)
