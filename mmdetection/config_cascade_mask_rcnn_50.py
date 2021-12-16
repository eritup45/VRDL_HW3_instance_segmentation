
# The new config inherits a base config to highlight the necessary modification

# # We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1),
#         mask_head=dict(num_classes=1)))

# _base_ = './configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco.py'

# _base_ = [
#     './configs/_base_/models/cascade_mask_rcnn_r50_fpn.py',
#     './configs/_base_/datasets/coco_instance.py',
#     './configs/_base_/schedules/schedule_20e.py', './configs/_base_/default_runtime.py'
# ]

_base_ = [
    './configs/_base_/models/cascade_mask_rcnn_r50_fpn.py',
    './configs/_base_/datasets/coco_instance.py',
    './configs/_base_/schedules/schedule_20e.py', './configs/_base_/default_runtime.py'
]
# gpu_assign_thr = 40
model = dict(
    roi_head=dict(
        mask_head=dict(num_classes=1),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                # gpu_assign_thr = 40,
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    # gpu_assign_thr = 40,
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    # gpu_assign_thr = 40,
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    # gpu_assign_thr = 40,
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]
    )
)


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('nucleus',)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        img_prefix='data/coco/train2017/train_pic/',
        classes=classes,
        ann_file='data/coco/annotations/train_coco.json'),
    val=dict(
        img_prefix='data/coco/val2017/val_pic/',
        classes=classes,
        ann_file='data/coco/annotations/val_coco.json'),
    test=dict(
        img_prefix='data/coco/test2017/test/',
        classes=classes,
        ann_file='data/coco/annotations/test_coco.json'))

# total_epochs = 70
# resume_from = './work_dirs/1206_ep20_config_cascade_mask_rcnn_50/latest.pth'

optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
# fp16 = dict(loss_scale=512.)

# 1 epoch for training and 1 epoch for validation will be run iteratively.
workflow = [('train', 1), ('val', 1)]

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = './checkpoints/cascade_mask_rcnn_r50_fpn_20e_coco_bbox_mAP-0.419__segm_mAP-0.365_20200504_174711-4af8e66e.pth'
# load_from = './work_dirs/1207_ep60_config_cascade_mask_rcnn_50/epoch_9.pth' # BEST

# data: 1212
# load_from = './work_dirs/1212_ep20_cascade_mask_rcnn/epoch_17.pth'
# load_from = './work_dirs/1212_ep20_cascade_mask_rcnn/epoch_7.pth'
load_from = './work_dirs/1212_ep40_cascade_mask_rcnn/epoch_1.pth'
