{
    "dataloader": {
        "train": {
            "dataset": {
                "names": "main_train",
                "_target_": "<function get_detection_dataset_dicts at 0x7f9ae24c8b80>",
            },
            "mapper": {
                "is_train": True,
                "augmentations": [
                    {
                        "min_scale": 0.1,
                        "max_scale": 2.0,
                        "target_height": 1024,
                        "target_width": 1024,
                        "_target_": "<class 'detectron2.data.transforms.augmentation_impl.ResizeScale'>",
                    },
                    {
                        "crop_size": [1024, 1024],
                        "_target_": "<class 'detectron2.data.transforms.augmentation_impl.FixedSizeCrop'>",
                    },
                    {
                        "horizontal": True,
                        "_target_": "<class 'detectron2.data.transforms.augmentation_impl.RandomFlip'>",
                    },
                ],
                "image_format": "BGR",
                "use_instance_mask": True,
                "_target_": "<class 'detectron2.data.dataset_mapper.DatasetMapper'>",
                "recompute_boxes": True,
                "instance_mask_format": "bitmask",
            },
            "total_batch_size": 1,
            "num_workers": 2,
            "_target_": "<function build_detection_train_loader at 0x7f9ae24c8e50>",
        },
        "test": {
            "dataset": {
                "names": "main_validation",
                "filter_empty": False,
                "_target_": "<function get_detection_dataset_dicts at 0x7f9ae24c8b80>",
            },
            "mapper": {
                "is_train": False,
                "augmentations": [
                    {
                        "short_edge_length": 800,
                        "max_size": 1333,
                        "_target_": "<class 'detectron2.data.transforms.augmentation_impl.ResizeShortestEdge'>",
                    }
                ],
                "image_format": "BGR",
                "_target_": "<class 'detectron2.data.dataset_mapper.DatasetMapper'>",
                "instance_mask_format": "bitmask",
                "use_instance_mask": True,
            },
            "num_workers": 2,
            "_target_": "<function build_detection_test_loader at 0x7f9ae24c9040>",
        },
        "evaluator": {
            "dataset_name": "${..test.dataset.names}",
            "_target_": "<class 'detectron2.evaluation.coco_evaluation.COCOEvaluator'>",
        },
    },
    "lr_multiplier": {
        "scheduler": {
            "values": [1.0, 0.1, 0.01],
            "milestones": [163889, 177546],
            "num_updates": 184375,
            "_target_": "<class 'fvcore.common.param_scheduler.MultiStepParamScheduler'>",
        },
        "warmup_length": 0.002711864406779661,
        "warmup_factor": 0.067,
        "_target_": "<class 'detectron2.solver.lr_scheduler.WarmupParamScheduler'>",
    },
    "model": {
        "backbone": {
            "bottom_up": {
                "stem_class": "<class 'detectron2.modeling.backbone.regnet.SimpleStem'>",
                "stem_width": 32,
                "block_class": "<class 'detectron2.modeling.backbone.regnet.ResBottleneckBlock'>",
                "depth": 23,
                "w_a": 38.65,
                "w_0": 96,
                "w_m": 2.43,
                "group_width": 40,
                "norm": "BN",
                "out_features": ["s1", "s2", "s3", "s4"],
                "_target_": "<class 'detectron2.modeling.backbone.regnet.RegNet'>",
            },
            "in_features": "${.bottom_up.out_features}",
            "out_channels": 256,
            "top_block": {
                "_target_": "<class 'detectron2.modeling.backbone.fpn.LastLevelMaxPool'>"
            },
            "_target_": "<class 'detectron2.modeling.backbone.fpn.FPN'>",
            "norm": "BN",
        },
        "proposal_generator": {
            "in_features": ["p2", "p3", "p4", "p5", "p6"],
            "head": {
                "in_channels": 256,
                "num_anchors": 3,
                "_target_": "<class 'detectron2.modeling.proposal_generator.rpn.StandardRPNHead'>",
                "conv_dims": [-1, -1],
            },
            "anchor_generator": {
                "sizes": [[32], [64], [128], [256], [512]],
                "aspect_ratios": [0.5, 1.0, 2.0],
                "strides": [4, 8, 16, 32, 64],
                "offset": 0.0,
                "_target_": "<class 'detectron2.modeling.anchor_generator.DefaultAnchorGenerator'>",
            },
            "anchor_matcher": {
                "thresholds": [0.3, 0.7],
                "labels": [0, -1, 1],
                "allow_low_quality_matches": True,
                "_target_": "<class 'detectron2.modeling.matcher.Matcher'>",
            },
            "box2box_transform": {
                "weights": [1.0, 1.0, 1.0, 1.0],
                "_target_": "<class 'detectron2.modeling.box_regression.Box2BoxTransform'>",
            },
            "batch_size_per_image": 128,
            "positive_fraction": 0.5,
            "pre_nms_topk": [2000, 1000],
            "post_nms_topk": [1000, 1000],
            "nms_thresh": 0.7,
            "_target_": "<class 'detectron2.modeling.proposal_generator.rpn.RPN'>",
        },
        "roi_heads": {
            "num_classes": 2,
            "batch_size_per_image": 128,
            "positive_fraction": 0.25,
            "proposal_matcher": {
                "thresholds": [0.5],
                "labels": [0, 1],
                "allow_low_quality_matches": False,
                "_target_": "<class 'detectron2.modeling.matcher.Matcher'>",
            },
            "box_in_features": ["p2", "p3", "p4", "p5"],
            "box_pooler": {
                "output_size": 7,
                "scales": [0.25, 0.125, 0.0625, 0.03125],
                "sampling_ratio": 0,
                "pooler_type": "ROIAlignV2",
                "_target_": "<class 'detectron2.modeling.poolers.ROIPooler'>",
            },
            "box_head": {
                "input_shape": ShapeSpec(channels=256, height=7, width=7, stride=None),
                "conv_dims": [256, 256, 256, 256],
                "fc_dims": [1024],
                "_target_": "<class 'detectron2.modeling.roi_heads.box_head.FastRCNNConvFCHead'>",
                "conv_norm": "<function <lambda> at 0x7f9ae01b0f70>",
            },
            "box_predictor": {
                "input_shape": ShapeSpec(
                    channels=1024, height=None, width=None, stride=None
                ),
                "test_score_thresh": 0.5,
                "box2box_transform": {
                    "weights": [10, 10, 5, 5],
                    "_target_": "<class 'detectron2.modeling.box_regression.Box2BoxTransform'>",
                },
                "num_classes": 2,
                "_target_": "<class 'detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers'>",
            },
            "mask_in_features": ["p2", "p3", "p4", "p5"],
            "mask_pooler": {
                "output_size": 14,
                "scales": [0.25, 0.125, 0.0625, 0.03125],
                "sampling_ratio": 0,
                "pooler_type": "ROIAlignV2",
                "_target_": "<class 'detectron2.modeling.poolers.ROIPooler'>",
            },
            "mask_head": {
                "input_shape": ShapeSpec(
                    channels=256, height=14, width=14, stride=None
                ),
                "num_classes": 2,
                "conv_dims": [256, 256, 256, 256, 256],
                "_target_": "<class 'detectron2.modeling.roi_heads.mask_head.MaskRCNNConvUpsampleHead'>",
                "conv_norm": "<function <lambda> at 0x7f9ae01b0f70>",
            },
            "_target_": "<class 'detectron2.modeling.roi_heads.roi_heads.StandardROIHeads'>",
        },
        "pixel_mean": [103.53, 116.28, 123.675],
        "pixel_std": [57.375, 57.12, 58.395],
        "input_format": "BGR",
        "_target_": "<class 'detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'>",
    },
    "optimizer": {
        "params": {
            "weight_decay_norm": 0.0,
            "_target_": "<function get_default_optimizer_params at 0x7f9ae03fa820>",
        },
        "lr": 0.00025,
        "momentum": 0.9,
        "weight_decay": 4e-05,
        "_target_": "<class 'torch.optim.sgd.SGD'>",
    },
    "train": {
        "output_dir": "/app/data/artifacts/detectron_data",
        "init_checkpoint": "/app/data/model_final_b7fbab.pkl",
        "max_iter": 5,
        "amp": {"enabled": True},
        "ddp": {
            "broadcast_buffers": False,
            "find_unused_parameters": False,
            "fp16_compression": True,
        },
        "checkpointer": {"period": 3, "max_to_keep": 100},
        "eval_period": 2,
        "log_period": 20,
        "device": "cuda:0",
        "cudnn_benchmark": True,
        "save_best_model": True,
        "max_to_keep": 3,
    },
    "model_id": 42047771,
    "test": {"vis_period": 1},
}
