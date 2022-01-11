
def get_pretrained_models():
    return {
        "COCO": [
            {
                "config": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl",
                "model": "R50-C4 (1x)",
                "train_time": 0.584,
                "inference_time": 0.110,
                "box": 36.8,
                "mask": 32.2,
                "model_id": 137259246
            },
            {
                "config": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/model_final_84107b.pkl",
                "model": "R50-DC5 (3x)",
                "train_time": 0.470,
                "inference_time": 0.076,
                "box": 40.0,
                "mask": 35.9,
                "model_id": 137849551
            },
            {
                "config": "../../../configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ/42047764/model_final_bb69de.pkl",
                "model": "R50-FPN (100)",
                "train_time": 0.376,
                "inference_time": 0.069,
                "box": 44.6,
                "mask": 40.3,
                "model_id": 42047764
            },
            {
                "config": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl",
                "model": "R50-FPN (400)",
                "train_time": 0.376,
                "inference_time": 0.069,
                "box": 47.4,
                "mask": 42.5,
                "model_id": 42019571
            },
            {
                "config": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_101_FPN_100ep_LSJ/42025812/model_final_4f7b58.pkl",
                "model": "R101-FPN (100)",
                "train_time": 0.376,
                "inference_time": 0.069,
                "box": 46.4,
                "mask": 41.6,
                "model_id": 42025812
            },
            {
                "config": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ/42073830/model_final_f96b26.pkl",
                "model": "R101-FPN (400)",
                "train_time": 0.376,
                "inference_time": 0.069,
                "box": 48.9,
                "mask": 43.7,
                "model_id": 42073830
            },
            {
                "config": "../../../configs/COCO-InstanceSegmentation/mask_rcnn_regnetx_4gf_dds_fpn_1x.py",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_100ep_LSJ/42047771/model_final_b7fbab.pkl",
                "model": "regnetx_4gf_dds_FPN (100)",
                "train_time": 0.474,
                "inference_time": 0.071,
                "box": 46.0,
                "mask": 41.3,
                "model_id": 42047771
            },
            {
                "config": "mask_rcnn_regnetx_4gf_dds_fpn_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ/42025447/model_final_f1362d.pkl",
                "model": "regnetx_4gf_dds_FPN (400)",
                "train_time": 0.474,
                "inference_time": 0.071,
                "box": 48.6,
                "mask": 43.5,
                "model_id": 42025447
            },
            {
                "config": "mask_rcnn_regnety_4gf_dds_fpn_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_100ep_LSJ/42047784/model_final_6ba57e.pkl",
                "model": "regnety_4gf_dds_FPN (100)",
                "train_time": 0.487,
                "inference_time": 0.073,
                "box": 46.1,
                "mask": 41.6,
                "model_id": 42047784
            },
            {
                "config": "mask_rcnn_regnety_4gf_dds_fpn_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ/42045954/model_final_ef3a80.pkl",
                "model": "regnety_4gf_dds_FPN (400)",
                "train_time": 0.487,
                "inference_time": 0.073,
                "box": 48.2,
                "mask": 43.3,
                "model_id": 42045954
            }

        ],

        "LVIS": [
            {
                "config": "mask_rcnn_R_50_FPN_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/144219072/model_final_571f7c.pkl",
                "model": "R50-FPN",
                "train_time": 0.292,
                "inference_time": 0.107,
                "box": 23.6,
                "mask": 24.4,
                "model_id": 144219072
            },
            {
                "config": "mask_rcnn_R_101_FPN_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x/144219035/model_final_824ab5.pkl",
                "model": "R101-FPN",
                "train_time": 0.371,
                "inference_time": 0.114,
                "box": 25.6,
                "mask": 25.9,
                "model_id": 144219035
            },
            {
                "config": "mask_rcnn_X_101_32x8d_FPN_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x/144219108/model_final_5e3439.pkl",
                "model": "X101-FPN",
                "train_time": 0.712,
                "inference_time": 0.151,
                "box": 26.7,
                "mask": 27.1,
                "model_id": 144219108
            }
        ],

        "Cityscapes": [
            {
                "config": "mask_rcnn_R_50_FPN.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl",
                "model": "R50-FPN",
                "train_time": 0.240,
                "inference_time": 0.078,
                "box": "-",
                "mask": 36.5,
                "model_id": 142423278
            }
        ],

        "Others": [
            {
                "config": "mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5/144998336/model_final_821d0b.pkl",
                "model": "Deformable Conv (3x)",
                "train_time": 0.349,
                "inference_time": 0.047,
                "box": 42.7,
                "mask": 38.5,
                "model_id": 144998336
            },
            {
                "config": "cascade_mask_rcnn_R_50_FPN_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl",
                "model": "Cascade R-CNN (3x)",
                "train_time": 0.328,
                "inference_time": 0.053,
                "box": 44.3,
                "mask": 38.5,
                "model_id": 144998488
            },
            {
                "config": "mask_rcnn_R_50_FPN_3x_gn.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/Misc/mask_rcnn_R_50_FPN_3x_gn/138602888/model_final_dc5d9e.pkl",
                "model": "GN (3x)",
                "train_time": 0.309,
                "inference_time": 0.060,
                "box": 42.6,
                "mask": 38.6,
                "model_id": 138602888
            },
            {
                "config": "mask_rcnn_R_50_FPN_3x_syncbn.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/Misc/mask_rcnn_R_50_FPN_3x_syncbn/169527823/model_final_3b3c51.pkl",
                "model": "SyncBN (3x)",
                "train_time": 0.345,
                "inference_time": 0.053,
                "box": 41.9,
                "mask": 37.8,
                "model_id": 169527823
            },
            {
                "config": "panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x/139797668/model_final_be35db.pkl",
                "model": "Panoptic FPN R101",
                "train_time": "-",
                "inference_time": 0.098,
                "box": 47.4,
                "mask": 41.3,
                "model_id": 139797668
            },
            {
                "config": "cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/model_0039999_e76410.pkl",
                "model": "Mask R-CNN X152",
                "train_time": "-",
                "inference_time": 0.234,
                "box": 50.2,
                "mask": 44.0,
                "model_id": 18131413
            }


        ]

    }


def get_table_columns():
    return [
        {"key": "model", "title": "model", "subtitle": None},
        {"key": "train_time", "title": "train time", "subtitle": "(s/im)"},
        {"key": "inference_time", "title": "inference time", "subtitle": "(s/im)"},
        {"key": "box", "title": "box", "subtitle": "AP"},
        {"key": "mask", "title": "mask", "subtitle": "AP"},
        {"key": "model_id", "title": "model id", "subtitle": None}

    ]
