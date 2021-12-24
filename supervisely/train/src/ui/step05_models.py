import errno
import os
import requests
from pathlib import Path

import sly_globals as g
import supervisely_lib as sly

progress5 = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progress5", "Download weights", is_size=True,
                                        min_report_percent=5)

local_weights_path = None

models_list = []


def init(data, state):
    state['pretrainedDataset'] = 'COCO'

    data["pretrainedModels"] = get_pretrained_models()
    data["modelColumns"] = get_table_columns()

    state["selectedModel"] = {pretrained_dataset: data["pretrainedModels"][pretrained_dataset][0]['Model']
                              for pretrained_dataset in data["pretrainedModels"].keys()}

    state["weightsInitialization"] = "pretrained"  # "custom"
    state["collapsed5"] = True
    state["disabled5"] = True

    progress5.init_data(data)

    state["weightsPath"] = ""
    data["done5"] = False


def get_pretrained_models():
    return {
        "COCO": [
            {
                "config": "mask_rcnn_R_50_C4_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl",
                "Model": "R50-C4(1x)",
                "inference_time": 0.110,
                "box": 36.8,
                "mask": 32.2
            },
            {
                "config": "mask_rcnn_R_50_DC5_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/model_final_4f86c3.pkl",
                "Model": "R50-DC5(1x)",
                "inference_time": 0.076,
                "box": 38.3,
                "mask": 34.2
            },
            {
                "config": "mask_rcnn_R_50_FPN_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl",
                "Model": "R50-FPN(1x)",
                "inference_time": 0.043,
                "box": 38.6,
                "mask": 35.2
            },
            {
                "config": "mask_rcnn_R_50_C4_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x/137849525/model_final_4ce675.pkl",
                "Model": "R50-C4(3x)",
                "inference_time": 0.111,
                "box": 39.8,
                "mask": 34.4
            },
            {
                "config": "mask_rcnn_R_50_DC5_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/model_final_84107b.pkl",
                "Model": "R50-DC5(3x)",
                "inference_time": 0.076,
                "box": 40.0,
                "mask": 35.9
            },
            {
                "config": "mask_rcnn_R_50_FPN_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
                "Model": "R50-FPN(3x)",
                "inference_time": 0.043,
                "box": 41.0,
                "mask": 37.2
            },
            {
                "config": "mask_rcnn_R_101_C4_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl",
                "Model": "R101-C4",
                "inference_time": 0.145,
                "box": 42.6,
                "mask": 36.7
            },
            {
                "config": "mask_rcnn_R_101_DC5_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x/138363294/model_final_0464b7.pkl",
                "Model": "R101-DC5",
                "inference_time": 0.092,
                "box": 41.9,
                "mask": 37.3
            },
            {
                "config": "mask_rcnn_R_101_FPN_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl",
                "Model": "R101-FPN",
                "inference_time": 0.056,
                "box": 42.9,
                "mask": 38.6
            },
            {
                "config": "mask_rcnn_X_101_32x8d_FPN_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl",
                "Model": "X101-FPN",
                "inference_time": 0.103,
                "box": 44.3,
                "mask": 39.5
            }
        ],

        "LVIS": [
            {
                "config": "mask_rcnn_R_50_FPN_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/144219072/model_final_571f7c.pkl",
                "Model": "R50-FPN",
                "inference_time": 0.107,
                "box": 23.6,
                "mask": 24.4
            },
            {
                "config": "mask_rcnn_R_101_FPN_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x/144219035/model_final_824ab5.pkl",
                "Model": "R101-FPN",
                "inference_time": 0.114,
                "box": 25.6,
                "mask": 25.9
            },
            {
                "config": "mask_rcnn_X_101_32x8d_FPN_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x/144219108/model_final_5e3439.pkl",
                "Model": "X101-FPN",
                "inference_time": 0.151,
                "box": 26.7,
                "mask": 27.1
            }
        ],

        "Cityscapes": [
            {
                "config": "mask_rcnn_R_50_FPN.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl",
                "Model": "R50-FPN",
                "inference_time": 0.078,
                "box": "",
                "mask": 36.5
            }
        ]

    }


def get_table_columns():
    return [
        {"key": "Model", "title": "Model", "subtitle": None},
        {"key": "inference_time", "title": "inference_time", "subtitle": "(s/im)"},
        {"key": "box", "title": "box", "subtitle": "AP"},
        {"key": "mask", "title": "mask", "subtitle": "AP"}

    ]


# def get_model_info_by_name(name):
#     models = get_models_list()
#     for info in models:
#         if info["model"] == name:
#             return info
#     raise KeyError(f"Model {name} not found")


def restart(data, state):
    data["done5"] = False


@g.my_app.callback("dataset_changed")
@sly.timeit
@sly.update_fields
@g.my_app.ignore_errors_and_show_dialog_window()
def dataset_changed(api: sly.Api, task_id, context, state, app_logger, fields_to_update):
    fields_to_update['state.selectedModel'] = get_pretrained_models()[state['pretrainedDataset']][0]['Model']


@g.my_app.callback("download_weights")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_weights(api: sly.Api, task_id, context, state, app_logger):
    # "https://download.pytorch.org/models/vgg11-8a719046.pth" to /root/.cache/torch/hub/checkpoints/vgg11-8a719046.pth
    # from train import model_list

    global local_weights_path
    try:
        if state["weightsInitialization"] == "custom":
            # raise NotImplementedError
            weights_path_remote = state["weightsPath"]
            if not weights_path_remote.endswith(".pth"):
                raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                                 f"Supported: '.pth'")
            #
            # # get architecture type from previous UI state
            # prev_state_path_remote = os.path.join(str(Path(weights_path_remote).parents[1]), "info/ui_state.json")
            # prev_state_path = os.path.join(g.my_app.data_dir, "ui_state.json")
            # api.file.download(g.team_id, prev_state_path_remote, prev_state_path)
            # prev_state = sly.json.load_json_file(prev_state_path)
            # api.task.set_field(g.task_id, "state.selectedModel", prev_state["selectedModel"])
            #
            g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
            if sly.fs.file_exists(g.local_weights_path) is False:
                file_info = g.api.file.get_info_by_path(g.team_id, weights_path_remote)
                if file_info is None:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path_remote)
                progress5.set_total(file_info.sizeb)
                g.api.file.download(g.team_id, weights_path_remote, g.local_weights_path, g.my_app.cache, progress5.increment)
                progress5.reset_and_update()
        else:
            # get_pretrained_models()[state['pretrainedDataset']][]

            models_by_dataset = get_pretrained_models()[state["pretrainedDataset"]]
            selected_model = next(item for item in models_by_dataset
                                  if item["Model"] == state["selectedModel"][state["pretrainedDataset"]])

            weights_url = selected_model.get('weightsUrl')
            if weights_url is not None:
                default_pytorch_dir = "/root/.cache/torch/hub/checkpoints/"
                # local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
                g.local_weights_path = os.path.join(default_pytorch_dir, sly.fs.get_file_name_with_ext(weights_url))
                if sly.fs.file_exists(g.local_weights_path) is False:
                    response = requests.head(weights_url, allow_redirects=True)
                    sizeb = int(response.headers.get('content-length', 0))
                    progress5.set_total(sizeb)
                    os.makedirs(os.path.dirname(g.local_weights_path), exist_ok=True)
                    sly.fs.download(weights_url, local_weights_path, g.my_app.cache, progress5.increment)
                    progress5.reset_and_update()
                sly.logger.info("Pretrained weights has been successfully downloaded",
                                extra={"weights": local_weights_path})
    except Exception as e:
        progress5.reset_and_update()
        raise e

    fields = [
        {"field": "data.done5", "payload": True},
        {"field": "state.collapsed6", "payload": False},
        {"field": "state.disabled6", "payload": False},
        {"field": "state.activeStep", "payload": 6},
    ]
    g.api.app.set_fields(g.task_id, fields)


def restart(data, state):
    data["done5"] = False
