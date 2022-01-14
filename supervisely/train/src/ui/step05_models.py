import errno
import json
import os
import requests
from pathlib import Path
from omegaconf import DictConfig

from detectron2.config import get_cfg
from detectron2.config import LazyConfig
from detectron2 import model_zoo

import sly_globals as g
import sly_functions as f
import supervisely_lib as sly


def init(data, state):
    state['pretrainedDataset'] = 'COCO'

    data["pretrainedModels"] = f.get_pretrained_models()
    data["modelColumns"] = get_table_columns()

    state["selectedModel"] = {pretrained_dataset: data["pretrainedModels"][pretrained_dataset][0]['model']
                              for pretrained_dataset in data["pretrainedModels"].keys()}

    state["weightsInitialization"] = "pretrained"  # "custom"
    state["collapsed5"] = True
    state["disabled5"] = True
    # state["disabled5"] = False

    state["loadingModel"] = False

    sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progress5", "Download weights", is_size=True,
                                min_report_percent=5).init_data(data)

    state["weightsPath"] = ""

    data["done5"] = False


def get_table_columns():
    return [
        {"key": "model", "title": "model", "subtitle": None},
        {"key": "train_time", "title": "train time", "subtitle": "(s/im)"},
        {"key": "inference_time", "title": "inference time", "subtitle": "(s/im)"},
        {"key": "box", "title": "box", "subtitle": "AP"},
        {"key": "mask", "title": "mask", "subtitle": "AP"},
        {"key": "model_id", "title": "model id", "subtitle": None}

    ]


def remove_not_scalars_dict(d):
    only_scalars_dict = {}
    for k, v in d.items():
        if isinstance(v, dict) or isinstance(v, DictConfig):
            only_scalars_dict[k] = remove_not_scalars_dict(v)
        elif type(v) in [float, int, str, bool, list]:
            only_scalars_dict[k] = v
        else:
            continue

    return only_scalars_dict


def filter_lazy_config(cfg):
    new_cfg = remove_not_scalars_dict(cfg)
    return json.dumps(new_cfg, indent=4)


def get_default_config_for_model(state):
    config_path = f.get_config_path(state)
    cfg = f.get_model_config(config_path, state)

    if config_path.endswith('.py'):
        return filter_lazy_config(cfg)
    else:
        return cfg.dump()


def restart(data, state):
    data["done5"] = False


def download_sly_file(remote_path, local_path, progress):
    if sly.fs.file_exists(local_path) is False:
        file_info = g.api.file.get_info_by_path(g.team_id, remote_path)
        if file_info is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), remote_path)
        progress.set_total(file_info.sizeb)
        g.api.file.download(g.team_id, remote_path, local_path, g.my_app.cache,
                            progress.increment)
        progress.reset_and_update()

        sly.logger.info(f"{remote_path} has been successfully downloaded",
                        extra={"weights": local_path})


def download_custom_config(state):
    progress = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progress5", "Download weights", is_size=True,
                                           min_report_percent=5)

    detectron_remote_dir = os.path.dirname(state["weightsPath"])
    g.model_config_local_path = os.path.join(g.my_app.data_dir, 'custom_local_model_config')

    for file_extension in ['.yaml', '.py']:
        config_remote_dir = os.path.join(detectron_remote_dir, f'model_config{file_extension}')
        if g.api.file.exists(g.team_id, config_remote_dir):
            g.model_config_local_path += file_extension
            download_sly_file(config_remote_dir, g.model_config_local_path, progress)
            break
    else:
        raise FileNotFoundError("Can't find config file for your custom model!")


@g.my_app.callback("dataset_changed")
@sly.timeit
@sly.update_fields
@g.my_app.ignore_errors_and_show_dialog_window()
def dataset_changed(api: sly.Api, task_id, context, state, app_logger, fields_to_update):
    fields_to_update['state.selectedModel'] = f.get_pretrained_models()[state['pretrainedDataset']][0]['model']


def load_advanced_config(state, fields_to_update):
    config_path = f.get_config_path(state)
    if config_path.endswith('.py'):
        fields_to_update['state.advancedConfig.options.mode'] = 'ace/mode/json'
    else:
        fields_to_update['state.advancedConfig.options.mode'] = 'ace/mode/yaml'

    fields_to_update['state.advancedConfig.content'] = get_default_config_for_model(state)
    fields_to_update['data.advancedConfigBackup'] = get_default_config_for_model(state)



@g.my_app.callback("download_weights")
@sly.timeit
@sly.update_fields
# @g.my_app.ignore_errors_and_show_dialog_window()
def download_weights(api: sly.Api, task_id, context, state, app_logger, fields_to_update):
    # "https://download.pytorch.org/models/vgg11-8a719046.pth" to /root/.cache/torch/hub/checkpoints/vgg11-8a719046.pth
    # from train import model_list
    fields_to_update['state.loadingModel'] = False

    progress = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progress5", "Download weights", is_size=True,
                                           min_report_percent=5)
    try:
        if state["weightsInitialization"] == "custom":
            # raise NotImplementedError
            weights_path_remote = state["weightsPath"]
            if not weights_path_remote.endswith(".pth"):
                raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                                 f"Supported: '.pth'")

            g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
            if sly.fs.file_exists(g.local_weights_path):
                os.remove(g.local_weights_path)

            download_sly_file(weights_path_remote, g.local_weights_path, progress)
            download_custom_config(state)

        else:
            # get_pretrained_models()[state['pretrainedDataset']][]
            models_by_dataset = f.get_pretrained_models()[state["pretrainedDataset"]]
            selected_model = next(item for item in models_by_dataset
                                  if item["model"] == state["selectedModel"][state["pretrainedDataset"]])

            weights_url = selected_model.get('weightsUrl')
            if weights_url is not None:
                # default_pytorch_dir = "/root/.cache/torch/hub/checkpoints/"
                g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
                # g.local_weights_path = os.path.join(default_pytorch_dir, sly.fs.get_file_name_with_ext(weights_url))
                if sly.fs.file_exists(g.local_weights_path) is False:
                    response = requests.head(weights_url, allow_redirects=True)
                    sizeb = int(response.headers.get('content-length', 0))
                    progress.set_total(sizeb)
                    os.makedirs(os.path.dirname(g.local_weights_path), exist_ok=True)
                    sly.fs.download(weights_url, g.local_weights_path, g.my_app.cache, progress.increment)
                    progress.reset_and_update()
                sly.logger.info("Pretrained weights has been successfully downloaded",
                                extra={"weights": g.local_weights_path})

            g.model_config_local_path = os.path.join(g.my_app.data_dir, 'local_model_config')

        load_advanced_config(state, fields_to_update)
    except Exception as e:
        progress.reset_and_update()
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
