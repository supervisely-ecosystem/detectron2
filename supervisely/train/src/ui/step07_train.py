import functools
import logging

import numpy as np
import pycocotools.mask
import yaml

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from detectron2.config import LazyConfig, instantiate
from detectron2.data.transforms import ResizeShortestEdge, Resize

import json
import os
import sys
import random
from functools import partial
from PIL import Image
import shutil
from itertools import groupby

from omegaconf import DictConfig
from yacs.config import CfgNode

import supervisely as sly
from supervisely.app.v1.widgets.compare_gallery import CompareGallery
from supervisely.app.v1.widgets.progress_bar import ProgressBar
from supervisely.app.v1.widgets.chart import Chart

import sly_plain_train_yaml_based
import sly_plain_train_python_based
import sly_train_results_visualizer
import sly_globals as g
import sly_functions as f
import step02_splits
import step03_classes
import step04_augs
import step05_models


_open_lnk_name = "open_app.lnk"
model_classes_path = os.path.join(g.info_dir, "model_classes.json")

train_vis_items_path = os.path.join(g.info_dir, "train_vis_items.json")
val_vis_items_path = os.path.join(g.info_dir, "val_vis_items.json")


def init(data, state):
    state["eta"] = None

    init_charts(data, state)
    init_progress_bars(data)

    state["collapsed7"] = True
    state["disabled7"] = True

    state["started"] = False

    data["outputName"] = None
    data["outputUrl"] = None

    state["setTimeIndexLoading"] = False

    gallery_custom = CompareGallery(g.task_id, g.api, f"data.galleryPreview", g.project_meta)
    data[f"galleryPreview"] = gallery_custom.to_json()

    data["previewPredLinks"] = []

    state["currEpochPreview"] = 0
    state["visStep"] = 0

    state["followLastPrediction"] = True

    data['metricsTable'] = None
    data['metricsForEpochs'] = []

    state['trainOnPause'] = False
    state['finishTrain'] = False

    data["done7"] = False


def restart(data, state):
    data["done7"] = False


def init_charts(data, state):
    state["smoothing"] = 0.6

    g.sly_charts = {
        'lr': Chart(g.task_id, g.api, "data.chartLR",
                                    title="LR", series_names=["LR"],
                                    yrange=[0, state["lr"] + state["lr"]],
                                    ydecimals=6, xdecimals=2),
        'loss': Chart(g.task_id, g.api, "data.chartLoss",
                                      title="Train Loss", series_names=["total", "mask", "box_reg"],
                                      smoothing=0.6, ydecimals=6, xdecimals=2),
        'val_ap': Chart(g.task_id, g.api, "data.chartAP",
                                        title="Validation AP", series_names=["AP", "AP50", "AP75"],
                                        yrange=[0, 1],
                                        smoothing=0.6, ydecimals=6, xdecimals=2)
    }

    for current_chart in g.sly_charts.values():
        current_chart.init_data(data)


def init_progress_bars(data):
    g.sly_progresses = {
        'iter': ProgressBar(g.task_id, g.api, "data.progressIter", "Iteration"),
        'other': ProgressBar(g.task_id, g.api, "data.progressOther", "Progress")
    }

    for current_progress in g.sly_progresses.values():
        current_progress.init_data(data)


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    local_path = os.path.join(local_dir, _open_lnk_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


def upload_artifacts_and_log_progress(experiment_name):
    _save_link_to_ui(g.artifacts_dir, g.my_app.app_url)

    def upload_monitor(monitor, api: sly.Api, task_id, progress: ProgressBar):

        if progress.get_total() is None:
            progress.set_total(monitor.len)
        else:
            progress.set(monitor.bytes_read)
        progress.update()

    progress_other = ProgressBar(g.task_id, g.api, "data.progressOther",
                                                 "Upload directory with training artifacts to Team Files",
                                                 is_size=True, min_report_percent=1)
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress_other)

    remote_dir = f"/detectron2/{g.task_id}_{experiment_name}"
    res_dir = g.api.file.upload_directory(g.team_id, g.artifacts_dir, remote_dir, progress_size_cb=progress_cb)
    progress_other.reset_and_update()
    return res_dir


def get_image_info(image_path):
    im = Image.open(image_path)
    width, height = im.size

    return width, height


def get_items_by_set_path(set_path):
    files_by_datasets = {}
    with open(set_path, 'r') as train_set_file:
        set_list = json.load(train_set_file)

        for row in set_list:
            existing_items = files_by_datasets.get(row['dataset_name'], [])
            existing_items.append(row['item_name'])
            files_by_datasets[row['dataset_name']] = existing_items

    return files_by_datasets


def convertStringToNumber(s):
    return int.from_bytes(s.encode(), 'little')


def get_all_classes(state):
    # project = sly.Project(directory=project_seg_dir_path, mode=sly.OpenMode.READ)
    # project_meta = project.meta

    # for class_index, obj_class in enumerate(project_meta.obj_classes):
    #     g.all_classes[obj_class.name] = class_index
    for class_index, selected_class in enumerate(state['selectedClasses']):
        g.all_classes[selected_class] = class_index

    # g.all_classes["__bg__"] = len(g.all_classes)


def convert_data_to_detectron(project_seg_dir_path, set_path):
    dataset_dicts = []

    project = sly.Project(directory=project_seg_dir_path, mode=sly.OpenMode.READ)
    project_meta = project.meta

    files_by_datasets = get_items_by_set_path(set_path=set_path)

    datasets_list = project.datasets
    for current_dataset in datasets_list:
        current_dataset_name = current_dataset.name

        items_in_dataset = files_by_datasets.get(current_dataset_name, [])

        for current_item in items_in_dataset:
            record = {
                "file_name": current_dataset.get_item_path(current_item),
                "image_id": convertStringToNumber(f"{set_path}_{current_dataset.get_item_path(current_item)}")
            }

            width, height = get_image_info(record["file_name"])
            record["height"] = height
            record["width"] = width

            ann_path = current_dataset.get_ann_path(current_item)
            ann = sly.Annotation.load_json_file(ann_path, project_meta)
            # ann = ann.filter_labels_by_classes(keep_classes=step03_classes.selected_classes)

            record["annotations"] = f.get_objects_on_image(ann, g.all_classes)
            record["sly_annotations"] = ann

            dataset_dicts.append(record)

    return dataset_dicts


def convert_supervisely_to_segmentation(state):
    project_dir_seg = os.path.join(g.my_app.data_dir, g.project_info.name + "_seg")

    if sly.fs.dir_exists(project_dir_seg) is False:  # for debug, has no effect in production
        sly.fs.mkdir(project_dir_seg, remove_content_if_exists=True)
        global progress_other
        progress_other = ProgressBar(g.task_id, g.api, "data.progressOther",
                                                     "Convert SLY annotations to segmentation masks",
                                                     sly.Project(g.project_dir, sly.OpenMode.READ).total_items)
        sly.Project.to_segmentation_task(
            g.project_dir, project_dir_seg,
            target_classes=state['selectedClasses'],
            progress_cb=progress_other.increment,
            segmentation_type='instance'
        )
        progress_other.reset_and_update()

    return project_dir_seg


def configure_datasets(state, project_seg_dir_path):
    get_all_classes(state)

    get_train = functools.partial(convert_data_to_detectron, project_seg_dir_path=project_seg_dir_path,
                                  set_path=step02_splits.train_set_path)
    get_validation = functools.partial(convert_data_to_detectron, project_seg_dir_path=project_seg_dir_path,
                                       set_path=step02_splits.val_set_path)

    if g.need_register_datasets:
        DatasetCatalog.register("main_train", get_train)
        DatasetCatalog.register("main_validation", get_validation)
        g.need_register_datasets = False

    MetadataCatalog.get("main_train").thing_classes = list(g.all_classes.keys())
    MetadataCatalog.get("main_validation").thing_classes = list(g.all_classes.keys())


def remove_all_multi_gpu_elements(cfg_dict):
    for key, value in cfg_dict.items():
        if isinstance(value, dict) or isinstance(value, DictConfig):
            cfg_dict[key] = remove_all_multi_gpu_elements(value)
        elif isinstance(value, str):
            if 'Sync' in value:
                cfg_dict[key] = value.replace('Sync', '')

    return cfg_dict


def set_trainer_parameters_by_state(state):
    # static
    config_path = f.get_config_path(state)
    cfg = f.get_model_config(config_path, state)

    if config_path.endswith('.py') or config_path.endswith('.json'):
        # from UI — train
        cfg.dataloader.train.num_workers = state['numWorkers']
        cfg.dataloader.test.num_workers = state['numWorkers']
        cfg.dataloader.train.total_batch_size = state['batchSize']
        cfg.model.proposal_generator.batch_size_per_image = state['batchSizePerImage']
        cfg.model.roi_heads.batch_size_per_image = state['batchSizePerImage']

        cfg.optimizer.lr = state['lr']
        cfg.train.max_iter = state['iters']

        cfg.train.device = f'cuda:{state["gpusId"]}'
        cfg.train.checkpointer.period = state['checkpointPeriod']

        # from UI — validation
        cfg.train.eval_period = state['evalInterval']
        cfg.model.roi_heads.box_predictor.test_score_thresh = state["visThreshold"]

        # if cuda devices == 1: turn on all multiGPU elements
        cfg = remove_all_multi_gpu_elements(cfg)

    else:
        # from UI — train
        cfg.DATALOADER.NUM_WORKERS = state['numWorkers']
        cfg.SOLVER.IMS_PER_BATCH = state['batchSize']
        cfg.SOLVER.BASE_LR = state['lr']
        cfg.SOLVER.MAX_ITER = state['iters']
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = state['batchSizePerImage']
        cfg.MODEL.DEVICE = f'cuda:{state["gpusId"]}' if state["gpusId"].isnumeric() else 'cpu'
        # cfg.MODEL.DEVICE = f'cuda:{state["gpusId"]}'
        cfg.SOLVER.CHECKPOINT_PERIOD = state['checkpointPeriod']

        # from UI — validation
        cfg.TEST.EVAL_PERIOD = state['evalInterval']
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = state["visThreshold"]

    return cfg


def set_trainer_parameters_by_advanced_config(state):
    config_path = f.get_config_path(state)
    cfg = f.get_model_config(config_path, state)

    config_content = state['advancedConfig']['content']

    if config_path.endswith('.py') or config_path.endswith('.json'):
        config_dict = json.loads(config_content)
        cfg = f.update_config_by_custom(cfg, config_dict)
    else:
        loaded_yaml = yaml.safe_load(config_content)
        yaml_cfg = CfgNode(loaded_yaml)
        cfg.merge_from_other_cfg(cfg_other=yaml_cfg)
    return cfg


def load_supervisely_parameters(cfg, state):
    config_path = f.get_config_path(state)
    if config_path.endswith('.py') or config_path.endswith('.json'):
        cfg['model_id'] = state['modelId']

        cfg.train.output_dir = os.path.join(g.artifacts_dir, 'detectron_data')
        cfg.train.init_checkpoint = g.local_weights_path
        cfg.train['save_best_model'] = state['checkpointSaveBest']
        cfg.train['max_to_keep'] = state['checkpointMaxToKeep']

        cfg.model.roi_heads.num_classes = len(g.all_classes)
        cfg.model.roi_heads.box_predictor.num_classes = len(g.all_classes)
        cfg.model.roi_heads.mask_head.num_classes = len(g.all_classes)

        cfg.dataloader.train.mapper['instance_mask_format'] = 'bitmask'
        cfg.dataloader.train.mapper["use_instance_mask"] = True
        cfg.dataloader.train.mapper["image_format"] = 'BGR'

        cfg.dataloader.test.mapper['instance_mask_format'] = 'bitmask'
        cfg.dataloader.test.mapper["use_instance_mask"] = True
        cfg.dataloader.test.mapper["image_format"] = 'BGR'

        cfg.dataloader.train.dataset.names = "main_train"
        cfg.dataloader.test.dataset.names = "main_validation"

        cfg['test'] = DictConfig({
            'vis_period': state['visStep']
        })
    else:
        cfg.INPUT.MASK_FORMAT = 'bitmask'

        cfg.OUTPUT_DIR = os.path.join(g.artifacts_dir, 'detectron_data')
        cfg.SAVE_BEST_MODEL = state['checkpointSaveBest']
        cfg.MAX_TO_KEEP = state['checkpointMaxToKeep']

        cfg.MODEL.WEIGHTS = g.local_weights_path

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(g.all_classes)
        cfg.DATASETS.TRAIN = ("main_train",)
        cfg.DATASETS.TEST = ("main_validation",)

        cfg.TEST.VIS_PERIOD = state['visStep']


def save_config_locally(cfg, config_path):
    if config_path.endswith('.py') or config_path.endswith('.json'):
        output_path = os.path.join(cfg.train.output_dir, 'model_config.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        clear_config = step05_models.remove_not_scalars_dict(cfg)

        with open(output_path, 'w') as file:
            json.dump(clear_config, fp=file, indent=4)

    else:
        output_path = os.path.join(cfg.OUTPUT_DIR, 'model_config.yaml')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)


def configure_trainer(state):
    config_path = f.get_config_path(state)

    if state['parametersMode'] == 'basic':
        cfg = set_trainer_parameters_by_state(state)
    else:
        cfg = set_trainer_parameters_by_advanced_config(state)

    load_supervisely_parameters(cfg, state)

    return cfg, config_path


def get_resize_transform(cfg):
    try:
        if g.resize_dimensions:
            h, w = g.resize_dimensions.get('h'), g.resize_dimensions.get('w')
            resize_transform = Resize([h, w])
        else:
            if isinstance(cfg, (LazyConfig, DictConfig)):
                test_mapper = cfg.dataloader.test.mapper
                resize_transform: ResizeShortestEdge = instantiate(test_mapper['augmentations'][0])
            elif isinstance(cfg, CfgNode):
                resize_transform = ResizeShortestEdge(
                    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
                )
            else:
                raise Exception(f"Unexpected config type: {type(cfg)}.")
    except Exception as exc:
        sly.logger.warn(f"Can't read resize_transform from config: {exc}."
                        " Using detectron2 defautls: size_min=800, size_max=1333.")
        resize_transform = ResizeShortestEdge([800, 800], 1333)
    print("resize_transform:", type(resize_transform), resize_transform.__dict__)
    return resize_transform


@g.my_app.callback("update_train_cycle")
@sly.update_fields
@sly.timeit
def update_train_cycle(api: sly.Api, task_id, context, state, app_logger, fields_to_update):
    g.training_controllers['pause'] = state['trainOnPause']
    g.training_controllers['stop'] = state['finishTrain']


@g.my_app.callback("previewByEpoch")
@sly.update_fields
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def preview_by_epoch(api: sly.Api, task_id, context, state, app_logger, fields_to_update):
    if len(g.api.app.get_field(g.task_id, 'data.previewPredLinks')) > 0:
        # fields_to_update['state.followLastPrediction'] = False

        index = int(state['currEpochPreview'] / state["evalInterval"])

        print(f'{state["evalInterval"]=}, {index=}, {state["currEpochPreview"]=}, {state["followLastPrediction"]=}')
        print("preview_len =", len(g.api.app.get_field(g.task_id, 'data.previewPredLinks')))

        gallery_preview = CompareGallery(g.task_id, g.api, f"data.galleryPreview", g.project_meta)
        sly_train_results_visualizer.update_preview_by_index(index, gallery_preview)
        sly_train_results_visualizer.update_metrics_table_by_by_index(state['currEpochPreview'])


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        sly.logger.debug(f"{g.need_convert_to_sly=}")
        if g.need_convert_to_sly:
            # convert project to segmentation masks
            project_dir_seg = convert_supervisely_to_segmentation(state)
            project_seg = sly.Project(project_dir_seg, sly.OpenMode.READ)
            g.seg_project_meta = project_seg.meta
            classes_json = project_seg.meta.obj_classes.to_json()
            classes_json = [current_class for current_class in classes_json if current_class['title'] != '__bg__']
            sly.json.dump_json_file(classes_json, model_classes_path)
            g.need_convert_to_sly = False
        else:
            project_dir_seg = os.path.join(g.my_app.data_dir, g.project_info.name + "_seg")

        configure_datasets(state, project_dir_seg)
        cfg, config_path = configure_trainer(state)

        sly.logger.info(f'{config_path=}')

        if config_path.endswith('.py') or config_path.endswith('.json'):
            g.sly_progresses['iter'].set_total(cfg.train.max_iter)
            output_dir = cfg.train.output_dir
        else:
            g.sly_progresses['iter'].set_total(cfg.SOLVER.MAX_ITER)
            output_dir = cfg.OUTPUT_DIR

        g.sly_progresses['iter'].set(value=0, force_update=True)

        if os.path.isdir(output_dir):
            for f in os.listdir(output_dir):
                path = os.path.join(output_dir, f)
                if os.path.isfile(path):
                    os.remove(path)

        save_config_locally(cfg, config_path)
        sly.logger.debug(config_path)
        sly.logger.debug(cfg)

        # TRAIN HERE
        # --------

        if config_path.endswith('.py') or config_path.endswith('.json'):
            sly.logger.debug("training with .py config")
            g.resize_transform = get_resize_transform(cfg)
            sly_plain_train_python_based.do_train(cfg=cfg)
        else:
            sly.logger.debug("training with .yaml config")
            g.resize_transform = get_resize_transform(cfg)
            sly_plain_train_yaml_based.do_train(cfg=cfg)

        # # --------

        g.sly_progresses['iter'].reset_and_update()

        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))
        remote_dir = upload_artifacts_and_log_progress(experiment_name=state["expName"])
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        api.task.set_output_directory(task_id, file_info.id, remote_dir)

        # show result directory in UI
        fields = [
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "data.done7", "payload": True},
            {"field": "state.started", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)
    except Exception as e:
        api.app.set_field(task_id, "state.started", False)
        raise e  # app will handle this error and show modal window

    # stop application
    g.my_app.show_modal_window(
        "Training is finished, app is still running and you can preview predictions dynamics over time."
        "Please stop app manually once you are finished with it.")
    # g.my_app.stop()


@g.my_app.callback("stop")
@sly.timeit
def stop(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "state.done7", "payload": True},
        {"field": "state.started", "payload": False},
    ]
    g.api.app.set_fields(g.task_id, fields)
