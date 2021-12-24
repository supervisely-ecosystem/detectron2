import functools

import numpy as np
import pycocotools.mask
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode

from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model

import json
import os
import sys
import random
from functools import partial
from PIL import Image
import shutil

import sly_train_results_visualizer
from supervisely_lib.app.widgets import CompareGallery

import step02_splits
import step04_augs
import step05_models
import supervisely_lib as sly
import sly_globals as g
import step03_classes

from itertools import groupby

import sly_plain_train_net
import sly_functions as f

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
    state["done7"] = False

    state["started"] = False

    data["outputName"] = None
    data["outputUrl"] = None


    state["setTimeIndexLoading"] = False

    gallery_custom = CompareGallery(g.task_id, g.api, f"data.galleryPreview", g.project_meta)
    data[f"galleryPreview"] = gallery_custom.to_json()

    data["previewPredLinks"] = []

    state["currEpochPreview"] = 1
    state["visStep"] = 0

    state["followLastPrediction"] = True


def restart(data, state):
    data["done7"] = False


def init_charts(data, state):
    state["smoothing"] = 0.6

    g.sly_charts = {
        'lr': sly.app.widgets.Chart(g.task_id, g.api, "data.chartLR",
                                    title="LR", series_names=["LR"],
                                    yrange=[0, state["lr"] + state["lr"]],
                                    ydecimals=6, xdecimals=2),
        'loss': sly.app.widgets.Chart(g.task_id, g.api, "data.chartLoss",
                                            title="Train Loss", series_names=["total", "mask", "box_reg"],
                                            smoothing=0.6, ydecimals=6, xdecimals=2),
        'val_ap': sly.app.widgets.Chart(g.task_id, g.api, "data.chartAP",
                                        title="Validation AP", series_names=["AP", "AP50", "AP75"],
                                        yrange=[0, 1],
                                        smoothing=0.6, ydecimals=6, xdecimals=2)
    }

    for current_chart in g.sly_charts.values():
        current_chart.init_data(data)


def init_progress_bars(data):
    g.sly_progresses = {
        'iter': sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progressIter", "Iteration"),
        'other': sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progressOther", "Progress")
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

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.app.widgets.ProgressBar):
        if progress.get_total() is None:
            progress.set_total(monitor.len)
        else:
            progress.set(monitor.bytes_read)
        progress.update()

    progress_other = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progressOther",
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

    g.all_classes["__bg__"] = len(g.all_classes)


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
            ann = ann.filter_labels_by_classes(keep_classes=step03_classes.selected_classes)

            record["annotations"] = f.get_objects_on_image(ann, g.all_classes)
            record["sly_annotations"] = ann

            dataset_dicts.append(record)

    return dataset_dicts


def convert_supervisely_to_segmentation(state):
    project_dir_seg = os.path.join(g.my_app.data_dir, g.project_info.name + "_seg")

    if sly.fs.dir_exists(project_dir_seg) is False:  # for debug, has no effect in production
        sly.fs.mkdir(project_dir_seg, remove_content_if_exists=True)
        global progress_other
        progress_other = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progressOther",
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

    DatasetCatalog.register("main_train", get_train)
    DatasetCatalog.register("main_validation", get_validation)

    MetadataCatalog.get("main_train").thing_classes = list(g.all_classes.keys())
    MetadataCatalog.get("main_validation").thing_classes = list(g.all_classes.keys())


def get_model_config_path(state):
    models_by_dataset = step05_models.get_pretrained_models()[state["pretrainedDataset"]]
    selected_model = next(item for item in models_by_dataset
                          if item["Model"] == state["selectedModel"][state["pretrainedDataset"]])

    if state["pretrainedDataset"] == 'COCO':
        par_folder = 'COCO-InstanceSegmentation'
        return os.path.join(par_folder, selected_model.get('config'))

    elif state["pretrainedDataset"] == 'LVIS':
        par_folder = 'LVISv0.5-InstanceSegmentation'
        return os.path.join(par_folder, selected_model.get('config'))

    elif state["pretrainedDataset"] == 'Cityscapes':
        par_folder = 'Cityscapes'
        return os.path.join(par_folder, selected_model.get('config'))


def configure_trainer(state):
    # static
    cfg = get_cfg()

    cfg.INPUT.MASK_FORMAT = 'bitmask'

    cfg.OUTPUT_DIR = os.path.join(g.artifacts_dir, 'detectron_data')

    models_by_dataset = step05_models.get_pretrained_models()[state["pretrainedDataset"]]
    selected_model = next(item for item in models_by_dataset
                          if item["Model"] == state["selectedModel"][state["pretrainedDataset"]])

    cfg.merge_from_file(model_zoo.get_config_file(get_model_config_path(state)))
    cfg.MODEL.WEIGHTS = selected_model.get('weightsUrl')

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(g.all_classes)
    cfg.DATASETS.TRAIN = ("main_train",)
    cfg.DATASETS.TEST = ("main_validation",)

    # from UI — train
    cfg.DATALOADER.NUM_WORKERS = state['numWorkers']
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = state['lr']
    cfg.SOLVER.MAX_ITER = state['iters']
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = state['batchSize']
    cfg.MODEL.DEVICE = f'cuda:{state["gpusId"]}'
    cfg.SOLVER.CHECKPOINT_PERIOD = state['checkpointPeriod']

    # from UI — validation
    cfg.TEST.EVAL_PERIOD = state['evalInterval']
    cfg.TEST.VIS_PERIOD = state['visStep']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = state["visThreshold"]

    return cfg


@g.my_app.callback("previewByEpoch")
@sly.update_fields
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def preview_by_epoch(api: sly.Api, task_id, context, state, app_logger, fields_to_update):
    if len(g.api.app.get_field(g.task_id, 'data.previewPredLinks')) > 0:
        # fields_to_update['state.followLastPrediction'] = False

        index = int(state['currEpochPreview'] / state["visStep"]) - 1

        gallery_preview = CompareGallery(g.task_id, g.api, f"data.galleryPreview", g.project_meta)
        sly_train_results_visualizer.update_preview_by_index(index, gallery_preview)


@g.my_app.callback("train")
@sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        # convert project to segmentation masks
        project_dir_seg = convert_supervisely_to_segmentation(state)

        # model classes = selected_classes + __bg__
        project_seg = sly.Project(project_dir_seg, sly.OpenMode.READ)
        g.seg_project_meta = project_seg.meta
        classes_json = project_seg.meta.obj_classes.to_json()

        # save model classes info + classes order. Order is used to convert model predictions to correct masks for every class
        sly.json.dump_json_file(classes_json, model_classes_path)

        g.sly_progresses['iter'].set_total(state['iters'])
        g.sly_progresses['iter'].set(value=0, force_update=True)

        # TRAIN HERE
        # --------

        configure_datasets(state, project_dir_seg)
        # configure_datasets(state, g.project_dir)
        cfg = configure_trainer(state)

        if os.path.isdir(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR)
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        model = build_model(cfg)
        sly_plain_train_net.do_train(cfg=cfg, model=model)

        # --------

        g.sly_progresses['iter'].reset_and_update()

        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))
        remote_dir = upload_artifacts_and_log_progress(experiment_name=state["expName"])
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        api.task.set_output_directory(task_id, file_info.id, remote_dir)

        # show result directory in UI
        fields = [
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "state.done7", "payload": True},
            {"field": "state.started", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)
    except Exception as e:
        api.app.set_field(task_id, "state.started", False)
        raise e  # app will handle this error and show modal window

    # stop application
    # g.my_app.show_modal_window("Training is finished, app is still running and you can preview predictions dynamics over time."
    #                           "Please stop app manually once you are finished with it.")
    g.my_app.stop()


@g.my_app.callback("stop")
@sly.timeit
def stop(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "state.done7", "payload": True},
        {"field": "state.started", "payload": False},
    ]
    g.api.app.set_fields(g.task_id, fields)
