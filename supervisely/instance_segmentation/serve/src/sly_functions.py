import errno
import json
import os
from pathlib import Path

import functools

import requests
import torch
import cv2
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.catalog import DatasetCatalog, Metadata, MetadataCatalog

import supervisely as sly
from supervisely.app.v1.widgets.progress_bar import ProgressBar

import sly_globals as g
import pretrained_models

from detectron2 import model_zoo  # config loaders
from detectron2.config import get_cfg
from detectron2.config import LazyConfig

from detectron2.modeling import build_model  # model builders
from detectron2.config import instantiate

import sly_apply_nn_to_video as nn_to_video


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            sly.logger.error(f"Error while processing data: {e}")
            request_id = kwargs["context"]["request_id"]
            # raise e
            try:
                g.my_app.send_response(request_id, data={"error": repr(e)})
                print(traceback.format_exc())
            except Exception as ex:
                sly.logger.exception(f"Cannot send error response: {ex}")
        return value

    return wrapper


@sly.process_image_roi
def inference_image_path(image_path, project_meta, context, state, app_logger):
    settings = state.get("settings", {})
    for key, value in g.default_settings.items():
        if key not in settings:
            app_logger.warn("Field {!r} not found in inference settings. Use default value {!r}".format(key, value))

    conf_thres = settings.get("conf_thres", g.default_settings["conf_thres"])

    app_logger.debug("Input path", extra={"path": image_path})

    im = cv2.imread(image_path)
    height, width = im.shape[:2]
    d = {"image": torch.as_tensor(im.transpose(2, 0, 1).astype("float32")).to(g.device)}

    outputs = g.model([d])
    res = outputs[0]["instances"].to("cpu")

    masks = res.pred_masks
    scores = res.scores
    classes = res.pred_classes

    labels = []

    classes_str = MetadataCatalog.get('eval').thing_classes
    for mask, score, curr_class_idx in zip(masks, scores, classes):
        if score < conf_thres:
            continue

        mask = mask.detach().cpu().numpy()
        if True in mask:
            bitmap = sly.Bitmap(mask)

            curr_class_name = classes_str[curr_class_idx]
            obj_class = g.meta.get_obj_class(curr_class_name)
            tag = sly.Tag(g.meta.get_tag_meta('confidence'), round(float(score), 4))
            label = sly.Label(bitmap, obj_class, sly.TagCollection([tag]))

            labels.append(label)

    ann = sly.Annotation(img_size=(height, width), labels=labels, )

    ann_json = ann.to_json()

    return ann_json


@g.my_app.callback("inference_image_url")
@sly.timeit
@send_error_data
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})

    image_url = state["image_url"]
    ext = sly.fs.get_file_ext(image_url)
    if ext == "":
        ext = ".jpg"
    local_image_path = os.path.join(g.my_app.data_dir, sly.rand_str(15) + ext)

    sly.fs.download(image_url, local_image_path)
    ann_json = inference_image_path(image_path=local_image_path, project_meta=g.meta,
                                    context=context, state=state, app_logger=app_logger)
    sly.fs.silent_remove(local_image_path)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=ann_json)


@g.my_app.callback("inference_image_id")
@sly.timeit
@send_error_data
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(g.my_app.data_dir, sly.rand_str(10) + image_info.name)
    api.image.download_path(image_id, image_path)
    ann_json = inference_image_path(image_path=image_path, project_meta=g.meta,
                                    context=context, state=state, app_logger=app_logger)
    sly.fs.silent_remove(image_path)
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=ann_json)


@g.my_app.callback("inference_batch_ids")
@sly.timeit
@send_error_data
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)
    paths = []
    for info in infos:
        paths.append(os.path.join(g.my_app.data_dir, sly.rand_str(10) + info.name))
    api.image.download_paths(infos[0].dataset_id, ids, paths)

    annotations = inference_images_dir(img_paths=paths,
                                       context=context,
                                       state=state,
                                       app_logger=app_logger)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=annotations)


@g.my_app.callback("inference_video_id")
@sly.timeit
@send_error_data
def inference_video_id(api: sly.Api, task_id, context, state, app_logger):
    video_info = g.api.video.get_info_by_id(state['videoId'])

    sly.logger.info(f'start inference video {state["videoId"]}')
    inf_video_interface = nn_to_video.InferenceVideoInterface(api=g.api,
                                                              start_frame_index=state.get('startFrameIndex', 0),
                                                              frames_count=state.get('framesCount',
                                                                                     video_info.frames_count - 1),
                                                              frames_direction=state.get('framesDirection', 'forward'),
                                                              video_info=video_info,
                                                              imgs_dir=os.path.join(g.my_app.data_dir, 'videoInference'))

    inf_video_interface.download_frames()

    annotations = inference_images_dir(img_paths=inf_video_interface.images_paths,
                                       context=context,
                                       state=state,
                                       app_logger=app_logger)

    g.my_app.send_response(context["request_id"], data={'ann': annotations})
    sly.logger.info(f'inference {video_info.id=} done, {len(annotations)} annotations created')


def inference_images_dir(img_paths, context, state, app_logger):
    annotations = []
    for image_path in img_paths:
        ann_json = inference_image_path(image_path=image_path,
                                        project_meta=g.meta,
                                        context=context,
                                        state=state,
                                        app_logger=app_logger)
        annotations.append(ann_json)
        sly.fs.silent_remove(image_path)
    return annotations


def construct_model_meta():
    current_meta = MetadataCatalog.get('eval')
    names = current_meta.thing_classes

    if hasattr(current_meta, 'thing_colors'):
        colors = current_meta.thing_colors
    else:
        colors = []
        for i in range(len(names)):
            colors.append(sly.color.generate_rgb(exist_colors=colors))

    obj_classes = [sly.ObjClass(name, sly.Bitmap, color) for name, color in zip(names, colors)]
    tags = [sly.TagMeta('confidence', sly.TagValueType.ANY_NUMBER)]

    g.meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes),
                             tag_metas=sly.TagMetaCollection(tags))


def download_sly_file(remote_path, local_path, progress):
    if sly.fs.file_exists(local_path) is False:
        file_info = g.api.file.get_info_by_path(g.TEAM_ID, remote_path)
        if file_info is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), remote_path)
        progress.set_total(file_info.sizeb)
        g.api.file.download(g.TEAM_ID, remote_path, local_path, g.my_app.cache,
                            progress.increment)
        progress.reset_and_update()

        sly.logger.info(f"{remote_path} has been successfully downloaded",
                        extra={"weights": local_path})


def download_model_weights():
    progress = ProgressBar(g.TASK_ID, g.api, "data.progress5", "Download weights", is_size=True,
                           min_report_percent=5)

    if g.weights_type == "custom":  # download from SLY FS
        if not g.custom_weights_url.endswith(".pth"):
            raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(g.custom_weights_url)}."
                             f"Supported: '.pth'")

        g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(g.custom_weights_url))
        download_sly_file(g.custom_weights_url, g.local_weights_path, progress)

        g.model_config_local_path = os.path.join(g.my_app.data_dir, 'model_config')

    else:  # download from Internet
        models_by_dataset = pretrained_models.get_pretrained_models()[g.selected_pretrained_dataset]
        selected_model = next(item for item in models_by_dataset
                              if item["model"] == g.selected_model)

        weights_url = selected_model.get('weightsUrl')
        if weights_url is not None:
            g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
            if sly.fs.file_exists(g.local_weights_path) is False:
                response = requests.head(weights_url, allow_redirects=True)
                sizeb = int(response.headers.get('content-length', 0))
                progress.set_total(sizeb)
                os.makedirs(os.path.dirname(g.local_weights_path), exist_ok=True)
                sly.fs.download(weights_url, g.local_weights_path, g.my_app.cache, progress.increment)
                progress.reset_and_update()

            g.model_config_local_path = selected_model.get('config')
            sly.logger.info("Pretrained weights has been successfully downloaded",
                            extra={"weights": g.local_weights_path})


def update_config_by_custom(cfg, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and cfg.get(k) is not None:
            cfg[k] = update_config_by_custom(cfg[k], v)
        else:
            cfg[k] = v

    return cfg


def get_model_path_by_id(model_id):
    models_by_datasets = pretrained_models.get_pretrained_models()

    for models_list_by_dataset in models_by_datasets.values():
        for current_model in models_list_by_dataset:
            if current_model['model_id'] == model_id:
                return current_model['config']


def get_model_config(custom_config_path):
    if custom_config_path.endswith('.py'):
        custom_config_path = os.path.join(g.models_configs_dir, custom_config_path)
        cfg = LazyConfig.load(custom_config_path)

    elif custom_config_path.endswith('.json'):
        with open(custom_config_path, 'r') as f:  # load custom config
            config_dict = json.load(f)

        base_config_path = os.path.join(g.models_configs_dir, get_model_path_by_id(config_dict['model_id']))
        cfg = LazyConfig.load(base_config_path)

        cfg = update_config_by_custom(cfg, config_dict)
    else:
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        if g.weights_type == 'custom':
            cfg.merge_from_file(custom_config_path)
        else:
            custom_config_path = os.path.join(g.models_configs_dir, custom_config_path)
            cfg.merge_from_file(custom_config_path)
    return cfg


def initialize_model(cfg, config_path):
    if config_path.endswith('.py') or config_path.endswith('.json'):
        model = instantiate(cfg.model)
    else:
        model = build_model(cfg)

    model.eval()
    return model


def download_custom_config():
    progress = ProgressBar(g.TASK_ID, g.api, "data.progress5", "Download weights", is_size=True,
                           min_report_percent=5)

    detectron_remote_dir = os.path.dirname(g.custom_weights_url)

    for file_extension in ['.yaml', '.json']:
        config_remote_dir = os.path.join(detectron_remote_dir, f'model_config{file_extension}')
        if g.api.file.exists(g.TEAM_ID, config_remote_dir):
            g.model_config_local_path += file_extension
            download_sly_file(config_remote_dir, g.model_config_local_path, progress)
            break


def initialize_weights():
    if g.weights_type == 'custom':
        download_custom_config()

    cfg = get_model_config(g.model_config_local_path)
    model = initialize_model(cfg, g.model_config_local_path)
    DetectionCheckpointer(model).load(g.local_weights_path)
    model.to(g.device)

    g.model = model


def hex_color_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return list((int(h[i:i + 2], 16) for i in (0, 2, 4)))


def init_model_meta():
    DatasetCatalog.register("eval", lambda: None)
    if g.weights_type == "custom":
        detectron_remote_dir = os.path.dirname(g.custom_weights_url)
        classes_info_json_url = os.path.join(str(Path(detectron_remote_dir).parents[0]), 'info', 'model_classes.json')
        local_classes_json_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(classes_info_json_url))
        if sly.fs.file_exists(local_classes_json_path) is False:
            file_info = g.api.file.get_info_by_path(g.TEAM_ID, classes_info_json_url)
            if file_info is None:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), g.custom_weights_url)
            g.api.file.download(g.TEAM_ID, classes_info_json_url, local_classes_json_path, g.my_app.cache)

        with open(local_classes_json_path) as file:
            classes_list = json.load(file)
            v = {k: [dic[k] for dic in classes_list] for k in classes_list[0] if k != 'id'}

            MetadataCatalog.get("eval").thing_classes = v['title']
            MetadataCatalog.get("eval").colors = v['color']

    else:
        if g.selected_pretrained_dataset == 'COCO':
            MetadataCatalog.get("eval").thing_classes = MetadataCatalog.get("coco_2017_val").thing_classes
        elif g.selected_pretrained_dataset == 'LVIS':
            MetadataCatalog.get("eval").thing_classes = MetadataCatalog.get("lvis_v1_val").thing_classes
        elif g.selected_pretrained_dataset == 'Cityscapes':
            MetadataCatalog.get("eval").thing_classes = MetadataCatalog.get(
                "cityscapes_fine_instance_seg_val").thing_classes
        else:
            raise NotImplementedError
