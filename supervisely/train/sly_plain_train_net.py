#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
import copy
import datetime
import logging
import os
from collections import OrderedDict
import time
from typing import Optional

import cv2
import torch
from detectron2.utils.visualizer import Visualizer

from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch, DefaultPredictor
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage, EventWriter, get_event_storage
from detectron2.data import DatasetMapper
from detectron2.data import DatasetCatalog, MetadataCatalog

import supervisely_lib as sly
import sly_globals as g
import sly_train_results_visualizer
import sly_functions as f


from detectron2.data import detection_utils as utils
import imgaug.augmenters as iaa


logger = logging.getLogger("detectron2")


class SuperviselyMetricPrinter(EventWriter):
    """
    Uses for print metrics to Supervisely application
    """

    def __init__(self, max_iter: Optional[int] = None, window_size: int = 20):
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
            window_size (int): the losses will be median-smoothed by this window size
        """
        self.logger = logging.getLogger(__name__)
        self._max_iter = max_iter
        self._window_size = window_size
        self._last_write = None  # (step, time) of last call to write(). Used to compute ETA

    def _get_eta(self, storage) -> Optional[str]:
        if self._max_iter is None:
            return ""
        iteration = storage.iter
        try:
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration - 1)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            return str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            # estimate eta on our own - more noisy
            eta_string = None
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                        iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())
            return eta_string

    def get_actual_stats(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None
        try:
            iter_time = storage.history("time").global_avg()
        except KeyError:
            iter_time = None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        eta_string = self._get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        return {
            'eta': f"eta: {eta_string}  " if eta_string else "--:--:--",
            'iter': iteration,
            'total_loss': storage.histories()['total_loss'].median(self._window_size),
            'loss_mask': storage.histories()['loss_mask'].median(self._window_size),
            'loss_box_reg': storage.histories()['loss_box_reg'].median(self._window_size),
            'losses': "  ".join(
                [
                    "{}: {:.4g}".format(k, v.median(self._window_size))
                    for k, v in storage.histories().items()
                    if "loss" in k
                ]
            ),
            'time': "time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
            'data_time': "data_time: {:.4f}  ".format(
                data_time) if data_time is not None else "",
            'lr': lr,
            'memory': "max_mem: {:.0f}M".format(
                max_mem_mb) if max_mem_mb is not None else "",
        }

    def write(self):
        actual_stats = self.get_actual_stats()

        self.update_charts(actual_stats)

        g.api.app.set_field(g.task_id, 'state.eta', actual_stats['eta'])  # rewrite
        g.sly_progresses['iter'].set(actual_stats['iter'], force_update=True)

    def update_charts(self, actual_stats):
        g.sly_charts['lr'].append(x=actual_stats['iter'], y=actual_stats['lr'], series_name='LR')
        g.sly_charts['loss'].append(x=actual_stats['iter'], y=actual_stats['total_loss'], series_name='total')
        g.sly_charts['loss'].append(x=actual_stats['iter'], y=actual_stats['loss_mask'], series_name='mask')
        g.sly_charts['loss'].append(x=actual_stats['iter'], y=actual_stats['loss_box_reg'], series_name='box_reg')


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
                torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
                torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, current_iter):
    results = OrderedDict()

    for dataset_name in cfg.DATASETS.TEST:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        # if os.path.isfile(f"{output_folder}/{dataset_name}_coco_format.json"):
        #     os.remove(f"{output_folder}/{dataset_name}_coco_format.json")

        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)

        #
        # evaluator = get_evaluator(
        #     cfg, dataset_name,
        # )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

    if len(results) == 1:
        results = list(results.values())[0]
        segm_res = dict(dict(results).get('segm',
                                          {}))  # contain keys: ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AP-kiwi', 'AP-lemon', 'AP-__bg__']
        # segm_res = res_to_vis.get('segm', {})

        # updating SLY charts
        g.sly_charts['val_ap'].append(x=current_iter, y=round((segm_res.get('AP', 0) / 100), 3),
                                      series_name='AP')  # SLY CODE
        g.sly_charts['val_ap'].append(x=current_iter, y=round((segm_res.get('AP50', 0) / 100), 3),
                                      series_name='AP50')  # SLY CODE
        g.sly_charts['val_ap'].append(x=current_iter, y=round((segm_res.get('AP75', 0) / 100), 3),
                                      series_name='AP75')  # SLY CODE

    return results


def get_visualizer(im, dataset_meta):
    return Visualizer(im[:, :, ::-1],
                      metadata=dataset_meta,
                      scale=1
                      # remove the colors of unsegmented pixels. This option is only available for segmentation models
                      )


def visualize_results(cfg, model):
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save("last_saved_model")  # save to output/last_saved_model.pth

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "last_saved_model.pth")  # path to the model we just trained

    test_ds_name = cfg.DATASETS.TEST[0]
    test_ds = DatasetCatalog.get(test_ds_name)

    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("main_validation")

    d = test_ds[0]
    # d = mapper(d)  # debug_augmentation

    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    gt_vis = get_visualizer(im, test_metadata)
    out_t = gt_vis.draw_dataset_dict(d)
    output_image_truth = out_t.get_image()[:, :, ::-1]

    pred_vis = get_visualizer(im, test_metadata)
    out_p = pred_vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image_pred = out_p.get_image()[:, :, ::-1]

    output_image_truth = cv2.cvtColor(output_image_truth, cv2.COLOR_BGR2RGB)
    output_image_pred = cv2.cvtColor(output_image_pred, cv2.COLOR_BGR2RGB)

    sly_train_results_visualizer.preview_predictions(gt_image=output_image_truth, pred_image=output_image_pred)


def apply_augmentation(augs: iaa.Sequential, img, boxes=None, masks=None):
    res = augs(images=[img], bounding_boxes=boxes, segmentation_maps=masks)
    # return image, boxes, masks
    return res[0][0], res[1], res[2]


def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    augmentations_config = sly.json.load_json_file(g.augs_config_path)
    augmentations = sly.imgaug_utils.build_pipeline(augmentations_config["pipeline"],
                                                    random_order=augmentations_config["random_order"])
    #
    _, res_img, res_ann = sly.imgaug_utils.apply(augmentations, g.seg_project_meta,
                                                 image, dataset_dict["sly_annotations"], segmentation_type='instance')

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = f.get_objects_on_image(res_ann, g.all_classes)

    instances = utils.annotations_to_instances(annos, res_img.shape[:2], mask_format="bitmask")
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    writers.append(SuperviselyMetricPrinter(max_iter))

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop

    if g.augs_config_path is not None:
        data_loader = build_detection_train_loader(cfg,  # AUGMENTATIONS HERE
                                                   mapper=mapper)
    else:
        data_loader = build_detection_train_loader(cfg)

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and iteration % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                do_test(cfg, model, iteration)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if (
                    cfg.TEST.VIS_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.VIS_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                visualize_results(cfg, model)
                comm.synchronize()

            if (iteration - start_iter > 5 and (iteration % 10 == 0 or iteration == max_iter - 1)) \
                    or iteration - start_iter == 0:
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

# def setup(args):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     default_setup(
#         cfg, args
#     )  # if you don't like any of the default setup, write your own setup code
#     return cfg

#
# def main(args):
#     cfg = setup(args)
#
#     model = build_model(cfg)
#     logger.info("Model:\n{}".format(model))
#     if args.eval_only:
#         DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#             cfg.MODEL.WEIGHTS, resume=args.resume
#         )
#         return do_test(cfg, model)
#
#     distributed = comm.get_world_size() > 1
#     if distributed:
#         model = DistributedDataParallel(
#             model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
#         )
#
#     do_train(cfg, model, resume=args.resume)
#     return do_test(cfg, model)

#
# if __name__ == "__main__":
#     args = default_argument_parser().parse_args()
#     print("Command Line Args:", args)
#     launch(
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )
