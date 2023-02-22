import datetime
import logging
import time
from typing import Optional

from detectron2.utils.events import EventWriter

from supervisely.app.v1.widgets.compare_gallery import CompareGallery

import sly_globals as g
import os


def preview_predictions(gt_image, pred_image):
    gallery_preview = CompareGallery(g.task_id, g.api, f"data.galleryPreview", g.project_meta)
    append_gallery(gt_image, pred_image)

    # follow_last_prediction = g.api.app.get_field(g.task_id, 'state.followLastPrediction')
    # if follow_last_prediction:
    #     update_preview_by_index(-1, gallery_preview)
    #     update_metrics_table_by_by_index(-1)


def update_preview_by_index(index, gallery_preview):
    previews_links = g.api.app.get_field(g.task_id, 'data.previewPredLinks')
    vis_threshold = g.api.app.get_field(g.task_id, 'state.visThreshold')
    gt_image_link = previews_links[index][0]
    pred_image_link = previews_links[index][1]

    gallery_preview.set_left('ground truth', gt_image_link)
    gallery_preview.set_right(f'predicted [threshold: {vis_threshold}]',
                              pred_image_link)

    gallery_preview.update(options=False)


def save_and_upload_image(temp_image, img_type):
    remote_preview_path = "/temp/{}_preview_segmentations.jpg"
    local_image_path = os.path.join(g.my_app.data_dir, f"{time.time()}_{img_type}.jpg")
    g.sly.image.write(local_image_path, temp_image)
    if g.api.file.exists(g.team_id, remote_preview_path.format(img_type)):
        g.api.file.remove(g.team_id, remote_preview_path.format(img_type))

    # @TODO: add ann in SLY format
    # class_lemon = g.sly.ObjClass('lemon', g.sly.Rectangle)
    # label_lemon = g.sly.Label(g.sly.Rectangle(200, 200, 500, 600), class_lemon)
    #
    # labels_arr = [label_lemon]
    # height, width = temp_image.shape[0], temp_image.shape[1]
    # ann = g.sly.Annotation((height, width), labels_arr)

    file_info = g.api.file.upload(g.team_id, local_image_path, remote_preview_path.format(img_type))
    return file_info


def append_gallery(gt_image, pred_image):
    file_info_gt = save_and_upload_image(gt_image, 'gt')
    file_info_pred = save_and_upload_image(pred_image, 'pred')

    fields = [
        {"field": "data.previewPredLinks",
         "payload": [[file_info_gt.storage_path, file_info_pred.storage_path]], "append": True},
    ]

    g.api.app.set_fields(g.task_id, fields)

    fields = [
        {"field": "state.currEpochPreview",
         "payload": (len(g.api.app.get_field(g.task_id, 'data.previewPredLinks')) - 1) *
                    g.api.app.get_field(g.task_id, 'state.evalInterval')},
    ]

    follow_last_prediction = g.api.app.get_field(g.task_id, 'state.followLastPrediction')
    if follow_last_prediction:
        g.api.app.set_fields(g.task_id, fields)


def update_metrics_table_by_by_index(index):
    print("update_metrics_table_by_by_index(), index =", index)
    current_results = g.metrics_for_each_epoch[index]

    table_to_upload = []
    for class_name, AP in current_results.items():
        try:
            table_to_upload.append({
                'class name': class_name[3:],
                'SEG AP [0.5:0.05:0.95] ⬆': AP
            })
        except:
            continue

    g.api.app.set_field(g.task_id, 'data.metricsTable', table_to_upload)

