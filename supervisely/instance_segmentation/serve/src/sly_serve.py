import errno
import os, cv2
import pathlib
import sys

import requests
import torch
import supervisely_lib as sly

from detectron2.engine import DefaultPredictor
from supervisely_lib.io.fs import get_file_name_with_ext
from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo
from pathlib import Path

import sly_globals as g
import sly_functions as f


@g.my_app.callback("get_output_classes_and_tags")
@sly.timeit
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_session_info")
@sly.timeit
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "Detectron2 serve",
        "device": str(g.device),
        "session_id": task_id,
        "classes_count": len(g.meta.obj_classes),
        "tags_count": len(g.meta.tag_metas),
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


@g.my_app.callback("get_custom_inference_settings")
@sly.timeit
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"settings": {}})  # send model config here


@sly.timeit
def init_model():
    f.download_model_weights()
    f.initialize_weights()

    f.init_model_meta()
    f.construct_model_meta()


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.TEAM_ID,
        "context.workspaceId": g.WORKSPACE_ID
    })

    init_model()
    sly.logger.info("🟩 Model has been successfully deployed")
    g.my_app.run()


# @TODO: add python configs


# @TODO: move inference methods to SDK

if __name__ == "__main__":
    sly.main_wrapper("main", main)
