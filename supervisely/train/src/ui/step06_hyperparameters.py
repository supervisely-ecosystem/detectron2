import json
import os

import yaml

import step05_models
import supervisely_lib as sly
import sly_globals as g
import sly_functions as f


def load_default_basic_config(state):
    state["expName"] = g.project_info.name
    state["numWorkers"] = 4  # 0 - for debug
    state["batchSize"] = 2
    state["batchSizePerImage"] = 128
    state["lr"] = 0.00025
    state["iters"] = 300
    state["gpusId"] = '0'
    state['evalInterval'] = 10

    state['checkpointPeriod'] = 100
    state['checkpointMaxToKeep'] = 3
    state['checkpointSaveBest'] = True

    state["visThreshold"] = 0.5


def init(data, state):
    state["parametersMode"] = 'basic'

    load_default_basic_config(state)

    state['advancedConfig'] = {
        "content": None,
        "options": {
            "height": '500px',
            "mode": "ace/mode/json",
            "readOnly": False,
            "showGutter": True,
            "highlightActiveLine": True
        }
    }

    data['advancedConfigBackup'] = None

    state["collapsed6"] = True
    state["disabled6"] = True
    data["done6"] = False


def restart(data, state):
    data["done6"] = False


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 (for this specific UNet-based models) and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0


def calc_visualization_step(iters):
    total_visualizations_count = 20

    vis_step = int(iters / total_visualizations_count) \
        if int(iters / total_visualizations_count) > 0 else 1
    g.api.app.set_field(g.task_id, 'state.visStep', vis_step)

    return vis_step


def get_iters_num(state):
    if state['parametersMode'] == 'basic':
        return state['iters']
    else:
        config_path = f.get_config_path(state)
        if config_path.endswith('.py') or config_path.endswith('.json'):
            return json.loads(state['advancedConfig']['content'])['train']['max_iter']
        else:
            return yaml.safe_load(state['advancedConfig']['content'])['SOLVER']['MAX_ITER']


@g.my_app.callback("reset_configuration")
@sly.timeit
@sly.update_fields
def reset_configuration(api: sly.Api, task_id, context, state, app_logger, fields_to_update):
    if state['parametersMode'] == 'basic':
        updated_state = {}
        load_default_basic_config(state=updated_state)

        for key, value in updated_state.items():
            fields_to_update[f'state.{key}'] = value
    else:
        advanced_config_backup = g.api.task.get_field(g.task_id, 'data.advancedConfigBackup')  # loading advanced config
        fields_to_update['state.advancedConfig.content'] = advanced_config_backup



@g.my_app.callback("use_hyp")
@sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    # input_height = state["imgSize"]["height"]
    # input_width = state["imgSize"]["width"]
    #
    # if not check_crop_size(input_height, input_width):
    #     raise ValueError('Input image sizes should be divisible by 32, but train '
    #                      'sizes (H x W : {train_crop_height} x {train_crop_width}) '
    #                      'are not.'.format(train_crop_height=input_height, train_crop_width=input_width))
    #
    # if not check_crop_size(input_height, input_width):
    #     raise ValueError('Input image sizes should be divisible by 32, but validation '
    #                      'sizes (H x W : {val_crop_height} x {val_crop_width}) '
    #                      'are not.'.format(val_crop_height=input_height, val_crop_width=input_width))
    vis_step = calc_visualization_step(get_iters_num(state))

    fields = [
        {"field": "state.visStep", "payload": vis_step},
        {"field": "data.done6", "payload": True},
        {"field": "state.collapsed7", "payload": False},
        {"field": "state.disabled7", "payload": False},
        {"field": "state.activeStep", "payload": 7},
    ]
    g.api.app.set_fields(g.task_id, fields)


def restart(data, state):
    data["done6"] = False
