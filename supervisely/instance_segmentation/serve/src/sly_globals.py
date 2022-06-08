import json
import os
import pathlib
import sys

import yaml

import supervisely as sly
from supervisely.app.v1.app_service import AppService
import torch.cuda

root_source_path = str(pathlib.Path(sys.argv[0]).parents[4])
sly.logger.info(f"Root source directory: {root_source_path}")  # /detectron2/
sys.path.append(root_source_path)

models_configs_dir = os.path.join(root_source_path, "configs")
print(f"Models configs directory: {models_configs_dir}")  # /detectron2/configs
sys.path.append(models_configs_dir)

my_app = AppService()
api = my_app.public_api

TASK_ID = my_app.task_id
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])

meta: sly.ProjectMeta = None
model = None
local_weights_path = None
model_config_local_path = None

device = os.environ['modal.state.device'] if 'cuda' in os.environ[
    'modal.state.device'] and torch.cuda.is_available() else 'cpu'
print(device)

weights_type = os.environ['modal.state.weightsInitialization']

selected_pretrained_dataset = os.environ['modal.state.pretrainedDataset']

selected_model = os.environ[f'modal.state.selectedModel.{selected_pretrained_dataset}']

# selected_model = json.loads(os.environ['modal.state.selectedModel'])

custom_weights_url = os.environ['modal.state.weightsPath']


settings_path = os.path.join(root_source_path, "supervisely/instance_segmentation/serve/custom_settings.yaml")
sly.logger.info(f"Custom inference settings path: {settings_path}")
with open(settings_path, 'r') as file:
    default_settings_str = file.read()
    default_settings = yaml.safe_load(default_settings_str)


