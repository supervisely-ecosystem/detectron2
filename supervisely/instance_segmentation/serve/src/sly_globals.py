import json
import os
import pathlib
import sys

import supervisely_lib as sly


root_source_path = str(pathlib.Path(sys.argv[0]).parents[4])
sly.logger.info(f"Root source directory: {root_source_path}")  # /detectron2/
sys.path.append(root_source_path)

models_configs_dir = os.path.join(root_source_path, "configs")
print(f"Models configs directory: {models_configs_dir}")  # /detectron2/configs
sys.path.append(models_configs_dir)

my_app = sly.AppService()
api = my_app.public_api

TASK_ID = my_app.task_id
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])

meta: sly.ProjectMeta = None
model = None
local_weights_path = None
model_config_local_path = None


device = f'cuda:{os.environ["modal.state.device"]}' if os.environ['modal.state.device'].isnumeric() else 'cpu'

weights_type = os.environ['modal.state.modelWeightsOptions']
selected_pretrained_dataset = os.environ['modal.state.pretrainedDataset']
selected_model = json.loads(os.environ['modal.state.selectedModel'])

custom_weights_url = os.environ['modal.state.weightsPath']
