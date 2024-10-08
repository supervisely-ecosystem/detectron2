import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import supervisely as sly
from supervisely.app.v1.app_service import AppService
from supervisely.nn.artifacts.detectron2 import Detectron2


root_source_dir = str(Path(sys.argv[0]).parents[3])
print(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

models_source_dir = os.path.join(root_source_dir, "custom_net")
print(f"Models source directory: {models_source_dir}")
sys.path.append(models_source_dir)

source_path = str(Path(sys.argv[0]).parents[0])
print(f"App source directory: {source_path}")
sys.path.append(source_path)

ui_sources_dir = os.path.join(source_path, "ui")
print(f"UI source directory: {ui_sources_dir}")
sys.path.append(ui_sources_dir)

models_configs_dir = os.path.join(root_source_dir, "configs")
print(f"Models configs directory: {models_configs_dir}")
sys.path.append(source_path)

# only for convenient debug
if not sly.is_production():
    debug_env_path = os.path.join(root_source_dir, "supervisely/train", "debug.env")
    secret_debug_env_path = os.path.join(root_source_dir, "supervisely/train", "secret_debug.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv(secret_debug_env_path)
    load_dotenv(debug_env_path)

my_app = AppService()

os.environ.get("DEBUG_APP_DIR", "")
api = my_app.public_api
task_id = my_app.task_id

# @TODO: for debug
# sly.fs.clean_dir(sly.app.get_data_dir())

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

sly_det2 = Detectron2(team_id)


project_info = api.project.get_info_by_id(project_id)
if project_info is None:  # for debug
    raise ValueError(f"Project with id={project_id} not found")
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

data_dir = my_app.data_dir # /app
artifacts_dir = sly.app.get_synced_data_dir() # /sly-app-data
# artifacts_dir = os.path.join(data_dir, "artifacts")

project_dir = os.path.join(data_dir, "sly_project")
info_dir = os.path.join(artifacts_dir, "info")
sly.fs.mkdir(info_dir)
checkpoints_dir = os.path.join(artifacts_dir, "checkpoints") 
sly.fs.mkdir(checkpoints_dir, remove_content_if_exists=True)  # remove content for debug, has no effect in production
my_app.logger.info(f"Create local paths: {checkpoints_dir}, {info_dir}")
visualizations_dir = os.path.join(artifacts_dir, "visualizations")
sly.fs.mkdir(visualizations_dir, remove_content_if_exists=True)  # remove content for debug, has no effect in production

augs_origin_config_path = os.path.join(info_dir, "augs_config.json")
augs_config_path = None
resize_dimensions = None

local_weights_path = None
model_config_local_path = None

sly_charts = {}
sly_progresses = {}

all_classes = {}

metrics_for_each_epoch = {}

training_controllers = {
    'pause': False,
    'stop': False
}

iterations_to_add = 0


seg_project_meta = None

need_convert_to_sly = True
need_register_datasets = True
resize_transform = None

sly_det2_generated_metadata = None # for project Workflow purposes