import os
from typing_extensions import Literal
from typing import List, Any, Dict
from pathlib import Path
import cv2
import json
from dotenv import load_dotenv
import torch
import supervisely as sly
import pretrained_models

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, LazyConfig, instantiate
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.modeling import build_model 

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()
root_source_path = str(Path(__file__).parents[4])
app_source_path = str(Path(__file__).parents[1])
models_configs_dir = os.path.join(root_source_path, "configs")

model_weights_option = os.environ['modal.state.weightsInitialization']
selected_pretrained_dataset = os.environ['modal.state.pretrainedDataset']
selected_model = os.environ[f'modal.state.selectedModel.{selected_pretrained_dataset}']
custom_weights = os.environ['modal.state.weightsPath']
device = os.environ['modal.state.device'] if 'cuda' in os.environ[
    'modal.state.device'] and torch.cuda.is_available() else 'cpu'
models_by_dataset = pretrained_models.get_pretrained_models()[selected_pretrained_dataset]
selected_model_dict = next(item for item in models_by_dataset if item["model"] == selected_model)
checkpoint_name = os.path.basename(selected_model_dict.get("config")).split(".")[0]

def update_config_by_custom(cfg, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and cfg.get(k) is not None:
            cfg[k] = update_config_by_custom(cfg[k], v)
        else:
            cfg[k] = v

    return cfg

class Detectron2Model(sly.nn.inference.InstanceSegmentation):
    def load_on_device(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        config_path = None
        weights_path = None

        # get model and config paths
        if model_weights_option == "custom":
            weights_path = self.location[0]
            config_path = self.location[1]
        elif model_weights_option == "pretrained":
            weights_path = self.location
            config_path = os.path.join(models_configs_dir, selected_model_dict.get('config'))
        
        # load config
        if config_path.endswith('.py'):
            cfg = LazyConfig.load(config_path)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:  # load custom config
                config_dict = json.load(f)
            base_config_path = None
            for models_list_by_dataset in models_by_dataset.values():
                for current_model in models_list_by_dataset:
                    if current_model['model_id'] == config_dict['model_id']:
                        base_config_path = current_model['config']
            base_config_path = os.path.join(models_configs_dir, base_config_path)
            cfg = LazyConfig.load(base_config_path)
            cfg = update_config_by_custom(cfg, config_dict)
        else:
            cfg = get_cfg()
            cfg.set_new_allowed(True)
            cfg.merge_from_file(config_path)
        
        if config_path.endswith('.py') or config_path.endswith('.json'):
            model = instantiate(cfg.model)
        else:
            model = build_model(cfg)

        model.eval()
        DetectionCheckpointer(model).load(weights_path)
        model.to(device)
        self.predictor = model

        DatasetCatalog.register("eval", lambda: None)
        if model_weights_option == "custom":
            detectron_remote_dir = os.path.dirname(custom_weights)
            classes_info_json_url = os.path.join(str(Path(detectron_remote_dir).parents[0]), 'info', 'model_classes.json')
            local_classes_json_path = os.path.join(sly.app.get_data_dir(), sly.fs.get_file_name_with_ext(classes_info_json_url))
            if not sly.fs.file_exists(local_classes_json_path):
                file_info = self.api.file.get_info_by_path(sly.env.team_id(), classes_info_json_url)
                if file_info is None:
                    raise FileNotFoundError("'model_classes.json' file not found.")
                self.api.file.download(sly.env.team_id(), classes_info_json_url, local_classes_json_path)

            with open(local_classes_json_path) as file:
                classes_list = json.load(file)
                v = {k: [dic[k] for dic in classes_list] for k in classes_list[0] if k != 'id'}

                MetadataCatalog.get("eval").thing_classes = v['title']
                MetadataCatalog.get("eval").colors = v['color']

        else:
            if selected_pretrained_dataset == 'COCO':
                MetadataCatalog.get("eval").thing_classes = MetadataCatalog.get("coco_2017_val").thing_classes
            elif selected_pretrained_dataset == 'LVIS':
                MetadataCatalog.get("eval").thing_classes = MetadataCatalog.get("lvis_v1_val").thing_classes
            elif selected_pretrained_dataset == 'Cityscapes':
                MetadataCatalog.get("eval").thing_classes = MetadataCatalog.get(
                    "cityscapes_fine_instance_seg_val").thing_classes
            else:
                raise NotImplementedError

        metadata = MetadataCatalog.get('eval')
        self.class_names = metadata.thing_classes

        if hasattr(metadata, 'thing_colors'):
            colors = metadata.thing_colors
        else:
            colors = []
            for i in range(len(self.class_names)):
                colors.append(sly.color.generate_rgb(exist_colors=colors))

        obj_classes = [sly.ObjClass(name, sly.Bitmap, color) for name, color in zip(self.class_names, colors)]

        self._model_meta = sly.ProjectMeta(
            obj_classes=sly.ObjClassCollection(obj_classes),
            tag_metas=sly.TagMetaCollection([self._get_confidence_tag_meta()])
        )
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def get_info(self):
        info = super().get_info()
        info["model_name"] = selected_model_dict.get("model") 
        info["checkpoint_name"] = checkpoint_name
        info["pretrained_on_dataset"] = selected_pretrained_dataset if model_weights_option == "pretrained" else "custom"
        info["device"] = device
        info["sliding_window_support"] = self.sliding_window_mode
        return info

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionMask]:
        confidence_threshold = settings.get("conf_thres", 0.5)
        image = cv2.imread(image_path)  # BGR
        input_data = [{"image": torch.as_tensor(image.transpose(2, 0, 1).astype("float32")).to(device)}]
        outputs = self.predictor(input_data)  # get predictions from Detectron2 model
        pred_classes = outputs["instances"].pred_classes.detach().numpy()
        pred_class_names = [self.class_names[pred_class] for pred_class in pred_classes]
        pred_scores = outputs["instances"].scores.detach().numpy().tolist()
        pred_masks = outputs["instances"].pred_masks.detach().numpy()

        results = []
        for score, class_name, mask in zip(pred_scores, pred_class_names, pred_masks):
            # filter predictions by confidence
            if score >= confidence_threshold and mask.any():
                results.append(sly.nn.PredictionMask(class_name, mask, score))
        return results

sly.logger.info("Script arguments", extra={
    "teamId": sly.env.team_id(),
    "workspaceId": sly.env.workspace_id(),
    "modal.state.modelWeightsOptions": model_weights_option,
    "modal.state.pretrainedDataset": selected_pretrained_dataset,
    "modal.state.selectedModel": selected_model,
    "modal.state.weightsPath": custom_weights
})

print("Using device:", device)

if model_weights_option == "custom":
    model_dir = os.path.dirname(custom_weights)
    config_path = None
    for file_extension in ['.yaml', '.json', '.py']:
        temp_config_path = os.path.join(model_dir, f'model_config{file_extension}')
        if api.file.exists(sly.env.team_id(), temp_config_path):
            config_path = temp_config_path
            break
    
    if config_path is None:
        raise FileNotFoundError("Config with name 'model_config' ('.yaml', '.json' or '.py') not found in model weights directory.")
    location = [
        custom_weights,
        config_path
    ]
elif model_weights_option == "pretrained":
    location = selected_model_dict.get('weightsUrl')

m = Detectron2Model(
    location=location, 
    custom_inference_settings=os.path.join(app_source_path, "custom_settings.yaml"),
)
m.load_on_device(device)

if sly.is_production():
    # this code block is running on Supervisely platform in production
    # just ignore it during development
    m.serve()
else:
    # for local development and debugging
    # TODO: add image
    image_path = "./demo/image_01.jpg"
    results = m.predict(image_path, settings={})
    vis_path = "./demo/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path)
    print(f"predictions and visualization have been saved: {vis_path}")
