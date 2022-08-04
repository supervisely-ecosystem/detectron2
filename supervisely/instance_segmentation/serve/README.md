<div align="center" markdown>

<img src="https://imgur.com/jIOW3zu.png"/>  

# Serve Detectron2 (Instance Segmentation)

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Pretrained-models">Pretrained models</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/detectron2)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve.png)](https://supervise.ly)

</div>

# Overview

Serve Detectron2 (Instance Segmentation) model as Supervisely Application.

Model serving allows to apply model to image (URL, local file, Supervisely image id) with 2 modes (full image, image ROI). Also app sources can be used as example how to use downloaded model weights outside Supervisely.

Application key points:
- Only Instance Segmentation models available
- Deployed on GPU or CPU
- Can be used with Supervisely Applications or [API](https://github.com/supervisely-ecosystem/gl-metric-learning/blob/main/supervisely/serve/src/demo_api_requests.py)



# Pretrained models


Detectron2 provides us **Mask R-CNN Instance Segmentation** baselines based on 3 different backbone combinations:
1. FPN: Use a ResNet+FPN backbone with standard conv and FC heads for mask and box prediction, respectively. It obtains the best speed/accuracy tradeoff, but the other two are still useful for research.
2. C4: Use a ResNet conv4 backbone with conv5 head. The original baseline in the Faster R-CNN paper.
3. DC5 (Dilated-C5): Use a ResNet conv5 backbone with dilations in conv5, and standard conv and FC heads for mask and box prediction, respectively. This is used by the Deformable ConvNet paper.
more about models

We have integrated popular architectures into this application. 

<details open>
  <summary><b> Show integrated models</b> 🔻</summary>
<br>
  
[ℹ️ You can find more information about each model here (use **model id**)](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)

### pretrained on COCO

|  model    |  train time  (s/im)   |  inference time  (s/im)   |  box  AP   |  mask  AP   |  model id    |
| --- | --- | --- | --- | --- | --- |
|   R50-C4 (1x)   |   0.584   |   0.11   |   36.8   |   32.2   |   137259246   |
|   R50-DC5 (3x)   |   0.47   |   0.076   |   40   |   35.9   |   137849551   |
|   R50-FPN (100)   |   0.376   |   0.069   |   44.6   |   40.3   |   42047764   |
|   R50-FPN (400)   |   0.376   |   0.069   |   47.4   |   42.5   |   42019571   |
|   R101-FPN (100)   |   0.376   |   0.069   |   46.4   |   41.6   |   42025812   |
|   R101-FPN (400)   |   0.376   |   0.069   |   48.9   |   43.7   |   42073830   |
|   regnetx_4gf_dds_FPN (100)   |   0.474   |   0.071   |   46   |   41.3   |   42047771   |
|   regnetx_4gf_dds_FPN (400)   |   0.474   |   0.071   |   48.6   |   43.5   |   42025447   |
|   regnety_4gf_dds_FPN (100)   |   0.487   |   0.073   |   46.1   |   41.6   |   42047784   |
|   regnety_4gf_dds_FPN (400)   |   0.487   |   0.073   |   48.2   |   43.3   |   42045954   |

### pretrained on LVIS

|  model    |  train time  (s/im)   |  inference time  (s/im)   |  box  AP   |  mask  AP   |  model id    |
| --- | --- | --- | --- | --- | --- |
|   R50-FPN   |   0.292   |   0.107   |   23.6   |   24.4   |   144219072   |
|   R101-FPN   |   0.371   |   0.114   |   25.6   |   25.9   |   144219035   |
|   X101-FPN   |   0.712   |   0.151   |   26.7   |   27.1   |   144219108   |


### pretrained on Cityscapes

|  model    |  train time  (s/im)   |  inference time  (s/im)   |  box  AP   |  mask  AP   |  model id    |
| --- | --- | --- | --- | --- | --- |
|   R50-FPN   |   0.24   |   0.078   |   -   |   36.5   |   142423278   |


### others

| model   | train time  (s/im)  | inference time  (s/im)  | box  AP  | mask  AP  | model id   |
| --- | --- | --- | --- | --- | --- |
|  Deformable Conv (3x)  |  0.349  |  0.047  |  42.7  |  38.5  |  144998336  |
|  Cascade R-CNN (3x)  |  0.328  |  0.053  |  44.3  |  38.5  |  144998488  |
|  GN (3x)  |  0.309  |  0.06  |  42.6  |  38.6  |  138602888  |
|  Mask R-CNN X152  |  -  |  0.234  |  50.2  |  44  |  18131413  |
  
  </details>


# How to Run

### 1. Add [Serve Detectron2](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve) to your team
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve" src="https://imgur.com/jKrRF7p.png" width="350px" style='padding-bottom: 10px'/>

### 2. Choose model, deploying device and press the **Run** button
<img src="https://imgur.com/DLDYMbk.png" width="80%" style='padding-top: 10px'>  

### 3. Wait for the model to deploy
<img src="https://imgur.com/KFdwTER.png" width="80%">  


# Related Apps

You can use served model in next Supervisely Applications ⬇️ 
  

- [Apply NN to Images Project](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analyse predictions and perform automatic data pre-labeling.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" height="70px" margin-bottom="20px"/>  

- [Apply NN to Videos Project](https://ecosystem.supervise.ly/apps/apply-nn-to-videos-project) - app allows to label your videos using served Supervisely models.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://imgur.com/LDo8K1A.png" height="70px" margin-bottom="20px" />

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployd NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>



# Acknowledgment

This app is based on the great work `Detectron2` ([github](https://github.com/facebookresearch/detectron2)). ![GitHub Org's stars](https://img.shields.io/github/stars/facebookresearch/detectron2?style=social)
