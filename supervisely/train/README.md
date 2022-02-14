
<div align="center" markdown>

<img src="https://imgur.com/T7nLLdH.png"/>  

# Train Detectron2 (Instance Segmentation)

<p align="center">
  <a href="#Overview">Overview</a> •
   <a href="#Pretrained-models">Pretrained models</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Demo">Demo</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/detectron2/supervisely/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/detectron2)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/detectron2/supervisely/train&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/detectron2/supervisely/train&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/detectron2/supervisely/train&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Train Detectron2 (Instance Segmentation) models in Supervisely.




Application key points:
- Only Instance Segmentation models available
- Define Train / Validation splits
- Select classes for training
- Define augmentations
- Use pretrained Detectron2 models
- Tune hyperparameters
- Monitor Metrics charts
- Preview model predictions in real time
- Save trained models to Team Files

# Pretrained models


Detectron2 provides us **Mask R-CNN Instance Segmentation** baselines based on 3 different backbone combinations:
1. FPN: Use a ResNet+FPN backbone with standard conv and FC heads for mask and box prediction, respectively. It obtains the best speed/accuracy tradeoff, but the other two are still useful for research.
2. C4: Use a ResNet conv4 backbone with conv5 head. The original baseline in the Faster R-CNN paper.
3. DC5 (Dilated-C5): Use a ResNet conv5 backbone with dilations in conv5, and standard conv and FC heads for mask and box prediction, respectively. This is used by the Deformable ConvNet paper.
more about models

We have integrated popular architectures into this application. 

<details>
  <summary><b> Show available models</b> 🔻</summary>
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

### 1. Add [Train Detectron2](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/detectron2/supervisely/train) to your team
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/detectron2/supervisely/train" src="https://imgur.com/hqpA5m1.png" width="350px" style='padding-bottom: 10px'/>

### 2. Run application from [labeled using [Bitmaps, Polygons] project](https://ecosystem.supervise.ly/projects/lemons-annotated)
<img src="https://imgur.com/W6pWc5L.png" width="80%" style='padding-top: 10px'>  


# Demo

<a data-key="sly-embeded-video-link" href="https://youtu.be/Uzpp7_xhbPQ" data-video-code="Uzpp7_xhbPQ">
    <img src="https://imgur.com/2j4lNti.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:80%;">
</a>

# Screenshot

<img src="https://imgur.com/3tGR1DX.png" width="100%" style='padding-top: 10px'>

# Acknowledgment

This app is based on the great work `Detectron2` ([github](https://github.com/facebookresearch/detectron2)). ![GitHub Org's stars](https://img.shields.io/github/stars/facebookresearch/detectron2?style=social)


