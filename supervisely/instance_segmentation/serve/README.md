<div align="center" markdown>

<img src="https://imgur.com/jIOW3zu.png"/>  

# Serve Detectron2 (Instance Segmentation)

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/detectron2)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Serve Detectron2 (Instance Segmentation) model as Supervisely Application.

Application key points:
- Only Instance Segmentation models available
- Deployed on GPU or CPU
- Can be used with Supervisely Applications or [API](https://github.com/supervisely-ecosystem/gl-metric-learning/blob/main/supervisely/serve/src/demo_api_requests.py)

# How to Run

### 1. Add [Serve Detectron2](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve) to your team
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve" src="https://imgur.com/jKrRF7p.png" width="350px" style='padding-bottom: 10px'/>

### 2. Choose model, deploying device and press the **Run** button
<img src="https://imgur.com/DLDYMbk.png" width="80%" style='padding-top: 10px'>  

### 3. Wait for the model to deploy
<img src="https://imgur.com/KFdwTER.png" width="80%">  


# Common apps

You can use served model in next Supervisely Applications ⬇️ 

<details>
  <summary style='font-size: 20px'>Applications List (click to show)</summary>  

- [Apply NN to images project ](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analyse predictions and perform automatic data pre-labeling.   
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" width="350px"/> 

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployd NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image. 
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" width="350px"/>
</details>


# Acknowledgment

This app is based on the great work `Detectron2` ([github](https://github.com/facebookresearch/detectron2)). ![GitHub Org's stars](https://img.shields.io/github/stars/facebookresearch/detectron2?style=social)
