# Clothing detection using YOLOv3 and DeepFashion2 datasets.

## Datasets

- DeepFashion2 dataset: https://github.com/switchablenorms/DeepFashion2 


## Models

- YOLOv3 trained with Darknet framework: https://github.com/AlexeyAB/darknet

- To do inference use a pytorch implementation of YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3.

- All the models trained with Resnet50 backbone, except YOLOv3 with Darknet53

## Weights

All weights and config files are in https://drive.google.com/drive/folders/1jXZZc5pp2OJCtmQYelzDgPzyuraAdxXP?usp=sharing

## Using
- Clone this repository into your computer
- Download weights and config files from the google drive link above and place it in the same folder as the cloned repository
- Run <code>new_image_demo.py</code>  to crop bounding boxes of images

All thanks to https://github.com/simaiden/Clothing-Detection, where we got the original trained model from!
