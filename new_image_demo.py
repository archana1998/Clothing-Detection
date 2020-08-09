import torch
import os
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
import glob
from tqdm import tqdm
import sys
from helpers.ImageLoader import load_images_from_folder



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

user_input = input("Please enter the name of the folder of images to crop: ")

#YOLO PARAMS
yolo_df2_params = {   "model_def" : "yolov3-df2.cfg",
"weights_path" : "yolov3-df2_15000.weights",
"class_path":"df2.names",
"conf_thres" : 0.5,
"nms_thres" :0.6,
"img_size" : 416,
"device" : device}



#DATASET
dataset = 'df2'

yolo_params = yolo_df2_params




#Classes
classes = load_classes(yolo_params["class_path"])

detectron = YOLOv3Predictor(params=yolo_params)


path = user_input
images, filenames = load_images_from_folder(path)
detections = []

for i in range (len(images)):
    detections.append(detectron.get_detections(images[i]))
    
    for x1, y1, x2, y2, cls_conf, cls_pred in detections[i]:
                
                
        if(classes[int(cls_pred)]=="short sleeve top" or classes[int(cls_pred)]=="long sleeve top"):
          
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            y1 = 0 if y1<0 else y1
           
            new_img=images[i][y1:y2,x1-5:x2+20]
            if(new_img.any()):    
                cv2.imwrite('Crops/'+'crop_'+filenames[i]+'.jpg', new_img)         
                img_id = path.split('/')[-1].split('.')[0]
            

print('Images Successfully Cropped and Saved')                

                
     