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


#YOLO PARAMS
yolo_df2_params = {   "model_def" : "/media/archana/Local/Flipkart GRiD/weights/yolov3-df2.cfg",
"weights_path" : "/media/archana/Local/Flipkart GRiD/weights/yolov3-df2_15000.weights",
"class_path":"/media/archana/Local/Flipkart GRiD/weights/df2.names",
"conf_thres" : 0.5,
"nms_thres" :0.6,
"img_size" : 416,
"device" : device}



#DATASET
dataset = 'df2'


if dataset == 'df2': #deepfashion2
    yolo_params = yolo_df2_params




#Classes
classes = load_classes(yolo_params["class_path"])

#Colors
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])


model = 'yolo'

if model == 'yolo':
    detectron = YOLOv3Predictor(params=yolo_params)


path = '/media/archana/Local/Flipkart GRiD/Amazon Images'
images, filenames = load_images_from_folder(path)
detections = []
count = 0
for i in range (len(images)):
    detections.append(detectron.get_detections(images[i]))
    
    for x1, y1, x2, y2, cls_conf, cls_pred in detections[i]:
                
                
        if(classes[int(cls_pred)]=="short sleeve top" or classes[int(cls_pred)]=="long sleeve top"):
          
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            y1 = 0 if y1<0 else y1
           
            new_img=images[i][y1:y2,x1-5:x2+20]
            if(new_img.any()):    
                cv2.imwrite('Crops/Amazon Images/'+'crop_'+filenames[i]+'.jpg', new_img)         
                img_id = path.split('/')[-1].split('.')[0]
            

print('Images Successfully Cropped and Saved')                

                
     