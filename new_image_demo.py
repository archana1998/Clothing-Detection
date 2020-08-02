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

"""yolo_modanet_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}"""


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


#Faster RCNN / RetinaNet / Mask RCNN




path = '/media/archana/Local/Flipkart GRiD/Amazon Images'
images = load_images_from_folder(path)
detections = []
for i in range (len(images)):
    detections.append(detectron.get_detections(images[i]))
      
for i in range (len(images)):
    if len(detections[i]) != 0 :
            detections[i].sort(reverse=False ,key = lambda x:x[4])
for i in range (len(images)):
    for x1, y1, x2, y2, cls_conf, cls_pred in detections[i]:
                
                
        if(classes[int(cls_pred)]=="short sleeve top" or classes[int(cls_pred)]=="long sleeve top"):
                    #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))           

                

            color = colors[int(cls_pred)]
                
            color = tuple(c*255 for c in color)
            color = (.7*color[2],.7*color[1],.7*color[0])       
                    
            font = cv2.FONT_HERSHEY_SIMPLEX   
            
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)
                
            cv2.rectangle(images[i],(x1,y1) , (x2,y2) , color,3)
            y1 = 0 if y1<0 else y1
            y1_rect = y1-25
            y1_text = y1-5



                    
            #remove y2 addition for flipkart
            new_img=images[i][y1:y2,x1+5:x2]
            cv2.imwrite('Crops/Amazon Images/'+'crop_'+str(i)+'.jpg', new_img)         
            #cv2.imshow('Detections',images[i])
            img_id = path.split('/')[-1].split('.')[0]

                

                
     