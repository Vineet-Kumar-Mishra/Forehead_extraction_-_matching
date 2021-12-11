import cv2
import os, sys
import time
from tqdm.notebook import tqdm
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt

# model builder ##############################################################################
def build_model(weights, cfg):
    net = cv2.dnn.readNet(weights, cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    return model


## Detection#####################################################################################
def detection(model, image, CONFIDENCE_THRESHOLD = 0.2, NMS_THRESHOLD = 0.2):
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    cls1, scr, boxs = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD) 
    while cv2.waitKey(1)<1:
        for (classid, score, box) in zip(cls1, scr, boxs):
            #color = COLORS[int(classid) % len(COLORS)]
            color = (0,255,255)
            label = "%s : %f" % ('forehead', score)
            print(box)
            cv2.rectangle(image, box, color, 2)
            cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("detections", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Image loader #################################################################################

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def IOU(box_1, box_2):
    pass

if __name__ == "__main__":
    print('Initializing................................................................\n\n\n')
    my_parser = argparse.ArgumentParser(description='Path to the image')
    my_parser.add_argument('Path', metavar='path', type =str, help = 'Path to the image')
    args = my_parser.parse_args()

    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.2
    class_names = ['forehead']


    img_path = args.Path

    print('Starting.....................................................................\n\n\n')
    model = build_model(weights = 'yolov4.weights', cfg = 'yolov4.cfg')
    
    print('Reading files.................................................................\n\n\n')


    image = load_image(img_path)
    clss, scr, box = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    box = [box[0][0], box[0][1],box[0][0]+box[0][2], box[0][1]+box[0][3]]
    image = cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]), color = (0,255,0),thickness=3)
    plt.figure(figsize=(15,15))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.savefig('Prediction.jpg')
    print('Finished doing..................................................................\n\n\n')
