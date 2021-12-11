import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model,Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.applications import MobileNetV2
import sys, os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
from forehead_detector import load_image, build_model, detection
from forehead_matcher import load_gpu, contrastive_loss, euclidean_distance, siamese_model



if __name__ == "__main__":
    
    print('Initializing......................................................................\n\n\n')
    my_parser = argparse.ArgumentParser(description='Give GPU availability, path to first image and path to second image')
    my_parser.add_argument('GPU', metavar='gpu', type = int, help = "Availability of GPU \n 1. GPU available\n 2.GPU unavailable")
    my_parser.add_argument('PATH_1', metavar='path_1', type = str, help = 'path to first Image')
    my_parser.add_argument('PATH_2', metavar='path_2', type = str, help = 'Path to second image')

    args = my_parser.parse_args()

    GPU_setting = args.GPU
    path_1 = args.PATH_1
    path_2 = args.PATH_2

    if GPU_setting:
        print('Activating GPU.................................................................\n\n\n')
        load_gpu()
    
    print('Building the model..............................................................\n\n\n')
    model_cropper = build_model('yolov4.weights', cfg = 'yolov4.cfg')
    my_model_siamese = siamese_model()

    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.2
    class_names = ['forehead']

    img_1 = load_image(path_1)
    img_2 = load_image(path_2)

    print('Cropping the forehead..............................................................\n\n\n')
    _, _, box_1 = model_cropper.detect(img_1, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    _, _, box_2 = model_cropper.detect(img_2, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    x11,y11,x21,y21 = box_1[0][0], box_1[0][1], box_1[0][0]+box_1[0][2], box_1[0][1]+box_1[0][3]
    x12,y12,x22,y22 = box_2[0][0], box_2[0][1], box_2[0][0]+box_2[0][2], box_2[0][1]+box_2[0][3]

    cropped_img_1 = img_1[y11:y21, x11:x21, :]
    cropped_img_2 = img_1[y12:y22, x12:x22, :]

    cropped_img_1 = cv2.resize(cropped_img_1,(64,64))
    cropped_img_2 = cv2.resize(cropped_img_2, (64,64))

    cropped_img_1 = np.expand_dims(cropped_img_1, axis = 0)
    cropped_img_2 = np.expand_dims(cropped_img_2, axis = 0)
    print('Finding the Distance.................................................................\n\n\n')
    pred_dist = my_model_siamese.predict([cropped_img_1, cropped_img_2])
    print('\n\n\n\n Distance = ', pred_dist[0][0])
