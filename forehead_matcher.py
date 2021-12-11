import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model,Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.applications import MobileNetV2
import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import argparse

def load_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("number of GPUs found ",len(physical_devices))
    print('\n Loaded on GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


img_size = 64
img_tuple = (img_size,img_size,3)

def contrastive_loss(y, preds, margin=1):
    y = tf.cast(y, preds.dtype)
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                       keepdims=True)
    # return the euclidean distance between the vectors
    
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def siamese_model():
    input_layer = layers.Input(shape = img_tuple)
    base_model = MobileNetV2(input_shape=img_tuple, weights = 'imagenet', include_top = False)(input_layer)
    output = layers.GlobalAveragePooling2D()(base_model)
    output = layers.Dense(1024, activation = 'relu')(output)
    output = layers.Dropout(0.25)(output)
    output = layers.Dense(128)(output)
    
    feature_extractor = Model(input_layer, output)

    img_1 = layers.Input(shape = img_tuple)
    img_2 = layers.Input(shape = img_tuple)

    feature_1 = feature_extractor(img_1)
    feature_2 = feature_extractor(img_2)
    distance = layers.Lambda(euclidean_distance, name = 'distance_layer')([feature_1, feature_2])
    model = Model(inputs = [img_1, img_2], outputs = distance)
    model.compile(loss = contrastive_loss, optimizer = 'adam')
    model.load_weights("./siamese_model.h5")
    
    return model





def load_image(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(64,64))
    return np.expand_dims((im/255).astype(np.float16), axis = 0)

if __name__ == "__main__":
    print('Initializing.....................................................................\n\n\n')
    my_parser = argparse.ArgumentParser(description='Give the GPU availability, path of first image and path of second image')
    my_parser.add_argument('GPU', metavar= 'gpu', type = int, help = 'Availability of GPU \n 1. GPU available\n 2.GPU unavailable')
    my_parser.add_argument('PATH_1', metavar='path_1', type = str, help = 'path to first Image')
    my_parser.add_argument('PATH_2', metavar='path_2', type = str, help = 'Path to second image')
    args = my_parser.parse_args()

    GPU_setting = args.GPU
    path_1 = args.PATH_1
    path_2 = args.PATH_2

    print('Starting GPU........................................................................... \n\n\n')
    if GPU_setting==1:
            load_gpu()

    print('Loading the model......................................................................\n\n\n')
    my_model = siamese_model()

    img_1 = load_image(path_1)
    img_2 = load_image(path_2)

    print('Making predictions....................................................................... \n\n\n')
    pred = my_model.predict([img_1, img_2])
    print('\n\n\n\n Distance is ', pred[0][0])


