#!/usr/bin/env python
# coding: utf-8
import numpy as np
np.random.seed(1337)
import tensorflow
tensorflow.random.set_seed(1337)
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import clone_model
from models import get_model, resnet_v1, resnet_v2
from util import select_clean_uncertain, combine_result, inject_noise
import time
import argparse
from util import other_class
from tensorflow.python.lib.io import file_io
from keras.utils import np_utils, multi_gpu_model
from keras import backend as K
from loss_acc_plot import loss_acc_plot
from sklearn.metrics import accuracy_score
from datetime import datetime
# from io import BytesIO

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = "" if no GPU
os.environ['PYTHONHASHSEED'] = '0'

NUM_CLASSES = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100, 'celeb': 20}
dataset = "celeb"
noise_ratio = 0
data_ratio = 100
repeat = 1


def eval_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    #print(predictions)
    #predictions = list(np.around(np.array(predictions),0))
    predictions_ = np.argmax(predictions, axis = 1)
    predictions_test = np.argmax(test_labels, axis = 1)

    accuracy = accuracy_score(predictions_test, predictions_)
    #print(accuracy)
    return accuracy

def get_data(noise_ratio):
    X_train = np.load('data/image_train_20.npy')
    X_test = np.load('data/image_test_20.npy')
    y_train = np.load('data/label_train_20.npy')
    y_test = np.load('data/label_test_20.npy')

    image_shape = 128
    X_train = X_train.reshape(-1, image_shape, image_shape, 3)
    X_test = X_test.reshape(-1, image_shape, image_shape, 3)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    means = X_train.mean(axis=0)
    X_train = (X_train - means)  # / std
    X_test = (X_test - means)  # / std

    # they are 2D originally in cifar
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    if noise_ratio > 0:
            n_samples = y_train.shape[0]
            n_noisy = int(noise_ratio*n_samples/100)
            np.random.seed()
            noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
            for i in noisy_idx:
                y_train[i] = other_class(n_classes=NUM_CLASSES[dataset], current_class=y_train[i])
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES[dataset])
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES[dataset])
    return X_train, X_test, y_train, y_test
# choix:
# 0: Fixed initialization of model, random noise
# 1: Random initialization of model, fixed noise
# 2: Random initialization of model, random noise
choix = 0
if choix == 0:
    for re in range(repeat):
        result_list = []
        for noise_ratio in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            X_train, X_test, y_train, y_test = get_data(noise_ratio)
            image_shape = X_train.shape[1:]
            model_quality = get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=NUM_CLASSES[dataset])

            optimizer = SGD(lr=0.01, decay=1e-4, momentum=0.9)
            model_quality.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
            epochs_init = 100
            batch_size = 128

            h_quality  =  model_quality.fit(X_train, y_train,
                                steps_per_epoch=X_train.shape[0]//batch_size, epochs=epochs_init, shuffle=False,
                                validation_data=(X_test, y_test)
                                )
            result_list.append(h_quality.history['val_accuracy'][-1])
        print("result_list: ",result_list)
        final_result_list.append(result_list)

elif choix == 1:
    # fixed noise
    for noise_ratio in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        X_train, X_test, y_train, y_test = get_data(noise_ratio)
        for re in range(3):
            result_list = []
            image_shape = X_train.shape[1:]
            model_quality = get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=NUM_CLASSES[dataset])
            optimizer = SGD(lr=0.01, decay=1e-4, momentum=0.9)
            model_quality.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
            epochs_init = 100
            batch_size = 128
            h_quality  =  model_quality.fit(X_train, y_train,
                                steps_per_epoch=X_train.shape[0]//batch_size, epochs=epochs_init, shuffle=False,
                                validation_data=(X_test, y_test)
                                )
            result_list.append(h_quality.history['val_accuracy'][-1])
        print("result_list: ",result_list)
        final_result_list.append(result_list)

elif choix == 2:
    tensorflow.random.set_seed(np.random.seed())
    for re in range(repeat):
        result_list = []
        for noise_ratio in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            X_train, X_test, y_train, y_test = get_data(noise_ratio)
            image_shape = X_train.shape[1:]
            model_quality = get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=NUM_CLASSES[dataset])

            optimizer = SGD(lr=0.01, decay=1e-4, momentum=0.9)
            model_quality.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
            epochs_init = 100
            batch_size = 128

            h_quality  =  model_quality.fit(X_train, y_train,
                                steps_per_epoch=X_train.shape[0]//batch_size, epochs=epochs_init, shuffle=False,
                                validation_data=(X_test, y_test)
                                )
            result_list.append(h_quality.history['val_accuracy'][-1])
        print("result_list: ",result_list)
        final_result_list.append(result_list)
