#!/usr/bin/env python
# coding: utf-8

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from collections import Counter

def eval_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    #print(predictions)
    predictions = list(np.around(np.array(predictions),0))
    accuracy = accuracy_score(test_labels, predictions)
    #print(accuracy)
    return accuracy

def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    #print(current_class)
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

def inject_noise(n_classes, y_, noise_level):
    y = y_.copy()
    if noise_level > 100 or noise_level < 0:
        raise ValueError('Noise level can not be bigger than 100 or smaller than 0')

    noisy_idx = np.random.choice(len(y), int(len(y)*noise_level/100.0), replace = False)
    for i in noisy_idx:
        y[i] = other_class(n_classes, y[i])

    return y

# Load testing data
dataset = "thermostat-original"
test_labels = np.loadtxt("data/" + dataset + "-test-labels.txt")[0:6000]
test_data = np.loadtxt("data/" + dataset + "-test-data.txt")[0:6000]
# Load training data
train_labels = np.loadtxt("data/" + dataset + "-train-labels.txt")
train_data = np.loadtxt("data/" + dataset + "-train-data.txt")
n_classes = len(list(set(list(train_labels))))


print(Counter(list(train_labels)).keys()) # equals to list(set(words))
print(Counter(list(train_labels)).values()) # counts the elements' frequency


# KNN thermostat test, 10 time averaged results are showed in the end of output
# KNN thermostat
acc_list_knn = []
repeat = 10
for i in range(repeat):
    for noise in [0,10,20,30,40,50,60,70,80,90,100]:
        train_noisy_labels = inject_noise(n_classes, train_labels, noise)
        knnmodel = KNeighborsClassifier(n_neighbors=4, weights = 'distance')
        knnmodel.fit(train_data, train_noisy_labels)
        acc = eval_model(knnmodel, test_data, test_labels)
        acc_list_knn.append(acc)
        print("noise_level, accuracy", noise, acc)
avg_result = np.zeros(11)
for i in range(repeat):
    for j in range(11):
        avg_result[j] += acc_list_knn[j+i*11]
average = avg_result/repeat
print(average)


# NearestCentroid thermostat test, 10 time averaged results are showed in the end of output
# NearestCentroid thermostat
acc_list_nc = []
repeat = 10
for i in range(repeat):
    for noise in [0,10,20,30,40,50,60,70,80,90,100]:
        train_noisy_labels = inject_noise(n_classes, train_labels, noise)
        ncmodel = NearestCentroid()
        ncmodel.fit(train_data, train_noisy_labels)
        acc = eval_model(ncmodel, test_data, test_labels)
        acc_list_nc.append(acc)
        print("noise_level, accuracy", noise, acc)
avg_result = np.zeros(11)
for i in range(repeat):
    for j in range(11):
        avg_result[j] += acc_list_nc[j+i*11]
average = avg_result/repeat
print(average)


# MLP thermostat test, 10 time averaged results are showed in the end of output
# Fixed initialization of model, random noise
acc_list_mlp = []
repeat = 10
for i in range(repeat):
    for noise in [0,10,20,30,40,50,60,70,80,90,100]:
        train_noisy_labels = inject_noise(n_classes, train_labels, noise)
        mlpmodel = MLPClassifier(solver='adam', hidden_layer_sizes=(28,28), random_state=1)
        mlpmodel.fit(train_data, train_noisy_labels)
        acc = eval_model(mlpmodel, test_data, test_labels)
        acc_list_mlp.append(acc)
        print("noise_level, accuracy", noise, acc)
avg_result = np.zeros(11)
for i in range(repeat):
    for j in range(11):
        avg_result[j] += acc_list_mlp[j+i*11]
average = avg_result/repeat
print(average)


# Random initialization of model, fixed noise
acc_list_mlp = []
repeat = 10
for noise in [0,10,20,30,40,50,60,70,80,90,100]:
    train_noisy_labels = inject_noise(n_classes, train_labels, noise)
    for i in range(repeat):
        mlpmodel = MLPClassifier(solver='adam', hidden_layer_sizes=(28,28))
        mlpmodel.fit(train_data, train_noisy_labels)
        acc = eval_model(mlpmodel, test_data, test_labels)
        acc_list_mlp.append(acc)
        print("noise_level, accuracy", noise, acc)
avg_result = np.zeros(11)
for i in range(repeat):
    for j in range(11):
        avg_result[j] += acc_list_mlp[j+i*11]
average = avg_result/repeat
print(average)

# Random initialization of model, random noise
acc_list_mlp = []
repeat = 10
for i in range(repeat):
    for noise in [0,10,20,30,40,50,60,70,80,90,100]:
        train_noisy_labels = inject_noise(n_classes, train_labels, noise)
        mlpmodel = MLPClassifier(solver='adam', hidden_layer_sizes=(28,28))
        mlpmodel.fit(train_data, train_noisy_labels)
        acc = eval_model(mlpmodel, test_data, test_labels)
        acc_list_mlp.append(acc)
        print("noise_level, accuracy", noise, acc)
avg_result = np.zeros(11)
for i in range(repeat):
    for j in range(11):
        avg_result[j] += acc_list_mlp[j+i*11]
average = avg_result/repeat
print(average)
