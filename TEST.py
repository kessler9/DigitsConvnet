#!/bin/python
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from MNIST_Convnet import cnn_model_fn
from scipy.misc import imread
import os
import sys

mnist_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir=os.popen("pwd").read().replace('\n', '') + "/MNIST_Model")

image = np.array([abs(x - 255.0) / 255.0 for x in imread(sys.argv[1], mode='L').reshape(784)]).astype('float32')

results = mnist_classifier.predict(image)

for result in results: print result

