import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from MNIST_Convnet import cnn_model_fn
from scipy.misc import imread
import os
import sys

images = [i for i in os.popen("find $(pwd) -name '*.bmp' | sort").read().split('\n') if i]

mnist_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir=os.popen("pwd").read().replace('\n', '') + "/MNIST_Model")

for image in images:



	image = np.array([abs(x - 255.0) / 255.0 for x in imread(image, mode='L').reshape(784)]).astype('float32')

	results = mnist_classifier.predict(image)

	for result in results: print result
