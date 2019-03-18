#!/bin/python
import numpy as np
import tensorflow as tf
from pprint import pprint
from tensorflow.contrib import learn
from run import cnn_model_fn
from scipy.misc import imread
import os
import sys

mnist_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir=os.popen("pwd").read().replace('\n', '') + "/MNIST_Model")

image = np.array([abs(x - 255.0) / 255.0 for x in imread(sys.argv[1], mode='L').reshape(784)]).astype('float32')

results = mnist_classifier.predict(image)

for result in results:
	if result['classes'] == int(sys.argv[1].split('.')[0]):
		print "%s PASSED" % sys.argv[1]
	else:
		print "%s FAILED" % sys.argv[1]
		pprint(result)
