#!/bin/python3
import numpy as np
import tensorflow as tf
from pprint import pprint
from tensorflow.contrib import learn
from run import cnn_model_fn
from scipy.misc import imread
from glob import glob
import os

file_names = glob('*.bmp')
file_names.sort()
for file_name in file_names:
	mnist_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir=os.popen("pwd").read().replace('\n', '') + "/MNIST_Model")

	image = np.array([abs(x - 255.0) / 255.0 for x in imread(file_name, mode='L').reshape(784)]).astype('float32')

	results = mnist_classifier.predict(image)

	for result in results:
		if result['classes'] == int(file_name.split('.')[0]):
			print("%s PASSED" % file_name)
		else:
			print("%s FAILED" % file_name)
			pprint(result)

