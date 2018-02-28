import tensorflow as tf
from tensorflow.python.client import device_lib
import random
import time
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #UNCOMMENT TO TURN OFF WARNINGS

N = int(sys.argv[1])
M = int(sys.argv[2])

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

#print get_available_gpus()

config = tf.ConfigProto(log_device_placement=True)
sess = tf.Session(config = config)

matrix1 = tf.constant([[float(x) for x in range(0, N)] for x in range(0, M)])
matrix2 = tf.constant([[float(x) for x in range(0, N)[::-1]] for x in range(0, M)[::-1]])

ts = time.time()

with tf.device('/' + sys.argv[3].lower() + ':0'):
	product = tf.matmul(matrix1, matrix2)

with sess:
	result = sess.run(product)
	print result

print sys.argv[3].upper() +  " MULTIPLIED 2 {0}x{1} MATRICIES IN {2} SECONDS".format(N, M, time.time() - ts)
