import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn
import time, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Turn off annoying warnings

DROPOUT_RATE = 0.875

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  input_layer = tf.reshape(features, [-1, 28, 28, 1])
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu6
  )
  pool1 = tf.layers.max_pooling2d(
      inputs=conv1,
      pool_size=[2, 2],
      strides=2
  )
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu6
  )
  pool2 = tf.layers.max_pooling2d(
      inputs=conv2,
      pool_size=[2, 2],
      strides=2
  )
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(
      inputs=pool2_flat,
      units=1024,
      activation=tf.nn.relu6
  )
  dropout = tf.layers.dropout(
      inputs=dense, 
      rate=DROPOUT_RATE, 
      training=mode == learn.ModeKeys.TRAIN
  )
  dense1 = tf.layers.dense(
      inputs=dropout,
      units=1024,
      activation=tf.nn.relu6
  )
  dropout1 = tf.layers.dropout(
      inputs=dense1, 
      rate=DROPOUT_RATE, 
      training=mode == learn.ModeKeys.TRAIN
  )
  logits = tf.layers.dense(inputs=dropout1, units=10)
  loss = None
  train_op = None
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
       onehot_labels=onehot_labels,
       logits=logits
    )
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.0125, 
        optimizer="SGD"
    )

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  return model_fn.ModelFnOps(
      mode=mode, 
      predictions=predictions, 
      loss=loss, 
      train_op=train_op
  )

def main(unused_argv):
  ts = time.time()
  mnist = learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images 
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images 
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, 
      model_dir= os.popen('pwd').read().replace('\n','') + "/MNIST_Model"
  )
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=50
  )
  mnist_classifier.fit(
    x=train_data,
    y=train_labels,
    batch_size=600,
    steps=8000,
    monitors=[logging_hook]
  )
  metrics = {
    "accuracy":
        learn.MetricSpec(
            metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }
  eval_results = mnist_classifier.evaluate(
      x=eval_data, 
      y=eval_labels, 
      metrics=metrics
  )
  print(eval_results)
  print("CONVNET TOOK {0}s TO TRAIN".format(time.time() - ts))

if __name__ == "__main__":
  tf.app.run()





  
      






