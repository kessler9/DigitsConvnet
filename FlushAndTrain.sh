#!/bin/bash
echo "FLUSHING DATA"
rm -rf MNIST-data
rm -rf MNIST_Model
echo "BEGINNING TRAINING"
python MNIST_Convnet.py > /dev/null 
echo "MNIST_Convnet HAS BEEN TRAINED"
echo "EVALUATING RESULTS..."
echo `python testall.py` | grep -E "\{|\}"
echo "ANALYSIS COMPLETE; RESULTS YIELDED ABOVE ^^"
