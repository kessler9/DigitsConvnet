#!/bin/bash
rm -rf MNIST-data
rm -rf MNIST-Model

stress --cpu 8 &
python MNIST_Convnet.py
python testall.py | grep -E "\{|\}"

killall stress
