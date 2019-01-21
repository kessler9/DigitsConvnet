#!/bin/bash
./flush.sh &&
python MNIST_Convnet.py &&
for i in `seq 9`; do
    python TEST.py "${i}.bmp" > "${i}_result.log"
done