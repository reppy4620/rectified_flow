#!/bin/bash

config=../configs/normal/mnist.yaml

python ../src/train.py \
    --config_file $config
