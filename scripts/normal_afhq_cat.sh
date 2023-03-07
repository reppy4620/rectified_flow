#!/bin/bash

config=../configs/normal/afhq_cat.yaml

python ../src/train.py \
    --config_file $config
