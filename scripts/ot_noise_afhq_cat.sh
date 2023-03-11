#!/bin/bash

config=../configs/ot_noise/afhq_cat.yaml

python ../src/train.py \
    --config_file $config
