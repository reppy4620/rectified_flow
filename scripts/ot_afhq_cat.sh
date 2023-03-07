#!/bin/bash

config=../configs/ot/afhq_cat.yaml

python ../src/train.py \
    --config_file $config
