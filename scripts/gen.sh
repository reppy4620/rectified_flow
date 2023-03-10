#!/bin/bash

root_dir=../out/ot/huggan/AFHQv2

python ../src/generate.py \
    --root_dir $root_dir \
    --output_dir $root_dir/gen
