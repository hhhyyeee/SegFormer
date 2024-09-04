#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29507}

#DEBUG
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_mim.py $CONFIG --wandb 1 --pid 2 --work-dir /ssd_data1/hyewon/SegFormer/work_dirs/project-2/20240605_120500 --launcher pytorch ${@:3}
