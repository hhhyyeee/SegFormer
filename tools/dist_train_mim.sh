#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29507}

#DEBUG
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_mim.py $CONFIG --wandb 1 --pid 5 --work-dir ./work_dirs/project-5/20240730_180500/ --launcher pytorch ${@:3}

# # experiment
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train_mim.py $CONFIG --wandb 0 --pid 2 --work-dir ./work_dirs/project-B1/debug/ --launcher pytorch ${@:3}
