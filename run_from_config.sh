#! /bin/bash

STAGE=$1
CONFIG=$2
GPUID=$3

export CUDA_VISIBLE_DEVICES=$GPUID

if [ $STAGE = "train" ] || [ $STAGE = "tr" ]; then
    python run_from_config.py fit --config $CONFIG
elif [ $STAGE = "val" ] || [ $STAGE = "v" ]; then
    python run_from_config.py validate --config $CONFIG
elif [ $STAGE = "test" ] || [ $STAGE = "te" ]; then
    python run_from_config.py test --config $CONFIG
elif [ $STAGE = "pred" ] || [ $STAGE = "p" ]; then
    python run_from_config.py predict --config $CONFIG
else
    echo "Invalid stage"
fi