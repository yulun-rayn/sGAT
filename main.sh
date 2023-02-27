#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate sgat-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name train-NSP15-dock"
PYARGS="$PYARGS --gpu 0" #PYARGS="$PYARGS --use_cpu"
PYARGS="$PYARGS --workers 12"
PYARGS="$PYARGS --data_path $DATA/datasets/NSP15_6W01_A_3_H.negonly_unique_300k.csv" # zinc_plogp_sorted.csv
PYARGS="$PYARGS --artifact_path $DATA/artifact/sgat"
PYARGS="$PYARGS --use_3d"

python main.py $PYARGS
