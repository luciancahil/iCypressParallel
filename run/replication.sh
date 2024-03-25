#!/usr/bin/env bash

ESET=$1
PATIENT=$2
CYTO=$3
GRID=$4
MODEL_PATH=$5


python generation.py ${ESET} ${PATIENT} ${CYTO} ${GRID}
wait

bash ./replications/run_replication_${ESET}_${CYTO}.sh $MODEL_PATH
