#!/usr/bin/env bash

ESET=$1
PATIENT=$2
CYTO=$3
GRID=$4
MAP=$5
HYPERPARAM=$6
SAVE=$7
NUM_GENES=$8
GET_EDGE_WEIGHTS=$9

STARTTIME=`date +"%Y-%m-%d %T"`


python generation.py ${ESET} ${PATIENT} ${CYTO} ${GRID} ${MAP} ${HYPERPARAM} ${SAVE} $NUM_GENES ${GET_EDGE_WEIGHTS}
wait

if [ $GRID = "True" ]; then
    bash ./customScripts/run_custom_batch_${ESET}_${CYTO}.sh
elif [ $GRID = "False" ]; then
    bash ./customScripts/run_custom_${ESET}_${CYTO}.sh
fi


ENDTIME=`date +"%Y-%m-%d %T"`

echo Start Time is: ${STARTTIME}
echo End Time is: ${ENDTIME}