#!/usr/bin/env bash

NODE_FILE=$1
EDGE_FILE=$2
NAME=$3
GRID=$4
HYPERPARAM=$5
SAVE=$6

STARTTIME=`date +"%Y-%m-%d %T"`



#python generate-node.py ${NODE_FILE} ${EDGE_FILE} ${NAME} ${GRID}${HYPERPARAM} ${SAVE}
python generate-node.py ${NODE_FILE} ${EDGE_FILE} ${NAME} ${GRID} ${HYPERPARAM} ${SAVE}
wait

if [ $GRID = "TRUE" ]; then
    echo "Here!!"
    bash ./customScripts/run_custom_batch_${NODE_FILE}_${NAME}.sh
elif [ $GRID = "FALSE" ]; then
    bash ./customScripts/run_custom_${NODE_FILE}_${NAME}.sh
fi


ENDTIME=`date +"%Y-%m-%d %T"`

echo Start Time is: ${STARTTIME}
echo End Time is: ${ENDTIME}