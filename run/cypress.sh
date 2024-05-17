#!/bin/bash

# Default values for parameters
ESET=""
PATIENT=""
CYTO="default_cyto_value"
GRID="default_grid_value"
MAP="default_map_value"
HYPERPARAM="default_hyperparam_value"
SAVE="False" 

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --eset)
      ESET="$2"
      shift # past argument
      shift # past value
      ;;
    --patient)
      PATIENT="$2"
      shift # past argument
      shift # past value
      ;;
    --cyto)
      CYTO="$2"
      shift # past argument
      shift # past value
      ;;
    --grid)
      GRID="$2"
      shift # past argument
      shift # past value
      ;;
    --map)
      MAP="$2"
      shift # past argument
      shift # past value
      ;;
    --hyperparam)
      HYPERPARAM="$2"
      shift # past argument
      shift # past value
      ;;
    --save)
      SAVE="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      echo "Unknown option $1."
      exit 1
      ;;
  esac
done

echo "ESET: $ESET"
echo "PATIENT: $PATIENT"
echo "CYTO: $CYTO"
echo "GRID: $GRID"
echo "MAP: $MAP"
echo "HYPERPARAM: $HYPERPARAM"
echo "SAVE: $SAVE"

# Continue with your script's logic here, using the variables ESET, PATIENT, etc.


STARTTIME=`date +"%Y-%m-%d %T"`



python generation.py ${ESET} ${PATIENT} ${CYTO} ${GRID} ${MAP} ${HYPERPARAM} ${SAVE} 
wait


if [ $GRID = "True" ]; then
    bash ./customScripts/run_custom_batch_${ESET}_${CYTO}.sh
elif [ $GRID = "False" ]; then
    bash ./customScripts/run_custom_${ESET}_${CYTO}.sh
fi


ENDTIME=`date +"%Y-%m-%d %T"`

echo Start Time is: ${STARTTIME}
echo End Time is: ${ENDTIME}