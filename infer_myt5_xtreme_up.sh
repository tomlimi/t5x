#!/bin/bash


T5X_DIR="/home/${ACCOUNT}/t5x"
MODEL_NAME="myt5"
MODEL_SIZE="small"
TASK="qa_in_lang"
TASK_TYPE="qa_tasks"
TRAIN_STEPS=256500
MODEL_DIR="gs://${BUCKET}/finetune/${MODEL_NAME}_${MODEL_SIZE}_${TASK}"
TSV_DATA_DIR="gs://${BUCKET}/data/xtreme_up/${TASK}"

python3 ${T5X_DIR}/t5x/eval.py \
  --gin_file=myt5/${MODEL_SIZE}_eval.gin \
  --gin_search_paths=${T5X_DIR}/xtreme_up/baseline \
  --gin_file=tasks_lib.gin \
  --gin.TASK_TSV_DATA_DIR=\'${TSV_DATA_DIR}\' \
  --gin.EVAL_OUTPUT_DIR=\'${MODEL_DIR}\' \
  --gin.CHECKPOINT_PATH=\'${MODEL_DIR}/checkpoint_${TRAIN_STEPS}\' \
  --gin.MIXTURE_OR_TASK_NAME=\'xtreme_up_${TASK}_${MODEL_NAME}\' \
  --gin.MIXTURE_OR_TASK_MODULE=\'xtreme_up.baseline.${TASK_TYPE}\' \
  --gin.utils.DatasetConfig.split=\'test\' \
  --gin.SPLIT=\'test\'
