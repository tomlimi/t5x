#!/bin/bash
BUCKET="t5-bucket-eur"
ACCOUNT="t_limisiewicz_gmail_com"


T5X_DIR="/home/${ACCOUNT}/t5x"
MODEL_NAME="myt5"
MODEL_SIZE="small"
TASK="qa_in_lang"
TASK_TYPE="qa_tasks"
TRAIN_STEPS=256500
CHECKPOINT="gs://${BUCKET}/checkpoints/${MODEL_NAME}_${MODEL_SIZE}/checkpoint_250000"
MODEL_DIR="gs://${BUCKET}/finetune/${MODEL_NAME}_${MODEL_SIZE}_${TASK}"
TSV_DATA_DIR="gs://${BUCKET}/data/xtreme_up/${TASK}"

python3 ${T5X_DIR}/t5x/train.py \
  --gin.MODEL_DIR=\'${MODEL_DIR}\' \
  --gin.INITIAL_CHECKPOINT_PATH=\'${MODEL_DIR}/checkpoint_250000\' \
  --gin.TASK_TSV_DATA_DIR=\'${TSV_DATA_DIR}\' \
  --gin_search_paths=${T5X_DIR}/xtreme_up/baseline \
  --gin_file=myt5/${MODEL_SIZE}_finetune.gin \
  --gin_file=tasks_lib.gin \
  --gin.MIXTURE_OR_TASK_NAME=\'xtreme_up_${TASK}_${MODEL_NAME}\' \
  --gin.MIXTURE_OR_TASK_MODULE=\'xtreme_up.baseline.${TASK_TYPE}\' \
  --gin.USE_CACHED_TASKS=False \
  --gin.BATCH_SIZE=64 \
  --gin.TASK_FEATURE_LENGTHS=\{\'inputs\':\ 1024,\ \'targets\':\ 128\} \
  --gin.TRAIN_STEPS=${TRAIN_STEPS} \
  --gin.EVAL_PERIOD=100000 \
  --gin.JSON_WRITE_N_RESULTS=20 \
  --gin.train.train_eval_dataset_cfg=None \
  --gin.train.infer_eval_dataset_cfg=None \
  --gin.utils.SaveCheckpointConfig.period=100000 \
  --gin.utils.DatasetConfig.pack=False
