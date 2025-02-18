#!/bin/bash
BUCKET="t5-bucket-eur"
ACCOUNT="t_limisiewicz_gmail_com"

MODEL_NAME=$1 # myt5 or byt5
MODEL_SIZE=$2 # small, base or large
TASK=$3 # qa_in_lang, qa_cross_lang, ner or translation
TRAIN_STEPS=$4 # train steps should be decided based on the validation results
SPLIT=$5 # dev or test

BUCKET="t5-bucket-eur"
ACCOUNT="t_limisiewicz_gmail_com"


T5X_DIR="/home/${ACCOUNT}/t5x"
MODEL_DIR="gs://${BUCKET}/finetune_new/${MODEL_NAME}_${MODEL_SIZE}_${TASK}"
TSV_DATA_DIR="gs://${BUCKET}/data/xtreme_up/${TASK}"

if [ $TASK = "qa_in_lang" ] || [ $TASK = "qa_cross_lang" ]
then
  TASK_TYPE="qa_tasks"
elif [ $TASK = "ner" ]
then
  TASK_TYPE="ner_tasks"
elif [ $TASK = "transliteration" ]
then
  TASK_TYPE="transliteration_tasks"
elif [ $TASK = "semantic_parsing" ]
then
  TASK_TYPE="semantic_parsing_tasks"
elif [ $TASK = "autocomplete" ]
then
  TASK_TYPE="autocomplete_tasks"
elif [ $TASK = "retrieval_in_lang" ] || [ $TASK = "retrieval_cross_lang" ]
then
  TASK_TYPE="retrieval_tasks"
elif [ $TASK = "translation" ]
then
  TASK_TYPE="mt_tasks"
elif [ $TASK = "translation_selected" ] || [ $TASK = "translation_all_langs" ]
then
  TASK_TYPE="mt_tasks"
  MODEL_DIR="gs://${BUCKET}/finetune_new/${MODEL_NAME}_${MODEL_SIZE}_translation_selected"
  TSV_DATA_DIR="gs://${BUCKET}/data/xtreme_up/translation"
elif [ $TASK = "translation_en_te" ]
then
  TASK_TYPE="mt_tasks"
  MODEL_DIR="gs://${BUCKET}/finetune_new/${MODEL_NAME}_${MODEL_SIZE}_translation_en_te"
  TSV_DATA_DIR="gs://${BUCKET}/data/xtreme_up/translation"
elif [ $TASK = "xnli" ]
then
  TASK_TYPE="xnli_tasks"
fi

if [ $MODEL_NAME = "myt5" ]
then
  GIN_DIR="myt5"
elif [ $MODEL_NAME = "byt5" ]
then
  GIN_DIR="byt5_copy"
fi

python3 ${T5X_DIR}/t5x/eval.py \
  --gin_file=${GIN_DIR}/${MODEL_SIZE}_eval.gin \
  --gin_search_paths=${T5X_DIR}/xtreme_up/baseline \
  --gin_file=tasks_lib.gin \
  --gin.TASK_TSV_DATA_DIR=\'${TSV_DATA_DIR}\' \
  --gin.EVAL_OUTPUT_DIR=\'${MODEL_DIR}\' \
  --gin.CHECKPOINT_PATH=\'${MODEL_DIR}/checkpoint_${TRAIN_STEPS}\' \
  --gin.MIXTURE_OR_TASK_NAME=\'xtreme_up_${TASK}_${MODEL_NAME}\' \
  --gin.MIXTURE_OR_TASK_MODULE=\'xtreme_up.baseline.${TASK_TYPE}\' \
  --gin.utils.DatasetConfig.split=\'${SPLIT}\' \
  --gin.SPLIT=\'${SPLIT}\'
