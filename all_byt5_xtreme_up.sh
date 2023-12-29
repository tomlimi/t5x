#!/bin/bash

for size in "small" "base" "large"
do
  echo "Finetuning and inference for ${size} model"
  for task in "qa_in_lang" "qa_cross_lang" "ner" "transliteration"
  do
    bash ft_xtreme_up.sh "by_t5" ${size} ${task}
    bash infer_xtreme_up.sh "by_t5" ${size} ${task}
  done
done