#!/bin/bash

task=$1

for size in "base" "small" "large"
do
  echo "Finetuning and inference for ${size} model"
  # for task in "qa_in_lang" "qa_cross_lang" "ner" "transliteration"
  # for task in "semantic_parsing" "ner"
  # do
  bash ft_xtreme_up.sh "myt5" ${size} ${task}
  # done
done