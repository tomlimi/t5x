#!/bin/bash

for size in "small" "base" "large"
do
  echo "Finetuning and inference for ${size} model"
  # for task in "qa_in_lang" "qa_cross_lang" "ner" "transliteration"
  for task in "qa_in_lang" "qa_cross_lang" "ner" "transliteration" "semantic_parsing" "autocomplete" "retrieval_in_lang" "retrieval_cross_lang"
  do
    bash ft_xtreme_up.sh "myt5" ${size} ${task}
    #bash infer_xtreme_up.sh "myt5" ${size} ${task}
  done
done