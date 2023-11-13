#!/bin/bash

LANG=$1

BUCKET=''
C4_LOCATION="${BUCKET}/data/c4"


cd ${C4_LOCATION} || exit
git lfs pull --include "multilingual/c4-${LANG}.*.json.gz"