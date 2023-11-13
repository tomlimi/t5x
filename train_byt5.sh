# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.


MODEL_DIR="gs://${BUCKET}/models"
T5X_DIR="/home/tomasz/t5x"  # directory where the T5X repo is cloned.
#TFDS_DATA_DIR="gs://allennlp-tensorflow-datasets/c4/multilingual/3.0.1?project=byt5-analysis"
TFDS_DATA_DIR="gs://${BUCKET}/data/c4/multilingual/3.1.0"

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="${T5X_DIR}/t5x/examples/pretrain_byt5_mc4.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --tfds_data_dir=${TFDS_DATA_DIR}