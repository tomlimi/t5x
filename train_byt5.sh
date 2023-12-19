# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
BUCKET="t5-bucket-eur" #
ACCOUNT=tomasz

EXPERIMENT="byt5_large"
MODEL_DIR="gs://${BUCKET}/models/${EXPERIMENT}"
T5X_DIR="/home/${ACCOUNT}/t5x"  # directory where the T5X repo is cloned.
TFDS_DATA_DIR="gs://${BUCKET}/data"

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="${T5X_DIR}/t5x/examples/pretrain_byt5_mc4.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --alsologtostderr