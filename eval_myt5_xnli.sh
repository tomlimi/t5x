# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.


EXPERIMENT="myt5_small"
CHECKPOINT_PATH="gs://${BUCKET}/models/${EXPERIMENT}"
EVAL_OUTPUT_DIR="gs://${BUCKET}/models/${EXPERIMENT}/eval"
T5X_DIR="/home/${ACCOUNT}/t5x"  # directory where the T5X repo is cloned.
TFDS_DATA_DIR="gs://${BUCKET}/data"

python3 ${T5X_DIR}/t5x/eval.py \
  --gin_file="${T5X_DIR}/t5x/examples/eval_myt5_xnli.gin" \
  --gin.EVAL_OUTPUT_DIR="'${EVAL_OUTPUT_DIR}'" \
  --gin.CHECKPOINT_PATH="'${CHECKPOINT_PATH}/checkpoint_100000'" \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --alsologtostderr