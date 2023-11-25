# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.


EXPERIMENT="byt5_small"
CHECKPOINT_PATH="gs://${BUCKET}/models/${EXPERIMENT}"
EVAL_OUTPUT_DIR="gs://${BUCKET}/eval/${EXPERIMENT}/results"
T5X_DIR="/home/${ACCOUNT}/t5x"  # directory where the T5X repo is cloned.
TFDS_DATA_DIR="gs://${BUCKET}/data"

python3 ${T5X_DIR}/eval.py \
  --gin_file="${T5X_DIR}/t5x/examples/eval_byt5_mc4.gin" \
  --gin.EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
  --gin.CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --alsologtostderr