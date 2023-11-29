# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.


EXPERIMENT="myt5_small"
CHECKPOINT_PATH="gs://${BUCKET}/models/${EXPERIMENT}"
EVAL_OUTPUT_DIR="gs://${BUCKET}/models/${EXPERIMENT}/eval"
T5X_DIR="/home/${ACCOUNT}/t5x"  # directory where the T5X repo is cloned.
TFDS_DATA_DIR="gs://${BUCKET}/data"

python3 ${T5X_DIR}/t5x/eval.py \
  --gin_file="${T5X_DIR}/t5x/examples/ft_myt5_xnli.gin" \
  --gin.MODEL_DIR="'${CHECKPOINT_PATH}/xnli_finetuned'" \
  --gin.INITIAL_CHECKPOINT_PATH="'${CHECKPOINT_PATH}/checkpoint_100000'" \
  --alsologtostderr