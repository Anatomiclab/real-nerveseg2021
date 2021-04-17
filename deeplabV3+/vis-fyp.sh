cd ..
# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
PQR_FOLDER="1211M1R"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/train"
DATASET="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/tfrecord"
PATH_TO_CHECKPOINT="${WORK_DIR}/${DATASET_DIR}/fyp/${EXP_FOLDER}/15000epoch-4batchsize-default-train-0.787iou-0.66loss-1:20-3class-xception-BNFalse-312"
PATH_TO_EVAL_DIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/eval"
PATH_TO_VIS_DIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/vis"

mkdir -p "${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/exp"
mkdir -p "${TRAIN_LOGDIR}"

python "${WORK_DIR}"/vis.py \
    --logtostderr \
    --vis_split="trainval"\
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --dataset="1211M1R" \
    --checkpoint_dir=${PATH_TO_CHECKPOINT} \
    --vis_logdir=${PATH_TO_VIS_DIR} \
    --dataset_dir=${DATASET}