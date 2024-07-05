#!/usr/bin/env bash

# Set number of GPUs, training algo etc
source ./deployment_cfg.sh

EXP_NAME=$1
SPLIT_ID=$2

echo "Saving results to "$VOC_BASE_SAVE_DIR

SAVE_DIR=$VOC_BASE_SAVE_DIR/${EXP_NAME}

# ------------------------------- Base Pre-training, 15 classes ---------------------------------- #
if [[ $FINETUNE != true ]] ;
then
  echo "Performing base pretraining"
python3 main.py --num-gpus $NUM_GPUS --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
           OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
else
  echo "Skipping base pretraining"
  mkdir -p ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
  cp ./model_base_voc/model_final${SPLIT_ID}.pth ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth
fi

# ----------------------------- Model Preparation --------------------------------- #
if [[ $PROVIDED_RANDINIT == true ]]
then
  # If you want to use same random initialisation for last layers as initial experiment
  if [[ $FINETUNE != true ]]; then echo "FINETUNE is not true, are you sure you want to use the pretrained surgery model?"; fi
  echo "Using provided model_reset_surgery"
  mkdir -p ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
  cp ./model_base_voc/model_reset_surgery${SPLIT_ID}.pth ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_surgery.pth
else
  # Perform it yourself, though weight initialisation random seed for the classifier will not be the same as original experiment
  echo "Creating model_reset_surgery.pth"
  python3 tools/model_surgery.py --dataset voc --method randinit                                \
      --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                    \
      --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
fi
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_surgery.pth


# ------------------------------ G-FSOD novel Fine-tuning, 15+5 classes ------------------------------- #
# Run model over TFA-derived data splits (0-9). For more robust results, run with more than one seed for the RNG
if [[ $GFSOD == true ]]
then
for seed in "${DATA_SPLIT_LIST[@]}"
do
    for shot in "${SHOT_LIST[@]}"
    do
        python3 tools/create_config.py --dataset voc --config_root configs/voc               \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus $NUM_GPUS --config-file ${CONFIG_PATH}                            \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} SEED $RNG_SEED TRAINER $TRAINER
        if [[ $KEEP_OUTPUTS != true ]]; then
            rm ${CONFIG_PATH}
            rm ${OUTPUT_DIR}/model_final.pth
        fi
    done
done
python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like --shot-list "${SHOT_LIST[@]}"  # summarize all results
fi
