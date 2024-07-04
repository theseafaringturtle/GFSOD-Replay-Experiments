#!/usr/bin/env bash

source ./deployment_cfg.sh

EXPNAME=$1

echo "Saving results to "$COCO_BASE_SAVE_DIR

SAVEDIR=$COCO_BASE_SAVE_DIR/${EXPNAME}

# ------------------------------- Base Pre-training, 60 classes ---------------------------------- #
if [[ $FINETUNE != true ]]
then
    echo "Performing base pretraining"
python3 main.py --num-gpus $NUM_GPUS --config-file configs/coco/defrcn_det_r101_base.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                         \
           OUTPUT_DIR ${SAVEDIR}/defrcn_det_r101_base
else
  echo "Skipping base pretraining"
  mkdir -p ${SAVEDIR}/defrcn_det_r101_base/
  cp ./model_base_coco/model_final.pth ${SAVEDIR}/defrcn_det_r101_base/model_final.pth
fi

# ----------------------------- Model Preparation --------------------------------- #

if [[ $PROVIDED_RANDINIT == true ]]
then
  # If you want to use same random initialisation for last layers as initial experiment
  if [[ $FINETUNE != true ]]; then echo "FINETUNE is not true, are you sure you want to use the pretrained surgery model?"; fi
  echo "Using provided model_reset_surgery"
  mkdir -p ${SAVEDIR}/defrcn_det_r101_base
  cp ./model_base_coco/model_reset_surgery.pth ${SAVEDIR}/defrcn_det_r101_base/model_reset_surgery.pth
else
  # Perform it yourself, though seed for randinit will not be the same as initial experiment
  echo "Creating model_reset_surgery.pth"
  python3 tools/model_surgery.py --dataset coco --method randinit                                \
      --src-path ${SAVEDIR}/defrcn_det_r101_base/model_final.pth                    \
      --save-dir ${SAVEDIR}/defrcn_det_r101_base
fi
BASE_WEIGHT=${SAVEDIR}/defrcn_det_r101_base/model_reset_surgery.pth


# ------------------------------ GFSOD Fine-tuning, 60+20 classes ------------------------------- #
# Run model over TFA-derived data splits (0-9). For more robust results, run with more than one seed for the RNG
if [[ $GFSOD == true ]]
then
for seed in "${DATA_SPLIT_LIST[@]}"
do
    for shot in "${SHOT_LIST[@]}"
    do
        python3 tools/create_config.py --dataset coco14 --config_root configs/coco     \
            --shot ${shot} --seed ${seed} --setting 'gfsod'
        CONFIG_PATH=configs/coco/defrcn_gfsod_r101_novel_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus $NUM_GPUS --config-file ${CONFIG_PATH}                      \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}               \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} SEED $RNG_SEED TRAINER $TRAINER
        if [[ $KEEP_OUTPUTS != true ]]; then
            rm ${CONFIG_PATH}
            rm ${OUTPUT_DIR}/model_final.pth
        fi
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like --shot-list "${SHOT_LIST[@]}"  # summarize all results
fi