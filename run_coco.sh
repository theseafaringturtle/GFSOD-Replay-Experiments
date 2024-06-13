#!/usr/bin/env bash

source ./deployment_cfg.sh

EXPNAME=$1

echo "Saving results to "$COCO_BASE_SAVE_DIR

SAVEDIR=$COCO_BASE_SAVE_DIR/${EXPNAME}

# ------------------------------- Base Pre-train ---------------------------------- #
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
  python3 tools/model_surgery.py --dataset coco --method remove                         \
    --src-path ${SAVEDIR}/defrcn_det_r101_base/model_final.pth                        \
    --save-dir ${SAVEDIR}/defrcn_det_r101_base
fi
BASE_WEIGHT=${SAVEDIR}/defrcn_det_r101_base/model_reset_remove.pth


# ------------------------------ Novel Fine-tuning -------------------------------- #
# --> 1. FSRW-like, using seed0 aka default files from TFA. Only one repeat since this is now deterministic across runs.
if [[ $FSRW == true ]]
then
for repeat_id in 0
do
    for shot in 1 2 3 5 10 30
    do
        for seed in 0
        do
            python3 tools/create_config.py --dataset coco14 --config_root configs/coco \
                --shot ${shot} --seed ${seed} --setting 'fsod'
            CONFIG_PATH=configs/coco/defrcn_fsod_r101_novel_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVEDIR}/defrcn_fsod_r101_novel/fsrw-like/${shot}shot_seed${seed}_repeat${repeat_id}
            python3 main.py --num-gpus $NUM_GPUS --config-file ${CONFIG_PATH}                  \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}           \
                       TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} SEED ${seed} TRAINER $TRAINER
            rm ${CONFIG_PATH}
            rm ${OUTPUT_DIR}/model_final.pth
        done
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/defrcn_fsod_r101_novel/fsrw-like --shot-list 1 2 3 5 10 30  # summarize all results
fi

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


# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
if [[ $GFSOD == true ]]
then
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10 30
    do
        python3 tools/create_config.py --dataset coco14 --config_root configs/coco     \
            --shot ${shot} --seed ${seed} --setting 'gfsod'
        CONFIG_PATH=configs/coco/defrcn_gfsod_r101_novel_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus $NUM_GPUS --config-file ${CONFIG_PATH}                      \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}               \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} SEED ${seed} TRAINER $TRAINER
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like --shot-list 1 2 3 5 10 30  # surmarize all results
fi

# ------------------------------ Novel Fine-tuning ------------------------------- #  not necessary, just for the completeness of defrcn
# --> 3. TFA-like, i.e. run seed0~9 for robust results
if [[ $FSOD_TFA == true ]]
then
BASE_WEIGHT=${SAVEDIR}/defrcn_det_r101_base/model_reset_remove.pth
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10 30
    do
        python3 tools/create_config.py --dataset coco14 --config_root configs/coco     \
            --shot ${shot} --seed ${seed} --setting 'fsod'
        CONFIG_PATH=configs/coco/defrcn_fsod_r101_novel_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/defrcn_fsod_r101_novel/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus $NUM_GPUS --config-file ${CONFIG_PATH}                      \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}               \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} SEED ${seed} TRAINER $TRAINER
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/defrcn_fsod_r101_novel/tfa-like --shot-list 1 2 3 5 10 30  # surmarize all results
fi
echo "End"
