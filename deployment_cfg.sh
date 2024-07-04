# This config is concerned with paths. Allows moving datasets over to another drive and specifying number of gpus in one file.

export VOC_BASE_SAVE_DIR="checkpoints/voc"
export COCO_BASE_SAVE_DIR="checkpoints/coco"

export IMAGENET_PRETRAIN="./ImageNetPretrained/MSRA/R-101.pkl"
export IMAGENET_PRETRAIN_TORCH="./ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth"

export DATASET_BASE_DIR="datasets"

export NUM_GPUS=4

# Whether to use base-trained model_final.pth by the authors, to be put in ./model_base_voc or ./model_base_coco
export FINETUNE=true
# Whether to use provided model surgery, same folder as model_final.pth
export PROVIDED_RANDINIT=true

# Not recommended if you cannot spare a few hundred MB model for each seed and shot. Best tweak SHOT_LIST or SEED_SPLIT_LIST if using this
export KEEP_OUTPUTS=false
# if final, 1 2 3 5 10
export SHOT_LIST=(1 2 3 5 10)
# TFA-derived data splits for few-shot data
export DATA_SPLIT_LIST=(0 1 2 3 4 5 6 7 8 9)
# RNG seed to run the experiments with.
export RNG_SEED=0 # (1 2 3)

export GFSOD=true

export TRAINER="DeFRCNTrainer"
