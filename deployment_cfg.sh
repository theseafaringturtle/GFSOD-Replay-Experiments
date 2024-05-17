# This config is concerned with paths. Allows moving datasets over to another drive and specifying number of gpus in one file.

export VOC_BASE_SAVE_DIR="checkpoints/voc"
export COCO_BASE_SAVE_DIR="checkpoints/coco"

export IMAGENET_PRETRAIN="./ImageNetPretrained/MSRA/R-101.pkl"
export IMAGENET_PRETRAIN_TORCH="./ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth"

export DATASET_BASE_DIR="datasets"

export NUM_GPUS=1

export FINETUNE=true

export TRAINER="DeFRCNTrainer"
