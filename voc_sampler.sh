#!/bin/bash

if (($# != 2)); then
  echo >&2 "Usage: bash voc_sampler.sh <exp_name> <class split> -s <sample pool size> -o <sample out size> -s <sampler class> -r <rng>"
  exit
fi

EXP_NAME=$1
VOC_CLASS_SPLIT=$2

SAMPLE_POOL_SIZE=100
SAMPLE_OUT_SIZE="1 5 10"
SAMPLER="AblationSampler"
RNGSEED=0

# Parse command line arguments
while getopts ":p:o:s:r:" opt; do
  case $opt in
  p) SAMPLE_POOL_SIZE="$OPTARG" ;;
  o) SAMPLE_OUT_SIZE="$OPTARG" ;;
  s) SAMPLER="$OPTARG" ;;
  r) RNGSEED="$OPTARG" ;;
  \?)
    echo "Invalid option: -$OPTARG" >&2
    exit 1
    ;;
  :)
    echo "Option -$OPTARG requires an argument." >&2
    exit 1
    ;;
  esac
done

for ((i = 0; i < 10; i++)); do
  python sampler.py --num-gpus 1 --config-file configs/voc/defrcn_det_r101_base${VOC_CLASS_SPLIT}.yaml \
    --sample_pool_size=$SAMPLE_POOL_SIZE --sample_out_size="$SAMPLE_OUT_SIZE" --sampler=$SAMPLER \
    --prev_dataseed $i --new_dataseed $((30 + i)) \
    --opts OUTPUT_DIR sampler_logs/voc/${EXP_NAME}/seed$((30 + i)) TEST.PCB_MODELPATH "./ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth" \
    SEED $RNGSEED
  if [ $? != 0 ]; then
    echo "Interrupting script due to last error" 1>&2
    break
  fi
done

ZIP_NAME="voc${VOC_CLASS_SPLIT}_${EXP_NAME}_pool${SAMPLE_POOL_SIZE}_s${SAMPLER}_rng${RNGSEED}"

zip -r "$ZIP_NAME.zip" datasets/vocsplit/seed3?
mkdir -p voc_splits
mv "$ZIP_NAME.zip" voc_splits
