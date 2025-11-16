[//]: # (<div align="center"><img src="assets/header.png" width="840"></div>)

## Introduction

This repo contains the source for "Prototype Distance Ratio Sampling for Generalised Few Shot Object Detection"


## Installation

**Datasets**

To automatically set up mamba and download the datasets:
```shell
source install_scripts/install_base.sh
mamba create -y -n tfa python=3.8 && source ~/.bashrc && mamba activate tfa
```
Training and validation splits for COCO should be sent to the same directory named `trainval2014`, with `val2014` being just a symlink to `trainval2014`.

Reminder that VOC is 4.3 GB, COCO is 20 GB, so plan accordingly.

**Dependencies**

Using `install_scripts/install_detectrion2.sh` should do the job. If it doesn't work, here are the manual steps below:


Downgrade numpy to avoid deprecated alias errors (np.bool etc), downgrade PIL
```shell
 pip install numpy==1.23.1
pip install pillow==9.5.0
 ```

Avoid recent bug with new setuptools compat https://github.com/aws-neuron/aws-neuron-sdk/issues/893

```shell
pip3 install setuptools==69.5.1
```

Install pytorch, detecton2 and dependencies
```shell

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -i https://download.pytorch.org/whl/cu117
pip install git+https://github.com/facebookresearch/fvcore
git clone https://github.com/facebookresearch/detectron2.git
pip install opencv-python scikit-learn
````
(Optional) Download determinism patch by M.Yousef explained in `https://github.com/facebookresearch/detectron2/issues/4723`, will ensure reproducibility on the same machine
```shell
cd detectron2
# Download determinism patch
curl -O https://storage.googleapis.com/gfsod_exp_data_mirror/roi_align_determinism.patch
# Apply patch to installed detectron. Easier to do on source install since it does not rely on python environment path.
patch -p1 <roi_align_determinism.patch
```
Install detectron2. Note: there are no binaries available since FB stopped releasing them years ago, but the build system works pretty well since they have integrated detectron2 into another project.
```
python -m pip install -e .
cd ..
```

**Sampling**

To generate different base splits, run `voc_sampler.sh` and `coco_sampler.sh`. Novel annotations will be copied automatically from the ones used in previous works (splits 0-9) keep it fair.

By default, all generated splits will have different numbers from the old ones in use by other G-FSOD works (10-19 for COCO, 30-39 for VOC).
To change which splits you're running, edit `DATA_SPLIT_LIST` in `deployment_cfg.sh`

**Training and Evaluation**

 * This follows the same structure as DeFRCN with a few differences: the scripts will use pre-trained weights by default for the base classes, unless told otherwise in deployment.cfg

* To reproduce the results on VOC, `EXP_NAME` can be any string (e.g defrcn, or something) and `SPLIT_ID` must be `1 or 2 or 3` (we consider 3 random splits like other papers).
  ```
  bash run_voc.sh EXP_NAME SPLIT_ID (1, 2 or 3)
  ```
* To reproduce the results on COCO, `EXP_NAME` can be any string (e.g defrcn, or something) 
  ```
  bash run_coco.sh EXP_NAME
  ```
* Don\'t forget to change `IMAGENET_PRETRAIN*` to the path to your downloaded pretrained ImageNet weights.

If you have less than 24GB VRAM at your disposal, edit run_voc.sh and run_coco.sh to append the SOLVER.IMS_PER_BATCH option.

## Acknowledgements
This repo is developed based on [DeFRCN](https://github.com/er-muyue/DeFRCN), [TFA](https://github.com/ucbdrive/few-shot-object-detection) and [Detectron2](https://github.com/facebookresearch/detectron2).
The training was originally run on [ARCHIE-WeST](https://www.archie-west.ac.uk/).
