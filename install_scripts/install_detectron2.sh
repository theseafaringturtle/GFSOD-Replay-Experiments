
# Use source rather than bash due to mamba activation 
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "Usage: source install_detectron2.sh , use source rather than separate execution"
    exit 1
fi

## Environment for TFA-like experiments only (TFA, DeFRCN, DiGeo)
mamba create -y -n tfa python=3.8 && mamba activate tfa
my_ip=$(curl ipinfo.io/ip)
country=$(curl ipinfo.io/$my_ip | jq -r '.country')

mamba create -y -n tfa python=3.8 && source ~/.bashrc && mamba activate tfa


# Install Pytorch. 
# If using detectron2 v0.4, use 1.8.2 (discontinued LTS version) to fix https://github.com/facebookresearch/detectron2/issues/2837 caused by https://github.com/pytorch/pytorch/issues/55027 
# Only alternatives are 1.7.1 which does not have determinism setting, and 1.9.0 which has to be compiled from source and has bugs
# pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# Alternatively, if installing latest detectron2 from source, just install torch 1.9.0, but GPU may not support latest arch
# pip install torch==1.9.0 torchvision==0.10.0 from CN mirror, but need to fix this with deterministic algos: https://github.com/pytorch/pytorch/issues/68525
# Best option is to use torch 1.11 or 1.13 and compile detectron2 from source. This script uses 1.13
# For 1.11, pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -i https://download.pytorch.org/whl/cu113
# Extra opt: Download https://download.pytorch.org/whl/lts/1.8/cu111/torch-1.8.2%2Bcu111-cp38-cp38-linux_x86_64.whl then upload to server

# Downgrade numpy to avoid deprecated alias errors (np.bool etc)
pip install numpy==1.23.1
# Downgrade PIL
pip3 install pillow==9.5.0

# Avoid recent bug with new setuptools compat https://github.com/aws-neuron/aws-neuron-sdk/issues/893
pip3 install setuptools==69.5.1

## Install Detectron2 from source: install fvcore, clone detectron2 repo
if [[ $country == "CN" ]]; then
# Dec 2022
pip install fvcore

mamba install pytorch=1.13.1=*cuda* torchvision=0.14.1 cudatoolkit=11.7 mkl==2024.0 -c pytorch # install from mamba, channel should be configured already since mamba was from mirror. MKL slightly downgraded due to https://github.com/pytorch/pytorch/issues/123097
pip install chardet
# May 2024
curl -O https://storage.googleapis.com/gfsod_exp_data_mirror/detectron2.zip && unzip detectron2.zip
else
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -i https://download.pytorch.org/whl/cu117
# Github version might include a few fixes
pip install git+https://github.com/facebookresearch/fvcore
git clone https://github.com/facebookresearch/detectron2.git
fi

# More dependencies
pip install opencv-python scikit-learn

cd detectron2
# Download determinism patch
curl -O https://storage.googleapis.com/gfsod_exp_data_mirror/roi_align_determinism.patch
# Apply patch to installed detectron. Easier to do on source install since it does not rely on python environment path.
patch -p1 <roi_align_determinism.patch
python -m pip install -e .
cd ..

# Install DeFRCN - omitted, now part of repo
# git clone https://github.com/theseafaringturtle/GFSOD-Replay-Experiments/
# cd DeFRCN

curl -O https://storage.googleapis.com/gfsod_exp_data_mirror/ImageNetPretrained.zip && unzip ImageNetPretrained.zip
curl -O https://storage.googleapis.com/gfsod_exp_data_mirror/model_base_voc.zip && unzip model_base_voc.zip
curl -O https://storage.googleapis.com/gfsod_exp_data_mirror/model_surgery_voc1.zip && unzip -o model_surgery_voc1.zip
curl -O https://storage.googleapis.com/gfsod_exp_data_mirror/model_surgery_voc2.zip && unzip -o model_surgery_voc2.zip
curl -O https://storage.googleapis.com/gfsod_exp_data_mirror/model_surgery_voc3.zip && unzip -o model_surgery_voc3.zip
curl -O  https://storage.googleapis.com/gfsod_exp_data_mirror/model_base_coco.zip && unzip model_base_coco.zip

# ln -s ~/datasets/ datasets

# Note: some server configs may require `export NCCL_SHM_DISABLE=1` (https://github.com/Lightning-AI/pytorch-lightning/issues/4420) and DATALOADER.NUM_WORKERS 0

# Note: for TFA, wget http://dl.yf.io/fs-det/models/voc/split1/base_model/model_final.pth
# Then python3 -m tools.ckpt_surgery --src1 model_final.pth --method randinit --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1
# python3 -m tools.train_net --num-gpus 4 --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1/model_reset_surgery.pth DATALOADER.NUM_WORKERS 0
# Had to remove prefetch_factor argument from defrcn/dataloader/build.py
